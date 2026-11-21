"""
Causal AI for Counterfactual Decision Intelligence  (Persevex project)
=======================================================================
ONE command runs everything:   python run_project.py

What it does (matches the 6 workflow steps of the project):
  1. Draws the causal diagram (DAG)
  2. Makes fake subscriber data (with hidden "answer key") + data dictionary
  3. Checks overlap with propensity scores
  4. Estimates the average effect (doubly robust, cross-fitted)
  5. Estimates uplift per customer (T-learner, DR-learner, Causal Forest)
  6. Turns uplift into decisions (20% budget, manual review for weak overlap)
  7. Compares strategies (uplift vs random vs churn-risk vs no offer), many seeds
  8. Writes a report with all the numbers

Everything lands in the folder  outputs/
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")                      # draw charts without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from econml.dml import CausalForestDML
from econml.dr import DRLearner

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# SETTINGS  (you can change these, then re-run)
# ----------------------------------------------------------------------
SEED = 42              # fixed random seed => same results every run
N = 8000               # number of fake subscribers
BUDGET = 0.20          # contact at most 20% of eligible subscribers
RENEWAL_MARGIN = 100.0 # money earned if one customer renews (e.g. Rs 100)
OFFER_COST = 10.0      # cost of sending one offer (e.g. Rs 10)
OVERLAP_LOW, OVERLAP_HIGH = 0.05, 0.95   # propensity outside this = weak overlap
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

FEATURES = ["tenure_months", "plan_tier", "recent_usage", "support_tickets", "late_payments"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------
# STEP A: make the fake data
# ----------------------------------------------------------------------
def make_data(n, seed, scenario="normal"):
    """Fake subscribers. Includes hidden truth (p0_true, p1_true) for evaluation only."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 61, n)                         # months subscribed
    plan = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])      # 0 basic, 1 standard, 2 premium
    usage = np.clip(rng.normal(50 + 8 * plan, 20, n), 0, 100)  # recent usage score
    tickets = rng.poisson(0.6, n)                           # support tickets (last 90 days)
    late = rng.poisson(0.4, n)                              # late payments

    low_usage = (50 - usage) / 20                           # bigger = lower usage

    # Baseline renewal probability WITHOUT offer. Low usage / tickets / late payments => lower.
    p0 = sigmoid(0.6 + 0.02 * tenure + 0.03 * (usage - 50) - 0.35 * tickets - 0.4 * late + 0.2 * plan)

    # TRUE offer effect (uplift) differs by customer type:
    #   strong uplift : low usage, shorter tenure   (persuadables)
    #   little uplift : middle customers
    #   negative uplift: loyal high-usage premium customers (offer annoys them / sleeping dogs)
    tau = 0.05 + 0.10 * np.clip(low_usage, -1, 2) - 0.002 * (tenure - 30)
    tau = np.where((usage > 70) & (plan == 2), -0.06, tau)          # sleeping dogs
    tau = np.where((tickets >= 2) & (late >= 1), 0.00, tau)        # lost causes: high risk, offer does nothing
    tau = np.clip(tau, -0.10, 0.35)
    p1 = np.clip(p0 + tau, 0.01, 0.99)

    # HISTORICAL POLICY: offers sent more to low-usage / recent support issues (this creates confounding!)
    slope = 1.0 if scenario == "normal" else 3.0           # weak-overlap scenario = extreme policy
    logit_t = -1.0 + slope * (0.9 * low_usage + 0.6 * np.minimum(tickets, 3) - 0.3)
    e_true = sigmoid(logit_t)
    treated = rng.binomial(1, e_true)

    # Observed outcome: only ONE of the two potential outcomes is seen
    renewed = rng.binomial(1, np.where(treated == 1, p1, p0))

    df = pd.DataFrame({
        "subscriber_id": np.arange(1, n + 1),
        "tenure_months": tenure, "plan_tier": plan, "recent_usage": usage.round(1),
        "support_tickets": tickets, "late_payments": late,
        "offer_sent": treated, "renewed_30d": renewed,
        "p0_true": p0, "p1_true": p1, "true_uplift": p1 - p0,   # EVALUATION ONLY
    })
    return df


DATA_DICTIONARY = """column,type,role,description
subscriber_id,int,id,Unique id of the subscriber
tenure_months,int,pre-treatment feature,Months the person has been subscribed
plan_tier,int,pre-treatment feature,0=basic 1=standard 2=premium
recent_usage,float,pre-treatment feature,Product usage score 0-100 in recent weeks
support_tickets,int,pre-treatment feature,Support tickets in last 90 days
late_payments,int,pre-treatment feature,Late payments in payment history
offer_sent,int,TREATMENT,1 if a retention offer was sent else 0 (opens/redemptions deliberately NOT included)
renewed_30d,int,OUTCOME,1 if renewed within 30 days of the decision date (same window for both groups)
p0_true,float,EVALUATION-ONLY ground truth,True renewal probability if NO offer sent
p1_true,float,EVALUATION-ONLY ground truth,True renewal probability if offer SENT
true_uplift,float,EVALUATION-ONLY ground truth,p1_true - p0_true
"""


# ----------------------------------------------------------------------
# STEP B: causal diagram
# ----------------------------------------------------------------------
def draw_dag(path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    pos = {"Customer features\n(tenure, plan, usage,\ntickets, payments)": (0.5, 0.85),
           "Offer sent\n(treatment)": (0.12, 0.2),
           "Renewed in 30 days\n(outcome)": (0.88, 0.2)}
    for name, (x, y) in pos.items():
        ax.text(x, y, name, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="#dbeafe", ec="#1d4ed8"))
    keys = list(pos.keys())
    arrows = [(keys[0], keys[1]), (keys[0], keys[2]), (keys[1], keys[2])]
    for a, b in arrows:
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="->", lw=2, shrinkA=38, shrinkB=38))
    ax.text(0.5, 0.02, "NOT in the model: offer opened / redeemed (they happen AFTER the decision)",
            ha="center", fontsize=9, color="crimson")
    ax.set_title("Causal diagram (DAG)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# STEP C: propensity + overlap
# ----------------------------------------------------------------------
def fit_propensity(df):
    X, t = df[FEATURES], df["offer_sent"]
    model = LogisticRegression(max_iter=1000)
    ps = cross_val_predict(model, (X - X.mean()) / X.std(), t, cv=5, method="predict_proba")[:, 1]
    return ps


def plot_overlap(df, ps, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ps[df["offer_sent"] == 1], bins=30, alpha=0.6, label="Got offer")
    ax.hist(ps[df["offer_sent"] == 0], bins=30, alpha=0.6, label="No offer")
    ax.axvline(OVERLAP_LOW, color="red", ls="--")
    ax.axvline(OVERLAP_HIGH, color="red", ls="--")
    ax.set_xlabel("Propensity score (chance of getting an offer)")
    ax.set_ylabel("Number of customers")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# STEP D: average effect (doubly robust / AIPW with cross-fitting)
# ----------------------------------------------------------------------
def doubly_robust_ate(df, seed=0):
    X = df[FEATURES].values
    t = df["offer_sent"].values
    y = df["renewed_30d"].values
    n = len(df)
    ps = np.zeros(n)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):   # cross-fitting
        Xs = (X - X[tr].mean(0)) / X[tr].std(0)
        ps[te] = LogisticRegression(max_iter=1000).fit(Xs[tr], t[tr]).predict_proba(Xs[te])[:, 1]
        for arm, store in ((0, mu0), (1, mu1)):
            idx = tr[t[tr] == arm]
            m = RandomForestRegressor(150, min_samples_leaf=20, random_state=seed, n_jobs=-1)
            store[te] = m.fit(X[idx], y[idx]).predict(X[te])
    ps = np.clip(ps, 0.02, 0.98)
    psi = mu1 - mu0 + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    return ate, ate - 1.96 * se, ate + 1.96 * se


# ----------------------------------------------------------------------
# STEP E: uplift models
# ----------------------------------------------------------------------
class TLearner:
    """Two models: one for offered customers, one for not-offered. Uplift = difference."""
    def __init__(self, seed=0):
        mk = lambda: RandomForestRegressor(200, min_samples_leaf=25, random_state=seed, n_jobs=-1)
        self.m0, self.m1 = mk(), mk()

    def fit(self, df):
        X, t, y = df[FEATURES].values, df["offer_sent"].values, df["renewed_30d"].values
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def predict_both(self, df):
        X = df[FEATURES].values
        return self.m0.predict(X), self.m1.predict(X)

    def uplift(self, df):
        a, b = self.predict_both(df)
        return b - a


def fit_dr_learner(df, seed=0):
    m = DRLearner(
        model_propensity=LogisticRegression(max_iter=1000),
        model_regression=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
        model_final=GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=seed),
        cv=5, random_state=seed)
    m.fit(df["renewed_30d"].values, df["offer_sent"].values, X=df[FEATURES].values)
    return m


def fit_causal_forest(df, seed=0):
    m = CausalForestDML(
        model_y=RandomForestRegressor(100, min_samples_leaf=20, random_state=seed, n_jobs=-1),
        model_t=RandomForestClassifier(100, min_samples_leaf=20, random_state=seed, n_jobs=-1),
        discrete_treatment=True, n_estimators=300, min_samples_leaf=20, cv=3, random_state=seed)
    m.fit(df["renewed_30d"].values, df["offer_sent"].values, X=df[FEATURES].values)
    return m


# ----------------------------------------------------------------------
# STEP F: decisions & policy evaluation
# ----------------------------------------------------------------------
def pick_top(score, budget, require_positive=None):
    """Pick the top `budget` fraction by score. Optionally only those where require_positive is True."""
    k = int(round(budget * len(score)))
    s = pd.Series(score).copy()
    if require_positive is not None:
        s[~np.asarray(require_positive)] = -np.inf
    order = s.sort_values(ascending=False).index[:k]
    chosen = np.zeros(len(score), dtype=bool)
    chosen[order] = np.isfinite(s.loc[order].values)
    return chosen


def evaluate(chosen, test, margin, cost):
    """Use the hidden truth to count extra renewals and money."""
    contacted = chosen.sum()
    inc = test.loc[chosen, "true_uplift"].sum()
    net = inc * margin - contacted * cost
    n = len(test)
    return dict(contacted=int(contacted),
                incremental_renewals_per_1000=inc / n * 1000,
                net_value_per_1000=net / n * 1000)


def compute_scores(train, test, seed):
    """Fit the uplift model (causal forest) and the churn-risk model once, return scores for test rows."""
    cf = fit_causal_forest(train, seed)
    uplift = cf.effect(test[FEATURES].values)
    # churn-risk model: predicts renewal from features ignoring the offer -> target lowest renewal chance
    churn_model = RandomForestClassifier(200, min_samples_leaf=25, random_state=seed, n_jobs=-1)
    churn_model.fit(train[FEATURES].values, train["renewed_30d"].values)
    churn_risk = 1 - churn_model.predict_proba(test[FEATURES].values)[:, 1]
    return uplift, churn_risk


def run_strategies(uplift, churn_risk, test, budget, margin, cost, seed):
    rng = np.random.default_rng(seed)
    value = uplift * margin - cost
    chosen = {
        "Uplift targeting": pick_top(value, budget, require_positive=value > 0),
        "Churn-risk targeting": pick_top(churn_risk, budget),
        "Random targeting": pick_top(rng.random(len(test)), budget),
        "No offer": np.zeros(len(test), dtype=bool),
    }
    return {name: evaluate(c, test, margin, cost) for name, c in chosen.items()}


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    report = []
    log = lambda s="": (print(s), report.append(s))

    # ---- 1. DAG
    draw_dag(f"{OUT}/01_causal_diagram.png")

    # ---- 2. Data
    df = make_data(N, SEED, "normal")
    df.to_csv(f"{OUT}/synthetic_subscribers.csv", index=False)
    with open(f"{OUT}/data_dictionary.csv", "w") as f:
        f.write(DATA_DICTIONARY)

    log("# Causal AI for Counterfactual Decision Intelligence - Results\n")
    log(f"Settings: seed={SEED}, subscribers={N}, budget={BUDGET:.0%}, "
        f"renewal margin={RENEWAL_MARGIN}, offer cost={OFFER_COST}\n")

    # ---- 3. Explore the history (the trap)
    raw = df.groupby("offer_sent")["renewed_30d"].mean()
    log("## 1. Raw comparison (the trap)")
    log(f"- Renewal rate WITH offer:    {raw[1]:.1%}")
    log(f"- Renewal rate WITHOUT offer: {raw[0]:.1%}")
    naive = raw[1] - raw[0]
    log(f"- Naive difference: {naive*100:+.1f} percentage points (misleading: offers went to low-usage / "
        f"support-issue customers who renew less anyway)")
    feat_means = df.groupby("offer_sent")[["recent_usage", "support_tickets", "tenure_months"]].mean().round(2)
    log("\nAverage features by group:\n")
    log(feat_means.to_markdown())

    # ---- 4. Overlap
    ps = fit_propensity(df)
    plot_overlap(df, ps, f"{OUT}/02_overlap_normal.png", "Overlap - normal scenario")
    weak = (ps < OVERLAP_LOW) | (ps > OVERLAP_HIGH)
    log(f"\n## 2. Overlap check\n- Customers with weak overlap (propensity outside "
        f"{OVERLAP_LOW}-{OVERLAP_HIGH}): {weak.sum()} of {len(df)} ({weak.mean():.1%})")

    # ---- 5. Average effect
    ate, lo, hi = doubly_robust_ate(df, SEED)
    true_ate = df["true_uplift"].mean()
    log("\n## 3. Average effect of sending an offer (percentage points)")
    log(f"- Naive (unadjusted):        {naive*100:+.2f}")
    log(f"- Doubly robust (cross-fit): {ate*100:+.2f}   (95% CI {lo*100:+.2f} to {hi*100:+.2f})")
    log(f"- TRUE average effect:       {true_ate*100:+.2f}")
    log(f"- Error of DR estimate: {abs(ate-true_ate)*100:.2f} pp, error of naive: {abs(naive-true_ate)*100:.2f} pp")

    # ---- 6. Individual uplift models
    train, test = train_test_split(df, test_size=0.4, random_state=SEED)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    tl = TLearner(SEED).fit(train)
    dr = fit_dr_learner(train, SEED)
    cf = fit_causal_forest(train, SEED)
    Xte = test[FEATURES].values
    up = {"T-learner": tl.uplift(test), "DR-learner": dr.effect(Xte), "Causal forest": cf.effect(Xte)}
    log("\n## 4. How well does each model find uplift? (on held-out customers)")
    rows = []
    for name, u in up.items():
        corr = np.corrcoef(u, test["true_uplift"])[0, 1]
        rmse = np.sqrt(np.mean((u - test["true_uplift"]) ** 2))
        rows.append({"model": name, "correlation_with_truth": round(corr, 3), "RMSE_pp": round(rmse * 100, 2)})
    log(pd.DataFrame(rows).to_markdown(index=False))

    # uplift by score band (decile groups, using the best-performing model for the decision = DR-learner)
    test["uplift_est"] = up["Causal forest"]
    test["band"] = pd.qcut(test["uplift_est"].rank(method="first"), 5,
                           labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    band = test.groupby("band", observed=True).agg(est_uplift_pp=("uplift_est", lambda s: s.mean() * 100),
                                                    true_uplift_pp=("true_uplift", lambda s: s.mean() * 100)).round(2)
    log("\nUplift by score band (Causal forest = our decision model) - estimate should rise together with the truth:\n")
    log(band.to_markdown())
    fig, ax = plt.subplots(figsize=(7, 4))
    band.plot(kind="bar", ax=ax)
    ax.set_ylabel("Uplift (percentage points)")
    ax.set_title("Uplift by score band (estimated vs true)")
    plt.xticks(rotation=0)
    fig.savefig(f"{OUT}/03_uplift_by_band.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 7. Representative customer profiles
    ps_model = LogisticRegression(max_iter=1000)
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    ps_model.fit((train[FEATURES] - mu) / sd, train["offer_sent"])
    profiles = pd.DataFrame([
        [6, 0, 20, 3, 1], [12, 0, 35, 1, 0], [30, 1, 50, 0, 0],
        [48, 2, 85, 0, 0], [60, 2, 90, 0, 0], [3, 0, 10, 4, 2],
    ], columns=FEATURES)
    profiles.insert(0, "profile", ["New, low usage, many tickets", "1yr basic, low usage", "Average customer",
                                   "Loyal premium, high usage", "5yr premium, very high usage",
                                   "Brand new, very low usage, tickets+late"])
    a, b = tl.predict_both(profiles)
    p_prof = ps_model.predict_proba((profiles[FEATURES] - mu) / sd)[:, 1]
    profiles["renew_prob_no_offer"] = a.round(3)
    profiles["renew_prob_with_offer"] = b.round(3)
    profiles["est_uplift_pp"] = ((b - a) * 100).round(1)
    profiles["propensity"] = p_prof.round(3)
    profiles["overlap_ok"] = (p_prof >= OVERLAP_LOW) & (p_prof <= OVERLAP_HIGH)
    log("\n## 5. Representative customer profiles (estimates for SIMILAR customers, not verified individual outcomes)\n")
    log(profiles.drop(columns=FEATURES).to_markdown(index=False))
    profiles.to_csv(f"{OUT}/representative_profiles.csv", index=False)

    # ---- 8. Decision export
    ps_test = ps_model.predict_proba((test[FEATURES] - mu) / sd)[:, 1]
    weak_test = (ps_test < OVERLAP_LOW) | (ps_test > OVERLAP_HIGH)
    value = test["uplift_est"].values * RENEWAL_MARGIN - OFFER_COST
    chosen = pick_top(value, BUDGET, require_positive=(value > 0) & ~weak_test)
    action = np.where(weak_test, "MANUAL REVIEW", np.where(chosen, "SEND OFFER", "NO OFFER"))
    reason = np.where(weak_test, "Weak overlap: too few comparable offered/not-offered customers",
             np.where(chosen, "Positive expected value and inside the 20% contact limit",
             np.where(value <= 0, "Expected value <= 0 (uplift too small for the cost)",
                      "Positive value but outside the 20% contact limit")))
    decisions = pd.DataFrame({"subscriber_id": test["subscriber_id"],
                              "estimated_uplift": test["uplift_est"].round(4),
                              "expected_incremental_value": value.round(2),
                              "recommended_action": action, "decision_reason": reason})
    decisions.to_csv(f"{OUT}/decision_export.csv", index=False)
    log("\n## 6. Decisions on held-out customers")
    log(decisions["recommended_action"].value_counts().to_frame("count").to_markdown())
    log(f"\n- Routed to manual review (weak overlap): {int(weak_test.sum())}")

    # ---- 9. Policy evaluation on held-out set (main seed)
    churn_main = None
    uplift_main = test["uplift_est"].values
    cm0 = RandomForestClassifier(200, min_samples_leaf=25, random_state=SEED, n_jobs=-1)
    cm0.fit(train[FEATURES].values, train["renewed_30d"].values)
    churn_main = 1 - cm0.predict_proba(test[FEATURES].values)[:, 1]
    res = run_strategies(uplift_main, churn_main, test, BUDGET, RENEWAL_MARGIN, OFFER_COST, SEED)
    log("\n## 7. Policy comparison on held-out set (same 20% limit for every strategy)")
    log(pd.DataFrame(res).T.round(2).to_markdown())

    # ---- 10. Repeat across seeds
    seeds = [1, 2, 3, 4, 5]
    allres = []
    for s in seeds:
        d = make_data(N, s)
        tr, te = train_test_split(d, test_size=0.4, random_state=s)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        u_s, c_s = compute_scores(tr, te, s)
        r = run_strategies(u_s, c_s, te, BUDGET, RENEWAL_MARGIN, OFFER_COST, s)
        for name, v in r.items():
            allres.append({"seed": s, "strategy": name, **v})
    allres = pd.DataFrame(allres)
    summ = allres.groupby("strategy")[["incremental_renewals_per_1000", "net_value_per_1000"]].agg(["mean", "std"]).round(2)
    summ.columns = ["_".join(c) for c in summ.columns]
    log("\n## 8. Repeated over 5 seeds (mean and std)")
    log(summ.to_markdown())
    allres.to_csv(f"{OUT}/policy_results_by_seed.csv", index=False)

    means = allres.groupby("strategy")[["net_value_per_1000"]].mean()["net_value_per_1000"]
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Uplift targeting", "Churn-risk targeting", "Random targeting", "No offer"]
    ax.bar(order, [means[o] for o in order], color=["#16a34a", "#f59e0b", "#94a3b8", "#cbd5e1"],
           yerr=[allres[allres.strategy == o]["net_value_per_1000"].std() for o in order], capsize=4)
    ax.set_ylabel("Net value per 1,000 subscribers")
    ax.set_title("Policy comparison (mean of 5 seeds)")
    plt.xticks(rotation=15)
    fig.savefig(f"{OUT}/04_policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 11. Sensitivity to cost and budget (single seed, fast)
    sens = []
    for cost in [5, 10, 20, 40]:
        for budget in [0.05, 0.10, 0.20, 0.30]:
            r = run_strategies(uplift_main, churn_main, test, budget, RENEWAL_MARGIN, cost, SEED)
            best = max(r, key=lambda k: r[k]["net_value_per_1000"])
            sens.append({"offer_cost": cost, "budget": f"{budget:.0%}", "best_strategy": best,
                         "uplift_net_value": round(r["Uplift targeting"]["net_value_per_1000"], 1),
                         "churn_net_value": round(r["Churn-risk targeting"]["net_value_per_1000"], 1),
                         "random_net_value": round(r["Random targeting"]["net_value_per_1000"], 1)})
    sens = pd.DataFrame(sens)
    sens.to_csv(f"{OUT}/sensitivity_cost_budget.csv", index=False)
    log("\n## 9. Sensitivity: how the ranking changes with offer cost and budget (net value per 1,000)")
    log(sens.to_markdown(index=False))

    # ---- 12. Weak-overlap scenario
    dw = make_data(N, SEED, "weak_overlap")
    psw = fit_propensity(dw)
    plot_overlap(dw, psw, f"{OUT}/05_overlap_weak_scenario.png", "Overlap - weak-overlap scenario")
    weak_w = (psw < OVERLAP_LOW) | (psw > OVERLAP_HIGH)
    log("\n## 10. Weak-overlap scenario (second scenario, extreme historical policy)")
    log(f"- Customers routed to manual review: {weak_w.sum()} of {len(dw)} ({weak_w.mean():.1%}) "
        f"vs {weak.sum()} ({weak.mean():.1%}) in the normal scenario")

    # ---- 13. Where churn-risk and uplift disagree
    u, cr = uplift_main, churn_main
    k = int(BUDGET * len(test))
    top_u = set(np.argsort(-u)[:k])
    top_c = set(np.argsort(-cr)[:k])
    overlap_share = len(top_u & top_c) / k
    only_c = test.iloc[list(top_c - top_u)]
    only_u = test.iloc[list(top_u - top_c)]
    log("\n## 11. Where churn-risk targeting and uplift targeting differ")
    log(f"- Only {overlap_share:.0%} of the contacted customers are the same under both strategies.")
    log(f"- Churn-risk picks that uplift skips: average TRUE uplift = {only_c['true_uplift'].mean()*100:.1f} pp "
        f"(avg usage {only_c['recent_usage'].mean():.0f}) - high risk, but the offer does little.")
    log(f"- Uplift picks that churn-risk skips: average TRUE uplift = {only_u['true_uplift'].mean()*100:.1f} pp "
        f"(avg usage {only_u['recent_usage'].mean():.0f}) - persuadable customers.")

    # ---- 14. Assumptions + pilot
    log("""
## 12. Assumptions to check before using this on REAL customer data
1. **No unmeasured confounding**: all factors that drove both "who got an offer" and "who renewed" are in our features.
   (If a sales rep used a gut feeling we didn't record, estimates will be biased.)
2. **Overlap / positivity**: every type of customer had a real chance of both getting and not getting an offer.
3. **Consistent treatment (SUTVA)**: "offer sent" means the same thing for everyone, and one customer's offer
   does not change another customer's renewal.
4. **Correct timing**: features are measured BEFORE the decision; the 30-day window starts at the same date for both groups.
5. **No post-decision variables** (opens, redemptions) in the model.
6. **Stable world**: customer behaviour in the future resembles the historical period.

## 13. Proposed randomized pilot
- Take a fresh group of eligible subscribers (e.g. 10,000).
- Randomly split: 50% policy group (follow the uplift recommendations, within the 20% limit) /
  50% control group (random 20% get offers, or business-as-usual).
- Run for one renewal window (30 days) and compare renewals and net value between groups.
- Because assignment is random, the difference is a fair test. Roll out only if the policy beats control
  by a pre-agreed margin; keep some customers randomized permanently to keep monitoring.
""")

    with open(f"{OUT}/REPORT.md", "w") as f:
        f.write("\n".join(report))
    print(f"\nDONE. Open the '{OUT}' folder to see everything.")


if __name__ == "__main__":
    main()
