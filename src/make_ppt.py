"""Builds a clean PowerPoint from the results already in outputs/ . Run: python src/make_ppt.py"""
import re
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image

OUT = "outputs"
rep = open(f"{OUT}/REPORT.md").read()

def grab(pattern, default="n/a"):
    m = re.search(pattern, rep)
    return m.group(1) if m else default

with_offer = grab(r"WITH offer:\s+([\d.]+%)")
no_offer = grab(r"WITHOUT offer:\s+([\d.]+%)")
naive = grab(r"Naive \(unadjusted\):\s+([+-][\d.]+)")
dr = grab(r"Doubly robust \(cross-fit\):\s+([+-][\d.]+)")
ci = grab(r"\(95% CI ([^)]+)\)")
true_ate = grab(r"TRUE average effect:\s+([+-][\d.]+)")
weak_normal = grab(r"\(propensity outside [\d.\-]+\): (\d+ of \d+ \([\d.]+%\))")
review = grab(r"Routed to manual review \(weak overlap\): (\d+)")
weak_scn = grab(r"Customers routed to manual review: (\d+ of \d+ \([\d.]+%\))")

pol = {}
sec7 = rep.split("## 7.")[1].split("## 8.")[0]
for line in sec7.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 4 and cells[0] in ("Uplift targeting", "Churn-risk targeting", "Random targeting", "No offer"):
        pol[cells[0]] = (cells[2], cells[3])

NAVY, GREEN, GREY = RGBColor(0x1E, 0x29, 0x3B), RGBColor(0x16, 0xA3, 0x4A), RGBColor(0x47, 0x55, 0x69)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]

def text(slide, s, x, y, w, h, size=18, bold=False, color=GREY, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        p.space_after = Pt(10)
        if align:
            p.alignment = align
    return tb

def new_slide(title):
    s = prs.slides.add_slide(blank)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.15))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()
    text(s, title, 0.6, 0.4, 12.1, 0.9, size=30, bold=True, color=NAVY)
    return s

def image(slide, name, x, y, max_w, max_h):
    w, h = Image.open(f"{OUT}/{name}").size
    scale = min(max_w / w, max_h / h)
    slide.shapes.add_picture(f"{OUT}/{name}", Inches(x), Inches(y), Inches(w * scale), Inches(h * scale))

# 1 title
s = prs.slides.add_slide(blank)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, "Causal AI for Counterfactual Decision Intelligence", 0.8, 2.3, 11.7, 1.8, size=44, bold=True, color=RGBColor(255, 255, 255))
text(s, "Who should get a retention offer? Finding customers whose renewal changes because of the offer.", 0.8, 4.4, 11.7, 1.2, size=22, color=RGBColor(0xCB, 0xD5, 0xE1))

# 2 problem
s = new_slide("The problem")
text(s, ["A subscription company can send retention offers to only 20% of eligible customers.",
         "A churn model tells us who is likely to leave. It does not tell us whose decision the offer would change.",
         "Uplift = renewal chance WITH the offer minus renewal chance WITHOUT it.",
         "Goal: send offers to the customers with the highest uplift, within budget, and prove it beats the usual methods."],
     0.8, 1.7, 11.7, 5, size=22)

# 3 DAG
s = new_slide("Causal assumptions (DAG)")
image(s, "01_causal_diagram.png", 0.6, 1.5, 7.4, 5.4)
text(s, ["Customer features affect both who got an offer and who renews (confounding).",
         "We adjust for these pre-decision features only.",
         "Offer opens and redemptions are excluded: they happen after the decision.",
         "Assumptions: no hidden confounders, overlap, consistent treatment."], 8.3, 1.7, 4.5, 5, size=17)

# 4 trap
s = new_slide("The trap: raw comparison misleads")
text(s, [f"Renewal rate with offer: {with_offer}   |   without offer: {no_offer}",
         f"Naive difference: {naive} percentage points (offers look harmful)",
         "Why: offers were sent mostly to low-usage / support-issue customers, who renew less anyway.",
         f"Doubly robust estimate (cross-fitted): {dr} pp, 95% CI {ci}",
         f"True average effect (from the simulator): {true_ate} pp",
         "Adjusting for confounders brings us close to the truth."], 0.8, 1.7, 11.7, 5, size=22)

# 5 overlap
s = new_slide("Overlap check")
image(s, "02_overlap_normal.png", 0.6, 1.5, 7.2, 5.3)
text(s, [f"Weak overlap in normal scenario: {weak_normal}",
         "Weak overlap = almost every similar customer got (or did not get) an offer, so there is nothing fair to compare.",
         f"Held-out customers routed to manual review: {review}",
         f"Weak-overlap scenario (extreme past policy): {weak_scn} routed to review"], 8.1, 1.7, 4.8, 5, size=17)

# 6 uplift
s = new_slide("Who benefits? Uplift by score band")
image(s, "03_uplift_by_band.png", 0.6, 1.5, 7.2, 5.3)
text(s, ["Models: T-learner (baseline), DR-learner, causal forest. The causal forest tracked true uplift best.",
         "Customers in the highest band gain most from an offer; the lowest band can even react negatively.",
         "Profile estimates describe similar customers, not verified individual outcomes."], 8.1, 1.7, 4.8, 5, size=17)

# 7 policy
s = new_slide("Policy comparison (same 20% contact limit)")
image(s, "04_policy_comparison.png", 0.6, 1.5, 7.2, 5.3)
lines = [f"{k}: {v[0]} extra renewals / 1,000, net value {v[1]}" for k, v in pol.items()]
text(s, lines + ["Churn-risk targeting wastes offers on 'lost causes' and people who would renew anyway."], 8.1, 1.7, 4.8, 5, size=17)

# 8 assumptions & pilot
s = new_slide("Before using this on real data")
text(s, ["Check: no unmeasured confounding, overlap, consistent offer definition, features measured before the decision.",
         "Same 30-day window for treated and untreated customers; stable customer behaviour over time.",
         "Proposed randomized pilot: split fresh customers 50/50 between the uplift policy and a control group.",
         "Compare renewals and net value after one 30-day window; roll out only if the policy wins by a pre-agreed margin."],
     0.8, 1.7, 11.7, 5, size=22)

prs.save(f"{OUT}/Causal_AI_Presentation.pptx")
print("Done: outputs/Causal_AI_Presentation.pptx")
