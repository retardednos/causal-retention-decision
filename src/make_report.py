import markdown
md = open("outputs/REPORT.md").read()
html = markdown.markdown(md, extensions=["tables"])
imgs = ["01_causal_diagram.png","02_overlap_normal.png","03_uplift_by_band.png","04_policy_comparison.png","05_overlap_weak_scenario.png"]
charts = "<h2>Charts</h2>" + "".join(f'<p><img src="{i}" style="max-width:100%"></p>' for i in imgs)
css = "<style>body{font-family:Arial;max-width:900px;margin:40px auto;line-height:1.5}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
open("outputs/REPORT.html","w").write("<html><head><meta charset='utf-8'>"+css+"</head><body>"+html+charts+"</body></html>")
print("Done")
