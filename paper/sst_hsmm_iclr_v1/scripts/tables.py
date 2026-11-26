"""Generate LaTeX table bodies from data/results.json -> generated/*.tex"""
import json, os
R = json.load(open("data/results.json"))
os.makedirs("generated", exist_ok=True)
D = ("en2zh", "zh2en")
def g(root, d, m, key="NEW_vs_B_b"):
    return R[root][key][d][m]
def absv(root, d, m, key="NEW_vs_B_b"):   # common B reference + the paired difference of this system
    return g("p3hk", d, m)["ref"] + g(root, d, m, key)["diff"]
B = {d: {m: g("p3hk", d, m)["ref"] for m in ("asr", "text_bleu", "d2", "EndOffset", "NP_in", "sim", "pesq")} for d in D}
# Table 1
rows = [("B: base drafter, LA-2", None, None),
        ("LA-2", "p3x", None), ("LA-2+CV", "p3hk", "N_vs_B_b"),
        ("Verifier $g$ as translator", "p3t", None),
        ("Hard gate (support)", "p3v_b", None),
        ("Hard gate (joint conf.)", "p3w", None),
        ("HSMM, unanchored", "p3n1", None),
        ("\\ours{} (quality, $\\theta{=}0.75$)", "p3hk", None),
        ("\\ours{} (latency)", "p3hk2", None)]
L = []
for name, root, key in rows:
    cells = []
    for d in D:
        for m, f in (("asr", "{:.2f}"), ("text_bleu", "{:.2f}"), ("d2", "{:.2f}")):
            v = B[d][m] if root is None else absv(root, d, m, key or "NEW_vs_B_b")
            cells.append(f.format(v))
    L.append(name + " & " + " & ".join(cells) + " \\\\")
    if root is None or name.startswith("HSMM"):
        L.append("\\midrule")
open("generated/tab_main.tex", "w").write("\n".join(L) + "\n")
# Table 2 per cell (ASC quality point and latency point): dASR / dLAAL, bold if CI excludes 0
cells = ["realsi/clean", "realsi/noisy", "obsi/clean", "obsi/id2", "obsi/ood1"]
def fmt(x, dec):
    s = f"{x['diff']:+.{dec}f}"
    lo, hi = x["ci95"]
    return f"\\textbf{{{s}}}" if (lo > 0 or hi < 0) else s
L = []
for root, lab in (("p3hk", "\\ours{} ($\\theta{=}0.75$)"), ("p3hk2", "\\ours{} (latency)")):
    for d, dl in (("en2zh", "\\enzh{}"), ("zh2en", "\\zhen{}")):
        a = [fmt(g(root, d, "asr")["cells"][c], 2) for c in cells]
        l = [fmt(g(root, d, "d2")["cells"][c], 2) for c in cells]
        L.append(f"{lab} & {dl} & " + " & ".join(f"{x} / {y}" for x, y in zip(a, l)) + " \\\\")
open("generated/tab_cells.tex", "w").write("\n".join(L) + "\n")
# B absolute per cell
L = []
for d, dl in (("en2zh", "\\enzh{}"), ("zh2en", "\\zhen{}")):
    a = g("p3hk", d, "asr")["cells"]; l = g("p3hk", d, "d2")["cells"]
    L.append(f"B (absolute) & {dl} & " + " & ".join(f"{a[c]['arm'] - a[c]['diff']:.2f} / {l[c]['arm'] - l[c]['diff']:.2f}" for c in cells) + " \\\\")
open("generated/tab_cells_b.tex", "w").write("\n".join(L) + "\n")
# Table 5 ablations: diffs vs B
abl = [("\\ours{} (quality point)", "p3hk"), ("$-$ boundary healing", "p3ts_u1t3"), ("$-$ healing, $-$ talker rate", "p3h8"),
       ("$-$ healing, $-$ rate, $\\theta{=}0.80$", "p3h6"), ("$-$ anchoring (unanchored HSMM)", "p3n1"),
       ("$+$ distance-to-draft-end input", "p3h7"), ("$+$ on-policy refit", "p3h9")]
L = []
for name, root in abl:
    c = []
    for d in D:
        for m, f in (("asr", "{:+.2f}"), ("d2", "{:+.3f}"), ("EndOffset", "{:+.2f}"), ("NP_in", "{:+.2f}")):
            c.append(f.format(g(root, d, m)["diff"]))
    L.append(name + " & " + " & ".join(c) + " \\\\")
open("generated/tab_abl.tex", "w").write("\n".join(L) + "\n")
# full metrics appendix
full = [("LA-2", "p3x", None), ("LA-2+CV", "p3hk", "N_vs_B_b"), ("Verifier as translator", "p3t", None), ("Hard gate (support)", "p3v_b", None),
        ("Hard gate (joint conf.)", "p3w", None), ("HSMM, per-direction, no anchor", "p3z2", None), ("HSMM, unanchored", "p3n1", None),
        ("\\ours{} w/o healing, $\\theta{=}0.80$", "p3h6", None), ("\\ours{} w/o healing", "p3h8", None), ("\\ours{} (quality)", "p3hk", None), ("\\ours{} (latency)", "p3hk2", None)]
M = ("asr", "text_bleu", "d2", "d1", "EndOffset", "NP_in", "PT_in", "sim", "pesq")
L = []
for d, dl in (("en2zh", "\\enzh{}"), ("zh2en", "\\zhen{}")):
    L.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{dl}}}}} \\\\")
    L.append("B (absolute) & " + " & ".join(f"{g('p3hk', d, m)['ref']:.3f}" if m in ('d2','d1','sim') else f"{g('p3hk', d, m)['ref']:.2f}" for m in M) + " \\\\")
    for name, root, key in full:
        L.append(name + " & " + " & ".join((f"{g(root, d, m, key or 'NEW_vs_B_b')['diff']:+.3f}" if m in ('d2','d1','sim') else f"{g(root, d, m, key or 'NEW_vs_B_b')['diff']:+.2f}") for m in M) + " \\\\")
    L.append("\\midrule")
open("generated/tab_full.tex", "w").write("\n".join(L[:-1]) + "\n")
print(open("generated/tab_main.tex").read()); print(open("generated/tab_cells.tex").read()); print(open("generated/tab_abl.tex").read())
