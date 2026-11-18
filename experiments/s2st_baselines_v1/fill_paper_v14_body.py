"""Body tables of the single-speaker QABC paper v14 (paper/bayes_simul_s2st_v14/generated/body_*.tex).

Every raw number is copied from a v13 generated table (paper/bayes_simul_s2st_v13_baselines/generated/) and checked
against that file before anything is written; means and standard deviations over adapter seeds or checkpoints are
computed here. The plan is paper/bayes_simul_s2st_v14_revision_plan.zh-CN.md (sections 17-20).

Usage: python3 experiments/s2st_baselines_v1/fill_paper_v14_body.py [--force]
Without --force an existing output file is left untouched and reported.
"""
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V13 = REPO / "paper/bayes_simul_s2st_v13_baselines/generated"
OUT = REPO / "paper/bayes_simul_s2st_v14/generated"
FORCE = "--force" in sys.argv
_src_cache = {}


def src(name):
    if name not in _src_cache:
        _src_cache[name] = (V13 / name).read_text()
    return _src_cache[name]


def chk(value, name):
    """value (a string as printed in the v13 table) must occur in the v13 file `name`."""
    if value not in src(name).replace("$-$", "-").replace("{-}", "-"):
        raise SystemExit(f"value {value!r} not found in {name}")
    return value


def fmt_signed(x, nd=2):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def tex_num(s):
    return s.replace("-", "$-$") if s.startswith("-") or "[-" in s or ",-" in s else s


def write(name, text):
    p = OUT / name
    if p.exists() and not FORCE:
        print(f"exists, not overwritten (use --force): {p}")
        return
    p.write_text(text)
    print(f"wrote {p}")


# ---------------------------------------------------------------- Table 1: offline anchor (CVSS-T), compact
cv = "tab_cvss.tex"
cvss_rows = [  # label, params, ASR en2zh, zh2en, Text en2zh, zh2en
    (r"Cascade, 2-stage$^\dagger$", "2.8B", "26.94", "20.86", "27.38", "22.20"),
    (r"GPT-4o$^\dagger$", "--", "31.64", "19.27", "--", "--"),
    (r"Seamless-m4t-v2-large$^\ddagger$", "2.3B", "20.86", "22.25", "20.75", "22.60"),
    (r"UniSS (Q)$^\ddagger$", "1.5B", "32.04", "24.72", "32.95", "25.51"),
    (r"SimulS2ST-Omni, Thinker--Talker$^\ddagger$", "3B", "31.12", "25.18", "33.73", "26.41"),
    (r"Qwen3-Omni, own Talker$^\P$", "30B-A3B", "36.59", "25.91", "38.25", "30.56"),
    (r"Ours: Thinker + Talker", "30B-A3B", "36.60", "28.17", "38.45", "30.79"),
    (r"\quad + \qabc{} adapter", "30B-A3B", "37.10", "29.07", "38.91", "31.05"),
]
for r in cvss_rows:
    for v in r[2:]:
        if v != "--":
            chk(v, cv)
best = [max(float(r[i]) for r in cvss_rows if r[i] != "--") for i in range(2, 6)]
lines = []
for i, r in enumerate(cvss_rows):
    cells = []
    for j, v in enumerate(r[2:]):
        cells.append(r"\textbf{" + v + "}" if v != "--" and float(v) == best[j] else v)
    if i == 6:
        lines.append(r"\midrule")
    lines.append(f"{r[0]} & {r[1]} & " + " & ".join(cells) + r" \\")
t1 = r"""\begin{table}[!tb]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{2}{c}{ASR-BLEU$\uparrow$} & \multicolumn{2}{c}{Text-BLEU$\uparrow$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & Params & En$\rightarrow$Zh & Zh$\rightarrow$En & En$\rightarrow$Zh & Zh$\rightarrow$En \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Offline S2ST on CVSS-T test (4{,}897 utterances a direction), the anchor for the simultaneous results.
$^\dagger$Reported by UniSS~\citep{uniss2025} under its own protocol with the same recognisers (2-stage: SeamlessM4T-v2-large
S2TT + CosyVoice~2). $^\ddagger$Reported or re-evaluated by \citet{he2026simuls2st}. $^\P$Run by us (its own Talker, a
preset voice). Ours: the Thinker translates each whole utterance greedily and our Talker speaks it in the source voice.
Our Thinker is far larger than the S2ST models, so the table places the anchor rather than ranking offline recipes.
Bold: best in the column.}
\label{tab:cvss}
\end{table}
"""
write("body_cvss.tex", t1)

# ---------------------------------------------------------------- Table 2: latency tiers (RealSI + ACL60/60)
rm = "tab_realsi_main.tex"
am = "tab_acl_main.tex"
# columns: Omni, SeamlessStreaming, Hy-SimulMT (our Talker), Hy+Qwen3-TTS, X-Translator, LA2, Hold-n, Vote, QABC
tiers = {
    "RealSI En$\\rightarrow$Zh": [
        ("m1", "1.18", ["19.80", "--", "--", "--", "--", "--", "--", "--", "25.33 (1.13)"]),
        ("m2", "1.98", ["24.21", "13.21 (1.98)", "25.67 (1.72)", "25.72 (1.90)", "--", "27.42 (1.84)", "25.67 (1.89)", "--", "28.26 (1.77)"]),
        ("m3", "2.46", ["25.01", "15.09 (2.41)", "27.96 (2.05)", "27.93 (2.17)", "--", "29.28 (2.41)", "26.96 (2.29)", "--", "30.33 (2.25)"]),
        ("m4", "2.98", ["25.54", "15.09 (2.41)", "27.96 (2.05)", "27.93 (2.17)", "23.76 (2.53)", "30.14 (2.76)", "28.60 (2.78)", "28.32 (2.73)", "30.51 (2.75)"]),
    ],
    "RealSI Zh$\\rightarrow$En": [
        ("m1", "1.18", ["14.89", "--", "--", "--", "--", "--", "--", "--", "17.83 (1.14)"]),
        ("m2", "1.92", ["19.55", "13.73 (1.84)", "21.88 (1.75)", "--", "--", "20.31 (1.78)", "16.73 (1.65)", "--", "21.42 (1.70)"]),
        ("m3", "2.40", ["21.07", "13.92 (2.24)", "21.88 (1.75)", "20.11 (2.11)", "16.23 (2.27)", "21.50 (2.24)", "20.62 (2.05)", "--", "22.46 (2.18)"]),
        ("m4", "2.98", ["22.47", "14.09 (2.85)", "21.88 (1.75)", "20.11 (2.11)", "16.36 (2.67)", "23.47 (2.80)", "21.58 (2.47)", "--", "22.65 (2.67)"]),
    ],
    "ACL60/60 En$\\rightarrow$Zh": [
        ("m1", "1.30", ["40.80", "n/a", "n/a", "n/a", "n/a", "--", "n/a", "n/a", "43.94 (1.23)"]),
        ("m2", "1.90", ["51.51", "n/a", "n/a", "n/a", "n/a", "45.06 (1.87)", "n/a", "n/a", "48.76 (1.88)"]),
        ("m3", "2.48", ["51.94", "n/a", "n/a", "n/a", "n/a", "49.09 (2.18)", "n/a", "n/a", "50.98 (2.35)"]),
        ("m4", "2.89", ["53.24", "n/a", "n/a", "n/a", "n/a", "50.85 (2.58)", "n/a", "n/a", "51.49 (2.67)"]),
    ],
}
for name, rows in tiers.items():
    f = am if name.startswith("ACL") else rm
    for tier, budget, cells in rows:
        for c in cells:
            if c not in ("--", "n/a"):
                v, _, lat = c.partition(" (")
                chk(v, f)
                if lat:
                    chk(lat.rstrip(")"), f)
body = []
for name, rows in tiers.items():
    body.append(r"\midrule")
    body.append(r"\multicolumn{11}{l}{\textit{" + name + (" (ASR-BLEU, segment LAAL)" if name.startswith("RealSI") else " (BLEU, StreamLAAL)") + r"}} \\")
    for tier, budget, cells in rows:
        ours = [c for c in cells[1:]]
        vals = [float(c.split(" (")[0]) for c in ours if c not in ("--", "n/a")]
        top = max(vals)
        out = [cells[0]]
        for c in ours:
            if c not in ("--", "n/a") and float(c.split(" (")[0]) == top:
                v, _, lat = c.partition(" (")
                out.append(r"\textbf{" + v + r"}\,{\scriptsize(" + lat)
                out[-1] = out[-1][:-1] + ")}" if lat else out[-1]
            elif c not in ("--", "n/a") and " (" in c:
                v, _, lat = c.partition(" (")
                out.append(v + r"\,{\scriptsize(" + lat.rstrip(")") + ")}")
            else:
                out.append(c)
        body.append(f"{tier} & {budget} & " + " & ".join(out) + r" \\")
t2 = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
Tier & Budget (s) & SimulS2ST-Omni$^\dagger$ & SeamlessStreaming$^\P$ & Hy-SimulMT$^\P$ & Hy-SimulMT + Qwen3-TTS$^\P$ & X-Translator$^\P$ & LA2 & Hold-$n$ & Vote & \qabc \\
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\caption{Simultaneous systems in the latency tiers of SimulS2ST-Omni (tier $m$: an $m$-second read window). Each cell is
a system's best operating point within the tier's latency budget (latency in parentheses); RealSI: mean of three talker
seeds. ``--'': no point fits; n/a: not run. $^\dagger$Reported by \citet{he2026simuls2st} for its own model with beam
search; the budgets are its latencies (Appendix~\ref{app:protocol}). $^\P$Run by us with our protocol, inference only:
SeamlessStreaming and X-Translator speak with their own speech generators; Hy-SimulMT outputs text, spoken by our Talker
or by Qwen3-TTS. LA2, Hold-$n$, Vote and \qabc{} share our Thinker and Talker. Bold: best among the systems we ran.}
\label{tab:realsi-main}
\end{table*}
"""
write("body_tiers.tex", t2)

# ---------------------------------------------------------------- Table 3: primary comparison + five adapter seeds
pa = "tab_prereg_all.tex"
sd_src = "tab_v10_seeds.tex"
prim = [  # name, s0 E1 [CI], s0 E2, per-seed E1 (s0..s4), five-seed mean, CI, E2 mean
    ("RealSI En$\\rightarrow$Zh (ASR-BLEU)", "+0.76", "[+0.00,+1.54]", "+0.21",
     ["+0.74", "+0.62", "+0.08", "+0.00", "+0.34"], "+0.36", "[-0.34,+1.20]", "-0.04"),
    ("RealSI Zh$\\rightarrow$En (ASR-BLEU)", "+0.78", "[+0.12,+1.79]", "+0.12",
     ["+0.78", "+0.65", "+1.33", "+1.12", "+1.63"], "+1.10", "[+0.38,+2.02]", "+0.16"),
    ("ACL60/60 En$\\rightarrow$Zh (BLEU)", "+2.63", "[+1.54,+5.76]", "+0.35",
     ["+2.63", "+2.60", "+1.66", "+1.93", "+1.84"], "+2.13", "[+1.15,+4.16]", "+0.20"),
]
if src("tab_v10_seeds_main.tex").count("5/5") != 3:
    raise SystemExit("expected 5/5 seeds ahead on all three test sets in tab_v10_seeds_main.tex")
rows3 = []
for name, e1, ci, e2, seeds, mean, ci5, e25 in prim:
    chk(e1, pa); chk(ci, pa); chk(e2, pa)
    for s in seeds:
        chk(s, sd_src)
    chk(mean, sd_src); chk(ci5, sd_src); chk(e25, sd_src)
    xs = [float(s) for s in seeds]
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    if abs(m - float(mean)) > 0.006:
        raise SystemExit(f"mean mismatch {name}: {m} vs {mean}")
    ahead = 5  # "Seeds >0" column of tab_v10_seeds_main (5/5 for all three; the En-Zh worst seed is +0.003)
    rows3.append(rf"{name} & adapter seed 0 & {tex_num(e1)} {tex_num(ci)} & {tex_num(e2)} & -- \\")
    rows3.append(rf" & five seeds & {tex_num(mean)}\,$\pm$\,{sd:.2f} {tex_num(ci5)} & {tex_num(e25)} & {ahead}/5 \\")
    rows3.append(r"\addlinespace")
rows3 = rows3[:-1]
t3 = r"""\begin{table}[!tb]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccc}
\toprule
\qabc{} vs.\ LA2 & Adapter & E1 [95\% CI] & E2 (s) & Ahead \\
\midrule
""" + "\n".join(rows3) + r"""
\bottomrule
\end{tabular}}
\caption{Primary comparison on the same Thinker and Talker: \qabc{} ($b_0{=}1$) against LocalAgreement over
LocalAgreement's 15 read windows. E1: mean quality gain at equal latency up to 3\,s; E2: mean latency saved at equal
quality. Adapter seed 0 was fixed before the test sets were run (interval: paired bootstrap over sentences or reference
segments, three talker seeds). Five seeds: mean\,$\pm$\,sd over adapters trained with different seeds (interval: seeds and
sentences resampled); Ahead: adapters with a positive E1.}
\label{tab:primary}
\end{table}
"""
write("body_primary.tex", t3)

# ---------------------------------------------------------------- Table 4: queue ablation + controls
qs = "tab_queue.tex"
ctl = "tab_controls.tex"
qrows = [  # dir, Delta, b0, LAAL qabcc, LAAL qabc, gain, CI
    ("En$\\rightarrow$Zh", "0.5", "$-$1", "1.17", "1.13", "+2.96", "[+1.64,+4.43]"),
    ("", "0.5", "0", "1.30", "1.29", "+1.87", "[+0.95,+2.64]"),
    ("", "0.5", "1", "1.77", "1.77", "+1.15", "[+0.53,+1.72]"),
    ("", "1", "0", "1.68", "1.70", "+0.48", "[-0.07,+1.03]"),
    ("", "1", "1", "2.24", "2.25", "+0.44", "[-0.04,+1.02]"),
    ("Zh$\\rightarrow$En", "0.5", "$-$1", "1.12", "1.14", "+2.33", "[+0.27,+5.00]"),
    ("", "0.5", "0", "1.31", "1.32", "+1.91", "[+0.15,+4.00]"),
    ("", "0.5", "1", "1.71", "1.72", "+0.72", "[-0.30,+1.69]"),
    ("", "1", "0", "1.67", "1.70", "+0.85", "[+0.26,+1.46]"),
    ("", "1", "1", "2.13", "2.18", "+0.58", "[-0.08,+1.24]"),
]
qsrc = src(qs).replace("$-$", "-")
for r in qrows:
    for v in r[3:]:
        if v not in qsrc:
            raise SystemExit(f"queue value {v} not in {qs}")
ctrl = [  # name, en2zh E1, E2, zh2en E1, E2
    (r"\qabc{} vs.\ LA2, on the adapted Thinker's drafts", "+2.01", "+0.30", "+0.17", "-0.01"),
    (r"LA2 with a queue gate vs.\ LA2", "+0.01", "-0.05", "+0.07", "-0.00"),
]
csrc = src(ctl).replace("$-$", "-")
for r in ctrl:
    for v in r[1:]:
        if v not in csrc:
            raise SystemExit(f"control value {v} not in {ctl}")
maxdl = max(abs(float(r[3]) - float(r[4])) for r in qrows)
pairs = [(D, b) for d, D, b, *_ in qrows[:5]]
qb = []
for k, (D, b) in enumerate(pairs):
    e, z = qrows[k], qrows[k + 5]
    assert (e[1], e[2]) == (z[1], z[2])
    qb.append(rf"{D} & {b} & {tex_num(e[5])} {{\scriptsize{tex_num(e[6])}}} & {tex_num(z[5])} {{\scriptsize{tex_num(z[6])}}} \\")
cb = [rf"{n} & {tex_num(a)} / {tex_num(b)} & {tex_num(c)} / {tex_num(e)} \\" for n, a, b, c, e in ctrl]
t4 = r"""\begin{table}[!tb]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{cccc}
\toprule
$\Delta$ (s) & $b_0$ & En$\rightarrow$Zh: \qabc{} $-$ \qabcc{} [95\% CI] & Zh$\rightarrow$En: \qabc{} $-$ \qabcc{} [95\% CI] \\
\midrule
""" + "\n".join(qb) + r"""
\bottomrule
\end{tabular}}

\vspace{4pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc}
\toprule
Control (committed text) & En$\rightarrow$Zh E1 / E2 & Zh$\rightarrow$En E1 / E2 \\
\midrule
""" + "\n".join(cb) + r"""
\bottomrule
\end{tabular}}
\caption{The queue-aware threshold. Top: ASR-BLEU of \qabc{} minus its constant-delay variant \qabcc{} on RealSI with the
same adapter, read window $\Delta$ and base bias $b_0$ (three talker seeds); the two variants' segment LAAL differ by at
most """ + f"{maxdl:.2f}" + r"""\,s. Bottom: controls on the committed text over LocalAgreement's 15 read windows (E1 in BLEU,
E2 in s): \qabc's rule against LocalAgreement on the same drafts of the adapted Thinker, and LocalAgreement committing
nothing while a whole read window of speech is queued.}
\label{tab:queue}
\end{table}
"""
write("body_queue.tex", t4)

# ---------------------------------------------------------------- Table 5: amortized vs test-time posterior + cost
rf_ = "tab_range_tests_full.tex"
cost = "tab_v2_cost.tex"
rng = {  # range -> rows (name, en2zh, zh2en, acl)
    "Budgets 2--3\\,s": [
        (r"\ttb{} (test-time, frozen Thinker)", "-0.27 [-0.76,+0.33]", "+0.47 [-0.25,+1.02]", "+0.83 [+0.02,+1.80]"),
        (r"\qabcc{} (amortized, constant delay)", "+0.35 [-0.64,+1.11]", "-0.21 [-1.26,+0.79]", "+1.53 [+0.36,+2.78]"),
        (r"\qabc{} (amortized, queue-aware)", "+0.78 [-0.22,+1.55]", "+0.15 [-0.82,+1.14]", "+1.63 [+0.41,+2.85]"),
    ],
    "Budgets below 2\\,s": [
        (r"\ttb{} (test-time, frozen Thinker)", "-0.52 [-1.41,+0.41]", "+1.01 [-0.30,+2.07]", "+2.39 [+0.74,+3.98]"),
        (r"\qabcc{} (amortized, constant delay)", "-0.25 [-1.43,+0.75]", "-0.00 [-1.22,+1.41]", "+5.13 [+3.35,+7.38]"),
        (r"\qabc{} (amortized, queue-aware)", "+0.90 [-0.26,+1.79]", "+0.72 [-0.61,+2.27]", "+5.65 [+3.79,+7.65]"),
    ],
}
rsrc = src(rf_).replace("$-$", "-")
for rows in rng.values():
    for r in rows:
        for v in r[1:]:
            g, _, ci = v.partition(" ")
            if g not in rsrc or ci not in rsrc:
                raise SystemExit(f"range value {v} not in {rf_}")
costs = [("LA2", "72", "76"), ("Vote", "135", "107"), (r"\ttb", "132", "136"), (r"\qabcc", "57", "59"), (r"\qabc", "59", "58")]
for n, a, b in costs:
    chk(a, cost); chk(b, cost)


def cell(v):
    g, _, ci = v.partition(" ")
    return tex_num(g) + r" {\scriptsize" + tex_num(ci) + "}"


b5 = []
for rname, rows in rng.items():
    b5.append(r"\multicolumn{4}{l}{\textit{" + rname + r"}} \\")
    for n, a, b, c in rows:
        b5.append(f"{n} & {cell(a)} & {cell(b)} & {cell(c)}" + r" \\")
b5c = " & ".join([f"{n}{{}} {a}/{b}" for n, a, b in costs])
t5 = r"""\begin{table}[!tb]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Mean gain over LA2 & RealSI En$\rightarrow$Zh & RealSI Zh$\rightarrow$En & ACL60/60 \\
\midrule
""" + "\n".join(b5) + r"""
\bottomrule
\end{tabular}}

\vspace{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Thinker time per read (ms), En$\rightarrow$Zh / Zh$\rightarrow$En & LA2 & Vote & \ttb & \qabcc & \qabc \\
\midrule
 & """ + " & ".join(f"{a}/{b}" for n, a, b in costs) + r""" \\
\bottomrule
\end{tabular}}
\caption{Amortized against test-time posterior. Top: mean quality gain over LocalAgreement across latency budgets, one
fixed setting per policy ($b_0{=}1$), with 95\% intervals (RealSI ASR-BLEU, three talker seeds; ACL60/60 BLEU). Bottom:
Thinker computation per read on one H200 (\ttb{} includes its second draft, Vote its eight samples); adding it to the
latency raises LAAL by at most 0.14\,s for any policy.}
\label{tab:range}
\end{table}
"""
chk("+0.14", cost)
write("body_range.tex", t5)

# ---------------------------------------------------------------- Table 6: label source and training length
ab = "tab_v2_ablation_text.tex"
abl = {
    "En$\\rightarrow$Zh": {
        "steps": [["27.62", "29.05", "31.45", "31.90"], ["28.29", "30.17", "31.24", "31.51"],
                  ["28.22", "29.89", "31.46", "32.15"], ["27.69", "29.78", "31.50", "31.96"]],
        "cons": [["25.04", "27.16", "28.67", "30.76"], ["28.19", "29.40", "29.85", "29.85"]],
    },
    "Zh$\\rightarrow$En": {
        "steps": [["20.31", "21.51", "21.51", "23.32"], ["19.53", "21.22", "22.70", "22.97"],
                  ["20.56", "21.33", "22.76", "23.74"], ["19.98", "21.72", "22.72", "23.42"]],
        "cons": [["18.04", "20.22", "21.48", "22.81"], ["18.92", "20.97", "21.94", "21.94"]],
    },
}
asrc = re.sub(r"\\textbf\{([^}]*)\}", r"\1", src(ab))
for d in abl.values():
    for grp in d.values():
        for row in grp:
            if " & ".join(row) not in asrc:
                raise SystemExit(f"ablation row {row} not contiguous in {ab}")


def msd(rows, j):
    xs = [float(r[j]) for r in rows]
    return f"{statistics.mean(xs):.2f}\\,$\\pm$\\,{statistics.stdev(xs):.2f}"


b6 = []
for dname, d in abl.items():
    b6.append(r"\multicolumn{5}{l}{\textit{" + dname + r"}} \\")
    b6.append(r"Full-context labels (step 200, deployed) & " + " & ".join(d["steps"][0]) + r" \\")
    b6.append(r"Consensus labels (two runs) & " + " & ".join(msd(d["cons"], j) for j in range(4)) + r" \\")
    b6.append(r"Full-context labels, steps 200--800 & " + " & ".join(msd(d["steps"], j) for j in range(4)) + r" \\")
T6_PENDING = True

# ---------------------------------------------------------------- Table 7: speech rate r
rt = "tab_v2_rate.tex"
rate = [  # dir, qabcc (laal), [(qabc, laal, gain) x3]
    ("En$\\rightarrow$Zh", "24.63", "1.30", [("25.59", "1.28", "+0.97"), ("26.12", "1.29", "+1.49"), ("27.15", "1.38", "+2.52")]),
    ("Zh$\\rightarrow$En", "18.65", "1.31", [("19.10", "1.31", "+0.45"), ("19.74", "1.32", "+1.09"), ("20.07", "1.40", "+1.42")]),
]
for d, a, l, xs in rate:
    chk(a, rt); chk(l, rt)
    for q, ql, g in xs:
        chk(q, rt); chk(ql, rt); chk(g, rt)
b7 = [rf"{d} & " + " & ".join(rf"{g} ({ql})" for q, ql, g in xs) + r" & \\" for d, a, l, xs in rate]
t6 = r"""\begin{table}[!tb]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Committed-text BLEU within budget & ${\le}1.5$\,s & ${\le}2.0$\,s & ${\le}2.5$\,s & ${\le}3.0$\,s \\
\midrule
""" + "\n".join(b6) + r"""
\midrule
Speech rate (gain of \qabc{} over \qabcc) & $r{\times}0.75$ & $r{\times}1$ & $r{\times}1.25$ & \\
\midrule
""" + "\n".join(b7) + r"""
\bottomrule
\end{tabular}}
\caption{Ablations. Top: posterior learning on RealSI (\qabcc, committed text, best BLEU among the operating points whose
latency fits each budget). Consensus labels: adapters trained on the agreement among sampled outputs instead of the
full-context continuation (mean\,$\pm$\,sd over two training runs); steps 200--800: mean\,$\pm$\,sd over four checkpoints
of the same run. Bottom: the speech rate $r$ of the queue model, Eq.~\eqref{eq:queue}; RealSI ASR-BLEU gain over \qabcc{}
(segment LAAL of \qabc, s) at $\Delta{=}0.5$\,s, $b_0{=}0$, one talker seed, where \qabcc{} scores """ + rate[0][1] + " (" + rate[0][2] + r""") in En$\rightarrow$Zh
and """ + rate[1][1] + " (" + rate[1][2] + r""") in Zh$\rightarrow$En.}
\label{tab:ablation}
\end{table}
"""
write("body_ablation.tex", t6)
print("all values checked against v13 tables")
