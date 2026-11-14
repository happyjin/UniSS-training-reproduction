"""Paper v9 tables, figures, macros and the prereg_v9 tests. Reads only evaluation outputs:

  eval_outputs/.../gstar_v9_20261104/scores/<set>_<system>_<refset>.json   reports/gstar_si_v9/rescore_v9.py
  .../gstar_v9_20261104/{sensitivity,m10,cost}.json, p19/, p12/             the v9 CPU / GPU analyses
Tests (prereg_v9 §4): exact sign-flip permutation over meetings, 10k meeting-bootstrap intervals with a decoder-noise draw
(make_tables_v8.bleu_test / ratio_test / pol_test), Holm over the 11 tests. Missing inputs print as "--".
Writes generated/macros_v9.tex (\\num{v9:...}), generated/tab_v9_*.tex, generated/fig_v9_pareto.tex, generated/v9_tests.json.

    python scripts/make_tables_v9.py      (from the paper folder)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables_v8 as M8  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
GEN = PAPER / "generated"
ROOT = PAPER.parents[1]
V9 = ROOT / "eval_outputs/uniss_qwen3_omni_v1/gstar_v9_20261104"
SC = V9 / "scores"
SETS = ("en2zh", "zh2en", "ali")
ARROW = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": "AliMeeting"}
MAIN_REF = {"en2zh": "std", "zh2en": "std", "ali": "hum"}   # AliMeeting: the DiariST human references (prereg_v9)
COMP = ("LA2", "Lag1", "Lag2", "Lag3", "Blk2")
MAC = {}


def put(key, v, fmt="{:.2f}"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        MAC[key] = "--"
    else:
        MAC[key] = fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def S(st, name, ref=None):
    p = SC / f"{st}_{name}_{ref or MAIN_REF[st]}.json"
    return json.load(open(p)) if p.exists() else None


SUBSET = {"Lag1", "Lag3", "Blk2", "D2"}   # prereg_v9 amendment 2: compared with S on the subset meetings


def pareto(st):
    pairs = {}
    for c in COMP:
        s = S(st, "S_sub" if c in SUBSET else "S")
        rc = S(st, c)
        if s is None or rc is None:
            pairs[c] = None
            continue
        q = M8.bleu_test(s, rc, "all")
        lt = M8.pol_test(s, rc)
        p = min(1.0, 2 * min(q["p"], lt["p"]))
        pairs[c] = {"p": p, "bleu_diff": q["diff"], "bleu_ci": q["ci"], "bleu_p": q["p"],
                    "lag_diff": lt["diff"], "lag_ci": lt["ci"], "lag_p": lt["p"]}
    if any(v is None for v in pairs.values()):
        return None, pairs
    return {"p": max(v["p"] for v in pairs.values()), "pairs": pairs,
            "ci": [min(v["bleu_ci"][0] for v in pairs.values()), max(v["bleu_ci"][1] for v in pairs.values())]}, pairs


def ovcap(st, kind):
    out = {}
    for d in ("D1", "D2"):
        s = S(st, "S_sub" if d in SUBSET else "S")
        rd = S(st, d)
        if s is None or rd is None:
            return None
        out[d] = M8.bleu_test(s, rd, "ov") if kind == 1 else M8.ratio_test(s, rd, "pairs_recalled", "pairs")
    return {"p": max(v["p"] for v in out.values()), "diff": min(v["diff"] for v in out.values()),
            "ci": [min(v["ci"][0] for v in out.values()), max(v["ci"][1] for v in out.values())], "vs": out}


def main():
    GEN.mkdir(exist_ok=True)
    T = {}
    for st in SETS:
        T[f"E-pareto-{st}"], _ = pareto(st)
        T[f"E-ovcap1-{st}"] = ovcap(st, 1)
        T[f"E-ovcap2-{st}"] = ovcap(st, 2)
    a, b = S("en2zh", "S_sdm", "std"), S("en2zh", "Rel_sdm", "std")
    T["E-far"] = M8.bleu_test(a, b, "ov") if a and b else None
    a, b = S("ali", "S", "hum"), S("ali", "Rel", "hum")
    T["E-ali-hum"] = M8.bleu_test(a, b, "ov") if a and b else None
    M8.holm(T)
    for k, v in T.items():
        if v is not None:
            v["holds"] = bool(v["p_holm"] < 0.05 and v["ci"][0] > 0) if not k.startswith("E-pareto") else bool(v["p_holm"] < 0.05)
        key = "v9:reg:" + k[2:]
        put(f"{key}:holm", M8.fp(v["p_holm"]) if v else None)
        put(f"{key}:holds", ("holds" if v["holds"] else "tie / fails") if v else None)
        if v and "diff" in v:
            put(f"{key}:diff", v["diff"], "{:.3f}" if "ovcap2" in k else "{:.2f}")
            put(f"{key}:ci", f"[{v['ci'][0]}, {v['ci'][1]}]")
    (GEN / "v9_tests.json").write_text(json.dumps(T, indent=1, default=float))
    DESC = {"pareto": r"\sys{}-Simul not dominated by LA2 / lag 1, 2, 3\,s / block 2\,s (quality, text lag)",
            "ovcap1": r"tcpBLEU-ov, \sys{}-Simul $>$ D1 and D2", "ovcap2": r"concurrent recall, \sys{}-Simul $>$ D1 and D2",
            "far": r"SDM far field: tcpBLEU-ov, \sys{}-Simul $>$ causal relabel",
            "ali-hum": r"AliMeeting, human references: tcpBLEU-ov, \sys{}-Simul $>$ causal relabel"}
    rows = []
    for k, v in T.items():
        parts = k[2:].split("-")
        h = "ali-hum" if k == "E-ali-hum" else parts[0]
        stn = ARROW.get(parts[-1], "AMI SDM" if k == "E-far" else "AliMeeting")
        diff = "--" if not v or "diff" not in v else (f"{v['diff']:.3f}" if h == "ovcap2" else f"{v['diff']:.2f}")
        ci = "--" if not v or "diff" not in v else f"[{v['ci'][0]}, {v['ci'][1]}]"
        rows.append(" & ".join([k, stn, DESC[h], diff, ci, M8.fp(v["p_holm"]) if v else "--",
                                ("holds" if v["holds"] else "tie / fails") if v else "not run"]) + r" \\")
    (GEN / "tab_v9_reg.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{\begin{tabular}{lllrrrl}
\toprule
Test & Set & Comparison (whole meetings, online) & $\Delta$ & 95\% interval & $p_\text{Holm}$ & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{The eleven confirmatory tests of prereg\_v9 (committed before any v9 decode), Holm-corrected together. A test
holds when $p_\text{Holm}<0.05$ and, for differences, the interval excludes 0. E-pareto: for every competitor, \sys{}-Simul
must be better on quality or on text lag (two-dimension Bonferroni); the set's $p$ is the largest over competitors.
LA2 and lag 2\,s: all meetings; lag 1 and 3\,s, block 2\,s and D2: the v7 meeting subset (8 AMI / 10 AliMeeting
meetings; prereg\_v9 amendment 2), with \sys{}-Simul scored on the same meetings.
E-ovcap: the larger $p$ of the comparisons with D1 and D2; $\Delta$ and the interval are the less favourable of the two.}
\label{tab:reg9}
\end{table*}
""", encoding="utf-8")

    # ---- policy comparison on the same prefix-trained model (Pareto data) and its table
    prow, coords = [], {st: [] for st in SETS}
    for st in SETS:
        for name, lab in (("S", r"\sys{}-Simul (learned)"), ("LA2", "LocalAgreement-2"), ("Lag1", r"fixed lag 1\,s"),
                          ("Lag2", r"fixed lag 2\,s"), ("Lag3", r"fixed lag 3\,s"), ("Blk2", r"block commit 2\,s")):
            r = S(st, name)
            if r is None:
                continue
            lag = (r.get("lag_ov") or {}).get("p50"), (r.get("lag_1spk") or {}).get("p50")
            allp50 = float(np.median(r["lag_values"]["ov"] + r["lag_values"]["1spk"])) if r.get("lag_values") else None
            put(f"v9:pol:{st}:{name}:all", r["tcpBLEU_all"])
            put(f"v9:pol:{st}:{name}:ov", r["tcpBLEU_ov"])
            put(f"v9:pol:{st}:{name}:cr", 100 * r["CR"], "{:.1f}")
            put(f"v9:pol:{st}:{name}:lag", allp50)
            put(f"v9:pol:{st}:{name}:lagov", lag[0])
            coords[st].append((name, allp50, r["tcpBLEU_all"]))
            prow.append(" & ".join([ARROW[st] if name == "S" else "", lab, f"{r['tcpBLEU_all']:.2f}", f"{r['tcpBLEU_ov']:.2f}",
                                    f"{100 * r['CR']:.1f}", f"{allp50:.2f}" if allp50 is not None else "--",
                                    "yes" if name != "S" else "no"]) + r" \\")
    (GEN / "tab_v9_policy.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrc}
\toprule
Set & Commit policy (same model) & all & overlap & CR & lag p50 & decodes to end \\
\midrule
""" + "\n".join(prow) + r"""
\bottomrule
\end{tabular}}
\caption{Commit policies on the same prefix-trained model, whole meetings online: tcpBLEU, concurrent recall (\%) and the
median text completion lag (s, commit of a unit's last segment minus the unit's end). AliMeeting: human references.
Policies that decode to the end at every read cost about twice a learned-policy read (\S\ref{sec:res-latency}).}
\label{tab:policy9}
\end{table}
""", encoding="utf-8")
    plots = []
    for st in SETS:
        if not coords[st]:
            continue
        pts = " ".join(f"({x:.2f},{y:.2f})" for n, x, y in coords[st] if x is not None and n != "S")
        sp = [(x, y) for n, x, y in coords[st] if n == "S" and x is not None]
        labels = "\n".join(rf"\node[font=\tiny,anchor=south west] at (axis cs:{x:.2f},{y:.2f}) {{{n}}};"
                           for n, x, y in coords[st] if x is not None and n != "S")
        plots.append(r"""\begin{tikzpicture}\begin{axis}[width=0.36\textwidth,height=3.4cm,title={""" + ARROW[st] + r"""},
title style={font=\small},xlabel={text lag p50 (s)},ylabel={tcpBLEU},label style={font=\scriptsize},
tick label style={font=\scriptsize}]
\addplot[only marks,mark=o] coordinates {""" + pts + r"""};
""" + (rf"\addplot[only marks,mark=*,mark size=2pt] coordinates {{({sp[0][0]:.2f},{sp[0][1]:.2f})}};" if sp else "") + "\n" + labels + r"""
\end{axis}\end{tikzpicture}""")
    (GEN / "fig_v9_pareto.tex").write_text(r"""\begin{figure*}[t]
\centering
""" + "\n".join(plots) + r"""
\caption{Quality against text lag for commit policies on the same prefix-trained model (whole meetings online; filled:
\sys{}-Simul's learned policy). AliMeeting with human references.}
\label{fig:pareto9}
\end{figure*}
""", encoding="utf-8")

    # ---- overlap-capable controls, AliMeeting human references, SDM, D1/D2
    orow = []
    for st in SETS:
        for name, lab in (("S", r"\sys{}-Simul"), ("Rel", "speaker-blind + causal relabel"),
                          ("D1", "D1: + multi-label split"), ("D2", "D2: turn-segmented input"), ("OLA2", r"offline \sys{} + LA2")):
            r = S(st, name)
            if r is None:
                continue
            sh = r["short_recall"]
            for f in ("all", "ov", "1spk"):
                put(f"v9:ov:{st}:{name}:{f}", r[f"tcpBLEU_{f}"])
            put(f"v9:ov:{st}:{name}:cr", 100 * r["CR"], "{:.1f}")
            put(f"v9:ov:{st}:{name}:dup", 100 * r["duplicate_rate"], "{:.1f}")
            put(f"v9:ov:{st}:{name}:s05", 100 * sh.get("le0.5_ov", {}).get("recall", float("nan")), "{:.1f}")
            orow.append(" & ".join([ARROW[st] if name == "S" else "", lab, f"{r['tcpBLEU_all']:.2f}", f"{r['tcpBLEU_ov']:.2f}",
                                    f"{100 * r['CR']:.1f}", f"{100 * sh.get('le0.5_ov', {}).get('recall', float('nan')):.1f}",
                                    f"{100 * r['duplicate_rate']:.1f}"]) + r" \\")
    (GEN / "tab_v9_ovcap.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrr}
\toprule
Set & System (whole meetings, online) & all & overlap & CR & $\le$0.5\,s & dup. \\
\midrule
""" + "\n".join(orow) + r"""
\bottomrule
\end{tabular}}
\caption{\sys{}-Simul against controls that may give overlapping speech two speakers, built on the same speaker-blind
prefix-trained model (prereg\_v9 \S3): D1 splits a segment where the dominant diarizer slot changes; D2 translates every
diarizer turn on its own (a turn-level cascade with the same translator). AliMeeting: human references. CR: concurrent
recall; $\le$0.5\,s: recall of overlapped short utterances; dup.: units translated twice (\%).}
\label{tab:ovcap9}
\end{table}
""", encoding="utf-8")
    # AliMeeting: human vs machine references, direction agreement
    for name in ("S", "Rel", "LA2", "OLA2", "D1", "D2", "Lag2", "C-ft2", "C-turn", "O2", "O2-nosd-relabel", "S2dl_d1", "C2_O"):
        for ref in ("hum", "std", "full"):
            r = S("ali", name, ref)
            if r:
                put(f"v9:ali:{name}:{ref}:all", r["tcpBLEU_all"])
                put(f"v9:ali:{name}:{ref}:ov", r["tcpBLEU_ov"])
                put(f"v9:ali:{name}:{ref}:cr", 100 * r["CR"], "{:.1f}")
    al = json.load(open(ROOT / "data/diarist_alimeeting_refs_v1/align_report.json"))
    put("v9:ali:align:rows", al["diarist_rows"], "{:,d}")
    put("v9:ali:align:matched", al["matched"], "{:,d}")
    put("v9:ali:align:scored", al["of_which_human_matched"], "{:,d}")
    put("v9:ali:align:full", al["full_scored_utterances"], "{:,d}")
    for name in ("S_sdm", "Rel_sdm"):
        r = S("en2zh", name, "std")
        if r:
            for f in ("all", "ov"):
                put(f"v9:sdm:{name}:{f}", r[f"tcpBLEU_{f}"])
            put(f"v9:sdm:{name}:cr", 100 * r["CR"], "{:.1f}")
    for st in ("en2zh", "zh2en", "ali"):
        for name in ("S", "Rel", "OLA2", "LA2"):
            r = S(st, name, "full")
            if r:
                put(f"v9:full:{st}:{name}:all", r["tcpBLEU_all"])
                put(f"v9:full:{st}:{name}:ov", r["tcpBLEU_ov"])
                put(f"v9:full:{st}:{name}:units", r["units_ov"] + r["units_1spk"], "{:,d}")

    # exploratory (not in Holm): AliMeeting offline systems on the human references, and both reference kinds
    X = {}
    for a_, b_ in (("O2", "C-ft2"), ("O2", "O2-nosd-relabel"), ("O2", "C-turn"), ("S", "Rel"), ("S", "OLA2"), ("S", "LA2")):
        for ref in ("hum", "std"):
            ra, rb = S("ali", a_, ref), S("ali", b_, ref)
            for reg in ("all", "ov"):
                t = M8.bleu_test(ra, rb, reg, one_sided=False) if ra and rb else None
                X[f"ali-{a_}-{b_}-{ref}-{reg}"] = t
                put(f"v9:x:ali:{a_}-{b_}:{ref}:{reg}:diff", t["diff"] if t else None)
                put(f"v9:x:ali:{a_}-{b_}:{ref}:{reg}:p", M8.fp(t["p"]) if t else None)
                put(f"v9:x:ali:{a_}-{b_}:{ref}:{reg}:ci", f"[{t['ci'][0]}, {t['ci'][1]}]" if t else None)
    agree = [np.sign((X[f"ali-{a_}-{b_}-hum-{r}"] or {}).get("diff", 0)) == np.sign((X[f"ali-{a_}-{b_}-std-{r}"] or {}).get("diff", 0))
             for a_, b_ in (("O2", "C-ft2"), ("O2", "O2-nosd-relabel"), ("S", "Rel"), ("S", "OLA2")) for r in ("all", "ov")]
    put("v9:x:ali:agree", sum(agree), "{:d}")
    put("v9:x:ali:agreen", len(agree), "{:d}")
    (GEN / "v9_exploratory.json").write_text(json.dumps(X, indent=1, default=float))

    # ---- sensitivity, M10, cost
    f = V9 / "sensitivity.json"
    if f.exists():
        R = json.load(open(f))
        for st in SETS:
            r = [x for x in R if x["set"] == st]
            put(f"v9:sens:{st}:min", min(x["ov_diff"] for x in r))
            put(f"v9:sens:{st}:max", max(x["ov_diff"] for x in r))
            put(f"v9:sens:{st}:excl", sum(x["ov_diff_ci"][0] > 0 for x in r), "{:d}")
            put(f"v9:sens:{st}:n", len(r), "{:d}")
            put(f"v9:sens:{st}:crmin", 100 * min(x["CR_diff"] for x in r), "{:.1f}")
            put(f"v9:sens:{st}:crmax", 100 * max(x["CR_diff"] for x in r), "{:.1f}")
    f = V9 / "m10.json"
    if f.exists():
        R = json.load(open(f))
        for st in SETS:
            put(f"v9:m10:{st}:slow", R[st]["slowest_p50_median"], "{:.1f}")
            put(f"v9:m10:{st}:slowmax", R[st]["slowest_p50_max"], "{:.0f}")
            put(f"v9:m10:{st}:sim", 100 * R[st]["simultaneous_share_mean"], "{:.0f}")
    f = V9 / "cost.json"
    crow = []
    if f.exists():
        R = json.load(open(f))
        for k, v in R["training"].items():
            if v:
                crow.append(rf"Training: {k} & {v['gpus']} & {v['hours']:.1f} & {v['gpu_hours']:.0f} \\")
        for k, v in R["inference"].items():
            st, s = k.split("_", 1)
            lab = r"\sys{}-Simul" if s == "S2dl" else r"\sys{} + LA2"
            crow.append(rf"Inference: {lab}, {ARROW[st]} & 2 & RTF {v['rtf_simulated']:.2f} & {v['gpu_hours_per_meeting_hour']:.2f}/h \\")
            put(f"v9:cost:{k}:gpuh", v["gpu_hours_per_meeting_hour"])
    (GEN / "tab_v9_cost.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrr}
\toprule
 & H200 & hours & GPU-hours \\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}}
\caption{Cost. Training: logged wall-clock times the GPUs used. Inference: GPU-hours per meeting-hour under the
queued-read simulation (median single-stream read time measured on two H200 cards, batch 1; every read re-encodes the
window prefix without a key-value cache). No wall-clock replay on dedicated hardware was run.}
\label{tab:cost9}
\end{table}
""", encoding="utf-8")
    # ---- strata (M15): overlap-share bins and short utterances, S vs Rel
    srow = []
    for st in SETS:
        for name, lab in (("S", r"\sys{}-Simul"), ("Rel", "causal relabel")):
            r = S(st, name)
            if r is None:
                continue
            bins = [r.get(f"tcpBLEU_bin{b}") for b in ("0", "0-10", "10-30", "30-100")]
            sh = r["short_recall"]
            srow.append(" & ".join([ARROW[st] if name == "S" else "", lab] + [f"{x:.2f}" if x is not None else "--" for x in bins] +
                                   [f"{100 * sh.get(k, {}).get('recall', float('nan')):.1f}" for k in ("le0.5_1spk", "le0.5_ov", "0.5-1_ov")]) + r" \\")
    (GEN / "tab_v9_strata.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrrrr}
\toprule
 & & \multicolumn{4}{c}{tcpBLEU by overlapped share of a unit} & \multicolumn{3}{c}{short-utterance recall (\%)} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-9}
Set & System & 0 & 0--10\% & 10--30\% & $>$30\% & $\le$0.5\,s & $\le$0.5\,s ovl. & 0.5--1\,s ovl. \\
\midrule
""" + "\n".join(srow) + r"""
\bottomrule
\end{tabular}}
\caption{Strata (review M15), whole meetings online. The sets also differ in source naturalness: real English (AMI
En$\rightarrow$Zh), the same meetings re-voiced in Chinese (Zh$\rightarrow$En), real Mandarin (AliMeeting, human references).}
\label{tab:strata9}
\end{table}
""", encoding="utf-8")
    lines = [r"\providecommand{\num}[1]{\csname n@#1\endcsname}"]
    lines += [rf"\expandafter\def\csname n@{k}\endcsname{{{v}}}" for k, v in sorted(MAC.items())]
    (GEN / "macros_v9.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in T.items():
        print(k, None if v is None else {x: v[x] for x in ("p", "p_holm", "holds") if x in v}, (v or {}).get("diff"), (v or {}).get("ci"))
    print(len(MAC), "macros")


if __name__ == "__main__":
    main()
