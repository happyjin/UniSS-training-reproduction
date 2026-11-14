"""Paper v8 tables, macros and statistics (prereg_v8). Reads the v8 outputs only:

  eval_outputs/.../gstar_v8_20261103/scores_ov2/<set>_<system>[ _fullref].json   overlap_score.py (per-meeting stats)
  eval_outputs/.../gstar_v8_20261103/latency/*.json                              listener_latency_v8.py
  eval_outputs/.../gstar_v8_20261103/b7/*.json, b5/*, c2/*                        B7 / B5 / C2
Writes generated/macros_v8.tex (\\num{v8:...}), generated/tab_v8_*.tex, generated/fig_v8_overlap.tex and
generated/v8_tests.json. Statistics: exact sign-flip permutation over meetings (16 / 20, all 2^M flips) and 10,000
meeting-bootstrap resamples, each adding one decoder-noise draw N(0, sigma_dec(n)^2) (v7's sigma_dec) to the BLEU
difference; the interval is the 2.5 / 97.5 percentile of those noisy differences. Holm over the 12 tests of prereg_v8.
Missing inputs (runs not finished) are written as "--".

    python scripts/make_tables_v8.py      (from the paper folder)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_v7 import bleu_vec, sig_dec  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
GEN = PAPER / "generated"
ROOT = PAPER.parents[1]
V8 = ROOT / "eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103"
SO = V8 / "scores_ov2"
SETS = ("en2zh", "zh2en", "ali")
ARROW = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": "AliMeeting"}
B, RNG = 10000, np.random.default_rng(20261103)
M = {}


def fp(p):
    """p-values: three decimals, '<0.001' below (review M14)."""
    return "--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def put(key, v, fmt="{:.2f}"):
    M[key] = "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else (fmt.format(v) if isinstance(v, (int, float)) else v)


def load(name):
    p = SO / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def per_meeting_stats(r, key):
    ms = sorted(r["per_meeting"])
    return ms, np.array([r["per_meeting"][m][f"{key}_stats"] or [0.0] * 10 for m in ms], dtype=np.float64)


def flips(M_):
    idx = np.arange(2 ** M_)
    return ((idx[:, None] >> np.arange(M_)[None, :]) & 1).astype(np.float64)


def bleu_test(ra, rb, key, one_sided=True):
    """BLEU(a) - BLEU(b) on region `key`: exact sign-flip p (one-sided a > b), noisy bootstrap CI."""
    if ra is None or rb is None:
        return None
    ma, Sa = per_meeting_stats(ra, key)
    mb, Sb = per_meeting_stats(rb, key)
    common = sorted(set(ma) & set(mb))
    Sa = Sa[[ma.index(m) for m in common]]
    Sb = Sb[[mb.index(m) for m in common]]
    diff = float(bleu_vec(Sa.sum(0))[0] - bleu_vec(Sb.sum(0))[0])
    F = flips(len(common))
    D = Sb - Sa
    A = Sa.sum(0) + F @ D
    Bm = Sa.sum(0) + Sb.sum(0) - A
    d = bleu_vec(A) - bleu_vec(Bm)
    p = float((d >= diff - 1e-9).mean()) if one_sided else float((np.abs(d) >= abs(diff) - 1e-9).mean())
    J = RNG.integers(0, len(common), (B, len(common)))
    n = int(sum(1 for _ in common))   # noise scale follows the number of windows; meetings here
    nwin = int(ra.get("units_ov", 0) + ra.get("units_1spk", 0)) or 1000
    ds = np.array([bleu_vec(Sa[j].sum(0))[0] - bleu_vec(Sb[j].sum(0))[0] for j in J]) + RNG.normal(0, sig_dec(nwin), B)
    return {"diff": round(diff, 2), "p": p, "ci": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
            "meetings": n}


def ratio_test(ra, rb, num, den):
    """pooled ratio(a) - ratio(b) (e.g. concurrent-speaker recall), one-sided sign-flip over meetings."""
    if ra is None or rb is None:
        return None
    common = sorted(set(ra["per_meeting"]) & set(rb["per_meeting"]))
    na = np.array([ra["per_meeting"][m][num] for m in common], float)
    da = np.array([ra["per_meeting"][m][den] for m in common], float)
    nb = np.array([rb["per_meeting"][m][num] for m in common], float)
    db = np.array([rb["per_meeting"][m][den] for m in common], float)
    obs = na.sum() / max(da.sum(), 1) - nb.sum() / max(db.sum(), 1)
    F = flips(len(common))
    An, Ad = F @ (nb - na) + na.sum(), F @ (db - da) + da.sum()
    Bn, Bd = na.sum() + nb.sum() - An, da.sum() + db.sum() - Ad
    d = An / np.maximum(Ad, 1) - Bn / np.maximum(Bd, 1)
    J = RNG.integers(0, len(common), (B, len(common)))
    ds = np.array([na[j].sum() / max(da[j].sum(), 1) - nb[j].sum() / max(db[j].sum(), 1) for j in J])
    return {"diff": round(float(obs), 4), "p": float((d >= obs - 1e-12).mean()),
            "ci": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)], "meetings": len(common)}


def lag_meeting(r, reg):
    return {m: float(np.median(v["lag"][reg])) for m, v in r["per_meeting"].items() if v.get("lag", {}).get(reg)}


def signflip_mean(x, one_sided_greater=True):
    x = np.asarray(x, float)
    F = 2 * flips(len(x)) - 1
    obs = x.mean()
    d = (F * np.abs(x)[None, :]).mean(1) if False else (F * x[None, :]).mean(1)
    return float((d >= obs - 1e-12).mean()) if one_sided_greater else float((d <= obs + 1e-12).mean())


def lat_test(r, bound=0.5):
    """non-inferiority: per-meeting median lag (ov) - median lag (1spk) < bound; one-sided sign-flip on d - bound."""
    if r is None:
        return None
    a, b = lag_meeting(r, "ov"), lag_meeting(r, "1spk")
    common = sorted(set(a) & set(b))
    d = np.array([a[m] - b[m] for m in common])
    p = signflip_mean(d - bound, one_sided_greater=False)   # small p: mean(d - bound) significantly < 0
    J = RNG.integers(0, len(d), (B, len(d)))
    ds = d[J].mean(1)
    return {"diff": round(float(d.mean()), 3), "p": p, "ci": [round(float(np.percentile(ds, 2.5)), 3),
            round(float(np.percentile(ds, 97.5)), 3)], "bound": bound, "meetings": len(d)}


def pol_test(ra, rb):
    """per-meeting median text completion lag of a lower than b (one-sided)."""
    if ra is None or rb is None:
        return None
    a, b = lag_meeting(ra, "all"), lag_meeting(rb, "all")
    common = sorted(set(a) & set(b))
    d = np.array([b[m] - a[m] for m in common])   # > 0: a earlier
    p = signflip_mean(d, one_sided_greater=True)
    J = RNG.integers(0, len(d), (B, len(d)))
    ds = d[J].mean(1)
    return {"diff": round(float(d.mean()), 3), "p": p, "ci": [round(float(np.percentile(ds, 2.5)), 3),
            round(float(np.percentile(ds, 97.5)), 3)], "meetings": len(d)}


def holm(tests):
    ks = [k for k, v in tests.items() if v is not None]
    order = sorted(ks, key=lambda k: tests[k]["p"])
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * tests[k]["p"]))
        tests[k]["p_holm"] = round(run, 4)
    return tests


def main():
    GEN.mkdir(exist_ok=True)
    T = {}
    for s in SETS:
        sim = load(f"{s}_on_S2dl")
        rel = load(f"{s}_on_{s}_S2dlnosd")
        pla2 = load(f"{s}_on_{s}_S2dlLA2")
        T[f"E-ov1-{s}"] = bleu_test(sim, rel, "ov")
        T[f"E-ov2-{s}"] = ratio_test(sim, rel, "pairs_recalled", "pairs")
        T[f"E-lat-{s}"] = lat_test(sim)
        T[f"E-pol-{s}"] = pol_test(sim, pla2)
        if T[f"E-pol-{s}"] is not None:
            T[f"E-pol-{s}"]["bleu"] = bleu_test(sim, pla2, "all", one_sided=False)
    holm(T)
    for k, v in T.items():
        ok = None
        if v is not None:
            if k.startswith("E-lat"):
                ok = v["p_holm"] < 0.05 and v["ci"][1] < v["bound"]
            else:
                ok = v["p_holm"] < 0.05 and v["ci"][0] > 0
        if v is not None:
            v["holds"] = bool(ok)
        key = k.replace("E-", "v8:reg:")
        put(f"{key}:diff", v["diff"] if v else None, "{:.2f}" if "ov2" not in k else "{:.3f}")
        put(f"{key}:holm", fp(v["p_holm"]) if v else None)
        put(f"{key}:holds", (r"\checkmark" if v["holds"] else "tie") if v else None)
        if v:
            put(f"{key}:ci", f"[{v['ci'][0]}, {v['ci'][1]}]")
    # exploratory (not in Holm): offline overlap comparisons on reference windows, and online Simul vs LA2 on overlap
    X = {}
    for s in SETS:
        o2, rel, cas = load(f"{s}_O2"), load(f"{s}_O2-nosd-relabel"), load(f"{s}_C-ft2")
        X[f"ov-O2-rel-{s}"] = bleu_test(o2, rel, "ov", one_sided=False)
        X[f"sp-O2-rel-{s}"] = bleu_test(o2, rel, "1spk", one_sided=False)
        X[f"cr-O2-rel-{s}"] = ratio_test(o2, rel, "pairs_recalled", "pairs")
        X[f"ov-O2-cas-{s}"] = bleu_test(o2, cas, "ov", one_sided=False)
        X[f"ov-simul-ola2-{s}"] = bleu_test(load(f"{s}_on_S2dl"), load(f"{s}_on_O2_LA2"), "ov", one_sided=False)
        X[f"ov-simul-pla2-{s}"] = bleu_test(load(f"{s}_on_S2dl"), load(f"{s}_on_{s}_S2dlLA2"), "ov", one_sided=False)
    for k, v in X.items():
        put(f"v8:x:{k}:diff", v["diff"] if v else None, "{:.3f}" if k.startswith("cr") else "{:.2f}")
        put(f"v8:x:{k}:p", fp(v["p"]) if v else None)
        put(f"v8:x:{k}:ci", f"[{v['ci'][0]}, {v['ci'][1]}]" if v else None)
    T["exploratory"] = X
    (GEN / "v8_tests.json").write_text(json.dumps(T, indent=1))
    T.pop("exploratory")

    # ---- overlap table: reference windows (offline + simultaneous) and whole meetings
    rows = []
    SYS_REF = [("C-ft2", "Cascade, same data"), ("C-turn", "Cascade, turn-trained"), ("O2-nosd-relabel", "Speaker-blind + relabel (offline)"),
               ("O2", r"\sys{} (offline)"), ("O2-relabel", r"\sys{} text, diarizer relabelling"), ("C2_O", r"\sys{} (offline) + fallback"), ("S2dl_LA2_full|S2dlZ_LA2_d1", r"\sys{}-Simul model + LA2"),
               ("S2dl_d1", r"\sys{}-Simul")]
    for s in SETS:
        rows.append(r"\multicolumn{8}{l}{\emph{" + ARROW[s] + r", reference windows}} \\")
        for keys, lab in SYS_REF:
            r = next((load(f"{s}_{k}") for k in keys.split("|") if load(f"{s}_{k}")), None)
            if r is None:
                continue
            sh = r["short_recall"]
            rows.append(" & ".join([lab, f"{r['tcpBLEU_all']:.2f}", f"{r['tcpBLEU_ov']:.2f}", f"{r['tcpBLEU_1spk']:.2f}",
                                    f"{100 * r['CR']:.1f}", f"{100 * r['collapse']:.1f}",
                                    f"{100 * sh.get('le0.5_ov', {}).get('recall', float('nan')):.1f}",
                                    f"{100 * r['duplicate_rate']:.1f}"]) + r" \\")
            tag = keys.split("|")[0]
            for f in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk"):
                put(f"v8:ov:{s}:{tag}:{f.split('_')[1]}", r[f])
            put(f"v8:ov:{s}:{tag}:cr", 100 * r["CR"], "{:.1f}")
            put(f"v8:ov:{s}:{tag}:coll", 100 * r["collapse"], "{:.1f}")
            put(f"v8:ov:{s}:{tag}:dup", 100 * r["duplicate_rate"], "{:.1f}")
            put(f"v8:ov:{s}:{tag}:s05ov", 100 * sh.get("le0.5_ov", {}).get("recall", float("nan")), "{:.1f}")
            put(f"v8:ov:{s}:{tag}:pre", 100 * r["pre_end_share"], "{:.1f}")
        rows.append(r"\multicolumn{8}{l}{\emph{" + ARROW[s] + r", whole meetings online}} \\")
        for k, lab in (("on_O2_LA2", r"\sys{} (offline) + LA2"), (f"on_{s}_S2dlLA2", r"\sys{}-Simul model + LA2"),
                       (f"on_{s}_S2dllag2", r"\sys{}-Simul model, lag 2\,s$^\dagger$"), (f"on_{s}_S2dlnosd", "Speaker-blind Simul + causal relabel"),
                       ("on_S2dl", r"\sys{}-Simul")):
            r = load(f"{s}_{k}")
            if r is None:
                continue
            sh = r["short_recall"]
            rows.append(" & ".join([lab, f"{r['tcpBLEU_all']:.2f}", f"{r['tcpBLEU_ov']:.2f}", f"{r['tcpBLEU_1spk']:.2f}",
                                    f"{100 * r['CR']:.1f}", f"{100 * r['collapse']:.1f}",
                                    f"{100 * sh.get('le0.5_ov', {}).get('recall', float('nan')):.1f}",
                                    f"{100 * r['duplicate_rate']:.1f}"]) + r" \\")
            tag = k.replace(f"on_{s}_", "on_")
            for f in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk"):
                put(f"v8:ov:{s}:{tag}:{f.split('_')[1]}", r[f])
            put(f"v8:ov:{s}:{tag}:cr", 100 * r["CR"], "{:.1f}")
            put(f"v8:ov:{s}:{tag}:coll", 100 * r["collapse"], "{:.1f}")
            put(f"v8:ov:{s}:{tag}:pre", 100 * r["pre_end_share"], "{:.1f}")
            for reg in ("ov", "1spk"):
                put(f"v8:ov:{s}:{tag}:lag{reg}", (r.get(f"lag_{reg}") or {}).get("p50"))
        if s == "en2zh":
            for k in ("on_S2dl", "on_O2_LA2"):
                r = load(f"{s}_{k}_fullref")
                if r:
                    put(f"v8:full:{s}:{k}:all", r["tcpBLEU_all"])
                    put(f"v8:full:{s}:{k}:ov", r["tcpBLEU_ov"])
                    put(f"v8:full:{s}:{k}:units", r["units_ov"] + r["units_1spk"], "{:d}")
    (GEN / "tab_v8_overlap.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{3}{c}{tcpBLEU$\uparrow$} & \multicolumn{2}{c}{concurrent speakers} & short & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
System & all & overlap & single & recall$\uparrow$ & collapse$\downarrow$ & $\le$0.5\,s$\uparrow$ & dup.$\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Overlap-aware scores (\S\ref{sec:eval-overlap}). Overlap: reference units with $\ge$20\% of their time overlapped
by another speaker; single: the rest. Concurrent speakers: pairs of utterances of two speakers overlapping $\ge$0.5\,s;
recall: both translated and attributed; collapse: one side missing and covered by the other side's output speaker. Short:
recall of overlapped utterances $\le$0.5\,s. Dup.: units translated twice (chrF $\ge$50 between two segments, \%).
$^\dagger$\,On the v7 meeting subset only (8 AMI / 10 AliMeeting meetings); compare with Appendix~\ref{app:v8}, not with the
other rows.}
\label{tab:overlap8}
\end{table*}
""", encoding="utf-8")

    # ---- compact main-text overlap table: systems x (set: all / overlap / concurrent recall)
    MAIN = [("ref", "C-ft2", "Cascade, same data"), ("ref", "O2-nosd-relabel", "Speaker-blind + relabel"),
            ("ref", "O2", r"\sys{} (offline)"), ("ref", "C2_O", r"\quad + short-turn fallback"),
            ("on", "on_O2_LA2", r"Offline model + LA2"), ("on", "on_{s}_S2dlLA2", r"Simul model + LA2"),
            ("on", "on_{s}_S2dlnosd", r"Speaker-blind Simul + causal relabel"), ("on", "on_S2dl", r"\sys{}-Simul")]
    mrows, last = [], None
    for grp, k, lab in MAIN:
        if grp != last:
            mrows.append(r"\multicolumn{10}{l}{\emph{" + ("Reference windows, offline" if grp == "ref" else
                                                          "Whole meetings, online, 1\\,s reads") + r"}} \\")
            last = grp
        cells = [lab]
        for s in SETS:
            r = load(f"{s}_{k.format(s=s)}")
            cells += [f"{r['tcpBLEU_all']:.2f}", f"{r['tcpBLEU_ov']:.2f}", f"{100 * r['CR']:.1f}"] if r else ["--"] * 3
        mrows.append(" & ".join(cells) + r" \\")
    (GEN / "tab_v8_ovmain.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{En$\rightarrow$Zh} & \multicolumn{3}{c}{Zh$\rightarrow$En} & \multicolumn{3}{c}{AliMeeting} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
System & all & overlap & CR & all & overlap & CR & all & overlap & CR \\
\midrule
""" + "\n".join(mrows) + r"""
\bottomrule
\end{tabular}
\caption{Overlap-aware evaluation (\S\ref{sec:eval-overlap}): tcpBLEU over all units and over overlapped units, and
concurrent-speaker recall (CR, \%). Relabelling attaches one diarizer speaker to each translated segment. All rows and
metrics: Table~\ref{tab:overlap8}.}
\label{tab:ovmain8}
\end{table*}
""", encoding="utf-8")

    # ---- overlap-share curve (figure): tcpBLEU per overlap bin, reference windows, En->Zh and Zh->En
    bins = ["0", "0-10", "10-30", "30-100"]
    plots = []
    for s in ("en2zh", "zh2en"):
        series = []
        for k, lab, sty in (("C-ft2", "Cascade", "dashed"), ("O2-nosd-relabel", "Relabel", "dotted"), ("O2", r"\sys{}", "solid")):
            r = load(f"{s}_{k}")
            if r is None:
                continue
            pts = " ".join(f"({i},{r.get('tcpBLEU_bin' + b) or 0:.2f})" for i, b in enumerate(bins))
            series.append(rf"\addplot[{sty},mark=*,mark size=1.2pt] coordinates {{{pts}}}; \addlegendentry{{{lab}}}")
            for b in bins:
                put(f"v8:bin:{s}:{k}:{b}", r.get("tcpBLEU_bin" + b))
        plots.append(r"""\begin{tikzpicture}\begin{axis}[width=0.52\columnwidth,height=3.6cm,xtick={0,1,2,3},
xticklabels={0,0--10,10--30,30+},xlabel={overlapped share of unit (\%)},ylabel={tcpBLEU},title={""" + ARROW[s] + r"""},
title style={font=\small},label style={font=\scriptsize},tick label style={font=\scriptsize},legend style={font=\tiny,at={(0.02,0.02)},anchor=south west}]
""" + "\n".join(series) + r"""
\end{axis}\end{tikzpicture}""")
    (GEN / "fig_v8_overlap.tex").write_text(r"""\begin{figure}[t]
\centering
""" + "\n".join(plots) + r"""
\caption{tcpBLEU by how much of a reference unit overlaps another speaker (reference windows, offline systems). The gap
between \sys{} and post-hoc relabelling opens where speakers overlap.}
\label{fig:overlap8}
\end{figure}
""", encoding="utf-8")

    # ---- latency table: queued reads, both channel layouts
    L = V8 / "latency"
    lrows = []
    for s in SETS:
        for k, lab in (("S2dl", r"\sys{}-Simul"), ("O2_LA2", r"\sys{} + LA2")):
            q1 = L / f"{s}_{k}_single_queue.json"
            q2 = L / f"{s}_{k}_per-speaker_queue.json"
            if not (q1.exists() and q2.exists()):
                continue
            a, b = json.load(open(q1)), json.load(open(q2))
            ov = load(f"{s}_on_{k}")
            lrows.append(" & ".join([ARROW[s] if k == "S2dl" else "", lab, f"{a['rtf_mean']:.2f}",
                                     f"{(ov.get('lag_ov') or {}).get('p50', float('nan')):.2f}" if ov else "--",
                                     f"{(ov.get('lag_1spk') or {}).get('p50', float('nan')):.2f}" if ov else "--",
                                     f"{b['comp_ov']['p50']:.1f}", f"{b['comp_1spk']['p50']:.1f}",
                                     f"{b['comp_all']['p90']:.1f}", f"{a['comp_all']['p50']:.1f}"]) + r" \\")
            for nm, x in (("single", a), ("ps", b)):
                put(f"v8:lat:{s}:{k}:{nm}:rtf", x["rtf_mean"])
                put(f"v8:lat:{s}:{k}:{nm}:p50", x["comp_all"]["p50"], "{:.1f}")
                put(f"v8:lat:{s}:{k}:{nm}:p90", x["comp_all"]["p90"], "{:.1f}")
                put(f"v8:lat:{s}:{k}:{nm}:ov", x["comp_ov"]["p50"], "{:.1f}")
                put(f"v8:lat:{s}:{k}:{nm}:sp", x["comp_1spk"]["p50"], "{:.1f}")
                put(f"v8:lat:{s}:{k}:{nm}:late10", 100 * x["comp_all"]["late10"], "{:.0f}")
                put(f"v8:lat:{s}:{k}:{nm}:start", x["start_p50"])
                put(f"v8:lat:{s}:{k}:{nm}:atd", x["atd_mean"])
                put(f"v8:lat:{s}:{k}:{nm}:cens", x["censored_streams"], "{:d}")
            nq = L / f"{s}_{k}_per-speaker_noqueue.json"
            if nq.exists():
                put(f"v8:lat:{s}:{k}:ps_noq:p50", json.load(open(nq))["comp_all"]["p50"], "{:.1f}")
    (GEN / "tab_v8_latency.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.2pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrrrr}
\toprule
 & & & \multicolumn{2}{c}{text lag p50 (s)} & \multicolumn{3}{c}{per-speaker channel (s)} & 1 ch. \\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}
Set & System & RTF & overlap & single & ovl. p50 & single p50 & p90 & p50 \\
\midrule
""" + "\n".join(lrows) + r"""
\bottomrule
\end{tabular}}
\caption{Whole meetings with queued reads. RTF: read computation per second of meeting ($>1$: falls behind). Text
lag: commit of a unit's last segment minus the unit's end. Per-speaker channel and one channel (1 ch.): completion lag
of the speech heard.}
\label{tab:lat8}
\end{table}
""", encoding="utf-8")

    # ---- pre-registered tests table
    DESC = {"ov1": r"tcpBLEU-ov, \sys{}-Simul $-$ causal relabel", "ov2": r"concurrent-speaker recall, same pair",
            "lat": r"text lag, overlapped $-$ single ($<$0.5\,s)", "pol": r"text lag, LA2 (same model) $-$ \sys{}-Simul"}
    rrows = []
    for k, v in T.items():
        _, h, s = k.split("-")
        cells = [k, ARROW[s], DESC[h]]
        if v is None:
            cells += ["--"] * 4
        else:
            cells += [f"{v['diff']:.3f}" if h == "ov2" else f"{v['diff']:.2f}", f"[{v['ci'][0]}, {v['ci'][1]}]",
                      fp(v["p_holm"]), "holds" if v["holds"] else "tie / fails"]
        rrows.append(" & ".join(cells) + r" \\")
    (GEN / "tab_v8_reg.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllrrrl}
\toprule
Test & Set & Comparison (whole meetings, online) & $\Delta$ & 95\% interval & $p_\text{Holm}$ & \\
\midrule
""" + "\n".join(rrows) + r"""
\bottomrule
\end{tabular}
\caption{The twelve confirmatory tests of prereg\_v8, Holm-corrected together. A test holds when $p_\text{Holm}<0.05$ and
the interval excludes 0 (E-lat: lies below the 0.5\,s bound). Exact sign-flip permutation over meetings; intervals from
10,000 meeting resamples, BLEU differences with a decoder-noise draw per resample.}
\label{tab:reg8}
\end{table*}
""", encoding="utf-8")

    # ---- scheduling (B7), C2 and B5 summaries
    for s in SETS:
        f = V8 / "b7" / f"{s}_S2dl.json"
        if f.exists():
            x = json.load(open(f))
            for p in "abcde":
                put(f"v8:b7:{s}:{p}:p50", x[p]["p50"], "{:.1f}")
                put(f"v8:b7:{s}:{p}:p90", x[p]["p90"], "{:.1f}")
                put(f"v8:b7:{s}:{p}:cov", 100 * x[p]["coverage_audio"], "{:.0f}")
    PL = {"a": "one channel, as is", "b": "short interjections as captions", "c": r"skip blocks $>$10\,s late",
          "d": "b + c", "e": "one channel per speaker, mixed"}
    brows = []
    for p in "abcde":
        cells = [PL[p]]
        for s in SETS:
            f = V8 / "b7" / f"{s}_S2dl.json"
            x = json.load(open(f))[p] if f.exists() else None
            cells += [f"{x['p50']:.1f}", f"{x['p90']:.1f}", f"{100 * x['coverage_audio']:.0f}"] if x else ["--"] * 3
        brows.append(" & ".join(cells) + r" \\")
    (GEN / "tab_v8_sched.tex").write_text(r"""\begin{table*}[t]
\centering\small
\begin{tabular}{lrrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{En$\rightarrow$Zh} & \multicolumn{3}{c}{Zh$\rightarrow$En} & \multicolumn{3}{c}{AliMeeting} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Playback & p50 & p90 & spoken & p50 & p90 & spoken & p50 & p90 & spoken \\
\midrule
""" + "\n".join(brows) + r"""
\bottomrule
\end{tabular}
\caption{Playback scheduling of \sys{}-Simul's whole-meeting output (reads queued; nothing re-synthesised, so the speech
of every played block is unchanged): listener completion lag (s) of the units played, and the share of units spoken
(\%; the rest are shown as captions). prereg\_v8 amendment 1.}
\label{tab:sched8}
\end{table*}
""", encoding="utf-8")
    # B5: threshold flips
    b5 = V8 / "b5"
    if (b5 / "flips.json").exists():
        F5 = json.load(open(b5 / "flips.json"))
        put("v8:b5:frames", sum(len(r["flipped_frames"]) for r in F5), "{:d}")
        put("v8:b5:windows", sum(len(r["windows_with_flip"]) for r in F5), "{:d}")
        put("v8:b5:bound_same", "all" if all(r["windows_same"] for r in F5) else "not all")
        same = diff = dsame = dn = 0
        for S in SETS:
            f, g = b5 / f"{S}_hyp.jsonl", b5 / f"{S}_dup_hyp.jsonl"
            if f.exists():
                H = {h["id"]: h for h in map(json.loads, open(f))}
                for k in [x for x in H if not x.endswith("_flip")]:
                    if k + "_flip" in H:
                        same += H[k]["raw"] == H[k + "_flip"]["raw"]
                        diff += 1
            if g.exists():
                H = {h["id"]: h for h in map(json.loads, open(g))}
                for k in [x for x in H if not x.endswith("_dup")]:
                    if k + "_dup" in H:
                        dsame += H[k]["raw"] == H[k + "_dup"]["raw"]
                        dn += 1
        put("v8:b5:flip_same", f"{same}/{diff}")
        put("v8:b5:dup_same", f"{dsame}/{dn}" if dn else None)
    b8 = V8 / "b8"
    for tag in ("8s", "1s"):
        f = b8 / f"voiceU_{tag}_wavlm.json"
        if f.exists():
            v = json.load(open(f))
            put(f"v8:b8:{tag}:sim", v["SIM_mean"])
            put(f"v8:b8:{tag}:conf", 100 * v["confusion_rate"], "{:.1f}")
            put(f"v8:b8:{tag}:id", 100 * v["identification_in_meeting"], "{:.1f}")
            put(f"v8:b8:{tag}:simci", f"[{v['SIM_mean_ci95'][0]:.2f}, {v['SIM_mean_ci95'][1]:.2f}]")
    if (b8 / "asr_vs_input.json").exists():
        a8 = json.load(open(b8 / "asr_vs_input.json"))
        for tag in ("8s", "1s"):
            put(f"v8:b8:{tag}:asr", a8[tag]["asr_bleu_vs_talker_input"])
    if (b8 / "wait.json").exists():
        w8 = json.load(open(b8 / "wait.json"))
        put("v8:b8:wait:streams", 100 * w8["streams_delayed"], "{:.1f}")
        put("v8:b8:wait:blocks", 100 * w8["blocks_delayed"], "{:.2f}")
        put("v8:b8:wait:p50", w8["added_delay_p50"], "{:.1f}")
        put("v8:b8:wait:p90", w8["added_delay_p90"], "{:.0f}")
        put("v8:b8:avail1", 100 * w8["prompt_1s_available_at_commit"], "{:.1f}")
        put("v8:b8:avail8", 100 * w8["prompt_8s_full_at_commit"], "{:.1f}")
    # exploratory: fixed lag 2 s on the 8-meeting subset (prereg_v8 amendment 2), with Simul and LA2 on the same meetings
    for st in ("en2zh", "zh2en", "ali"):
        for tag in ("on_S2dl", "lag2", "pla2"):
            r = load(f"{st}_sub_{tag}")
            if r:
                put(f"v8:sub:{st}:{tag}:all", r["tcpBLEU_all"])
                put(f"v8:sub:{st}:{tag}:ov", r["tcpBLEU_ov"])
                put(f"v8:sub:{st}:{tag}:lagov", (r.get("lag_ov") or {}).get("p50"))
    # B6 (prereg_v8 amendment 3): wall-clock replay of one meeting per set, under contention (RTF is an upper bound)
    b6s = V8 / "b6" / "b6_summary.json"
    b6ok = False
    if b6s.exists():
        B6 = json.load(open(b6s))["runs"]
        rt = []
        for k, r in B6.items():
            if r.get("windows") and r.get("status") != "complete":
                put(f"v8:b6:{k}:done", r["windows"] - r["windows_missing"], "{:d}")
                put(f"v8:b6:{k}:of", r["windows"], "{:d}")
            if r.get("status") != "complete":
                continue
            put(f"v8:b6:{k}:rtf", r["rtf"])
            put(f"v8:b6:{k}:readp50", r["read_ms_p50"], "{:.0f}")
            put(f"v8:b6:{k}:lag50", r["text_lag_wall"]["p50"])
            put(f"v8:b6:{k}:lag90", r["text_lag_wall"]["p90"])
            put(f"v8:b6:{k}:late10", 100 * r["text_lag_wall"]["late10"], "{:.1f}")
            put(f"v8:b6:{k}:over", 100 * r["reads_over_cadence"], "{:.0f}")
            put(f"v8:b6:{k}:meeting", r["meeting"].replace("_", r"\_"))
            if k.endswith("_S2dl"):
                rt.append(r["rtf"])
        ct = json.load(open(b6s)).get("cotenancy", {})
        if ct:
            put("v8:b6:foreign_gib", np.mean([v["foreign_mib_mean"] for v in ct.values()]) / 1024, "{:.0f}")
        if len(rt) == 3:
            b6ok = max(rt) < 1
            put("v8:b6:simul:rtfmax", max(rt))
            put("v8:b6:simul:rtfmin", min(rt))
    put("v8:b6:keeps", "yes" if b6ok else "no")
    # AliMeeting fixed lag: windows of the subset decoded before the run was stopped
    sub_ali = set(json.load(open(ROOT / "data/online_v6_20261031/subset_meetings.json"))["ali"])
    wa = {json.loads(l)["id"] for l in open(ROOT / "data/online_v6_20261031/windows_test_online_ali.jsonl")
          if json.loads(l)["meeting"] in sub_ali}
    got = set()
    for f in (V8 / "runs/on_ali_S2dllag2").glob("hyp_*.jsonl"):
        for l in open(f):
            try:
                got.add(json.loads(l)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    put("v8:lag:ali:done", len(got & wa), "{:d}")
    # exploratory: training seeds, whole meetings online (seed 1 = the paper's model)
    for st, tags in (("en2zh", ("seed2", "seed3")), ("zh2en", ("seed2",)), ("ali", ("seed2",))):
        rs = [load(f"{st}_on_S2dl")] + [load(f"{st}_on_{st}_S2dl_{t}") for t in tags]
        if all(rs):
            a = np.array([r["tcpBLEU_all"] for r in rs]); o = np.array([r["tcpBLEU_ov"] for r in rs])
            put(f"v8:seed:{st}:n", len(rs), "{:d}")
            put(f"v8:seed:{st}:all", f"{a.mean():.2f}\\,$\\pm$\\,{a.std(ddof=1):.2f}")
            put(f"v8:seed:{st}:ov", f"{o.mean():.2f}\\,$\\pm$\\,{o.std(ddof=1):.2f}")
            put(f"v8:seed:{st}:min", float(a.min())); put(f"v8:seed:{st}:max", float(a.max()))
    vx = json.load(open(GEN / "v6_extras.json")).get("voices", {})
    for s in SETS:
        if s in vx:
            put(f"v8:vp:{s}:wait", vx[s].get("wait_to_1s_alone_p50_s"), "{:.1f}")
            put(f"v8:vp:{s}:ever", 100 * vx[s].get("slot_ever_alone_in_session", float("nan")), "{:.1f}")
    lines = [r"\providecommand{\num}[1]{\csname n@#1\endcsname}"]
    lines += [rf"\expandafter\def\csname n@{k}\endcsname{{{v}}}" for k, v in sorted(M.items())]
    (GEN / "macros_v8.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: (None if v is None else {kk: vv for kk, vv in v.items() if kk != "bleu"}) for k, v in T.items()}, indent=1))
    print(len(M), "macros")


if __name__ == "__main__":
    main()
