"""Numbers and tables added in paper v7 (prereg_v7), read from the evaluation outputs; nothing typed by hand.
make_tables.py / make_tables_v6.py are left as they are; this script writes generated/macros_v7.tex and
generated/tab_v7_*.tex. A number whose source does not exist yet prints as "--".

  generated/bootstrap_v7.json  scripts/bootstrap_v7.py      generated/v7_extras.json  scripts/v7_analyses.py
  generated/v7_online/*.json   scripts/online_analyses_v7.py generated/v7_nchk.json
  eval_outputs/.../gstar_v7_20261101/
      scores/{on,win,sp_on}_<set>_<sys>_full.json   reports/gstar_si_v6/meeting_score.py
      latency/<set>_<sys>_full[_ps][_ex].json       reports/gstar_si_v6/listener_latency.py
      <dir>_<sys>/score.json                        gstar_score;  qe/*.jsonl  reports/gstar_si_v7/qe_gemba.py

    python scripts/make_tables_v7.py      (from the paper folder)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables_v6 import f1, f2, load, neg, pct, pval  # noqa: E402
from systems_v7 import V7, hyp_dir  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
SC = V7 / "scores"
LAT = V7 / "latency"
M = {}
SETS = ("en2zh", "zh2en", "ali")
ARROW = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": r"AliMeeting"}


def put(key, v, fmt="{:.2f}"):
    M[key] = "--" if v is None else (fmt.format(v) if not isinstance(v, str) else v)


def score_of(dr, name):
    d = hyp_dir(dr, name)
    s = load(d / "score.json") if d else None
    return s["all"] if s else None


def table(name, body):
    (GEN / f"tab_v7_{name}.tex").write_text(body, encoding="utf-8")


def skey(k):
    return k.replace(":", "-").replace("+", "-")


NM = {"B-off": r"Qwen3-Omni, prompted", "C-ft2": r"Cascade, same-data translator", "C-turn": r"Cascade, turn-trained translator",
      "CC": r"Cascade, turns + 2\,s context", "OC": r"Cascade, reference turns (oracle)", "O2-nosd": r"\sys{} w/o speaker stream",
      "O2-nosd-relabel": r"\quad + diarizer relabelling", "O2": r"\sys{} (offline)", "O2_LA2_d1": r"\sys{} (offline) + LA2",
      "PLA2": r"\sys{}-Simul model + LA2", "SC": r"Streaming cascade (turns + LA2)", "SEEDS": r"\sys{}-Simul",
      "O2-oracle": r"\sys{} (offline), oracle stream"}
PLA2 = {"en2zh": "S2dl_LA2_full", "zh2en": "S2dlZ_LA2_d1", "ali": "S2dlZ_LA2_d1"}
SEEDS = {"en2zh": ["S2dl_d1", "S2dl_seed2_full", "S2dl_seed3_d1"], "zh2en": ["S2dl_d1", "S2dlZ_seed2_d1"],
         "ali": ["S2dl_d1", "S2dlZ_seed2_d1"]}


def seed_scores(dr):
    return [x for x in (score_of(dr, n) for n in SEEDS[dr]) if x]


def cells7(dr, key, BS, mark=None):
    """LAAL, cpBLEU_sess, BLEU_blind, Spk, DER of one system (seed mean +- sd for SEEDS)."""
    if key == "SEEDS":
        xs = seed_scores(dr)
        if not xs:
            return ["--"] * 5
        cp = [x["cpBLEU_session"] for x in xs]
        sd = f"$\\pm$" + f"{np.std(cp, ddof=1):.2f}" if len(cp) > 1 else ""
        cpc = f"{np.mean(cp):.2f}{sd}$^{{{len(cp)}}}$"
        mean = lambda f: float(np.mean([f(x) for x in xs]))   # noqa: E731
        return [f2(mean(lambda x: x["LAAL_s"])), cpc + (mark or ""), f2(mean(lambda x: x["BLEU_speaker_blind"])),
                pct(mean(lambda x: x["count_exact"])), f1(mean(lambda x: x["DER_output"]["DER"]))]
    s = score_of(dr, PLA2[dr] if key == "PLA2" else key)
    if not s:
        return ["--"] * 5
    return [f2(s.get("LAAL_s")) if s.get("LAAL_s") is not None else "--", f2(s["cpBLEU_session"]) + (mark or ""),
            f2(s["BLEU_speaker_blind"]), pct(s["count_exact"]), f1(s["DER_output"]["DER"])]


def verdict(BS, a, b):
    r = BS.get(f"{a}_vs_{b}")
    return r.get("verdict") if r else None


def main():
    BS = {k: v for k, v in (load(GEN / "bootstrap_v7.json") or {}).items() if not k.startswith("_")}
    X = load(GEN / "v7_extras.json") or {}

    # ---- whole meetings (E-on), window references, speech, listener lag
    rows = []
    for st in SETS:
        for s in ("S2dl", "O2_LA2"):
            on, win, sp = load(SC / f"on_{st}_{s}_full.json"), load(SC / f"win_{st}_{s}_full.json"), load(SC / f"sp_on_{st}_{s}_full2.json")
            k = f"v7:on:{st}:{s}"
            put(f"{k}:tcp", on and on["tcpBLEU"])
            put(f"{k}:mcp", on and on["mcpBLEU"])
            put(f"{k}:omit", on and pct(on["omitted"]))
            put(f"{k}:ins", on and pct(on["inserted"]))
            put(f"{k}:lag50", on and on["end_lag_s"]["p50"])
            put(f"{k}:lag90", on and on["end_lag_s"]["p90"])
            put(f"{k}:meetings", on and str(on["meetings"]))
            put(f"{k}:windows", on and f"{on['windows']:,}".replace(",", "{,}"))
            put(f"v7:win:{st}:{s}:tcp", win and win["tcpBLEU"])
            put(f"v7:sp:on:{st}:{s}:tcp", sp and sp["tcpBLEU"])
            for lay in ("", "_ps"):
                L = load(LAT / f"{st}_{s}_full{lay}_ex.json") or load(LAT / f"{st}_{s}_full{lay}.json")
                for f in ("comp_p50", "comp_p90", "comp_p95", "onset_p50", "backlog_p95", "backlog_max", "backlog_growth_s_per_min", "speech_share"):
                    put(f"v7:lat:{st}:{s}{lay}:{f}", L and L.get(f), "{:.1f}" if f != "speech_share" else "{:.2f}")
            L1, L2 = load(LAT / f"{st}_{s}_full_ex.json") or load(LAT / f"{st}_{s}_full.json"), load(LAT / f"{st}_{s}_full_ps_ex.json") or load(LAT / f"{st}_{s}_full_ps.json")
            name = r"\sys{}-Simul" if s == "S2dl" else r"\sys{} + LA2"
            rows.append(f"{ARROW[st] if s == 'S2dl' else ''} & {name} & {M[f'v7:on:{st}:{s}:meetings']} & {M[f'v7:win:{st}:{s}:tcp']} & "
                        f"{M[f'{k}:tcp']} & {M[f'v7:sp:on:{st}:{s}:tcp']} & {M[f'{k}:omit']} & {M[f'{k}:ins']} & {M[f'{k}:lag50']} & "
                        f"{M[f'{k}:lag90']} & {f1(L1 and L1['comp_p50'])} & {f1(L2 and L2['comp_p50'])} & {f1(L2 and L2['comp_p90'])} \\\\")
        rows.append(r"\midrule")
    table("online", "\n".join([
        r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.0pt}", r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrrrr}", r"\toprule",
        r" & & & \multicolumn{3}{c}{tcpBLEU} & & & \multicolumn{2}{c}{text end lag (s)} & \multicolumn{3}{c}{listener lag, completion (s)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){9-10}\cmidrule(lr){11-13}",
        r"Set & System & Mtgs & windows & online & speech & Omit & Ins. & p50 & p90 & 1 ch.\ p50 & per-spk p50 & per-spk p90 \\",
        r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Whole meetings run online, all test meetings (AMI 16 per direction, AliMeeting 20; no reference boundary). "
        r"tcpBLEU of the same system on the reference windows of the same meetings (windows), run online (text), and of the ASR "
        r"of its synthesised speech (speech); Omit / Ins.: unscored reference units / inserted segments (\%); text end lag: last "
        r"commit of a reference unit minus its end, including the delay until a window's end is detected; listener lag: end of a "
        r"block's speech minus the end of its source segment, with measured computation and synthesis, on one interpreter channel "
        r"(1 ch.) or one channel per speaker (Appendix~\ref{app:online7}).}", r"\label{tab:online7}", r"\end{table*}"]) + "\n")

    for st in SETS:
        w, o = load(SC / f"win_{st}_S2dl_full.json"), load(SC / f"on_{st}_S2dl_full.json")
        put(f"v7:gap:{st}", w and o and w["tcpBLEU"] - o["tcpBLEU"])

    # ---- reference-window systems of v7
    for dr in SETS:
        for name in ("O2", "C-ft2", "C-turn", "OC", "CC", "SC", "S2dl_d1", "O2_LA2_d1", "S2dl_LA2_full", "S2dlZ_LA2_d1",
                     "S2dl_seed2_full", "S2dl_seed3_d1", "S2dlZ_seed2_d1", "S2dl_lag1_sub", "S2dl_lag2_sub", "O2-nosd-relabel"):
            s = score_of(dr, name)
            k = f"v7:tx:{dr}:{name}"
            put(f"{k}:cp", s and s["cpBLEU_session"])
            put(f"{k}:win", s and s["cpBLEU_window"])
            put(f"{k}:spk", s and pct(s["count_exact"]))
            put(f"{k}:der", s and s["DER_output"]["DER"], "{:.1f}")
            put(f"{k}:laal", s and s.get("LAAL_s"))

    def cp(dr, n):
        x = score_of(dr, n)
        return x and x["cpBLEU_session"]
    for dr in ("en2zh", "zh2en"):
        g = cp(dr, "O2") - cp(dr, "C-ft2")
        put(f"v7:share:seg:{dr}", cp(dr, "OC") and max(0.0, 100 * (cp(dr, "OC") - cp(dr, "C-ft2")) / g), "{:.0f}")
        put(f"v7:share:turn:{dr}", cp(dr, "C-turn") and max(0.0, 100 * (cp(dr, "C-turn") - cp(dr, "C-ft2")) / g), "{:.0f}")
    a, b, c = cp("en2zh", "O2_LA2_d1"), cp("en2zh", "S2dl_LA2_full"), cp("en2zh", "S2dl_d1")
    put("v7:share:prefla2", a and b and c and 100 * (b - a) / (c - a), "{:.0f}")

    # ---- statistics
    reg_rows = []
    for key, r in BS.items():
        k = f"v7:bs:{skey(key)}"
        put(f"{k}:diff", neg(f2(r["diff"])))
        put(f"{k}:lo", neg(f2(r["ci95"][0])))
        put(f"{k}:hi", neg(f2(r["ci95"][1])))
        put(f"{k}:nlo", neg(f2(r["ci95_noise"][0])))
        put(f"{k}:nhi", neg(f2(r["ci95_noise"][1])))
        put(f"{k}:perm", pval(r["perm_p_exact"]))
        put(f"{k}:verdict", r.get("verdict", "--"))
        if "lomo" in r:
            put(f"{k}:lomolo", neg(f2(r["lomo"][0])))
            put(f"{k}:lomohi", neg(f2(r["lomo"][1])))
        if "per_seed" in r:
            put(f"{k}:seeds", ", ".join(neg(f2(x)) for x in r["per_seed"]))
        if "label" in r:
            lab = r["label"]
            put(f"v7:reg:{lab}:diff", neg(f2(r["diff"])))
            put(f"v7:reg:{lab}:holm", pval(r["p_holm"]))
            put(f"v7:reg:{lab}:perm", pval(r["perm_p_exact"]))
            put(f"v7:reg:{lab}:nlo", neg(f2(r["ci95_noise"][0])))
            put(f"v7:reg:{lab}:nhi", neg(f2(r["ci95_noise"][1])))
            put(f"v7:reg:{lab}:verdict", r["verdict"])
            what = {"E-on1": r"En$\rightarrow$Zh whole meetings, \sys{}-Simul vs.\ \sys{} + LA2 (tcpBLEU)",
                    "E-on2": r"Zh$\rightarrow$En whole meetings, same", "E-on3": r"AliMeeting whole meetings, same",
                    "E-cas1": r"En$\rightarrow$Zh, \sys{} vs.\ turn-trained cascade (cpBLEU$_\text{sess}$)",
                    "E-cas2": r"Zh$\rightarrow$En, same", "E-cas3": r"AliMeeting, same",
                    "E-seed-en": r"En$\rightarrow$Zh, \sys{}-Simul (seed mean) vs.\ \sys{} + LA2",
                    "E-seed-zh": r"Zh$\rightarrow$En, same"}[lab]
            reg_rows.append((lab, f"{lab} & {what} & {r['meetings']} & {neg(f2(r['diff']))} & [{neg(f2(r['ci95'][0]))}, {neg(f2(r['ci95'][1]))}] & "
                                  f"[{neg(f2(r['ci95_noise'][0]))}, {neg(f2(r['ci95_noise'][1]))}] & {pval(r['perm_p_exact'])} & "
                                  f"{pval(r['p_holm'])} & {r['verdict']} \\\\"))
    order = ["E-on1", "E-on2", "E-on3", "E-cas1", "E-cas2", "E-cas3", "E-seed-en", "E-seed-zh"]
    reg_rows = [x for _, x in sorted(reg_rows, key=lambda t: order.index(t[0]))]
    table("reg", "\n".join([r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
                            r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrl}", r"\toprule",
                            r"Test & Comparison & Mtgs & $\Delta$ & 95\% CI & + decoder noise & $p_\text{perm}$ & $p_\text{Holm}$ & verdict \\",
                            r"\midrule"] + reg_rows + [r"\bottomrule", r"\end{tabular}}",
                            r"\caption{The eight pre-registered tests of v7 (prereg\_v7). $\Delta$: first minus second system; 95\% CI: cluster "
                            r"bootstrap over meetings; + decoder noise: the same interval widened by the run-to-run noise of decoding "
                            r"(Appendix~\ref{app:stats7}); $p_\text{perm}$: exact two-sided permutation test over meeting sign flips; "
                            r"$p_\text{Holm}$: Holm-adjusted over the eight. A difference is called real only if $p_\text{Holm}<0.05$ "
                            r"and the widened interval excludes 0; otherwise ``tie''.}", r"\label{tab:reg7}", r"\end{table*}"]) + "\n")

    # ---- extras
    for k, v in X.get("count", {}).items():
        for f in ("exact", "under", "over", "wrong_speakers"):
            put(f"v7:count:{k}:{f}", pct(v[f]))
        put(f"v7:count:{k}:n", str(v["multi_windows"]))
        for b, u in v["under_by_least_speaker"].items():
            put(f"v7:count:{k}:under:{b}", pct(u["under"]))
            put(f"v7:count:{k}:share:{b}", pct(u["windows"] / max(v["multi_windows"], 1)))
    for k, v in X.get("zerolen", {}).items():
        put(f"v7:zl:{k}:zero", pct(v["zero_len"]))
        put(f"v7:zl:{k}:at20", pct(v["zero_len_at_20s_end"]))
        put(f"v7:zl:{k}:short", pct(v["zero_len_short_ref_le1s"]))
        for w in ("before", "after_fix"):
            put(f"v7:zl:{k}:{w}:cp", v[w]["cpBLEU_session"])
            put(f"v7:zl:{k}:{w}:der", v[w]["DER"], "{:.2f}")
    for dr, v in X.get("funnel", {}).items():
        put(f"v7:fun:{dr}:meeth", v["meeting_hours_upto_last_window"], "{:.1f}")
        put(f"v7:fun:{dr}:winh", v["window_audio_hours"], "{:.1f}")
        for f in ("kept_speech", "kept_overlap", "kept_short"):
            put(f"v7:fun:{dr}:{f}", pct(v[f]))
        for f in ("speech_hours", "overlap_hours", "short_hours"):
            put(f"v7:fun:{dr}:{f}", v[f], "{:.1f}")
    nchk = load(GEN / "v7_nchk.json")
    if nchk:
        put("v7:nchk:v7", nchk["tcp_v7"])
        put("v7:nchk:v6", nchk["tcp_v6"])
        put("v7:nchk:diff", neg(f2(nchk["diff"])))
        put("v7:nchk:same", pct(nchk["identical_windows"] / nchk["windows"]))
    for f in sorted((GEN / "v7_online").glob("*_full.json")):
        o = load(f)
        k = f"v7:oa:{o['set']}:{o['name'].replace('_full', '')}"
        put(f"{k}:capshare", pct(o["windows_cap_share"]))
        for c in ("silence", "cap"):
            if c in o["close"]:
                put(f"{k}:{c}:cov", pct(o["close"][c]["coverage"]))
                put(f"{k}:{c}:covnear", pct(o["close"][c]["coverage_near_boundary"]))
                put(f"{k}:{c}:covin", pct(o["close"][c]["coverage_inside"]))
                put(f"{k}:{c}:rep", pct(o["close"][c]["repeated_in_two_windows"]))
            if c in o["close_latency_s"]:
                put(f"{k}:{c}:closelat", o["close_latency_s"][c]["p50"])
        att = [v["attribution"] for v in o["drift_5min"].values() if v["covered_utts"] >= 50]
        put(f"{k}:drift:min", att and pct(min(att)))
        put(f"{k}:drift:max", att and pct(max(att)))
        put(f"{k}:reentry", pct(o["reentry"]["attribution"]))
        put(f"{k}:reentry:n", str(o["reentry"]["utts"]))
        for h in ("nested", "other"):
            if h in o["hol"]:
                put(f"{k}:hol:{h}", o["hol"][h]["commit_after_end_p50_s"])
                put(f"{k}:hol:{h}:n", str(o["hol"][h]["segments"]))
    qe = V7 / "qe"
    rows = []
    for f in sorted(qe.glob("ali_ref_gemba*.jsonl")):
        if "smoke" in f.name:
            continue
        rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    sc = np.array([r["score"] for r in rows if r.get("score") is not None])
    if len(sc):
        put("v7:qe:n", f"{len(sc):,}".replace(",", "{,}"))
        put("v7:qe:mean", float(sc.mean()), "{:.1f}")
        put("v7:qe:median", float(np.median(sc)), "{:.0f}")
        put("v7:qe:lt50", pct(float((sc < 50).mean())))
        put("v7:qe:lt70", pct(float((sc < 70).mean())))
    qer = load(GEN / "v7_qe_rank.json")
    if qer:
        for k, v in qer.items():
            put(f"v7:qerank:{k}", neg(f2(v)) if isinstance(v, (int, float)) else str(v))

    # ---- Table 1 (v7): AMI text, with the v7 cascade controls, prefix LA2, streaming cascade and the seeds
    def strongest(dr, keys):
        sc = {k: score_of(dr, PLA2[dr] if k == "PLA2" else k) for k in keys}
        sc = {k: v["cpBLEU_session"] for k, v in sc.items() if v}
        return max(sc, key=sc.get) if sc else None
    rows = []
    marks = {}
    for dr in ("en2zh", "zh2en"):
        bo = strongest(dr, ["C-ft2", "C-turn", "CC"])
        bs = strongest(dr, ["O2_LA2_d1", "PLA2", "SC"])
        put(f"v7:best:{dr}:off", NM.get(bo, "--") if bo else "--")
        put(f"v7:best:{dr}:sim", NM.get(bs, "--") if bs else "--")
        bsn = PLA2[dr] if bs == "PLA2" else bs
        marks[(dr, "O2")] = "$^\\dagger$" if bo and verdict(BS, f"{dr}:O2", f"{dr}:{bo}") == "a better" else ""
        marks[(dr, "SEEDS")] = "$^\\dagger$" if bs and verdict(BS, f"{dr}:S2dl_d1", f"{dr}:{bsn}") == "a better" else ""
        put(f"v7:bestname:{dr}:sim", bsn or "--")
        r = BS.get(f"{dr}:O2_vs_{dr}:{bo}") if bo else None
        put(f"v7:gainoff:{dr}", r and neg(f2(r["diff"])))
        r = BS.get(f"{dr}:S2dl_d1_vs_{dr}:{bsn}") if bsn else None
        put(f"v7:gainsim:{dr}", r and neg(f2(r["diff"])))
        put(f"v7:gainsim:{dr}:perm", r and pval(r["perm_p_exact"]))
    blocks = [("Offline", ["C-ft2", "C-turn", "CC", "O2-nosd", "O2-nosd-relabel", "O2"]),
              (r"Simultaneous ($\Delta{=}1$\,s)", ["O2_LA2_d1", "PLA2", "SC", "SEEDS"]),
              ("Oracle input (reference turns or activity), same translator", ["OC", "O2-oracle"])]
    for title, keys in blocks:
        rows.append(f"\\multicolumn{{11}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for k in keys:
            c = cells7("en2zh", k, BS, marks.get(("en2zh", k))) + cells7("zh2en", k, BS, marks.get(("zh2en", k)))
            if k in ("C-ft2", "C-turn", "CC", "O2-nosd", "O2-nosd-relabel", "O2", "OC", "O2-oracle"):
                c[0] = c[5] = "--"
            rows.append(f"{NM[k]} & " + " & ".join(c) + " \\\\")
        rows.append(r"\addlinespace")
    table("main", "\n".join([r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccc@{\hskip 10pt}ccccc}", r"\toprule",
        r" & \multicolumn{5}{c}{En$\rightarrow$Zh} & \multicolumn{5}{c}{Zh$\rightarrow$En} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"System & LAAL\,s & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk & DER & LAAL\,s & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk & DER \\",
        r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Speaker-attributed translation (text) on the reference windows of the AMI test meetings (2{,}375 "
        r"En$\rightarrow$Zh and 3{,}317 Zh$\rightarrow$En windows, 16 meetings). One bidirectional \sys{} (offline) serves "
        r"both directions; \sys{}-Simul is trained per direction and reported as the mean $\pm$ s.d.\ over its training "
        r"seeds (superscript: number of seeds; per seed in Table~\ref{tab:seeds7}). cpBLEU$_\text{sess}$ (primary) matches "
        r"output speakers to reference speakers once per meeting; BLEU$_\text{blind}$ ignores speakers. Spk: windows with the "
        r"right number of speakers (\%); DER (\%) of the output segments. LAAL includes the diarizer's 1.04\,s look-ahead "
        r"and is computation-unaware. $^\dagger$: better than the strongest non-oracle baseline of its block (exact "
        r"permutation test over meetings $p<0.05$ and the decoder-noise-widened interval excludes 0; Table~\ref{tab:sig7}).}",
        r"\label{tab:main7}", r"\end{table*}"]) + "\n")

    # ---- real Mandarin (AliMeeting), v7 rows
    rows = []
    for k in ["C-ft2", "C-turn", "O2-nosd", "O2-nosd-relabel", "O2", "O2_LA2_d1", "PLA2", "SEEDS"]:
        mk = ""
        if k == "O2" and verdict(BS, "ali:O2", "ali:C-ft2") == "a better":
            mk = "$^\\dagger$"
        c = cells7("ali", k, BS, mk)
        if k in ("C-ft2", "C-turn", "O2-nosd", "O2-nosd-relabel", "O2"):
            c[0] = "--"
        rows.append(f"{NM[k]} & " + " & ".join(c) + " \\\\")
        if k == "O2":
            rows.append(r"\midrule")
    table("real", "\n".join([r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.6pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{lccccc}", r"\toprule",
        r"System & LAAL\,s & cpBLEU$_\text{sess}$ & BLEU$_\text{blind}$ & Spk & DER \\", r"\midrule"] + rows +
        [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Real Mandarin meetings (AliMeeting test, Zh$\rightarrow$En, 20 meetings, 4{,}243 reference windows; "
        r"zero-shot: no AliMeeting data in training). The English references are machine translations of the human "
        r"transcripts (their automatic quality estimate: Appendix~\ref{app:qe7}).}", r"\label{tab:real7}", r"\end{table}"]) + "\n")

    # ---- seeds (appendix)
    rows = []
    for dr in SETS:
        for i, n in enumerate(SEEDS[dr]):
            sc = score_of(dr, n)
            rows.append(f"{ARROW[dr] if i == 0 else ''} & {i + 1} & " + (" & ".join([f2(sc["cpBLEU_session"]), f2(sc["cpBLEU_window"]), f2(sc["LAAL_s"]),
                        pct(sc["count_exact"]), pct(sc["empty"])]) if sc else " & ".join(["--"] * 5)) + " \\\\")
        xs = seed_scores(dr)
        if len(xs) > 1:
            cp = [x["cpBLEU_session"] for x in xs]
            rows.append(f" & mean $\\pm$ s.d. & {np.mean(cp):.2f} $\\pm$ {np.std(cp, ddof=1):.2f} & & {np.mean([x['LAAL_s'] for x in xs]):.2f} & & \\\\")
            put(f"v7:seed:{dr}:mean", float(np.mean(cp)))
            put(f"v7:seed:{dr}:sd", float(np.std(cp, ddof=1)))
            put(f"v7:seed:{dr}:n", str(len(cp)))
            put(f"v7:seed:{dr}:min", float(min(cp)))
            put(f"v7:seed:{dr}:max", float(max(cp)))
        rows.append(r"\midrule")
    table("seeds", "\n".join([r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{llccccc}", r"\toprule", r"Set & Seed & cpBLEU$_\text{sess}$ & cpBLEU$_\text{win}$ & LAAL\,s & Spk & Empty \\",
        r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabular}}",
        r"\caption{\sys{}-Simul per training seed ($\Delta{=}1$\,s, reference windows; AliMeeting uses the Zh$\rightarrow$En "
        r"models). Seed 1 is the model of v6; seeds 2 and 3 differ only in the random seed (seed 3 of En$\rightarrow$Zh was "
        r"retrained from scratch, prereg\_v7 amendment 6). Empty: windows with no output (failures).}", r"\label{tab:seeds7}",
        r"\end{table}"]) + "\n")

    # ---- all v7 comparisons (appendix)
    rows = []
    lab = {"C-turn": "turn-trained casc.", "C-ft2": "same-data casc.", "CC": "context casc.", "OC": "ref.-turn casc.",
           "O2-nosd-relabel": "relabelling", "O2_LA2_d1": "\\sys{}+LA2", "S2dl_LA2_full": "Simul model+LA2",
           "S2dlZ_LA2_d1": "Simul model+LA2", "SC": "stream. casc.", "S2dl_lag1_sub": "fixed lag 1\\,s",
           "S2dl_lag2_sub": "fixed lag 2\\,s", "S2dl_d1": "\\sys{}-Simul", "O2": "\\sys{}"}
    def nm(x):
        x = x.removeprefix("sp:")
        d, n = x.split(":", 1)
        return lab.get(n, n.replace("_", "\\_"))
    for key, r in BS.items():
        if key.startswith("on:") or key.startswith("seeds:") or "perm_p_exact" not in r:
            continue
        a, b = r["a"], r["b"]
        st = a.removeprefix("sp:").split(":")[0]
        rows.append(f"{ARROW[st]}{' (speech)' if a.startswith('sp:') else ''} & {nm(a)} vs.\\ {nm(b)} & {r['meetings']} & {neg(f2(r['diff']))} & "
                    f"[{neg(f2(r['ci95'][0]))}, {neg(f2(r['ci95'][1]))}] & [{neg(f2(r['ci95_noise'][0]))}, {neg(f2(r['ci95_noise'][1]))}] & "
                    f"{pval(r['perm_p_exact'])} & {r.get('verdict', '--')} \\\\")
    for key, r in BS.items():
        if key.startswith("on:") and "perm_p_exact" in r and "label" not in r:
            st = key.split(":")[1]
            rows.append(f"{ARROW[st]} whole mtgs & {r['a'].replace('_', '\\_')} vs.\\ {r['b'].replace('_', '\\_')} & {r['meetings']} & {neg(f2(r['diff']))} & "
                        f"[{neg(f2(r['ci95'][0]))}, {neg(f2(r['ci95'][1]))}] & [{neg(f2(r['ci95_noise'][0]))}, {neg(f2(r['ci95_noise'][1]))}] & "
                        f"{pval(r['perm_p_exact'])} & {r.get('verdict', '--')} \\\\")
    table("sig", "\n".join([r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrl}", r"\toprule", r"Set & Comparison & Mtgs & $\Delta$ & 95\% CI & + decoder noise & $p_\text{perm}$ & verdict \\",
        r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Exploratory comparisons of v7 (cpBLEU$_\text{sess}$ on reference windows, ASR-cpBLEU for speech, tcpBLEU "
        r"for whole meetings), on the windows both systems decoded. Columns as Table~\ref{tab:reg7}; no multiple-test "
        r"correction (exploratory). Fixed lag is on the 8-meeting subset (prereg\_v7 amendment 2).}", r"\label{tab:sig7}",
        r"\end{table*}"]) + "\n")

    # ---- coverage funnel (appendix)
    rows = []
    for dr in SETS:
        v = X.get("funnel", {}).get(dr)
        if not v:
            continue
        rows.append(f"{ARROW[dr]} & {v['meetings']} & {f1(v['meeting_hours_upto_last_window'])} & {f1(v['speech_hours'])} & "
                    f"{f1(v['window_audio_hours'])} & {pct(v['kept_speech'])} & {f1(v['overlap_hours'])} & {pct(v['kept_overlap'])} & "
                    f"{f1(v['short_hours'])} & {pct(v['kept_short'])} \\\\")
    table("funnel", "\n".join([r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r" & & \multicolumn{2}{c}{hours} & window & speech & \multicolumn{2}{c}{overlapped} & \multicolumn{2}{c}{$\le$1\,s utts} \\",
        r"Set & Mtgs & meetings & speech & audio h & kept & h & kept & h & kept \\", r"\midrule"] + rows +
        [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Coverage of the reference-window test sets: meeting time (to the last window), annotated speech, "
        r"reference-window audio, and the share of speech, overlapped speech and utterances of at most 1\,s that the windows "
        r"keep (\%). The whole-meeting online runs (Table~\ref{tab:online7}) cover all of it.}", r"\label{tab:funnel7}",
        r"\end{table}"]) + "\n")

    # ---- speaker counts on multi-speaker windows (appendix)
    rows = []
    for dr in SETS:
        for n in ("C-ft2", "O2", "S2dl_d1", "O2_LA2_d1"):
            v = X.get("count", {}).get(f"{dr}:{n}")
            if not v:
                continue
            u = v["under_by_least_speaker"]
            g = lambda b: pct(u[b]["under"]) if b in u else "--"   # noqa: E731
            rows.append(f"{ARROW[dr] if n == 'C-ft2' else ''} & {NM.get(n, NM['SEEDS'] if n == 'S2dl_d1' else n)} & {v['multi_windows']} & "
                        f"{pct(v['exact'])} & {pct(v['under'])} & {pct(v['over'])} & {pct(v['wrong_speakers'])} & {g('le0.5')} & {g('0.5-1')} & "
                        f"{g('1-2')} & {g('gt2')} & {g('overlapped')} & {g('not_overlapped')} \\\\")
        rows.append(r"\midrule")
    table("count", "\n".join([r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrrrr}", r"\toprule",
        r" & & & & & & & \multicolumn{6}{c}{under-count rate by the least-heard speaker} \\", r"\cmidrule(lr){8-13}",
        r"Set & System & Wins & Exact & Under & Over & Wrong & $\le$0.5\,s & 0.5--1\,s & 1--2\,s & $>$2\,s & overl. & not overl. \\",
        r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Speaker counts on multi-speaker reference windows (\%). Under / Over: fewer / more output speakers than "
        r"reference speakers; Wrong: right count, but a reference speaker gets no text after the session matching. The "
        r"under-count rate is split by how long the window's least-heard speaker talks and whether that speaker overlaps "
        r"another.}", r"\label{tab:count7}", r"\end{table*}"]) + "\n")

    # ---- zero-length segments (appendix)
    rows = []
    for k, v in X.get("zerolen", {}).items():
        dr, n = k.split(":")
        rows.append(f"{ARROW[dr]} & {NM.get(n, NM['SEEDS'] if n == 'S2dl_d1' else n)} & {pct(v['zero_len'])} & {pct(v['zero_len_at_20s_end'])} & "
                    f"{pct(v['zero_len_short_ref_le1s'])} & {f2(v['before']['cpBLEU_session'])} / {f2(v['after_fix']['cpBLEU_session'])} & "
                    f"{v['before']['DER']:.2f} / {v['after_fix']['DER']:.2f} \\\\")
    table("zerolen", "\n".join([r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrcc}", r"\toprule",
        r"Set & System & $\alpha{=}\beta$ & at 20\,s & $\le$1\,s ref. & cpBLEU before / after & DER before / after \\", r"\midrule"] + rows +
        [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Zero-length output segments (start token = end token, \% of segments), the share of them at the 20\,s end "
        r"of the time vocabulary and the share whose best-overlapping reference utterance lasts at most 1\,s (\% of zero-length "
        r"segments), and the scores before / after enforcing $\beta\ge\alpha+1$ (end moved one 80\,ms bin later, or start one "
        r"earlier at the window end).}", r"\label{tab:zerolen7}", r"\end{table}"]) + "\n")

    # ---- online analyses (appendix)
    rows = []
    for st in SETS:
        for s in ("S2dl", "O2_LA2"):
            k = f"v7:oa:{st}:{s}"
            if M.get(f"{k}:capshare", "--") == "--":
                continue
            rows.append(f"{ARROW[st] if s == 'S2dl' else ''} & {'\\sys{}-Simul' if s == 'S2dl' else '\\sys{} + LA2'} & {M[k + ':capshare']} & "
                        f"{M.get(k + ':silence:cov', '--')} & {M.get(k + ':cap:cov', '--')} & {M.get(k + ':silence:covnear', '--')} & "
                        f"{M.get(k + ':silence:covin', '--')} & {M.get(k + ':silence:rep', '--')} & {M.get(k + ':silence:closelat', '--')} & "
                        f"{M.get(k + ':cap:closelat', '--')} & {M.get(k + ':drift:min', '--')}--{M.get(k + ':drift:max', '--')} & "
                        f"{M.get(k + ':reentry', '--')} & {M.get(k + ':hol:nested', '--')} / {M.get(k + ':hol:other', '--')} \\\\")
    table("oa", "\n".join([r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrrrr}", r"\toprule",
        r" & & 20\,s & \multicolumn{2}{c}{coverage} & \multicolumn{2}{c}{silence-closed} & & \multicolumn{2}{c}{close lat.\ (s)} & attribution & re-entry & commit lag (s) \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){9-10}",
        r"Set & System & windows & silence & 20\,s cut & $\le$2\,s bound. & inside & repeat & silence & 20\,s & per 5 min & ($\ge$60\,s) & nested / other \\",
        r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}}",
        r"\caption{Whole-meeting online runs, by window type (\%, except lags). 20\,s windows: windows cut at the 20\,s cap. "
        r"Coverage: scored utterances that get an output segment, for utterances in silence-closed windows and in 20\,s-cut "
        r"windows, and (silence-closed) within 2\,s of a window boundary or inside; repeat: covered utterances translated in two "
        r"windows (the 0.2\,s / 0.3\,s padding). Close lat.: from the window's last reference speech to the moment its end is "
        r"known (median). Attribution: share of covered utterances whose segment carries the right speaker after the meeting "
        r"matching, lowest--highest over 5-minute bins with at least 50 utterances; re-entry: utterances of a speaker silent for "
        r"at least 60\,s. Commit lag (median, s after the segment's end) of segments nested in another speaker's segment vs.\ "
        r"the rest (head-of-line blocking of the start-ordered output).}", r"\label{tab:oa7}", r"\end{table*}"]) + "\n")

    # ---- pre-registration commits
    log_ = subprocess.run(["git", "log", "--reverse", "--format=%h %ad", "--date=format-local:%Y-%m-%d %H:%M",
                           "--", "docs/prereg_v7.zh-CN.md"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1], env={"TZ": "UTC", "PATH": "/usr/bin:/bin"}).stdout.split("\n")
    log_ = [l for l in log_ if l.strip()]
    names = ["registration"] + [f"amendment {i}" for i in range(1, len(log_))]
    put("v7:prereg:list", "; ".join(f"{n} \\texttt{{{l.split()[0]}}} ({' '.join(l.split()[1:])} UTC)" for n, l in zip(names, log_)), "{}")

    lines = [r"\providecommand{\num}[1]{\csname n@#1\endcsname}"]
    lines += [rf"\expandafter\def\csname n@{k}\endcsname{{{v}}}" for k, v in sorted(M.items())]
    (GEN / "macros_v7.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GEN / "numbers_v7.json").write_text(json.dumps(M, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(M), "macros")


if __name__ == "__main__":
    main()
