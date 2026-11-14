"""Numbers and tables added in paper v6 (review of v5), read from the evaluation outputs; nothing typed by hand.
make_tables.py (v5's tables) is left as it is and still writes generated/macros.tex; this script writes
generated/macros_v6.tex and generated/tab_v6_*.tex.

  generated/v6_extras.json          scripts/v6_analyses.py (forced commits, filtering, same-bin, relabelling, segments,
                                    voice prompts)
  generated/bootstrap_v6.json       scripts/bootstrap_v6.py
  eval_outputs/.../gstar_v6_20261031/
      scores/{on,win}_<set>_<sys>.json   reports/gstar_si_v6/meeting_score.py
      latency/<set>_<sys>.json           reports/gstar_si_v6/listener_latency.py
      <dir>_<sys>/score.json             gstar_score of the new reference-window systems
      voice/<dir>_<sys>_{wavlm,titanet}.json, causality/cuts.jsonl, analysis/deadline_audit.json

    python scripts/make_tables_v6.py      (from the paper folder)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems_v6 import SP6, V6, hyp_dir  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
SC = V6 / "scores"
M = {}


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def put(key, v, fmt="{:.2f}"):
    M[key] = "--" if v is None else (fmt.format(v) if not isinstance(v, str) else v)


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def f1(x):
    return "--" if x is None else f"{x:.1f}"


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def neg(s):
    return s.replace("-", "$-$") if s.startswith("-") else s


def score_of(dr, name):
    d = hyp_dir(dr, name)
    s = load(d / "score.json") if d else None
    return s["all"] if s else None


def pval(p):
    if p is None:
        return "--"
    return "{<}0.001" if p < 0.001 else f"{p:.3f}"


def table(name, body):
    (GEN / f"tab_v6_{name}.tex").write_text(body, encoding="utf-8")


def main():
    X = load(GEN / "v6_extras.json") or {}
    BS = load(GEN / "bootstrap_v6.json") or {}

    # ---- filtering, same-bin, forced commits
    for dr, f in X.get("filter", {}).items():
        put(f"v6:filt:{dr}:all:dur", pct(f["all"]["kept_dur"]))
        put(f"v6:filt:{dr}:short:dur", pct(f["short_le1s"]["kept_dur"]))
        put(f"v6:filt:{dr}:ovl:dur", pct(f["overlapped"]["kept_dur"]))
        put(f"v6:filt:{dr}:hours", f"{f['all']['hours']:.1f}")
    for k, v in X.get("sameb", {}).items():
        put(f"v6:sameb:{k}", pct(v["start_eq_end"]))
    for k, v in X.get("flush", {}).items():
        if isinstance(v, dict) and "final_read_share" in v:
            put(f"v6:flush:{k}", pct(v["final_read_share"]))
    if "labels:zh2en" in X.get("flush", {}):
        put("v6:flush:labels:zh2en", pct(X["flush"]["labels:zh2en"]["deadline_after_window_end"]))

    # ---- relabelling (M06)
    for dr, v in X.get("relabel", {}).items():
        if v:
            put(f"v6:relabel:{dr}:cp", v["cpBLEU_session"])
            put(f"v6:relabel:{dr}:moved", pct(v.get("relabelled_share")))

    # ---- voice prompts (M09.1)
    for dr, v in X.get("voices", {}).items():
        put(f"v6:prompt:{dr}:avail", pct(v["causal_prompt_available"]))
        put(f"v6:prompt:{dr}:fallback", pct(v["fallback"]))
        put(f"v6:prompt:{dr}:len", f"{v['prompt_s_mean']:.1f}")
        if v.get("prompt_other_slot_mean") is not None:
            put(f"v6:prompt:{dr}:other", f"{v['prompt_other_slot_mean']:.3f}")
    # ---- batch invariance (T13)
    t13 = [load(V6 / f"t13_zh2en_{b}/score.json") for b in ("b16", "b4")]
    if all(t13):
        import glob as _g
        A = {json.loads(l)["id"]: json.loads(l)["raw"] for l in open(V6 / "t13_zh2en_b16/hyp.jsonl")}
        B = {json.loads(l)["id"]: json.loads(l)["raw"] for l in open(V6 / "t13_zh2en_b4/hyp.jsonl")}
        put("v6:t13:same", pct(sum(A[k] == B[k] for k in A) / len(A)))
        put("v6:t13:b16", t13[0]["all"]["cpBLEU_session"])
        put("v6:t13:b4", t13[1]["all"]["cpBLEU_session"])

    # ---- segment attribution (M13): appendix table
    seg = X.get("segments", {})
    rows = []
    for key, v in seg.items():
        dr, name = key.split(":")
        put(f"v6:attr:{key}:f1", pct(v["attr_F1"]))
        put(f"v6:attr:{key}:reentry", pct(v.get("covered_reentry")))
        put(f"v6:attr:{key}:short", pct(v.get("covered_short")))
        rows.append((dr, name, v))
    names = {"B-off": "Prompted", "C-ft2": "Cascade, same data", "O2-nosd": "w/o speaker stream", "O2": r"\sys{}",
             "S2dl_d1": r"\sys{}-Simul", "O2_LA2_d1": r"\sys{} + LA2", "S2dlm_d1": r"\sys{}-Simul-M"}
    arrow = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": "AliMeeting"}
    body = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
            r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrr}", r"\toprule",
            r"Set & System & P & R & F1 & Omit & Misattr. & Onset & Offset & Re-entry & $\le$1\,s \\",
            r"\midrule"]
    last = None
    for dr, name, v in rows:
        if last is not None and dr != last:
            body.append(r"\midrule")
        body.append(f"{arrow[dr] if dr != last else ''} & {names.get(name, name)} & {pct(v['attr_precision'])} & "
                    f"{pct(v['attr_recall'])} & {pct(v['attr_F1'])} & {pct(v['omitted'])} & {pct(v['misattributed'])} & "
                    f"{v['onset_med_s']:.2f} & {v['offset_med_s']:.2f} & {pct(v.get('covered_reentry'))} & "
                    f"{pct(v.get('covered_short'))} \\\\")
        last = dr
    body += [r"\bottomrule", r"\end{tabular}}",
             r"\caption{Segment-level attribution on the test windows. Each reference utterance is checked against the "
             r"output segments overlapping it (0.5\,s collar): recall (R) = covered by its own speaker after the "
             r"session matching, Omit = no segment at all, Misattr. = only other speakers' segments; precision (P) = "
             r"segments that overlap an utterance of their matched speaker. Onset / offset: median error of the nearest "
             r"segment (s). Re-entry: recall on utterances of a speaker silent for $\ge$60\,s before; $\le$1\,s: recall "
             r"on utterances of at most one second.}", r"\label{tab:attr}", r"\end{table*}"]
    table("attr", "\n".join(body) + "\n")

    # ---- deadline audit (M08)
    da = load(V6 / "analysis/deadline_audit.json")
    if da:
        for dr, rules in (("en2zh", ["word_fraction"]), ("zh2en", ["duration_v5", "char_fraction_v6"])):
            for r in rules:
                v = da[dr][r]
                put(f"v6:dl:{dr}:{r}:mae", v["mae_s"])
                put(f"v6:dl:{dr}:{r}:early", pct(v["earlier_than_evidence"]))
                put(f"v6:dl:{dr}:{r}:before", pct(v["before_evidence_even_with_1s_lag"]))
                put(f"v6:dl:{dr}:{r}:p90", v["p90_s"])

    # ---- causality (M02): cuts_v2.jsonl of reports/gstar_si_v6/causality_compare.py (80 ms pooled, as the dumps)
    cf = V6 / "causality/cuts_v2.jsonl"
    if cf.exists():
        cuts = [json.loads(l) for l in open(cf)]
        vd = [c["vs_dump"] for c in cuts]
        vp = [c["vs_longer_prefix"] for c in cuts if "vs_longer_prefix" in c]
        put("v6:caus:cuts", str(len(cuts)), "{}")
        put("v6:caus:frames", f"{sum(c['frames'] for c in vd):,}", "{}")
        put("v6:caus:maxabs", max(c["max_abs"] for c in vd), "{:.1e}")
        ndiff = sum(round(c["frac_decision_diff"] * c["frames"]) for c in vd)
        put("v6:caus:ndec", str(ndiff), "{}")
        put("v6:caus:arg", f"{100 * min(c['argmax_agree_active'] for c in vd):.2f}", "{}")
        put("v6:caus:ovlframes", f"{sum(c['frames_overlap'] for c in vd):,}", "{}")
        put("v6:caus:ovldiff", str(sum(c["dec_diff_overlap"] for c in vd)), "{}")
        put("v6:caus:mean", float(np.mean([c["mean_abs"] for c in vd])), "{:.1e}")
        put("v6:caus:slotset", f"{100 * min(c['slot_set_agree'] for c in vd):.2f}", "{}")
        put("v6:caus:selfpairs", str(len(vp)), "{}")
        put("v6:caus:selfmax", max(c["max_abs"] for c in vp) if vp else None, "{:.1e}")

    # ---- online whole meetings (M01/M03)
    for st in ("en2zh", "zh2en", "ali"):
        for s in ("S2dl", "O2_LA2", "S2dlm", "O2", "C-ft2"):
            for kind in ("on", "win", "winsub", "sp"):
                v = load(SC / f"{kind}_{st}_{s}.json")
                if v:
                    put(f"v6:{kind}:{st}:{s}:tcp", v["tcpBLEU"])
                    put(f"v6:{kind}:{st}:{s}:mcp", v["mcpBLEU"])
                    put(f"v6:{kind}:{st}:{s}:omit", pct(v["omitted"]))
                    put(f"v6:{kind}:{st}:{s}:ins", pct(v["inserted"]))
                    if "end_lag_s" in v:
                        put(f"v6:{kind}:{st}:{s}:lag50", v["end_lag_s"]["p50"])
                        put(f"v6:{kind}:{st}:{s}:lag90", v["end_lag_s"]["p90"])
            lat = load(V6 / f"latency/{st}_{s}.json")
            if lat:
                for k2, v2 in lat.items():
                    if isinstance(v2, (int, float)):
                        put(f"v6:lat:{st}:{s}:{k2}", v2)

    # ---- E-zh dev gate and subset scores (prereg_v6 §4, amendment 6)
    d0, d1 = load(V6 / "dev_zh2en_S2dl3k/score.json"), load(V6 / "dev_zh2en_S2dlchar/score.json")
    if d0 and d1:
        a0, a1 = d0["all"]["cpBLEU_session"], d1["all"]["cpBLEU_session"]
        put("v6:zhgate", f"{'passed' if a1 >= a0 else 'failed'}: {a1:.2f} vs.\\ {a0:.2f} on 800 dev windows", "{}")
    for n in ("zh2en_S2dl3k_d1", "zh2en_O2_LA2_d1"):
        v = load(V6 / f"subset/{n}_score.json")
        if v:
            put(f"v6:sub:{n}:cp", v["all"]["cpBLEU_session"])
            put(f"v6:sub:{n}:laal", v["all"]["LAAL_s"])
    for n in ("zh2en_S2dlchar_d1", "en2zh_S2dl_seed1_d1"):
        v = load(V6 / f"{n}/score.json")
        if v:
            put(f"v6:sub:{n}:cp", v["all"]["cpBLEU_session"])
            put(f"v6:sub:{n}:laal", v["all"]["LAAL_s"])

    # ---- AliMeeting speech (M07)
    for n in ("O2", "C-ft2", "S2dl_d1", "O2_LA2_d1"):
        v = load(SP6 / f"ali_{n}/score.json")
        if v:
            put(f"v6:sp:ali:{n}:cp", v["all"].get("cpASR_BLEU_session"))

    # ---- new reference-window systems
    for dr in ("en2zh", "zh2en", "ali"):
        for name in ("S2dl_LA2_d1", "S2dl_lag1", "S2dl_lag2", "SC", "OC", "CC", "C-turn", "S2dlabc_d1", "S2dlchar_d1"):
            sc = score_of(dr, name)
            if sc:
                put(f"v6:tx:{dr}:{name}:cp", sc["cpBLEU_session"])
                put(f"v6:tx:{dr}:{name}:blind", sc["BLEU_speaker_blind"])
                put(f"v6:tx:{dr}:{name}:der", f1(sc["DER_output"]["DER"]))
                if "LAAL_s" in sc:
                    put(f"v6:tx:{dr}:{name}:laal", sc["LAAL_s"])

    # ---- decoder run-to-run noise (scripts/decoder_noise.py): sd of the difference two re-decodes show on N windows
    NZ = load(GEN / "decoder_noise.json") or {}
    ONN = {"en2zh": 1412, "zh2en": 1410, "ali": 1439}

    def noise_sd(key, r):
        if not NZ:
            return None
        n = ONN[key.split(":")[1]] if key.startswith("on:") else r["windows"]
        sizes = sorted(int(k) for k in NZ if k.isdigit())
        nn = min(sizes, key=lambda x: abs(x - n))
        return NZ[str(nn)] * (nn / n) ** 0.5

    def with_noise(key, r):
        sd = noise_sd(key, r)
        if sd is None:
            return None
        lo, hi = r["meeting"]["ci95"]
        se = (hi - lo) / 3.92
        h = 1.96 * (se ** 2 + sd ** 2) ** 0.5
        return r["diff"] - h, r["diff"] + h, sd
    put("v6:noise:t13diff", f"{abs(NZ.get('diff_t13', 0)):.2f}" if NZ else None, "{}")
    for n in ("1410", "2375", "3317"):
        if n in NZ:
            put(f"v6:noise:sd{n}", NZ[n])

    # ---- bootstrap
    for key, r in BS.items():
        k = key.replace(":", "-")
        wn = with_noise(key, r)
        if wn:
            put(f"v6:bsn:{k}:lo", neg(f"{wn[0]:.2f}"))
            put(f"v6:bsn:{k}:hi", neg(f"{wn[1]:.2f}"))
            put(f"v6:bsn:{k}:beyond", "yes" if (wn[0] > 0 or wn[1] < 0) else "no", "{}")
        put(f"v6:bs:{k}:diff", neg(f"{r['diff']:+.2f}"))
        put(f"v6:bs:{k}:pm", pval(r["meeting"]["p_two_sided"]))
        put(f"v6:bs:{k}:pw", pval(r["window"]["p_two_sided"]))
        put(f"v6:bs:{k}:perm", pval(r["perm_p_two_sided"]))
        put(f"v6:bs:{k}:lo", neg(f"{r['meeting']['ci95'][0]:.2f}"))
        put(f"v6:bs:{k}:hi", neg(f"{r['meeting']['ci95'][1]:.2f}"))

    # ---- voice (M09)
    for f in sorted((V6 / "voice").glob("*.json")) if (V6 / "voice").exists() else []:
        if f.name.endswith(".rows.jsonl"):
            continue
        v = load(f)
        k = f.stem
        put(f"v6:vo:{k}:sim", v.get("SIM_mean"))
        put(f"v6:vo:{k}:conf", pct(v.get("confusion_rate")))
        put(f"v6:vo:{k}:id", pct(v.get("identification_in_meeting")))

    # voice table (M09 / T14): both SV models, confusion with meeting-bootstrap interval
    vrows = []
    for dr in ("en2zh", "zh2en"):
        for name, label in (("C-ft2", "Cascade, same data"), ("O2", r"\sys{}"), ("O2_LA2_d1", r"\sys{} + LA2"),
                            ("S2dl_d1", r"\sys{}-Simul")):
            cells = []
            for sv in ("wavlm", "titanet"):
                v = load(V6 / "voice" / f"{dr}_{name}_{sv}.json")
                if not v:
                    cells += ["--"] * 3
                    continue
                ci = v.get("confusion_rate_ci95") or [None, None]
                cells += [f"{v['SIM_mean']:.2f}", f"{pct(v['confusion_rate'])} [{pct(ci[0])}, {pct(ci[1])}]",
                          pct(v.get("identification_in_meeting"))]
            vrows.append((dr, label, cells))
    arrow2 = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En"}
    body = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r" & & \multicolumn{3}{c}{WavLM-SV} & \multicolumn{3}{c}{TitaNet-large} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"Dir. & System & SIM & Confusion [95\% CI] & ID & SIM & Confusion [95\% CI] & ID \\", r"\midrule"]
    last = None
    for dr, label, cells in vrows:
        if last is not None and dr != last:
            body.append(r"\midrule")
        body.append(f"{arrow2[dr] if dr != last else ''} & {label} & " + " & ".join(cells) + r" \\")
        last = dr
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Voice of each output speaker's synthesised speech, matched to reference speakers per meeting by the "
             r"target-language text (v5 matched Zh$\rightarrow$En output against the Chinese source text, which made its "
             r"Zh$\rightarrow$En matching close to arbitrary; corrected here). SIM: cosine to the attributed speaker's own "
             r"clean speech. Confusion: share of streams in multi-speaker windows closer to another speaker of the window "
             r"(interval: meeting bootstrap). ID: the attributed speaker is the closest of all speakers of the meeting. "
             r"Cosines of the two models are not comparable in scale.}", r"\label{tab:voice}", r"\end{table*}"]
    table("voice", "\n".join(body) + "\n")

    # ---- significance (replaces v5's Table 9): two-sided, meeting level, Holm over the registered family
    from pairs_v6 import REGISTERED
    reg = [(k, BS[k]["perm_p_two_sided"]) for k in REGISTERED if k in BS]
    holm = {}
    for i, (k, p_) in enumerate(sorted(reg, key=lambda kv: kv[1])):
        holm[k] = max([min(1.0, (len(reg) - j) * pp) for j, (_, pp) in enumerate(sorted(reg, key=lambda kv: kv[1])[: i + 1])])
    for k, v in holm.items():
        put(f"v6:holm:{REGISTERED[k]}", pval(v))
        put(f"v6:reg:{REGISTERED[k]}:diff", neg(f"{BS[k]['diff']:+.2f}"))
        put(f"v6:reg:{REGISTERED[k]}:lo", neg(f"{BS[k]['meeting']['ci95'][0]:.2f}"))
        put(f"v6:reg:{REGISTERED[k]}:hi", neg(f"{BS[k]['meeting']['ci95'][1]:.2f}"))
        put(f"v6:reg:{REGISTERED[k]}:perm", pval(BS[k]["perm_p_two_sided"]))

    AR = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": "Ali"}
    NM = {"O2": r"\sys{}", "C-ft2": "cascade", "O2-nosd-relabel": "relabel", "OC": "ref.-turn casc.",
          "CC": "context casc.", "C-turn": "turn-trained casc.", "S2dl_d1": r"\sys{}-Simul", "O2_LA2_d1": r"\sys{}+LA2",
          "S2dl_LA2_d1": "Simul model+LA2", "S2dlchar_d1": "char. deadlines", "S2dl_seed1_d1": "Simul, seed 2",
          "S2_b0_d1": "stable-prefix", "on_S2dl": r"\sys{}-Simul", "on_O2_LA2": r"\sys{}+LA2", "on_S2dlm": r"\sys{}-Simul-M"}

    def name_of(x):
        sp = x.startswith("sp:")
        x = x.removeprefix("sp:")
        n = x.split(":")[-1]
        return NM.get(n, n) + (" (speech)" if sp else "")

    rows = []
    for k, r in BS.items():
        if k.startswith("on:"):
            dr = k.split(":")[1]
            a = f"{AR[dr]} whole meetings: {name_of(r['a'])} vs {name_of(r['b'])}"
        else:
            dr = r["a"].removeprefix("sp:").split(":")[0]
            a = f"{AR[dr]}: {name_of(r['a'])} vs {name_of(r['b'])}"
        fam = REGISTERED.get(k, "")
        rows.append((0 if fam else 1, fam, a, k, r))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    body = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
            r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrr}", r"\toprule",
            r"Test & Comparison (A vs B) & $\Delta$ & 95\% CI (meetings) & + decoder noise & $p_\text{perm}$ & $p_\text{boot,m}$ & $p_\text{boot,w}$ & $p_\text{Holm}$ \\",
            r"\midrule"]
    first_expl = True
    for grp, fam, a, k, r in rows:
        if grp == 1 and first_expl:
            body.append(r"\midrule \multicolumn{9}{l}{\emph{Exploratory (not corrected)}} \\")
            first_expl = False
        lo, hi = r["meeting"]["ci95"]
        dif = neg(f"{r['diff']:+.2f}")
        los, his = neg(f"{lo:.2f}"), neg(f"{hi:.2f}")
        wn = with_noise(k, r)
        nz = f"[{neg(f'{wn[0]:.2f}')}, {neg(f'{wn[1]:.2f}')}]" if wn else "--"
        body.append(f"{fam} & {a} & {dif} & [{los}, {his}] & {nz} & "
                    f"{pval(r['perm_p_two_sided'])} & {pval(r['meeting']['p_two_sided'])} & {pval(r['window']['p_two_sided'])} & "
                    f"{pval(holm[k]) if k in holm else ''} \\\\")
    body += [r"\bottomrule", r"\end{tabular}}",
             r"\caption{Paired comparisons of cpBLEU$_\text{sess}$ (whole meetings: tcpBLEU; speech: ASR-cpBLEU$_\text{sess}$), "
             r"A minus B. CI: 2.5--97.5\% of 1{,}000 bootstrap resamples of whole meetings (slots are matched per meeting, so "
             r"this equals re-matching on every resample); none of the $p$ values and only the ``+ decoder noise'' interval account for the decoder's run-to-run non-determinism: that interval widens the meeting interval by the spread two re-decodes of one system show on a test of the same size (Appendix~\ref{app:noise}), and we call a difference real only if it excludes 0. $p_\text{perm}$: two-sided permutation test that swaps the two "
             r"systems' outputs per meeting (10{,}000 draws); $p_\text{boot,m}$ / $p_\text{boot,w}$: two-sided bootstrap over "
             r"meetings / windows. $p_\text{Holm}$: Holm correction of $p_\text{perm}$ over the five pre-registered tests "
             r"(E-on1--3, E-seed, E-zh; prereg\_v6 and its five amendments). Everything below the line is exploratory. "
             r"$p<0.001$: fewer than one resample in 1{,}000.}", r"\label{tab:sig6}", r"\end{table*}"]
    table("sig", "\n".join(body) + "\n")

    # ---- new baselines (main table, review M04/M05/M06)
    def cp(dr, name):
        sc = score_of(dr, name)
        return sc["cpBLEU_session"] if sc else None

    def lat(dr, name):
        sc = score_of(dr, name)
        return sc.get("LAAL_s") if sc else None

    def d_vs(a, b):
        r = BS.get(f"{a}_vs_{b}")
        if r is None:
            return "--"
        p_ = r["perm_p_two_sided"]
        mark = "$^{***}$" if p_ < 0.001 else "$^{**}$" if p_ < 0.01 else "$^{*}$" if p_ < 0.05 else ""
        return neg(f"{r['diff']:+.2f}") + mark
    off = [("C-ft2", "Cascade, same-data translator"), ("C-turn", "Cascade, turn-trained translator (not run)"),
           ("CC", "Cascade, turns + 2\\,s context (not run)"), ("OC", "Cascade, reference turns and speakers (not run)"),
           ("O2-nosd-relabel", "No speaker stream + diarizer relabelling"), ("O2", r"\sys{} (offline)")]
    body = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{\begin{tabular}{lrrrrrr}",
            r"\toprule", r" & \multicolumn{2}{c}{En$\rightarrow$Zh} & \multicolumn{2}{c}{Zh$\rightarrow$En} & \multicolumn{2}{c}{AliMeeting} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Offline system & cpBLEU & $\Delta_{\sys{}}$ & cpBLEU & $\Delta_{\sys{}}$ & cpBLEU & $\Delta_{\sys{}}$ \\", r"\midrule"]
    for n, lab in off:
        cells = []
        for dr in ("en2zh", "zh2en", "ali"):
            v = cp(dr, n)
            cells += [f2(v), "" if n == "O2" or v is None else d_vs(f"{dr}:O2", f"{dr}:{n}")]
        body.append(f"{lab} & " + " & ".join(cells) + r" \\")
    body += [r"\midrule", r"Simultaneous ($\Delta{=}1$\,s) & cpBLEU & LAAL & cpBLEU & LAAL & cpBLEU & LAAL \\", r"\midrule"]
    sim = [("O2_LA2_d1", r"\sys{} (offline) + LA2"), ("S2dl_d1", r"\sys{}-Simul"),
           ("S2dl_seed1_d1", r"\sys{}-Simul, second seed$^\dagger$"), ("S2dlchar_d1", r"\sys{}-Simul, character-timed deadlines$^\dagger$")]
    for n, lab in sim:
        cells = []
        for dr in ("en2zh", "zh2en", "ali"):
            cells += [f2(cp(dr, n)), f2(lat(dr, n))]
        body.append(f"{lab} & " + " & ".join(cells) + r" \\")
    sub = load(V6 / "subset/en2zh_S2dl_d1_score.json")
    lsub = score_of("en2zh", "S2dl_LA2_d1")
    osub = load(V6 / "subset/en2zh_O2_LA2_d1_score.json")
    for k_, v_ in (("simul", sub["all"] if sub else None), ("prefla2", lsub), ("offla2", osub["all"] if osub else None)):
        if v_:
            put(f"v6:sub:{k_}:cp", v_["cpBLEU_session"])
            put(f"v6:sub:{k_}:laal", v_["LAAL_s"])
            put(f"v6:sub:{k_}:blind", v_["BLEU_speaker_blind"])
    body += [r"\bottomrule", r"\end{tabular}}",
             r"\caption{Stronger baselines on the reference windows (cpBLEU$_\text{sess}$). $^\dagger$On the 8-meeting subset of \S\ref{sec:res-online} only. En$\rightarrow$Zh on the same subset: \sys{} + LA2 " + (f2(osub["all"]["cpBLEU_session"]) if osub else "--")
             + r", LA2 on the prefix-trained \sys{}-Simul model " + (f2(lsub["cpBLEU_session"]) if lsub else "--")
             + r", \sys{}-Simul " + (f2(sub["all"]["cpBLEU_session"]) if sub else "--")
             + r" ($\Delta$ vs.\ the former " + d_vs("en2zh:S2dl_d1", "en2zh:S2dl_LA2_d1") + r"). $\Delta_{\sys{}}$: \sys{} minus the "
             r"system, with the two-sided meeting permutation test ($^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$; "
             r"Table~\ref{tab:sig6}). ``--'': not run (prereg\_v6 amendments; the three cascade controls were pre-registered and cancelled for compute in amendments 5 and 6).}", r"\label{tab:base6}", r"\end{table*}"]
    table("base", "\n".join(body) + "\n")

    # ---- whole meetings online (main table, review M01/M03)
    body = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{llrrrrrrrrr}",
            r"\toprule",
            r" & & \multicolumn{2}{c}{tcpBLEU} & & & \multicolumn{2}{c}{text end lag (s)} & \multicolumn{3}{c}{listener lag, one channel (s)} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
            r"Set & System & windows & online & Omit & Ins. & p50 & p90 & p50 & p90 & backlog p95 \\", r"\midrule"]
    for st, lab in (("en2zh", r"En$\rightarrow$Zh"), ("zh2en", r"Zh$\rightarrow$En"), ("ali", "AliMeeting")):
        for s_, nm in (("S2dl", r"\sys{}-Simul"), ("O2_LA2", r"\sys{} + LA2")):
            w = load(SC / f"winsub_{st}_{s_}.json")
            o = load(SC / f"on_{st}_{s_}.json")
            lt = load(V6 / f"latency/{st}_{s_}.json")
            el = (o or {}).get("end_lag_s", {})
            body.append(f"{lab if s_ == 'S2dl' else ''} & {nm} & {f2(w['tcpBLEU']) if w else '--'} & {f2(o['tcpBLEU']) if o else '--'} & "
                        f"{pct(o['omitted']) if o else '--'} & {pct(o['inserted']) if o else '--'} & "
                        f"{f2(el.get('p50'))} & {f2(el.get('p90'))} & "
                        f"{f1(lt['comp_p50']) if lt else '--'} & {f1(lt['comp_p90']) if lt else '--'} & {f1(lt['backlog_p95']) if lt else '--'} \\\\")
        if st != "ali":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Whole meetings run online (8 AMI test meetings per direction, 10 AliMeeting meetings; no reference "
             r"boundary). tcpBLEU of the same system on the reference windows of the same meetings (windows) and run "
             r"online; Omit / Ins.: unscored reference units / inserted segments (\%); text end lag: last commit of a reference unit "
             r"minus its end, including the delay until a window's end is detected; listener lag: end of a block's speech "
             r"on one interpreter channel minus the end of its source segment, with measured computation and synthesis "
             r"(Appendix~\ref{app:online}).}", r"\label{tab:online}", r"\end{table*}"]
    table("online", "\n".join(body) + "\n")
    for st in ("en2zh", "zh2en", "ali"):
        for s_ in ("S2dl", "O2_LA2"):
            for suf in ("", "_ps"):
                lt = load(V6 / f"latency/{st}_{s_}{suf}.json")
                if lt:
                    for k2 in ("comp_p50", "comp_p90", "comp_p95", "onset_p50", "backlog_p95", "backlog_max", "speech_share",
                               "backlog_growth_s_per_min", "read_ms", "tok_rate"):
                        if lt.get(k2) is not None:
                            put(f"v6:lat:{st}:{s_}{suf}:{k2}", lt[k2], "{:.2f}" if k2 != "read_ms" else "{:.0f}")

    for st in ("en2zh", "zh2en", "ali"):
        w, o = load(SC / f"winsub_{st}_S2dl.json"), load(SC / f"on_{st}_S2dl.json")
        if w and o:
            put(f"v6:online:gap:{st}", w["tcpBLEU"] - o["tcpBLEU"], "{:.1f}")
    ON = Path(__file__).resolve().parents[3] / "data/online_v6_20261031"
    for st in ("en2zh", "zh2en", "ali"):
        f = ON / f"windows_test_online_{st}_sub.jsonl"
        if f.exists():
            put(f"v6:onwin:{st}", f"{sum(1 for _ in open(f)):,}", "{}")

    # ---- tcpBLEU collar sensitivity (appendix)
    cd = V6 / "scores/collar"
    crow = []
    for st in ("en2zh", "zh2en", "ali"):
        for s_ in ("S2dl", "O2_LA2"):
            for kind in ("winsub", "on"):
                name = f"{kind}_{st}_{s_}"
                vals = [load(cd / f"{name}_c{c}.json") for c in ("0", "0.5", "1", "2")]
                if any(vals):
                    crow.append((st, s_, kind, [f2(v["tcpBLEU"]) if v else "--" for v in vals]))
    AR2 = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En", "ali": "AliMeeting"}
    body = [r"\begin{table}[h]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{\begin{tabular}{lllrrrr}",
            r"\toprule", r"Set & System & Run & 0\,s & 0.5\,s & 1\,s & 2\,s \\", r"\midrule"]
    for st, s_, kind, v in crow:
        body.append(f"{AR2[st]} & {'Simul' if s_ == 'S2dl' else 'LA2'} & {'windows' if kind == 'winsub' else 'online'} & "
                    + " & ".join(v) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}}",
             r"\caption{tcpBLEU on the whole-meeting subset at four collars (Simul: \sys{}-Simul; LA2: \sys{} + LA2; windows: "
             r"the reference-window run of the same system, scored on the same meetings).}", r"\label{tab:collar}", r"\end{table}"]
    table("collar", "\n".join(body) + "\n")

    # ---- pre-registration commits (appendix): hash and UTC time of prereg_v6 and each amendment
    import subprocess
    log_ = subprocess.run(["git", "log", "--reverse", "--format=%h %ad", "--date=format-local:%Y-%m-%d %H:%M",
                           "--", "docs/prereg_v6.zh-CN.md"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1], env={"TZ": "UTC", "PATH": "/usr/bin:/bin"}).stdout.split("\n")
    log_ = [l for l in log_ if l.strip()]
    names = ["registration"] + [f"amendment {i}" for i in range(1, len(log_))]
    put("v6:prereg:list", "; ".join(f"{n} \\texttt{{{l.split()[0]}}} ({' '.join(l.split()[1:])} UTC)" for n, l in zip(names, log_)), "{}")

    lines = [r"\providecommand{\num}[1]{\csname n@#1\endcsname}"]
    lines += [rf"\expandafter\def\csname n@{k}\endcsname{{{v}}}" for k, v in sorted(M.items())]
    (GEN / "macros_v6.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GEN / "numbers_v6.json").write_text(json.dumps(M, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(M), "macros")


if __name__ == "__main__":
    main()
