"""v11 P0-2 -> paper: eval_outputs/mist_w3po_v1/v11_paper/alieval.json ({"switch": "AliEval", "final": true,
"macros": {"v11:ae:verdict", "v11:ae:hours"}, "table": <LaTeX table, label tab:alieval>}) and the same table as
eval_outputs/mist_w3po_v1/v11_paper/tab_v11_alieval.tex, from joint/alieval_analysis/analysis_v1.json. In v11 naming
B+PA is MIST and the write-aligned rows are MIST+WA. Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.alieval_paper_json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
A = REPO / "eval_outputs/mist_w3po_v1/joint/alieval_analysis/analysis_v1.json"
V = REPO / "eval_outputs/mist_w3po_v1/v11_paper"
SESS = REPO / "data/alimeeting_eval_si_v1/session_audio/near_mix"   # meeting hours as the test set's 10.8 h


def f2(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def sd(x, d=2):
    return "--" if x is None else (f"+{x:.{d}f}" if x >= 0 else f"$-${-x:.{d}f}")


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    for o in (V / "alieval.json", V / "tab_v11_alieval.tex"):
        if o.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    R = json.load(open(A))
    import soundfile as sf
    hours = sum(sf.info(str(w)).duration for w in SESS.glob("*.wav")) / 3600
    B, L, S, Q, P = (R["bleu"]["systems"], R["latency"]["systems"], R["speech"]["systems"], R["quality"]["systems"],
                     R["block_pauses"] or {})
    TL = R["text_laal"]
    rows = [("b", "B (deadline Thinker, base Talker)", "b"), ("e2", r"\sys{} (B + playback alignment)", "b"),
            ("m", r"\sys{}+WA, $\bias=-1$, step 3{,}500", "m"), ("m4", r"\sys{}+WA, $\bias=-1$, step 4{,}000 (rule)", "m4")]

    def pz(sy):
        s = [P.get(sy, {}).get(k, {}).get("pauses_all", {}).get("per_min") for k in ("s1", "s2")]
        s = [x for x in s if x is not None]
        return sum(s) / len(s) if s else None

    body = []
    for sy, lab, tx in rows:
        lat = L[sy]
        body.append(" & ".join([lab, f2(B[sy]["text"]["all"]), f2(B[sy]["asr_all"]), f2(TL.get(tx)),
                                f2(lat["slaal_blockstart"]["mean"]), f2(lat["slaal_perword"]["mean"]),
                                f2(lat["startoff"]["mean"]), f2(lat["endoff"]["mean"]),
                                f2(S[sy]["units_per_s"]), f2(pz(sy), 1),
                                f2(Q.get(sy, {}).get("utmos_mean"))]) + r" \\")
    C = R["confirmatory"]
    lab = {"H1a_asr_bleu": "ASR-BLEU", "H1b_laal_block": r"LAAL$_\text{block}$", "H2_endoffset": "EndOffset"}
    conf = "; ".join(f"{lab[k]} {sd(v['diff'])} [{sd(v['ci'][0])}, {sd(v['ci'][1])}], $p_\\text{{Holm}}$ {pv(v['p_holm'])}"
                     for k, v in C.items())
    table = "\n".join([
        r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        r" & text & ASR- & text & \multicolumn{4}{c}{speech latency (s)} & rate & pauses & \\",
        r"\cmidrule(lr){5-8}",
        r"System & tcpBLEU & BLEU & LAAL & LAAL$_\text{block}$ & LAAL$_\text{spread}$ & start & EndOff. & (w/s) & /min & UTMOS \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
        r"\caption{Locked test on the eight unused AliMeeting evaluation meetings (" + f"{hours:.1f}" +
        r"\,h, Zh$\rightarrow$En, DiariST human references), each system decoded once; ASR-BLEU and speech metrics "
        r"averaged over two TTS seeds (UTMOS: seed 1), latencies under the chunk protocol without computation. "
        r"Confirmatory tests of \sys{} $-$ B (Holm over three; exact sign-flip over 8 meetings, minimum $p$ = 2/256; "
        r"95\% meeting-bootstrap interval): " + conf + r".}",
        r"\label{tab:alieval}", r"\end{table*}", ""])
    e2b = R["bleu"]["paired"]["e2-b"]
    me2 = R["bleu"]["paired"]["m-e2"]
    m4e2 = R["bleu"]["paired"]["m4-e2"]
    nconf = sum(v["confirmed"] for v in C.values())
    lp = R["latency"]["paired"]["e2-b"]
    verdict = (
        f"Against B, \\sys{{}} changes ASR-BLEU by {sd(C['H1a_asr_bleu']['diff'])}, the conventional block-start speech "
        f"LAAL by {sd(C['H1b_laal_block']['diff'])}\\,s and EndOffset by {sd(C['H2_endoffset']['diff'])}\\,s "
        f"({nconf} of the three pre-registered tests confirmed after Holm correction"
        + ("" if nconf == 3 else "; with eight meetings the smallest attainable two-sided $p$ is 2/256") + "); "
        f"speech start moves by {sd(lp['startoff']['diff_pooled'])}\\,s and the speaking rate rises from "
        f"{f2(S['b']['units_per_s'])} to {f2(S['e2']['units_per_s'])} words per second. "
        f"Write alignment changes text tcpBLEU by {sd(me2['text_all']['diff'])} "
        f"({sd(me2['text_all']['ci'][0])} to {sd(me2['text_all']['ci'][1])}; step 4{{,}}000: "
        f"{sd(m4e2['text_all']['diff'])}) and ASR-BLEU by {sd(me2['asr_all']['diff'])}, exploratory comparisons.")
    V.mkdir(parents=True, exist_ok=True)
    (V / "tab_v11_alieval.tex").open("x").write(table)
    with open(V / "alieval.json", "x") as fh:
        json.dump({"switch": "AliEval", "final": True,
                   "macros": {"v11:ae:verdict": verdict, "v11:ae:hours": f"{hours:.1f}"},
                   "table": table, "table_path": str((V / "tab_v11_alieval.tex").relative_to(REPO)),
                   "source": str(A.relative_to(REPO))}, fh, indent=1)
    print(verdict)


if __name__ == "__main__":
    main()
