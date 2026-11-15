"""v11 P0-4 -> paper: eval_outputs/mist_w3po_v1/v11_paper/s4000.json ({"switch": "SfourK", "final": true,
"macros": {"v11:s4000:summary"}, "table": <LaTeX table, label tab:s4000>}) from the P0-4 test summary
(joint/wb_test_s4000_summary_v1.json: text tcpBLEU, ASR-BLEU per TTS seed) and the post-review latency analysis
(joint/post_review_v11_s4000/analysis_v1.json, chunk protocol, mean over the two seeds). Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.s4000_paper_json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
J = REPO / "eval_outputs/mist_w3po_v1/joint"
OUT = REPO / "eval_outputs/mist_w3po_v1/v11_paper/s4000.json"
RT1 = "joint_v8h6k_20261113T073658Z_s06000"
RUN = {"e2": ("b", RT1, "__rlplem"), "m": ("m", "joint_v2p_20261113T120415Z_s03500_wbzm1em1", "__rlplem"),
       "m4": ("m", "joint_v2p_20261113T120415Z_s04000_wbzm1em1", "__rlplem")}
SETS = (("ami", "AMI Zh$\\rightarrow$En"), ("en", "AMI En$\\rightarrow$Zh"), ("ali", "AliMeeting"))
NAME = {"e2": "\\sys{}", "m": "\\sys{}+WA, step 3{,}500", "m4": "\\sys{}+WA, step 4{,}000 (rule)"}


def sd(x):
    return f"+{x:.2f}" if x >= 0 else f"$-${-x:.2f}"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    S = json.loads((J / "wb_test_s4000_summary_v1.json").read_text())
    P = json.loads((J / "post_review_v11_s4000/analysis_v1.json").read_text())
    rows, val = [], {}
    for s, lab in SETS:
        for k in ("e2", "m", "m4"):
            p, r, sfx = RUN[k]
            t = f"jtest_{p}{s}_{r}"
            d = S.get(t, {})   # e2's text is B's (same summary key); ASR-BLEU from each speech run's score_asr.json
            if k == "e2":
                d = {"text_all": S[t]["text_all"]} if t in S else d
            sp = [json.loads((REPO / "eval_outputs/mist_w3po_v1/decode" / (t + sfx + x) / "score_asr.json").read_text())["tcpBLEU_all"]
                  for x in ("", "_tts2")]
            asr = sum(sp) / 2 if None not in sp else None
            L = P["latency"][s]["systems"].get(k, {})
            val[(s, k)] = (d.get("text_all"), asr)
            cells = [f"{d['text_all']:.2f}" if d.get("text_all") is not None else "--",
                     f"{asr:.2f}" if asr is not None else "--"]
            cells += [f"{L[m]['mean']:.2f}" if m in L else "--" for m in ("tlaal", "slaal_blockstart", "slaal_perword", "endoff")]
            rows.append((lab if k == "e2" else "", NAME[k], cells))
    dt = [val[(s, "m4")][0] - val[(s, "m")][0] for s, _ in SETS]
    da = [val[(s, "m4")][1] - val[(s, "m")][1] for s, _ in SETS]
    summary = (f"text tcpBLEU {' / '.join(sd(x) for x in dt)} and ASR-BLEU {' / '.join(sd(x) for x in da)} against "
               "step 3{,}500, lower on all three sets")
    body = "\n".join(f"{a} & {b} & " + " & ".join(c) + " \\\\" + ("\n\\midrule" if b == NAME["m4"] and a == "" and i < len(rows) - 1 else "")
                     for i, (a, b, c) in enumerate(rows))
    table = ("\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llrrrrrr}\n\\toprule\n"
             " & & text & ASR- & text & \\multicolumn{3}{c}{speech (s)} \\\\\n\\cmidrule(lr){6-8}\n"
             "Set & System & BLEU & BLEU & LAAL & LAAL$_\\text{block}$ & LAAL$_\\text{spread}$ & EndOff. \\\\\n\\midrule\n"
             + body + "\n\\bottomrule\n\\end{tabular}\n"
             "\\caption{The checkpoint chosen by the pre-set selection rule (step 4{,}000) against the one reported in v10 "
             "(step 3{,}500), both with $\\bias=-1$ and the playback layer, decoded once on each test set after the "
             "registration; \\sys{} for reference. Speech metrics: mean of two TTS seeds, chunk protocol.}\n"
             "\\label{tab:s4000}\n\\end{table}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"switch": "SfourK", "final": True, "macros": {"v11:s4000:summary": summary},
                               "table": table}, ensure_ascii=False, indent=1))
    print(summary)
    print(table)


if __name__ == "__main__":
    main()
