"""v11 P2 (fluency-constrained PA, PA-F; prereg reports/mist_w3po_v1/v11_p2_fluency_pa_prereg.zh-CN.md) -> paper:
eval_outputs/mist_w3po_v1/v11_paper/fluencypa.json ({"switch": "FluencyPA", "final": true, "macros": {"v11:flu:app"},
"table": <LaTeX table, label tab:fluencypa>}).

  ASR-BLEU  the registered gate (gate_2seed, meeting bootstrap, two TTS seeds) of PA-F against MIST (B+PA):
            crf/t_eval/<text>__rlplemf/gate_0_<text>__rlplem.json, dev (both directions) and test (AMI Zh->En, Ali)
  rate      units per second of speech (v11_factorial_analysis.rate_stats), mean of the two seeds
  pauses    >= 0.5 s gaps per minute of speech: dev logs/mist_v11/p2/pauses/<run>.json; test the factorial analysis rows
  LAAL      test only, LAAL_block from the factorial analysis (rows R5 / R6, paired R6-R5)
Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.p2_paper_json --analysis <factorial analysis json with R6>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import v11_factorial_analysis as FA

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "eval_outputs/mist_w3po_v1/v11_paper/fluencypa.json"
TE = REPO / "eval_outputs/mist_w3po_v1/crf/t_eval"
PZ = REPO / "logs/mist_v11/p2/pauses"
SETS = (("devz", "Zh$\\rightarrow$En dev"), ("deve", "En$\\rightarrow$Zh dev"), ("ami", "AMI Zh$\\rightarrow$En"),
        ("ali", "AliMeeting"))


def sd(x, d=2):
    return f"+{x:.{d}f}" if x >= 0 else f"$-${-x:.{d}f}"


def sn(x, d=2):
    return f"{x:.{d}f}" if x >= 0 else f"$-${-x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    R = json.loads(Path(a.analysis).read_text())["sets"]
    rows, gate = [], {}
    for s, lab in SETS:
        t = FA.TEXT[s]
        g = json.loads((TE / f"{t}__rlplemf" / f"gate_0_{t}__rlplem.json").read_text())["items"]["speech_all"]
        gate[s] = g
        rate = [np.mean([FA.rate_stats(s, n)[0]["units_per_s"] for n in FA.runs(s, sfx)]) for sfx in ("__rlplem", "__rlplemf")]
        if s.startswith("dev"):
            pz = [np.mean([json.loads((PZ / f"{n}.json").read_text())["pauses_all"]["per_min"] for n in FA.runs(s, sfx)])
                  for sfx in ("__rlplem", "__rlplemf")]
            lat = "--"
        else:
            r5, r6 = R[s]["rows"]["R5"], R[s]["rows"]["R6"]
            pz = [r5["pauses_per_min"], r6["pauses_per_min"]]
            lat = f"{r5['laal_block']:.2f} / {r6['laal_block']:.2f}"
        ok = g["diff"] >= -0.2 and g["p_le0"] < 0.95
        rows.append(f"{lab} & {g['A']:.2f} & {g['B']:.2f} & {sd(g['diff'])} [{sn(g['ci95'][0])}, {sn(g['ci95'][1])}] & "
                    f"{'yes' if ok else 'no'} & {rate[0]:.2f} / {rate[1]:.2f} & {pz[0]:.1f} / {pz[1]:.1f} & {lat} \\\\")
    rows.insert(2, "\\midrule")
    app = (f"On development meetings PA-F keeps ASR-BLEU in Zh$\\rightarrow$En ({sd(gate['devz']['diff'])}) but not in "
           f"En$\\rightarrow$Zh ({sd(gate['deve']['diff'])}, below the gate), so only the Zh$\\rightarrow$En Talker was "
           f"tested. On test it changes ASR-BLEU by {sd(gate['ami']['diff'])} on AMI and {sd(gate['ali']['diff'])} on "
           f"AliMeeting (95\\% interval [{sn(gate['ali']['ci95'][0])}, {sn(gate['ali']['ci95'][1])}]).")
    table = ("\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lrrlcccc}\n\\toprule\n"
             " & \\multicolumn{4}{c}{ASR-BLEU} & rate & pauses & LAAL$_\\text{block}$ \\\\\n\\cmidrule(lr){2-5}\n"
             "Set & \\sys{} & PA-F & $\\Delta$ [95\\%] & gate & \\multicolumn{3}{c}{\\sys{} / PA-F} \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
             "\\caption{PA-F, the PA Talker trained with a second budget on the expected speech duration per word, against "
             "\\sys{} on the same text (two TTS seeds). Gate: $\\Delta\\ge-0.2$ and $P(\\Delta\\le0)<0.95$ (meeting "
             "bootstrap), registered before decoding. Rate: words (En) or characters (Zh) per second of speech; pauses: "
             "$\\ge$0.5\\,s per minute of speech.}\n\\label{tab:fluencypa}\n\\end{table}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"switch": "FluencyPA", "final": True, "macros": {"v11:flu:app": app}, "table": table},
                              ensure_ascii=False, indent=1))
    print(app)
    print(table)


if __name__ == "__main__":
    main()
