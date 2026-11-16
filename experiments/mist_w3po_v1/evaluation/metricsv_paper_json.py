"""v11 P1-4 (definitions: reports/mist_w3po_v1/LOG.zh-CN.md 2026-11-15 09:35 Prague) -> paper:
eval_outputs/mist_w3po_v1/v11_paper/metricsv.json ({"switch": "MetricsV", "final": true, "macros":
{"v11:asr2:gain:<set>", "v11:metricsv:app"}, "table": <LaTeX table, label tab:metricsv11>}) from
v11_metrics/summary_v11_p14.json (v11_summarize.py): second-ASR ASR-BLEU (English: parakeet-tdt-0.6b-v2; Chinese:
Whisper-large-v3) and COMET (wmt22-comet-da, re-implemented inference checked against the package example), test sets,
B vs MIST (= "e2" in the summary) and MIST+WA (= "m") vs MIST; meeting-level tests as Table sig10. Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.metricsv_paper_json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "eval_outputs/mist_w3po_v1/v11_metrics/summary_v11_p14.json"
OUT = REPO / "eval_outputs/mist_w3po_v1/v11_paper/metricsv.json"
SETS = (("ami", "AMI Zh$\\rightarrow$En"), ("en", "AMI En$\\rightarrow$Zh"), ("ali", "AliMeeting"))


def sd(x):
    return f"+{x:.2f}" if x >= 0 else f"$-${-x:.2f}"


def star(p):
    return "$^{*}$" if p < 0.05 else "$^{\\circ}$"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    S = json.loads(SRC.read_text())
    T = S["tests"]
    cm = lambda s, k, f: (S["comet"][f"{s}__{k}__asr_tts1"]["mean"] + S["comet"][f"{s}__{k}__asr_tts2"]["mean"]) / 2 if f == "speech" \
        else S["comet"][f"{s}__{k}__text"]["mean"]
    rows, mac = [], {}
    for s, lab in SETS:
        a2, c = T["asr2"][f"{s}:e2-b"], T["comet"][f"{s}:speech:e2-b"]
        mac[f"v11:asr2:gain:{s}"] = sd(a2["diff"]) + star(a2["p_holm"])
        rows.append(f"{lab} & {S['asr1'][f'{s}:b']:.2f} & {S['asr1'][f'{s}:e2']:.2f} & "
                    f"{S['asr2'][f'{s}:b']:.2f} & {S['asr2'][f'{s}:e2']:.2f} & {sd(a2['diff'])}{star(a2['p_holm'])} & "
                    f"{cm(s, 'b', 'speech'):.2f} & {cm(s, 'e2', 'speech'):.2f} & {sd(c['diff'])}{star(c['p_holm9'])} & "
                    f"{cm(s, 'b', 'text'):.2f} \\\\")
    wa = [T["asr2"][f"{s}:m-e2"] for s, _ in SETS] + [T["comet"][f"{s}:speech:m-e2"] for s, _ in SETS]
    wa_sig = any(t.get("p_holm", t.get("p_holm9", 1)) < 0.05 for t in wa)
    mac["v11:metricsv:app"] = (
        "Against B, \\sys{} gains on the second recogniser by "
        + " / ".join(mac[f"v11:asr2:gain:{s}"] for s, _ in SETS)
        + " and in speech COMET by " + " / ".join(sd(T["comet"][f"{s}:speech:e2-b"]["diff"]) + star(T["comet"][f"{s}:speech:e2-b"]["p_holm9"]) for s, _ in SETS)
        + " (AMI Zh$\\rightarrow$En / En$\\rightarrow$Zh / AliMeeting); text COMET is unchanged by construction. "
        + ("\\sys{}+WA differs from \\sys{} significantly on at least one of them."
           if wa_sig else "\\sys{}+WA differs from \\sys{} on neither metric significantly."))
    table = ("\\begin{table*}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
             " & \\multicolumn{2}{c}{ASR-BLEU} & \\multicolumn{3}{c}{ASR-BLEU, second recogniser} & "
             "\\multicolumn{3}{c}{COMET of speech} & COMET \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
             "Test set & B & \\sys{} & B & \\sys{} & $\\Delta$ & B & \\sys{} & $\\Delta$ & of text \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
             "\\caption{The speech-layer gain under a second recogniser (English: Parakeet-TDT-0.6B-v2, Chinese: "
             "Whisper-large-v3; neither is the CTC recogniser used for selection) and under COMET (wmt22-comet-da, "
             "$\\times100$, per reference unit, empty hypotheses scored). ASR-BLEU: the paper's recognisers (Whisper-large-v3 "
             "/ Paraformer). Both TTS seeds pooled; B and \\sys{} share their text. $\\Delta$: \\sys{} $-$ B, meeting "
             "sign-flip test, Holm over the test family: $^{*}$ $p_\\text{Holm}<0.05$, $^{\\circ}$ not significant.}\n"
             "\\label{tab:metricsv11}\n\\end{table*}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"switch": "MetricsV", "final": True, "macros": mac, "table": table},
                              ensure_ascii=False, indent=1))
    print(json.dumps(mac, ensure_ascii=False, indent=1))
    print(table)


if __name__ == "__main__":
    main()
