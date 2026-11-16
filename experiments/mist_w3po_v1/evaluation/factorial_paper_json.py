"""v11 P0-1 -> paper: eval_outputs/mist_w3po_v1/v11_paper/factorial.json ({"switch": "Factorial", "final": true,
"macros": {"v11:fact:verdict"}, "table": <LaTeX table, label tab:factorial>}) from the factorial analysis JSON written by
v11_factorial_analysis.py on the test sets (ami / ali / en). The verdict follows the registration
(reports/mist_w3po_v1/v11_p01_factorial_prereg.zh-CN.md §5): "the learned part still adds after rate matching" only
when R5 - R3' ASR-BLEU is Holm-significant and positive on at least two test sets; otherwise the fallback wording.
Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.factorial_paper_json --analysis <analysis json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "eval_outputs/mist_w3po_v1/v11_paper/factorial.json"
SETS = ("ami", "en", "ali")
ROWS = (("R0", "B"), ("R1", "+ read-level merging"), ("R2", "+ backlog rate rule"), ("R3", "+ best-of-2 CTC"),
        ("R3p", "R3, rate-matched"), ("R4", "R2 + PA Talker"), ("R5", "\\sys{} (R4 + best-of-2)"))


def sd(x, d=2):
    return f"+{x:.{d}f}" if x >= 0 else f"$-${-x:.{d}f}"


def cell(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    R = json.loads(Path(a.analysis).read_text())
    S = R["sets"]
    lines = []
    for row, lab in ROWS:
        v = [S[s]["rows"].get(row, {}) for s in SETS]
        c = [cell(x.get("asr_all")) for x in v] + [cell(x.get("laal_block")) for x in v] + \
            [cell(v[0].get("rate")), cell(v[0].get("pauses_per_min"), 1)]
        name = row.replace("R3p", "R3$^\\prime$")
        lines.append(f"{name} {lab} & " + " & ".join(c) + " \\\\")
    H = R.get("primary_holm", {})
    P = {s: S[s]["paired"].get("R5-R3p", {}) for s in SETS}
    d_asr = [P[s]["asr_all"]["diff"] for s in SETS]
    d_lat = [P[s]["laal_block"]["diff_pooled"] for s in SETS]
    h_asr = [H[f"{s}:asr_all"]["p_holm"] for s in SETS]
    h_lat = [H[f"{s}:laal_block"]["p_holm"] for s in SETS]
    pos = sum(1 for s, h in zip(SETS, h_asr) if h < 0.05 and P[s]["asr_all"]["diff"] > 0
              and P[s]["asr_all"]["ci95"][0] > 0)
    fmt_p = lambda ps: " / ".join("$<$0.001" if p < 0.001 else f"{p:.3f}" for p in ps)
    lat_ns = all(h >= 0.05 for h in h_lat)
    lat_txt = (f"LAAL$_\\text{{block}}$ by at most {max(abs(x) for x in d_lat):.3f}\\,s (none significant)" if lat_ns else
               f"LAAL$_\\text{{block}}$ by {' / '.join(sd(x, 3) for x in d_lat)}\\,s ($p_\\text{{Holm}}$ {fmt_p(h_lat)})")
    nums = (f"Against R3$^\\prime$, \\sys{{}} changes ASR-BLEU by {' / '.join(sd(x) for x in d_asr)} "
            f"($p_\\text{{Holm}}$ {fmt_p(h_asr)}) and {lat_txt}")
    if pos >= 2:
        verdict = nums + (". The learned Talker therefore adds to what merging, a rate rule and selection give, also when "
                          "the rule speaks as fast.")
    else:
        # registered fallback (prereg section 5); the sets where R5 - R3' is nevertheless significant are named
        NAMES = {"ami": "AMI Zh$\\rightarrow$En", "en": "AMI En$\\rightarrow$Zh", "ali": "AliMeeting"}
        sig = [NAMES[s] for s, h in zip(SETS, h_asr) if h < 0.05 and P[s]["asr_all"]["diff"] > 0]
        step = lambda k: " / ".join(sd(S[s]["paired"][k]["asr_all"]["diff"]) for s in SETS)
        verdict = nums + (": most of the gain comes from the backlog-dependent rate (R2$-$R1 "
                          f"{step('R2-R1')}) and candidate selection (R3$-$R2 {step('R3-R2')}; exploratory), and the "
                          "learned Talker adds a significant gain " + (f"only on {sig[0]}." if len(sig) == 1 else
                                                                         "on none of the test sets."))
    rate3, rate5 = S["ami"]["rows"].get("R3p", {}).get("rate"), S["ami"]["rows"].get("R5", {}).get("rate")
    table = ("\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
             " & \\multicolumn{3}{c}{ASR-BLEU} & \\multicolumn{3}{c}{LAAL$_\\text{block}$ (s)} & rate & pause \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
             "Speech layer on B's text & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali & Z$\\rightarrow$E & E$\\rightarrow$Z & Ali & w/s & /min \\\\\n"
             "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
             "\\caption{Factorial ablation of the playback layer on the test sets, B's text throughout, each row decoded "
             "once with two TTS seeds. R3$^\\prime$ raises the rule's rate shift until its mean development rate matches "
             f"\\sys{{}}'s (shift {S['ami'].get('r3p_shift')} / {S['en'].get('r3p_shift')}; AMI Zh$\\rightarrow$En rate "
             f"{cell(rate3)} vs.\\ {cell(rate5)}). Rate and pauses: AMI Zh$\\rightarrow$En.}}\n"
             "\\label{tab:factorial}\n\\end{table}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"switch": "Factorial", "final": True, "macros": {"v11:fact:verdict": verdict},
                               "table": table}, ensure_ascii=False, indent=1))
    print(verdict)
    print(table)


if __name__ == "__main__":
    main()
