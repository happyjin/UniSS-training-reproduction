"""v11 P1-1 (prereg reports/mist_w3po_v1/v11_p11_wallclock_prereg.zh-CN.md) -> paper:
eval_outputs/mist_w3po_v1/v11_paper/wallclock.json ({"switch": "Wallclock", "final": true, "macros": {"v11:wc:verdict",
"v11:wc:app"}, "table": <LaTeX table, label tab:wallclock>}) from wallclock_v11/wallclock_summary.json
(wallclock_v11_launch.py). Run names: "b" = B's Thinker + PA Talker = v11's MIST; "m" = stage 2 + bias + PA = MIST+WA.
Verdict rule (registered): Thinker and Talker real-time factor both < 1 and |slope of the computation-aware minus
computation-free lag| < 0.05 s/min -> keeps up; otherwise it does not, as measured. Refuses to overwrite.

    python -m experiments.mist_w3po_v1.evaluation.wallclock_paper_json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "eval_outputs/mist_w3po_v1/wallclock_v11/wallclock_summary.json"
OUT = REPO / "eval_outputs/mist_w3po_v1/v11_paper/wallclock.json"
SETS = (("ami", "AMI Zh$\\rightarrow$En"), ("en", "AMI En$\\rightarrow$Zh"), ("ali", "AliMeeting"))
SYS = (("b", "\\sys{}"), ("m", "\\sys{}+WA"))
LAY = "per-speaker"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    D = json.loads(SRC.read_text())["runs"]
    rows, V = [], {}
    for k, name in SYS:
        for s, lab in SETS:
            R = D[f"{k}_{s}"]
            x = R["result"]
            th = {m: v for m, v in x["thinker"].items() if isinstance(v, dict) and "rtf" in v}
            L = x["layouts"][LAY]
            v = {"th": x["thinker_rtf_all"], "th_max": max(t["rtf"] for t in th.values()), "tk": x["talker"]["rtf"],
                 "over": sum(t["over_cadence"] * t["reads"] for t in th.values()) / sum(t["reads"] for t in th.values()),
                 "d50": L["ca_minus_cf_end"]["p50"], "d90": L["ca_minus_cf_end"]["p90"],
                 "slope": L["ca_minus_cf_slope_s_per_min"], "cf50": L["cf"]["end_lag"]["p50"],
                 "contended": R["talker_attempt"]["foreign_samples"] > 0 or R["thinker_attempt"]["foreign_samples"] > 0,
                 "n": len(th)}
            V[(k, s)] = v
            mark = "$^\\dagger$" if v["contended"] else ""
            rows.append(f"{name if s == 'ami' else ''} & {lab} ({v['n']}) & {v['th']:.2f} ({v['th_max']:.2f}) & "
                        f"{v['tk']:.2f}{mark} & {100 * v['over']:.0f} & {v['cf50']:.1f} & +{v['d50']:.1f} / +{v['d90']:.1f} & "
                        f"{v['slope']:+.2f} \\\\")
        rows.append("\\midrule")
    rows.pop()
    keeps = all(V[("b", s)]["th"] < 1 and V[("b", s)]["tk"] < 1 and abs(V[("b", s)]["slope"]) < 0.05 for s, _ in SETS)
    b = [V[("b", s)] for s, _ in SETS]
    j = lambda f, key: " / ".join(f.format(x[key]) for x in b)
    verdict = (f"Replayed with measured compute times on one GPU per system, \\sys{{}}'s Thinker needs "
               f"{j('{:.2f}', 'th')}\\,s of computation per second of meeting (AMI Zh$\\rightarrow$En / En$\\rightarrow$Zh "
               f"/ AliMeeting) and its PA Talker {j('{:.2f}', 'tk')}; computation delays the median block's end by a further "
               f"{j('{:.1f}', 'd50')}\\,s, and the delay grows over a meeting, so ")
    verdict += ("one GPU keeps up with a meeting" if keeps else
                "as implemented (batch-1 decoding that re-encodes the prefix at every read) it does not keep up in real time")
    app = ("The Thinker is the bottleneck: reads take longer than the 1\\,s cadence "
           f"{100 * min(x['over'] for x in b):.0f}--{100 * max(x['over'] for x in b):.0f}\\% of the time and the queue of "
           "unprocessed reads builds up in the densest meetings (one AliMeeting meeting needs "
           f"{V[('b', 'ali')]['th_max']:.2f}\\,s per second), while the Talker uses about a third of real time or less. "
           "Incremental encoding of the audio prefix (a KV cache across reads) and batching speakers would address the "
           "Thinker; we did not implement either.")
    table = ("\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llrrrrrr}\n\\toprule\n"
             " & & \\multicolumn{2}{c}{real-time factor} & reads & block end (s) & \\multicolumn{2}{c}{added by computation} \\\\\n"
             "\\cmidrule(lr){3-4}\\cmidrule(lr){7-8}\n"
             "System & Set (meetings) & Thinker (max) & Talker & $>$1\\,s (\\%) & no comp.\\ p50 & p50 / p90 (s) & s/min \\\\\n"
             "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
             "\\caption{Computation-aware replay of whole meetings (meetings fixed before measuring: the 7th--9th AMI test "
             "meetings by length, both directions, and the 10th--11th AliMeeting test meetings). One decoder stream "
             "(batch 1) and one Talker worker per meeting, each on its own GPU; real-time factor = computation / meeting "
             "span (Thinker: pooled over meetings, max over meetings in brackets); block end: from the source end of a "
             "block's text to the end of its playback, one channel per speaker; added: computation-aware minus "
             "computation-free, per block, and its slope over meeting time. $^\\dagger$ Talker timing taken while "
             "another process shared the GPU (three attempts each, all contended).}\n\\label{tab:wallclock}\n\\end{table}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"switch": "Wallclock", "final": True,
                               "macros": {"v11:wc:verdict": verdict, "v11:wc:app": app}, "table": table},
                              ensure_ascii=False, indent=1))
    print(verdict)
    print(app)
    print(table)


if __name__ == "__main__":
    main()
