"""Summary + tier assignment for the SeamlessStreaming RealSI grid (R1): reads paper/baselines_results/seamless/<tag>_<dir>.json
and writes paper/baselines_results/seamless/summary.md (refuses an existing file; pass --out for a new name).

Tier rule (paper/baselines_run_plan_v1.md §1, the paper's rule): within each SimulS2ST-Omni budget, the grid point with
the highest ASR-BLEU whose segment LAAL D1 fits the budget; "--" where none fits. The same assignment by D0w is shown
as a secondary column.

    PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
        experiments/s2st_baselines_v1/seamless/summarize_seamless.py --grid <grid.json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "paper/baselines_results/seamless"
BUDGETS = {"en2zh": [("m1", 1.18), ("m2", 1.98), ("m3", 2.46), ("m4", 2.98)],
           "zh2en": [("m1", 1.18), ("m2", 1.92), ("m3", 2.40), ("m4", 2.98)]}
OMNI = {"en2zh": {"m1": 19.80, "m2": 24.21, "m3": 25.01, "m4": 25.54},
        "zh2en": {"m1": 14.89, "m2": 19.55, "m3": 21.07, "m4": 22.47}}
NAME = {"en2zh": "En→Zh", "zh2en": "Zh→En"}


def tier(pts: list, key: str, budget: float):
    ok = [p for p in pts if p[key] is not None and p[key] <= budget and p.get("asr_bleu") is not None]
    return max(ok, key=lambda p: p["asr_bleu"]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", default=str(RES / "summary.md"))
    a = ap.parse_args()
    grid = json.load(open(a.grid, encoding="utf-8"))
    L = [f"# SeamlessStreaming on RealSI sentence level (R1): results and tiers",
         "",
         f"Written {time.strftime('%Y-%m-%d %H:%M', time.localtime())} (Prague). Registration: "
         f"`paper/baselines_run_plan_seamless_v1.md`. Official weights and agent, inference only; one run per point "
         f"(greedy, deterministic). ASR-BLEU: speech-only audio, Whisper-large-v3 (En) / Paraformer-large (Zh), "
         f"`evaluation.text_metrics.corpus_bleu`. D1 = the paper's segment LAAL (unit at its segment's playback start, "
         f"queue included); D0w = SimulEval word LAAL (unit at its segment's emission); D0 = SimulEval default speech LAAL "
         f"(one delay per speech segment), for reference only. Computation-unaware, as in the paper.", ""]
    for d in grid["directions"]:
        pts = []
        for p in grid["points"]:
            f = RES / f"{p['tag']}_{d}.json"
            if not f.exists():
                pts.append({"tag": p["tag"], "missing": True})
                continue
            r = json.load(open(f, encoding="utf-8"))
            r["tag"] = p["tag"]
            pts.append(r)
        have = [p for p in pts if not p.get("missing")]
        L += [f"## {NAME[d]} ({have[0]['n_sentences'] if have else '?'} sentences)", "",
              "| point | decision_threshold | min_starting_wait_w2vbert | min_unit_chunk_size | ASR-BLEU | text BLEU | "
              "D1 (s) | D0w (s) | D0 (s) | StartOffset p50 (s) | ASR cover |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p, g in zip(pts, grid["points"]):
            if p.get("missing"):
                L.append(f"| {g['tag']} | {g['decision_threshold']} | {g['min_starting_wait_w2vbert']} | "
                         f"{g['min_unit_chunk_size']} | missing | | | | | | |")
                continue
            L.append(f"| {p['tag']} | {g['decision_threshold']} | {g['min_starting_wait_w2vbert']} | "
                     f"{g['min_unit_chunk_size']} | {p.get('asr_bleu')} | {p['text_bleu']} | {p['laal_d1']:.2f} | "
                     f"{p['laal_d0w']:.2f} | {p['laal_d0_defs']:.2f} | {p['start_offset_s_p50']} | "
                     f"{p.get('asr_coverage')} |")
        L += ["", f"Tiers ({NAME[d]}; best ASR-BLEU within budget):", "",
              "| tier | budget (s) | SimulS2ST-Omni (reported) | SeamlessStreaming by D1: ASR-BLEU (D1) [point] | "
              "by D0w (secondary): ASR-BLEU (D0w) [point] |", "|---|---|---|---|---|"]
        for m, b in BUDGETS[d]:
            t1, tw = tier(have, "laal_d1", b), tier(have, "laal_d0w", b)
            c1 = f"{t1['asr_bleu']:.2f} ({t1['laal_d1']:.2f}) [{t1['tag']}]" if t1 else "--"
            cw = f"{tw['asr_bleu']:.2f} ({tw['laal_d0w']:.2f}) [{tw['tag']}]" if tw else "--"
            L.append(f"| {m} | {b:.2f} | {OMNI[d][m]:.2f} | {c1} | {cw} |")
        L.append("")
    with open(a.out, "x", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
