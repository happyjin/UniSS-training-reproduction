"""The fluency plan's comparison (docs/m2_fluency_plan_v1.zh-CN.md §4): for si_infer runs on the same RealSI sentences,
the paper's metrics and the fluency metrics side by side, and each run against a base run.

  paper      speech-only ASR-BLEU (the speech segments transcribed, as collect_results.py) and LAAL with every unit at
             its block's start (computation-unaware, ms -> s); also the Talker's text BLEU, first sound and end offset
  fluency    si_fluency_stats on the timeline audio: pauses, seams, queue, pace, blocks, Talker errors
  quality    UTMOS (speech only) when --utmos gives per-sentence scores {run, id, utmos}
  vs base    paired bootstrap of the ASR-BLEU difference (si_paired_bootstrap), and the paired UTMOS difference

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_compare --base m2 --run m2=<realsi_dir> \
        --run hc=<realsi_dir> [--utmos <jsonl>] --out <json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluation.text_metrics import corpus_bleu
from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import analyse
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units


def paper_metrics(run: Path) -> dict:
    recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(run / "asr_speech.jsonl", encoding="utf-8"))}
    lang = recs[0]["tgt_lang"]
    refs = [r["translation_ref"] for r in recs]
    seg, first, end = [], [], []
    for r in recs:
        ds = [b["start_ms"] for b in r["blocks"] for _ in units(b["text"], lang)]
        if ds:
            seg.append(laal(ds, r["source_duration_ms"], len(units(r["translation_ref"], lang))))
        if r["blocks"]:
            first.append(r["blocks"][0]["start_ms"] / 1000)
            end.append((max(b["start_ms"] + b["dur_ms"] for b in r["blocks"]) - r["source_duration_ms"]) / 1000)
    return {"asr_bleu": round(corpus_bleu([heard.get(r["id"], "") for r in recs], refs, language=lang)["score"], 2),
            "text_bleu": round(corpus_bleu([r["translated_text"] for r in recs], refs, language=lang)["score"], 2),
            "laal_s": round(float(np.mean(seg)) / 1000, 3), "first_block_s_p50": round(float(np.median(first)), 2),
            "end_offset_s_p50": round(float(np.median(end)), 2), "end_offset_s_p90": round(float(np.percentile(end, 90)), 2)}


def fluency_metrics(run: Path) -> dict:
    s, _ = analyse(run)
    b, q, sp, rp = s["breakdown"], s["queue"], s["speed"], s["repair"]
    return {
        "blocks_per_utt": round(s["blocks"]["per_utt"], 2), "one_unit_blocks": round(s["blocks"]["one_unit"], 3),
        "punct_only_blocks": round(s["blocks"]["no_unit"], 3),
        "excess_pause_s_per_100u": round(b["excess_pause_burden_s_per_100u"], 2),
        "utt_with_pause_ge1s": round(b["utt_with_pause_ge1s"], 3),
        "pauses_ge1_per_100u": round(b["pauses_ge1.0_per_100u"], 2), "pauses_ge0.25_per_100u": round(b["pauses_ge0.25_per_100u"], 2),
        "silence_ratio": round(1 - sp["phonation_ratio"], 3), "mean_length_of_run_u": round(sp["mean_length_of_run_u"], 2),
        "speech_rate_u_per_s": round(sp["speech_rate_u_per_s"], 2),
        "adjacent_rate_jump_gt1.5x": round(sp["adjacent_rate_jump_gt1.5x"], 3),
        "word_seam_excess": round(b["seams"].get("W", {}).get("excess_s_per_100u", 0.0), 2),
        "phrase_seam_excess": round(b["seams"].get("P", {}).get("excess_s_per_100u", 0.0)
                                    + b["seams"].get("P-fw", {}).get("excess_s_per_100u", 0.0), 2),
        "blocks_queued": round(q["blocks_queued"], 3), "queued_s_p90": round(q["queued_s_p50_p90_max"][1] or 0.0, 2),
        "tail_s_p50": round(q["tail_after_source_s_p50_p90"][0], 2),
        "talker_err": round(rp["talker_err"], 4) if rp["talker_err"] is not None else None,
        "utt_talker_err_gt0.2": round(rp["utt_talker_err_gt0.2"], 3) if rp["utt_talker_err_gt0.2"] is not None else None,
        "stutter_per_100u": round(rp["stutter_per_100u"], 3), "text_repeat_per_100u": round(rp["text_repeat_per_100u"], 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="name=realsi_<dir> of an si_infer run (with asr_speech.jsonl)")
    ap.add_argument("--base", required=True, help="the run every other is compared with")
    ap.add_argument("--utmos", action="append", default=[], help="jsonl of {run, id, utmos}; run = the --run name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    runs = dict(spec.split("=", 1) for spec in args.run)
    ut = {}
    for f in args.utmos:
        for r in map(json.loads, open(f)):
            ut.setdefault(r["run"], {})[r["id"]] = r["utmos"]
    res = {}
    for name, d in runs.items():
        res[name] = {**paper_metrics(Path(d)), **fluency_metrics(Path(d))}
        if name in ut:
            res[name]["utmos"] = round(float(np.mean(list(ut[name].values()))), 3)
        print(name, json.dumps(res[name]), flush=True)
    for name, d in runs.items():
        if name == args.base:
            continue
        p = subprocess.run([sys.executable, "-m", "experiments.uniss_qwen3_omni_v1.evaluation.si_paired_bootstrap",
                            "--a", runs[args.base], "--b", d], capture_output=True, text=True)
        res[name]["vs_base_bleu"] = json.loads(p.stdout.strip().splitlines()[-1])
        if name in ut and args.base in ut:
            ids = sorted(set(ut[name]) & set(ut[args.base]))
            dd = np.array([ut[name][i] - ut[args.base][i] for i in ids])
            rng = np.random.default_rng(0)
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            res[name]["vs_base_utmos"] = {"diff": round(float(dd.mean()), 3),
                                          "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
