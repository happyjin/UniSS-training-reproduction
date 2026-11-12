"""Synthesis jobs for Hy-SimulMT + Qwen3-TTS-12Hz-0.6B-Base on RealSI (registration:
paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md, tracked copy reports/s2st_baselines_v1/).

Input: R2's committed Hy-SimulMT blocks (eval_outputs/baselines_hysimulmt_v1/thinker/<point>/blocks_<dir>.jsonl, the
same files the paper-talker run spoke). Every block whose stripped text holds an alphanumeric character is one TTS
request (SimulInfer's tts.Speech.run skips the others; they keep zero speech). Writes, refusing existing files:
  <E>/refs/<point>/<dir>/<id>.wav   voice reference: the source heard up to the sentence's first commit (>= 1 s), i.e.
                                    the paper talker's clip (si_infer), peak-normalised as SimulInfer's SourceVoice
  <E>/jobs.jsonl                    one row per batch: blocks sorted by text length (fixed composition), torch seed
                                    = SEED_BASE[seed] + batch index; seed-major order

    PYTHONPATH=. venvs/tts_schemes/bin/python experiments/baselines_hysimulmt_qwen3tts_v1/make_jobs.py --batch 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/baselines_hysimulmt_v1"))
from hy_realsi import realsi_items  # noqa: E402  (si_infer.realsi_items, verbatim)

E = REPO / "eval_outputs/baselines_hysimulmt_qwen3tts_v1"
THINKER = REPO / "eval_outputs/baselines_hysimulmt_v1/thinker"
POINTS = ("hy-std-w4", "hy-sc-w4")
DIRS = ("en2zh", "zh2en")
LANGUAGE = {"en2zh": "Chinese", "zh2en": "English"}
SEED_BASE = {0: 20261112, 1: 1_000_000, 2: 2_000_000}


def speakable(text: str) -> bool:
    return any(c.isalnum() for c in text)


def reference(wav: np.ndarray) -> tuple[np.ndarray, float]:
    """SimulInfer tts_source.SourceVoice._snapshot's level rule: scale to 0.7 peak if the peak is < 0.5 or > 0.95."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if 1e-4 <= peak and (peak < 0.5 or peak > 0.95):
        wav = wav * (0.7 / peak)
    return wav.astype(np.float32), peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    jobs_path = E / "jobs.jsonl"
    if jobs_path.exists():
        sys.exit(f"refusing: {jobs_path} exists")
    per_pd, stats = {}, {}
    for point in POINTS:
        for d in DIRS:
            items = {it["id"]: it for it in realsi_items(d)}
            rows = [json.loads(l) for l in open(THINKER / point / f"blocks_{d}.jsonl", encoding="utf-8")]
            assert {r["id"] for r in rows} == set(items), (point, d)
            refdir = E / "refs" / point / d
            refdir.mkdir(parents=True, exist_ok=False)
            cache, blocks, low = {}, [], 0
            for r in rows:
                if not r["blocks"]:
                    continue
                it = items[r["id"]]
                if it["wav"] not in cache:
                    x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
                    assert sr == 16000, it["wav"]
                    cache = {it["wav"]: x.mean(axis=1)}
                wav = cache[it["wav"]][it["start"] * 16: it["end"] * 16]
                first = r["blocks"][0]["write_ms"]
                ref, peak = reference(wav[: max(first, 1000) * 16])
                low += peak < 1e-4
                ref_path = refdir / f"{r['id']}.wav"
                with open(ref_path, "xb") as fh:
                    sf.write(fh, ref, 16000, format="WAV", subtype="PCM_16")
                for k, b in enumerate(r["blocks"]):
                    text = b["text"].strip()
                    if speakable(text):
                        blocks.append({"id": r["id"], "k": k, "text": text, "ref": str(ref_path.relative_to(REPO))})
            blocks.sort(key=lambda b: (len(b["text"]), b["id"], b["k"]))
            per_pd[(point, d)] = blocks
            stats[f"{point}|{d}"] = {"sentences": sum(bool(r["blocks"]) for r in rows),
                                     "blocks": sum(len(r["blocks"]) for r in rows), "tts_blocks": len(blocks),
                                     "silent_refs": int(low)}
    out = []
    for seed in a.seeds:
        for (point, d), blocks in per_pd.items():
            for n, i in enumerate(range(0, len(blocks), a.batch)):
                out.append({"job": f"{point}_{d}_s{seed}_b{n:04d}", "point": point, "direction": d, "seed": seed,
                            "torch_seed": SEED_BASE[seed] + n, "language": LANGUAGE[d],
                            "out": str((E / "tts" / f"{point}_s{seed}" / d).relative_to(REPO)),
                            "blocks": blocks[i: i + a.batch]})
    with open(jobs_path, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print(json.dumps({"jobs": len(out), "batch": a.batch, "seeds": a.seeds, **stats}, indent=1))


if __name__ == "__main__":
    main()
