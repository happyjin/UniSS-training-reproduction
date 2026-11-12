"""Speech timeline of one (point, direction, seed) from the Qwen3-TTS block wavs, in si_infer's results format
(registration: paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md §2), plus the after-synthesis pipeline.

assemble: for every RealSI sentence (all have committed blocks), block k starts playing at max(write_ms_k, end of
block k-1's playback) and lasts its synthesised audio (24 kHz -> 16 kHz, librosa.resample default); a block without a
TTS request (punctuation only) has no audio. Writes, like si_infer (refuses an existing run folder):
  <run>/wav/<id>_simul.wav    the target channel on the source clock (16 kHz)
  <run>/wav/<id>_speech.wav   the speech alone, blocks back to back (16 kHz; what the speech-only ASR hears)
  <run>/results.jsonl         si_infer's fields used by speech_only.py / asr_transcribe / scoring (blocks with
                              write_ms, text, final, start_ms, dur_ms)
run folder: <E>/a1/runs/system_<point>[-seed<n>]_m<w>/realsi_<dir> (w = chunk length in s, a label as in R2).

pipeline (CPU + the ASR on --asr-cards): waits until every job of a run is done, assembles it, then runs
reports/uniss_si_v4/speech_only.py and evaluation.asr_transcribe --batch-size 4 (uniss-eval) on one card, at most one
ASR per card at a time; exits with PIPELINE_DONE when every run has asr_speech.jsonl.

    PYTHONPATH=. venvs/tts_schemes/bin/python experiments/baselines_hysimulmt_qwen3tts_v1/assemble.py pipeline \
        --asr-cards 4 5 6 7
    ... assemble.py one --point hy-std-w4 --direction en2zh --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/baselines_hysimulmt_v1"))
from hy_realsi import GRID, realsi_items  # noqa: E402

E = REPO / "eval_outputs/baselines_hysimulmt_qwen3tts_v1"
R = E / "a1"   # amendment 1 (per-block frame cap); the aborted first attempt stays in E/{jobs.jsonl,claims,tts}
THINKER = REPO / "eval_outputs/baselines_hysimulmt_v1/thinker"
LANGS = {"en2zh": ("eng", "cmn"), "zh2en": ("cmn", "eng")}
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
POINTS, DIRS, SEEDS = ("hy-std-w4", "hy-sc-w4"), ("en2zh", "zh2en"), (0, 1, 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_dir(point: str, d: str, seed: int) -> Path:
    w = GRID[point][0] / 1000
    return R / "runs" / f"system_{point}{f'-seed{seed}' if seed else ''}_m{w:g}" / f"realsi_{d}"


def tts_dir(point: str, d: str, seed: int) -> Path:
    return R / "tts" / f"{point}_s{seed}" / d


def n_jobs(point: str, d: str, seed: int, jobs: list[dict]) -> int:
    return sum(j["point"] == point and j["direction"] == d and j["seed"] == seed for j in jobs)


def assemble(point: str, d: str, seed: int) -> dict:
    import librosa
    out = run_dir(point, d, seed)
    out.mkdir(parents=True, exist_ok=False)
    (out / "wav").mkdir()
    synth = {}
    for f in sorted((tts_dir(point, d, seed) / "done").glob("*.json")):
        for b in json.load(open(f, encoding="utf-8"))["blocks"]:
            assert (b["id"], b["k"]) not in synth, (f, b["id"], b["k"])
            synth[(b["id"], b["k"])] = b
    src_lang, tgt = LANGS[d]
    items = realsi_items(d)
    rows = {r["id"]: r for r in map(json.loads, open(THINKER / point / f"blocks_{d}.jsonl", encoding="utf-8"))}
    cache, records, used = {}, [], 0
    for it in items:
        r = rows[it["id"]]
        if not r["blocks"]:
            continue
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000
            cache = {it["wav"]: x.mean(axis=1)}
        dur = len(cache[it["wav"]][it["start"] * 16: it["end"] * 16]) // 16
        assert dur == r["dur_ms"], (it["id"], dur, r["dur_ms"])
        pieces, blocks, end_prev = [], [], 0
        for k, b in enumerate(r["blocks"]):
            s = synth.get((it["id"], k))
            piece = np.zeros(0, dtype=np.float32)
            if s is not None:
                w24, sr24 = sf.read(REPO / s["wav"], dtype="float32")
                piece = librosa.resample(w24, orig_sr=sr24, target_sr=16000).astype(np.float32) if w24.size else piece
                used += 1
            start = max(b["write_ms"], end_prev)
            pieces.append((start, piece))
            end_prev = start + len(piece) / 16
            blocks.append({"write_ms": b["write_ms"], "text": b["text"], "final": b["final"], "start_ms": int(start),
                           "dur_ms": int(len(piece) / 16), "tts": s is not None,
                           "tts_frames": s["frames"] if s else 0, "tts_cap": s["cap"] if s else None,
                           "tts_hit_cap": bool(s and s["hit_cap"]), "tts_wav": s["wav"] if s else None})
        total = int(max(dur, end_prev)) + 300
        right = np.zeros(total * 16, dtype=np.float32)
        for start, piece in pieces:
            a = int(start * 16)
            right[a: a + len(piece)] += piece
        speech = np.concatenate([p for _, p in pieces]) if pieces else np.zeros(0, dtype=np.float32)
        name = it["id"].replace("/", "__")
        tgt_path, speech_path = out / "wav" / f"{name}_simul.wav", out / "wav" / f"{name}_speech.wav"
        sf.write(tgt_path, right, 16000)
        sf.write(speech_path, speech if speech.size else np.zeros(320, dtype=np.float32), 16000)
        gaps = sum(max(0.0, s - (ps + len(pp) / 16)) for (s, _), (ps, pp) in zip(pieces[1:], pieces[:-1]))
        sp = sum(len(p) / 16 for _, p in pieces)
        records.append({"id": it["id"], "mode": "si", "name": it["id"], "src_lang": src_lang, "tgt_lang": tgt,
                        "transcription_ref": it["src_text"], "translation_ref": it["trg_text"],
                        "translated_text": "".join(b["text"] for b in r["blocks"]), "audio_path": str(tgt_path),
                        "speech_path": str(speech_path), "audio_duration_seconds": total / 1000,
                        "source_duration_ms": dur, "system": f"Hy-SimulMT {point} + Qwen3-TTS-12Hz-0.6B-Base",
                        "tts_seed": seed, "error": None if speech.size else "no audio",
                        "start_offset_ms": int(pieces[0][0]) if speech.size else None,
                        "end_offset_ms": int(end_prev - dur), "silence_ratio": round(gaps / max(sp + gaps, 1e-6), 4),
                        "blocks": blocks})
    assert used == len(synth), (point, d, seed, used, len(synth))
    with open(out / "results.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
    return {"run": str(out.relative_to(REPO)), "sentences": len(records), "tts_blocks": used}


def pipeline(cards: list[int]) -> None:
    jobs = [json.loads(l) for l in open(R / "jobs.jsonl", encoding="utf-8")]
    runs = [(p, d, s) for s in SEEDS for p in POINTS for d in DIRS]
    asr = {}   # card -> (Popen, run)
    pool = ProcessPoolExecutor(4)
    assembling = {}
    while True:
        for c, (proc, key) in list(asr.items()):
            if proc.poll() is not None:
                o = run_dir(*key)
                log(f"asr {'done' if proc.returncode == 0 and (o / 'asr_speech.jsonl').exists() else 'FAILED'} "
                    f"{key} (card {c}, rc {proc.returncode})")
                del asr[c]
        for key, fut in list(assembling.items()):
            if fut.done():
                log(f"assembled {key}: {fut.result()}")   # an exception here stops the pipeline (visible in its log)
                del assembling[key]
        for key in runs:
            o, t = run_dir(*key), tts_dir(*key)
            done = len(list((t / "done").glob("*.json"))) if (t / "done").exists() else 0
            if not o.exists() and key not in assembling and done == n_jobs(*key, jobs):
                assembling[key] = pool.submit(assemble, *key)
                log(f"assembling {key}")
        for key in runs:
            o = run_dir(*key)
            if not (o / "results.jsonl").exists() or (o / "asr_speech.jsonl").exists() or (o / "asr.log").exists():
                continue
            free = [c for c in cards if c not in asr]
            if not free:
                break
            c = free[0]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(c), "PYTHONPATH": ".", "TOKENIZERS_PARALLELISM": "false",
                   "HF_HOME": "/opt/dlami/nvme/jasonleeeli/cache/huggingface", "OMP_NUM_THREADS": "8",
                   "MODELSCOPE_CACHE": "/opt/dlami/nvme/jasonleeeli/cache/modelscope"}
            cmd = (f"{PYE} reports/uniss_si_v4/speech_only.py {o} > /dev/null && {PYE} -m evaluation.asr_transcribe "
                   f"--input {o}/results_speechonly.jsonl --output {o}/asr_speech.jsonl --device cuda:0 --batch-size 4 "
                   f"> {o}/asr.log 2>&1")
            (o / "asr.log").touch(exist_ok=False)
            asr[c] = (subprocess.Popen(cmd, shell=True, cwd=REPO, env=env), key)
            log(f"asr started {key} on card {c}")
        if all((run_dir(*k) / "asr_speech.jsonl").exists() for k in runs) and not asr:
            log("PIPELINE_DONE")
            return
        time.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one")
    o.add_argument("--point", required=True)
    o.add_argument("--direction", required=True)
    o.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("--asr-cards", type=int, nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "one":
        print(json.dumps(assemble(a.point, a.direction, a.seed)))
    else:
        pipeline(a.asr_cards)


if __name__ == "__main__":
    main()
