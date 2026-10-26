"""Qwen3-ASR-1.7B transcripts for the cleaning plans, on vLLM (native Qwen3ASR support).

Prompt and parsing follow the qwen-asr package exactly: an empty system turn,
the audio as the user turn, and the model answers "language X<asr_text>...".
No context text is passed -- a reference in the prompt would pull the
transcript towards it, and the transcript is how the label is judged. The
language is not forced either, so the detected language is a check too.

Input: an items parquet (key, side, path); each shard takes every
num_shards-th item and writes part files of --flush items, so a stopped
shard resumes where it was.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.qwen3_asr_score \
        --items items.parquet --output out_dir --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

MODEL = ("/opt/dlami/nvme/jasonleeeli/projects/multi_speaker_pipeline_share/multi_speaker_pipeline_share/"
         "model_weights/Qwen3-ASR-1.7B")
PROMPT = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
          "<|im_start|>assistant\n")
SR = 16_000


def parse(raw: str) -> tuple[str, str]:
    """qwen_asr.inference.utils.parse_asr_output, without the forced-language case."""
    s = (raw or "").strip()
    if "<asr_text>" not in s:
        return "", s
    meta, text = s.split("<asr_text>", 1)
    if "language none" in meta.lower():
        return "", text.strip()
    lang = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            lang = line[len("language "):].strip()
            break
    return lang, text.strip()


def load(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--flush", type=int, default=4096, help="items per generate call and per part file")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--io-threads", type=int, default=24)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = pq.read_table(args.items, columns=["key", "side", "path"]).to_pylist()[args.shard :: args.num_shards]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob(f"shard{args.shard:02d}_part*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue   # a part cut short by a stop; its missing items are redone
            done.add((r["key"], r["side"]))
    todo = [it for it in items if (it["key"], it["side"]) not in done]
    print(f"shard {args.shard}: {len(items)} items, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    llm = LLM(model=MODEL, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
              limit_mm_per_prompt={"audio": 1}, max_num_seqs=256, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=512)
    part = 1 + max((int(f.stem.split("_part")[1]) for f in out.glob(f"shard{args.shard:02d}_part*.jsonl")),
                   default=-1)
    pool = ThreadPoolExecutor(args.io_threads)
    batches = [todo[s : s + args.flush] for s in range(0, len(todo), args.flush)]
    # The next batch is read while this one decodes: read and decode in turn
    # left the GPU at about 30%.
    pending = [pool.submit(load, it["path"]) for it in batches[0]]
    t0, n = time.time(), 0
    for b, batch in enumerate(batches):
        wavs = [f.result() for f in pending]
        if b + 1 < len(batches):
            pending = [pool.submit(load, it["path"]) for it in batches[b + 1]]
        results = llm.generate([{"prompt": PROMPT, "multi_modal_data": {"audio": (w, SR)}} for w in wavs],
                               params, use_tqdm=False)
        target = out / f"shard{args.shard:02d}_part{part:05d}.jsonl"
        tmp = target.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for it, w, res in zip(batch, wavs, results):
                raw = res.outputs[0].text
                lang, text = parse(raw)
                fh.write(json.dumps({"key": it["key"], "side": it["side"], "lang": lang, "text": text,
                                     "raw": raw, "seconds": round(len(w) / SR, 3)}, ensure_ascii=False) + "\n")
        tmp.rename(target)
        part += 1
        n += len(batch)
        audio_h = sum(len(w) for w in wavs) / SR / 3600
        print(f"shard {args.shard}: {n}/{len(todo)}  {n / (time.time() - t0):.1f} items/s  "
              f"(last batch {audio_h:.2f} h audio)", flush=True)


if __name__ == "__main__":
    main()
