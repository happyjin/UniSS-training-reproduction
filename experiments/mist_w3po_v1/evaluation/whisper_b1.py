"""Re-transcribe a talker run's speech-only clips with whisper-large-v3 at batch size 1 (GPU; reports/mist_w3po_v1/
text_to_speech_gap_plan.zh-CN.md §A3). Same model, dtype (fp16), transformers pipeline and call options as
evaluation/asr_transcribe.py, but every clip alone, so that no clip is decoded next to padding: the official run batches
clips > 2 s by 8, and some of those decode into runs of "....." or repeated words (the "hallucination tail").
--guard-streams <streams.jsonl>: for the tail streams of that file (13a ASR / text tokens > 1.5) also decode with
Whisper's fallback heuristics (temperature 0 .. 1 step 0.2, compression ratio 1.35, log-prob -1, no-speech 0.6).
Writes <out> (jsonl: id, asr_text[, asr_text_guarded]; refuses to overwrite). Shards: --shard k --num-shards n.

    python -m experiments.mist_w3po_v1.evaluation.whisper_b1 --speech <speech dir> --device cuda:N --out <jsonl> \
        [--guard-streams <streams.jsonl>] [--shard k --num-shards n]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guard-streams", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    from transformers import pipeline
    rows = [json.loads(l) for l in open(Path(args.speech) / "talker/asr_speech_merged.jsonl", encoding="utf-8")]
    rows = sorted(rows, key=lambda r: r["id"])[args.shard::args.num_shards]
    tail = set()
    if args.guard_streams:
        tail = {json.loads(l)["id"] for l in open(args.guard_streams, encoding="utf-8") if json.loads(l)["ratio13a"] > 1.5}
    rec = pipeline("automatic-speech-recognition", model=MODEL, torch_dtype=torch.float16, device=args.device)
    rec.feature_extractor.return_attention_mask = True
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            x, sr = sf.read(r["audio_path"], dtype="float32", always_2d=True)
            x = x.mean(1)
            opts = {"generate_kwargs": {"language": "english", "task": "transcribe"}}
            if len(x) / sr > 30:
                opts["return_timestamps"] = True
            o = rec({"raw": x, "sampling_rate": sr}, batch_size=1, **opts)
            rec_out = {"id": r["id"], "asr_text": str(o.get("text", "")).strip()}
            if r["id"] in tail:
                g = rec({"raw": x, "sampling_rate": sr}, batch_size=1, generate_kwargs={
                    "language": "english", "task": "transcribe", "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0, "no_speech_threshold": 0.6,
                    "condition_on_prev_tokens": False})
                rec_out["asr_text_guarded"] = str(g.get("text", "")).strip()
            fh.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
