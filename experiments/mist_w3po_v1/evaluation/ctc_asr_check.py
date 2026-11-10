"""A second, independent ASR on the talker's speech-only clips (CPU; reports/mist_w3po_v1/text_to_speech_gap_plan.zh-CN.md):
wav2vec 2.0 Large LV-60 self-trained, fine-tuned on 960 h LibriSpeech (facebook/wav2vec2-large-960h-lv60-self -- the
English model of fairseq's ust/asr_bleu that StreamSpeech's ASR-BLEU uses). Greedy CTC, no language model, so it cannot
continue a sentence it did not hear: a clip on which Whisper writes far more words than the talker was given, while CTC
writes about as many as the talker was given, is a Whisper hallucination; extra words in both point at the talker.

Clips: --ids-from <speech_gap_decomp streams.jsonl> picks every stream with 13a ASR / text tokens > 1.5 (the tail) plus
--n-rest random others (seed 0); --all takes every stream. Shards with --shard k --num-shards n. Writes <out> (jsonl:
id, ctc_text; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.ctc_asr_check --speech <speech dir> --ids-from <streams.jsonl> \
        --model <wav2vec2 dir> --out <jsonl> [--all] [--shard k --num-shards n]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--ids-from", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rest", type=int, default=600)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    torch.set_num_threads(args.threads)
    S = [json.loads(l) for l in open(args.ids_from, encoding="utf-8")]
    if args.all:
        ids = sorted(r["id"] for r in S)
    else:
        tail = sorted(r["id"] for r in S if r["ratio13a"] > 1.5)
        rest = sorted(r["id"] for r in S if r["ratio13a"] <= 1.5)
        rng = np.random.default_rng(0)
        ids = tail + sorted(rng.choice(rest, size=min(args.n_rest, len(rest)), replace=False).tolist())
        tailset = set(tail)
    ids = ids[args.shard::args.num_shards]
    proc = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval()
    wav = Path(args.speech) / "talker/wav"
    with open(out, "x", encoding="utf-8") as fh:
        for k in ids:
            p = wav / f"{k}_speech.wav"
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 16000:
                raise SystemExit(f"{p}: {sr} Hz")
            if len(x) < 400:
                text = ""
            else:
                inp = proc(x, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    lg = model(inp.input_values).logits
                text = proc.batch_decode(lg.argmax(-1))[0]
            fh.write(json.dumps({"id": k, "ctc_text": text, "dur_s": round(len(x) / 16000, 3),
                                 "group": "all" if args.all else ("tail" if k in tailset else "rest")}) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
