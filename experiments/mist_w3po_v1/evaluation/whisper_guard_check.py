"""Does Whisper's own hallucination guarding remove the ASR tail of the speech score? (CPU; reports/mist_w3po_v1/
text_to_speech_gap_plan.zh-CN.md §A3.) whisper-large-v3, batch 1, English, on the talker's speech-only clips:
  plain     greedy, no fallback (the evaluation's setting: transformers pipeline, language english, greedy)
  guarded   the Whisper paper's decoding heuristics: temperature fallback 0 .. 1.0 step 0.2 when the compression ratio
            of the output exceeds 1.35 (transformers' token-level threshold; 2.4 in the original gzip form) or the mean
            log-probability is below -1, no-speech probability 0.6 (with the log-probability test), no conditioning on
            earlier text
Clips: the tail streams (13a ASR / text tokens > 1.5) of a speech_gap_decomp streams.jsonl plus --n-rest random others
(seed 1). Writes <out> (jsonl: id, group, plain, guarded; refuses to overwrite). Shards: --shard k --num-shards n.

    python -m experiments.mist_w3po_v1.evaluation.whisper_guard_check --speech <speech dir> --streams <streams.jsonl> \
        --out <jsonl> [--n-rest 300] [--shard k --num-shards n]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rest", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(args.threads)
    S = [json.loads(l) for l in open(args.streams, encoding="utf-8")]
    tail = sorted(r["id"] for r in S if r["ratio13a"] > 1.5)
    rest = sorted(r["id"] for r in S if r["ratio13a"] <= 1.5)
    rng = np.random.default_rng(1)
    pick = [(k, "tail") for k in tail] + [(k, "rest") for k in sorted(rng.choice(rest, size=min(args.n_rest, len(rest)), replace=False).tolist())]
    pick = pick[args.shard::args.num_shards]
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    wav = Path(args.speech) / "talker/wav"
    with open(out, "x", encoding="utf-8") as fh:
        for k, g in pick:
            x, sr = sf.read(str(wav / f"{k}_speech.wav"), dtype="float32", always_2d=True)
            x = x.mean(1)
            feats = proc(x, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
            outs = {}
            with torch.no_grad():
                ids = model.generate(feats.input_features, attention_mask=feats.attention_mask, language="en",
                                     task="transcribe")
                outs["plain"] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                ids = model.generate(feats.input_features, attention_mask=feats.attention_mask, language="en",
                                     task="transcribe", temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                     compression_ratio_threshold=1.35, logprob_threshold=-1.0, no_speech_threshold=0.6,
                                     condition_on_prev_tokens=False, do_sample=False)
                outs["guarded"] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            fh.write(json.dumps({"id": k, "group": g, "dur_s": round(len(x) / sr, 3), **outs}, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
