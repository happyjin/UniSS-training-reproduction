"""Phase A gate: Qwen3-Omni-30B-A3B zero-shot S2TT on CVSS-T test.

The same question the Step 1 gate asked of Qwen2.5-Omni-3B, asked of a
bigger backbone: with no training at all, how well does the Thinker
translate? That number decides whether a backbone swap is worth any GPU
hours, because translation is the one axis our training has never moved
-- stage 1 froze the Thinker and stage 2's LoRA made Zh->En worse.

The protocol is imported from the 3B's gate rather than copied, so the
two numbers cannot drift apart: same instruction strings, same pairing of
direction to audio side, same normalised scorer. Only the model and
processor classes differ, and this is a new file so nothing the 3B path
runs on is touched.

    PYTHONPATH=. .../qwen3omni/bin/python -m \
      experiments.uniss_qwen3_omni_v1.evaluation.qwen3_zeroshot_s2tt \
        --manifest .../cvss_t_zh_en_test_pairs.jsonl --audio-root ... \
        --direction zh2en --shard 0 --num-shards 8 --output ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from evaluation import text_metrics
from experiments.uniss_omni_s2st_v1.evaluation.omni_zeroshot_s2tt import (
    INSTRUCTION,
    LANGUAGE,
    load_pairs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    rows = load_pairs(Path(args.manifest), Path(args.audio_root), args.direction)
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} pairs", flush=True)

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    # Only the Thinker translates. Its Talker and vocoder are the parts this
    # project discards -- it keeps UniSS's BiCodec path instead -- and they
    # cost memory a 30B model has no spare of.
    for attr in ("talker", "code2wav", "token2wav"):
        if hasattr(model, attr):
            delattr(model, attr)
    torch.cuda.empty_cache()
    model.eval()

    hypotheses: list[dict] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        waveforms = [sf.read(r["audio"], dtype="float32")[0] for r in batch]
        conversations = [
            [{"role": "user", "content": [
                {"type": "audio", "audio": r["audio"]},
                {"type": "text", "text": INSTRUCTION[args.direction]},
            ]}]
            for r in batch
        ]
        # One templating call for the whole batch: per-conversation calls
        # nest their output, which silently becomes a prompt of lists.
        text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text, audio=waveforms, sampling_rate=16_000,
            return_tensors="pt", padding=True,
        )
        inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            # thinker_max_new_tokens, not max_new_tokens: the latter is
            # ignored here and the model's own 1024 default wins, which on
            # the 3B turned a 26.65 into a 13.91.
            out = model.generate(
                **inputs, thinker_max_new_tokens=args.max_new_tokens,
                thinker_do_sample=False, return_audio=False,
            )
        for row, sequence in zip(batch, out):
            text_out = processor.tokenizer.decode(
                sequence[prompt_length:], skip_special_tokens=True
            ).strip()
            hypotheses.append({**row, "hypothesis": text_out})
        if start % (args.batch_size * 20) == 0:
            done = len(hypotheses)
            rate = done / max(1e-9, time.perf_counter() - started)
            print(f"  {done}/{len(rows)}  {rate:.1f}/s", flush=True)

    score = text_metrics.corpus_bleu(
        [h["hypothesis"] for h in hypotheses],
        [h["reference"] for h in hypotheses],
        language=LANGUAGE[args.direction],
    )
    report = {
        "model": args.model, "direction": args.direction,
        "pairs": len(hypotheses), "bleu": score["score"],
        "sys_len": score["sys_len"], "ref_len": score["ref_len"],
        "bp": score["bp"],
        "empty": sum(1 for h in hypotheses if not h["hypothesis"]),
        "shard": args.shard, "num_shards": args.num_shards,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"report": report, "rows": hypotheses}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
