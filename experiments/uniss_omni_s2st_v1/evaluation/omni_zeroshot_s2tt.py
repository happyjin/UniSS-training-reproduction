"""Zero-shot speech-to-text translation with Qwen2.5-Omni, on our test set.

Why this runs before anything is trained
----------------------------------------
The case for replacing the backbone rests on a number from someone else's
paper: Qwen-Omni2.5 scores 34.85 / 24.39 Text-BLEU on CVSS-T where this
project's 0.5B scores 24.15 / 15.33.  That comparison was made under their
protocol, on their decode, with their references.  Before spending weeks on a
data pipeline and three training stages, the same question gets asked on *our*
audio, with *our* references and *our* scorer: how much better is Omni here?

If the gap is close to the paper's ten points, the replacement is justified.
If it is small, the gap lives in the protocol or the data rather than the
backbone, and the plan should stop rather than proceed.

Nothing here trains, and nothing here touches the existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import soundfile as sf
import torch

from evaluation import text_metrics
from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
    load_audio_text_processor,
)

LANGUAGE = {"en2zh": "cmn", "zh2en": "eng"}
INSTRUCTION = {
    "en2zh": "Listen to the English speech and translate it into Chinese. "
             "Reply with the Chinese translation only.",
    "zh2en": "Listen to the Chinese speech and translate it into English. "
             "Reply with the English translation only.",
}


def load_pairs(manifest: Path, audio_root: Path, direction: str) -> list[dict]:
    """CVSS-T pairs, with the audio each direction actually consumes.

    zh->en reads the real Common Voice Chinese source; en->zh reads the
    synthesised English side, which is the same asymmetry the project's own
    CVSS-T tokenizer encodes.
    """
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = row.get("id") or row.get("sample_id")
        # The manifest carries resolved paths; the id still has its .mp3
        # suffix from Common Voice, so rebuilding a filename from it is a
        # needless way to get this wrong.
        if direction == "zh2en":
            audio = Path(row["source_zh_audio_path"])
            reference = row["target_en_text"]
        else:
            audio = Path(row["target_en_audio_path"])
            reference = row["source_zh_text"]
        if not audio.is_absolute():
            audio = audio_root / audio
        if audio.exists() and reference:
            rows.append({"id": sample, "audio": str(audio), "reference": reference})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--batch-size", type=int, default=8)
    # Omni's generate() declares thinker_max_new_tokens=1024 explicitly and
    # documents that the prefixed keyword wins over a bare max_new_tokens, so
    # the bare one is silently ignored. 256 is far above any CVSS-T reference
    # and bounds the greedy-decode repetition loops this model falls into on
    # rare inputs.
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration

    rows = load_pairs(Path(args.manifest), Path(args.audio_root), args.direction)
    if not rows:
        raise SystemExit("no pairs resolved; check the manifest and audio root")
    if args.limit and args.limit < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
    if args.num_shards > 1:
        # Shard after the limit/shuffle so every shard draws from the same
        # sample set regardless of how many workers run.
        rows = rows[args.shard :: args.num_shards]
    print(
        f"{len(rows)} pairs, direction {args.direction}"
        f" (shard {args.shard}/{args.num_shards})",
        flush=True,
    )

    processor = load_audio_text_processor(args.model)
    # Decoder-only generation with a padded batch has to pad on the left, or
    # the shorter members decode from the middle of their own padding.
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    # The speech-synthesis head is dead weight here: this measures translation
    # quality as text, which is what the decision gate is about.
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()

    started = time.perf_counter()
    hypotheses = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        conversations = [
            [{"role": "user", "content": [
                {"type": "audio", "audio": r["audio"]},
                {"type": "text", "text": INSTRUCTION[args.direction]},
            ]}]
            for r in batch
        ]
        # Templating the batch in one call returns a flat list of strings;
        # per-conversation calls each return a one-element list, which nests.
        texts = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        audios = [sf.read(r["audio"], dtype="float32")[0] for r in batch]
        inputs = processor(
            text=texts, audio=audios, return_tensors="pt", padding=True,
            sampling_rate=16_000,
        )
        inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, thinker_max_new_tokens=args.max_new_tokens,
                thinker_do_sample=False, return_audio=False,
            )
        for row, sequence in zip(batch, out):
            text = processor.tokenizer.decode(
                sequence[prompt_length:], skip_special_tokens=True,
            ).strip()
            hypotheses.append({**row, "hypothesis": text})
        if start % (args.batch_size * 20) == 0:
            done = len(hypotheses)
            rate = done / max(1e-9, time.perf_counter() - started)
            print(f"  {done}/{len(rows)}  {rate:.1f}/s", flush=True)

    language = LANGUAGE[args.direction]
    score = text_metrics.corpus_bleu(
        [h["hypothesis"] for h in hypotheses],
        [h["reference"] for h in hypotheses],
        language=language,
    )
    report = {
        "model": args.model, "direction": args.direction,
        "pairs": len(hypotheses), "bleu": score["score"],
        "sys_len": score["sys_len"], "ref_len": score["ref_len"],
        "bp": score["bp"], "empty": sum(1 for h in hypotheses if not h["hypothesis"]),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"report": report, "rows": hypotheses}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
