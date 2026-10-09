"""Stage 1: warm the Talker up on this project's BiCodec codebook.

The Talker's code embedding and output head were re-drawn at Stage 0 --
Omni's row k and BiCodec's row k mean different things -- so before any
joint training the new 8,192-way head has to converge on our code space.
The task is TTS: target text in, target codes out, no source audio
anywhere.

Frozen: the audio encoder (unused here) and the whole Thinker. Trained: the
Talker, which includes the re-drawn tensors and ``thinker_to_talker_proj``.
The Thinker runs under no_grad, so its 36 layers cost a forward pass and no
optimiser state.

The dev curve is cross-entropy on a fixed 1,000-pair sample of CVSS-T dev,
which is disjoint from the test split the project reports -- on id and on
source text both. Watching the test split during training would contaminate
the number the whole replacement is being judged by.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import (
    build_codec_sequence,
    build_text_stream,
)
from experiments.uniss_omni_s2st_v1.modeling.talker_surgery import (
    apply_output_mask,
    read_codec_layout,
    retarget_talker_codebook,
    valid_output_mask,
)
from experiments.uniss_omni_s2st_v1.runtime.omni_processor import (
    load_audio_text_processor,
)
from experiments.uniss_omni_s2st_v1.training.tts_data import (
    build_thinker_batch,
    read_rows,
    tts_prompt,
)


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if is_main():
        print(message, flush=True)


def encode_prompts(processor, rows: list[dict]) -> list[dict]:
    """Tokenise once up front; the text does not change between epochs."""
    tokenizer = processor.tokenizer
    prepared = []
    for row in rows:
        prompt = processor.apply_chat_template(
            [tts_prompt(row["text"], row["lang"])],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_text = prompt[0] if isinstance(prompt, list) else prompt
        prepared.append(
            {
                **row,
                "prompt_ids": tokenizer(prompt_text, add_special_tokens=False)["input_ids"],
                "reply_ids": tokenizer(row["text"], add_special_tokens=False)["input_ids"],
            }
        )
    return prepared


def batch_loss(model, processor, items, *, layout, output_mask, device, eos_id):
    """Teacher-forced code cross-entropy for one batch."""
    thinker, talker = model.thinker, model.talker
    batch = build_thinker_batch(
        [item["prompt_ids"] for item in items],
        [item["reply_ids"] for item in items],
        pad_id=processor.tokenizer.pad_token_id or 0,
        device=device,
    )
    with torch.no_grad():
        out = thinker(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1][
            :, batch.reply_start : batch.reply_start + batch.reply_ids.shape[1], :
        ]
        embeds = thinker.get_input_embeddings()(batch.reply_ids)
        eos_embed = thinker.get_input_embeddings()(
            torch.tensor([[talker.text_eos_token]], device=device)
        )
        pad_embed = thinker.get_input_embeddings()(
            torch.tensor([[talker.text_pad_token]], device=device)
        )

    codes = [torch.as_tensor(item["codes"], dtype=torch.long, device=device) for item in items]
    codec_ids, labels, codec_mask = build_codec_sequence(
        codes, bos_id=talker.codec_bos_token, pad_id=talker.codec_pad_token, eos_id=eos_id
    )
    steps = codec_ids.shape[1]

    # Each row's text stream must run out at *its own* reply length, not the
    # batch's; a short reply padded to the batch width would be conditioned
    # on the pad embeddings of a longer neighbour.
    streams = []
    for row in range(len(items)):
        length = int(batch.reply_mask[row].sum())
        streams.append(
            build_text_stream(
                hidden[row : row + 1, :length].float(),
                embeds[row : row + 1, :length].float(),
                steps=steps,
                eos_embed=eos_embed.float(),
                pad_embed=pad_embed.float(),
            )
        )
    text_stream = torch.cat(streams, dim=0).to(dtype=hidden.dtype)

    codec_embeds = talker.get_input_embeddings()(codec_ids)
    lm_input = talker.thinker_to_talker_proj(codec_embeds + text_stream)
    position_ids = (
        torch.arange(steps, device=device).view(1, 1, -1).expand(3, len(items), -1)
    )
    talker_out = talker.model(
        inputs_embeds=lm_input,
        attention_mask=codec_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    logits = apply_output_mask(talker.codec_head(talker_out.last_hidden_state), output_mask)
    return F.cross_entropy(
        logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--parquet-root", default="data/raw/UniST")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dev-parquet", required=True)
    ap.add_argument("--dev-subset", required=True)
    ap.add_argument("--rows-per-shard", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--dev-batches", type=int, default=24)
    ap.add_argument("--head-scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--max-codes", type=int, default=600)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.output)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    # token2wav is a vocoder; it has no role in a code-space objective and
    # its parameters would otherwise sit in memory on every rank.
    if hasattr(model, "token2wav"):
        del model.token2wav
    talker = model.talker
    layout = read_codec_layout(talker.config)
    retarget_talker_codebook(talker, seed=args.seed, head_scale=args.head_scale)
    output_mask = valid_output_mask(layout, device=device)

    model.thinker.requires_grad_(False)
    model.thinker.eval()
    talker.requires_grad_(True)
    talker.train()

    trainable = [p for p in talker.parameters() if p.requires_grad]
    log(
        f"talker trainable parameters:"
        f" {sum(p.numel() for p in trainable) / 1e6:.1f}M"
        f"   thinker frozen: {sum(p.numel() for p in model.thinker.parameters()) / 1e6:.1f}M"
    )

    wrapped = talker
    if world > 1:
        wrapped = DistributedDataParallel(
            talker, device_ids=[local_rank], find_unused_parameters=False
        )

    shards = [Path(args.parquet_root) / s for s in args.shards]
    rows = read_rows(shards, limit_per_file=args.rows_per_shard, max_codes=args.max_codes)
    rows = encode_prompts(processor, rows)
    log(f"{len(rows)} training utterances from {len(shards)} shards")

    dev_rows = load_dev(args.dev_parquet, args.dev_subset, args.max_codes)
    dev_rows = encode_prompts(processor, dev_rows)
    log(f"{len(dev_rows)} dev utterances")

    optimiser = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def learning_rate(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = rank * args.batch_size
    curve_path = out_dir / "curve.jsonl"
    started = time.perf_counter()

    for step in range(args.steps):
        for group in optimiser.param_groups:
            group["lr"] = learning_rate(step)
        optimiser.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            if cursor + args.batch_size > len(order):
                rng.shuffle(order)
                cursor = rank * args.batch_size
            items = [rows[i] for i in order[cursor : cursor + args.batch_size]]
            cursor += args.batch_size * world
            loss = batch_loss(
                model, processor, items,
                layout=layout, output_mask=output_mask, device=device,
                eos_id=layout.special_ids["eos"],
            )
            (loss / args.accum).backward()
            total += float(loss) / args.accum
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()

        if is_main() and (step % 20 == 0 or step == args.steps - 1):
            rate = (step + 1) / max(1e-9, time.perf_counter() - started)
            log(f"  step {step:5d}  loss {total:7.4f}  lr {learning_rate(step):.2e}  {rate:.2f} it/s")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            dev = evaluate(
                model, processor, dev_rows, args,
                layout=layout, output_mask=output_mask, device=device, world=world,
            )
            if is_main():
                entry = {
                    "step": step + 1,
                    "train_loss": total,
                    "dev_loss": dev,
                    "lr": learning_rate(step),
                    "elapsed_s": time.perf_counter() - started,
                }
                with curve_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                log(f"  [dev] step {step + 1}  dev_loss {dev:.4f}")
            talker.train()

        if is_main() and ((step + 1) % args.save_every == 0 or step == args.steps - 1):
            torch.save(
                {"talker": talker.state_dict(), "step": step + 1, "args": vars(args)},
                out_dir / "talker_last.pt",
            )

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    log("STAGE1 DONE")


def load_dev(dev_parquet: str, subset_manifest: str, max_codes: int) -> list[dict]:
    """CVSS-T dev rows, restricted to the fixed curve subset."""
    import pyarrow.parquet as pq

    wanted = {
        json.loads(line)["id"]
        for line in Path(subset_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    table = pq.read_table(
        dev_parquet, columns=["id", "translation", "tgt_lang", "target_bicodec"]
    )
    rows = []
    for record in table.to_pylist():
        if record["id"] not in wanted:
            continue
        codes = record["target_bicodec"]
        text = (record["translation"] or "").strip()
        if not text or codes is None or not 4 <= len(codes) <= max_codes:
            continue
        rows.append(
            {
                "id": record["id"],
                "text": text,
                "lang": record["tgt_lang"],
                "codes": np.asarray(codes, dtype=np.int16),
            }
        )
    return rows


@torch.no_grad()
def evaluate(model, processor, dev_rows, args, *, layout, output_mask, device, world):
    """Mean dev cross-entropy, averaged across ranks."""
    model.talker.eval()
    total = 0.0
    seen = 0
    for index in range(args.dev_batches):
        start = (index * world + int(os.environ.get("RANK", "0"))) * args.batch_size
        items = dev_rows[start : start + args.batch_size]
        if not items:
            continue
        total += float(
            batch_loss(
                model, processor, items,
                layout=layout, output_mask=output_mask, device=device,
                eos_id=layout.special_ids["eos"],
            )
        )
        seen += 1
    tensor = torch.tensor([total, float(seen)], device=device)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0] / max(1.0, float(tensor[1])))


if __name__ == "__main__":
    main()
