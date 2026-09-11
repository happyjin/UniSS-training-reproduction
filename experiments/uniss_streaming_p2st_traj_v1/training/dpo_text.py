"""Length-normalised DPO on the streaming text stream.

NaturalFlow's claim is that the pauses in a streaming speech translator are a
*policy* defect rather than a vocoder or data defect: the model decides, read
step by read step, how much target text to commit, and a model that commits a
single character and stops leaves the listener in silence until the next step.
So the preference is expressed over the text stream and nowhere else, and the
speech stream is left alone.

Concretely, a candidate's text stream is every MT read step it ran, and its
score is

    logp = sum_i log p(produced_i + <end> | mt_prompt(source_i, target_i))

normalised by the total number of produced tokens.  Normalising matters here
more than it does in the usual chat setting: the two candidates differ mostly
in *how much text they emit*, so an unnormalised objective would be dominated
by length and would simply prefer the shorter stream -- the opposite of what
we want.

The reference policy is this same module with the adapters switched off, which
makes ``policy == reference`` exact at step 0 and costs no extra memory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.uniss_streaming_p2st_traj_v1.training import lora
from experiments.uniss_streaming_p2st_traj_v1.training.mt_prompt import (
    mt_completion_ids,
    mt_prompt_ids,
)


def load_manifests(arm_root: Path) -> dict[str, dict[str, dict]]:
    """``arm -> sample_id -> record``, for every arm that carries mt_steps."""
    arms: dict[str, dict[str, dict]] = {}
    for manifest in sorted(arm_root.glob("*/MANIFEST.json")):
        rows = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
        keep = {r["sample_id"]: r for r in rows if r.get("mt_steps")}
        if keep:
            arms[manifest.parent.name] = keep
    return arms


def build_examples(pairs: list[dict], arms: dict[str, dict[str, dict]]) -> list[dict]:
    """Keep only pairs whose *both* sides recorded a text stream."""
    examples = []
    for pair in pairs:
        sample = pair["sample_id"]
        sides = {}
        for side in ("chosen", "rejected"):
            arm = pair[side]["arm"]
            record = arms.get(arm, {}).get(sample)
            if record is None:
                break
            sides[side] = record
        if len(sides) != 2:
            continue
        examples.append(
            {
                "sample_id": sample,
                "tgt_lang": sides["chosen"].get("tgt_lang", "cmn"),
                "chosen": sides["chosen"]["mt_steps"],
                "rejected": sides["rejected"]["mt_steps"],
            }
        )
    return examples


def stream_logprob(model, tokenizer, steps, *, tgt_lang, device, max_len=1024):
    """Total log-probability of one candidate's text stream, and its length.

    Every read step is one sequence; they are batched together because they
    share nothing but the model, and a padded batch of short sequences is far
    cheaper than a loop over them.
    """
    rows = []
    for step in steps:
        prompt = mt_prompt_ids(
            tokenizer,
            source_prefix=step["source_prefix"],
            target_prefix=step["target_prefix"],
            tgt_lang=tgt_lang,
        )
        completion = mt_completion_ids(step["produced"], ended=step["ended"])
        if not completion:
            continue
        ids = prompt + completion
        if len(ids) > max_len:
            # Drop the oldest source tokens rather than the decision itself.
            ids = ids[-max_len:]
            prompt_len = max(1, len(ids) - len(completion))
        else:
            prompt_len = len(prompt)
        rows.append((ids, prompt_len, len(completion)))
    if not rows:
        return None, 0

    width = max(len(ids) for ids, _, _ in rows)
    input_ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    attention = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    target = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for i, (ids, prompt_len, _) in enumerate(rows):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, device=device)
        attention[i, :n] = 1
        # Predicting position t from t-1, so the label for the completion token
        # at index j sits at index j-1 of the shifted target.
        target[i, prompt_len - 1 : n - 1] = torch.tensor(
            ids[prompt_len:n], device=device
        )

    logits = model(input_ids=input_ids, attention_mask=attention).logits
    token_logp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        target.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_logp = token_logp.reshape(target.shape)
    token_logp = token_logp * (target != -100)
    return token_logp.sum(), int(sum(n for _, _, n in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--arm-root", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=256.0)
    # A small likelihood term on the chosen stream.  Plain DPO can raise the
    # margin by pushing *both* sides down, which for a speech model shows up as
    # degenerate text; anchoring the winner is the standard guard and it is
    # cheap because the forward pass is already done.
    ap.add_argument("--nll-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    layers = lora.inject(model, rank=args.rank, alpha=args.alpha)
    params = lora.trainable_parameters(layers)
    print(
        f"lora: {len(layers)} layers, "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters",
        flush=True,
    )

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))["pairs"]
    arms = load_manifests(Path(args.arm_root))
    examples = build_examples(pairs, arms)
    if not examples:
        raise SystemExit(
            "no pair had a recorded text stream on both sides; "
            "the candidates predate --dump-codes carrying mt_steps"
        )
    print(
        f"pairs: {len(examples)} usable of {len(pairs)} "
        f"({len(arms)} arms carry a text stream)",
        flush=True,
    )

    optimiser = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0

    reference_cache: dict[tuple[int, str], tuple[float, int]] = {}
    history = []
    for step in range(1, args.steps + 1):
        optimiser.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "margin": 0.0, "accuracy": 0.0, "chosen_logp": 0.0}
        used = 0
        for _ in range(args.batch_pairs):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            index = order[cursor]
            cursor += 1
            example = examples[index]

            scores = {}
            for side in ("chosen", "rejected"):
                key = (index, side)
                if key not in reference_cache:
                    with torch.no_grad(), lora.adapters_disabled(layers):
                        value, length = stream_logprob(
                            model,
                            tokenizer,
                            example[side],
                            tgt_lang=example["tgt_lang"],
                            device=device,
                        )
                    reference_cache[key] = (
                        float(value) if value is not None else None,
                        length,
                    )
                ref_value, length = reference_cache[key]
                if ref_value is None or length == 0:
                    scores = {}
                    break
                policy, _ = stream_logprob(
                    model,
                    tokenizer,
                    example[side],
                    tgt_lang=example["tgt_lang"],
                    device=device,
                )
                scores[side] = (policy, ref_value, length)
            if len(scores) != 2:
                continue

            chosen_policy, chosen_ref, chosen_len = scores["chosen"]
            rejected_policy, rejected_ref, rejected_len = scores["rejected"]
            margin = (chosen_policy - chosen_ref) / chosen_len - (
                rejected_policy - rejected_ref
            ) / rejected_len
            loss = -F.logsigmoid(args.beta * margin)
            if args.nll_weight > 0.0:
                loss = loss - args.nll_weight * chosen_policy / chosen_len
            (loss / args.batch_pairs).backward()

            totals["loss"] += float(loss)
            totals["margin"] += float(margin)
            totals["accuracy"] += float(margin > 0)
            totals["chosen_logp"] += float(chosen_policy) / chosen_len
            used += 1

        if used == 0:
            continue
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        schedule.step()
        row = {k: v / used for k, v in totals.items()}
        row.update(step=step, grad_norm=float(norm), lr=schedule.get_last_lr()[0])
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:4d}  loss {row['loss']:.4f}  "
                f"margin {row['margin']:+.4f}  acc {row['accuracy']:.2f}  "
                f"chosen_logp/tok {row['chosen_logp']:+.3f}  "
                f"|g| {row['grad_norm']:.3f}",
                flush=True,
            )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    lora.merge(layers)
    model.config.use_cache = True
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
