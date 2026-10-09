"""Stage 0: assemble Thinker + Talker on BiCodec targets, and train nothing.

Three things are checked, and only one of them is a number.

**The assembly is causal and correctly offset.** This is the decisive test,
because it is the one a wiring mistake cannot pass. Position t is fed
``[bos, codes[0..t-1]]`` and must predict ``codes[t]``, so perturbing
``codes[t]`` must leave every logit at position <= t bit-identical and must
change the logits at t+1. Labels off by one, a text stream accidentally
aligned to the codes it predicts, or a broken causal mask all show up here.

**Codes are in range.** Every target code must fall inside the 8,192 slots
below the reserved block, or the codebook is silently truncated.

**The initial distribution is not confidently wrong.** With a random head
there is no "correct" cross-entropy to hit, so this is a sweep rather than a
threshold: ``--head-scales`` multiplies the head's draw, and the report says
what each choice costs. A head drawn at the pretrained scale is as confident
as a trained one while being random, which starts training by fighting its
own noise.

One thing that is *not* evidence of a fault: probability mass on the 256
reserved rows. Those rows keep their trained weights and are the only
trained thing in the head, so they fire. It falls as the code rows learn.

Nothing here trains, and nothing here is written back to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.uniss_omni_s2st_v1.data.dual_source import iter_utterances
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

INSTRUCTION = {
    ("en", "zh"): "Listen to the English speech and translate it into Chinese.",
    ("zh", "en"): "Listen to the Chinese speech and translate it into English.",
}


def thinker_reply_states(thinker, processor, waveform, instruction, reply_text, device):
    """Teacher-force the gold reply and return its hidden states and embeds."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "x.wav"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        [conversation], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt, audio=[waveform], sampling_rate=16_000, return_tensors="pt"
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    reply_ids = processor.tokenizer(
        reply_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    prompt_length = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat([inputs["input_ids"], reply_ids], dim=1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(reply_ids)], dim=1
        )

    with torch.no_grad():
        out = thinker(**inputs, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1][:, prompt_length:, :]
    embeds = thinker.get_input_embeddings()(reply_ids)
    return hidden, embeds


def talker_logits(talker, codec_ids, text_stream, mask, device):
    """One teacher-forced Talker forward, bypassing generation-time rope.

    ``talker.forward`` derives mrope positions from the generation-time
    layout (prompt ids, audio lengths, grids). A standalone code stream has
    none of that, so positions are plain and sequential here.
    """
    steps = codec_ids.shape[1]
    codec_embeds = talker.get_input_embeddings()(codec_ids)
    lm_input = talker.thinker_to_talker_proj(codec_embeds + text_stream)
    position_ids = torch.arange(steps, device=device).view(1, 1, -1).expand(3, codec_ids.shape[0], -1)
    out = talker.model(
        inputs_embeds=lm_input,
        attention_mask=mask,
        position_ids=position_ids,
        return_dict=True,
    )
    return talker.codec_head(out.last_hidden_state).float()


def causality_probe(talker, codec_ids, text_stream, mask, device, *, position, code_size):
    """Perturb one input code; report where the logits were allowed to move.

    Position t is fed ``[bos, codes[0..t-1]]`` and predicts ``codes[t]``, so
    ``codes[t]`` enters at input index t+1. Changing it must leave every
    logit at index <= t untouched and must move index t+1.
    """
    with torch.no_grad():
        base = talker_logits(talker, codec_ids, text_stream, mask, device)
        altered = codec_ids.clone()
        index = position + 1
        altered[0, index] = (int(altered[0, index]) + 1) % code_size
        moved = talker_logits(talker, altered, text_stream, mask, device)
    delta = (moved - base).abs().amax(dim=-1)[0]
    return {
        "perturbed_input_index": index,
        "max_delta_at_or_before": float(delta[: index].max()) if index else 0.0,
        "delta_at_perturbed": float(delta[index]),
        "max_delta_after": float(delta[index + 1 :].max())
        if index + 1 < delta.numel()
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default="data/unist_omni_v1/speech")
    ap.add_argument("--parquet-root", default="data/raw/UniST")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument(
        "--head-scales", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1, 0.0]
    )
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument(
        "--no-output-mask",
        action="store_true",
        help="score the raw 8,448-way softmax, to show what the mask is worth",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    thinker, talker = model.thinker, model.talker
    device = model.device

    layout = read_codec_layout(talker.config)
    print(
        f"codec layout: {layout.vocab_size} slots,"
        f" codes 0..{layout.code_size - 1},"
        f" {layout.vocab_size - layout.code_size} reserved, specials {layout.special_ids}",
        flush=True,
    )

    # The sweep re-draws the same rows repeatedly, and a draw takes its scale
    # from the rows it replaces -- so the originals have to be kept, or every
    # scale after the first would be measured against the previous draw.
    output_mask = (
        None if args.no_output_mask else valid_output_mask(layout, device=device)
    )
    if output_mask is not None:
        print(
            f"output mask: {int(output_mask.sum())} of {layout.vocab_size}"
            " slots reachable",
            flush=True,
        )

    pristine = {
        "embed": talker.get_input_embeddings().weight[: layout.code_size].clone(),
        "head": talker.codec_head.weight[: layout.code_size].clone(),
    }

    thinker_embed = thinker.get_input_embeddings()
    eos_embed = thinker_embed(torch.tensor([[talker.text_eos_token]], device=device)).detach()
    pad_embed = thinker_embed(torch.tensor([[talker.text_pad_token]], device=device)).detach()

    # The Thinker pass does not depend on the head draw, so it runs once.
    prepared = []
    for utterance in iter_utterances(
        args.manifest,
        audio_root=args.audio_root,
        parquet_root=args.parquet_root,
        limit=args.samples,
    ):
        waveform, _ = utterance.waveform(args.audio_root)
        instruction = INSTRUCTION[(utterance.source_lang, utterance.target_lang)]
        hidden, embeds = thinker_reply_states(
            thinker, processor, waveform, instruction, utterance.target_text, device
        )
        codes = torch.as_tensor(utterance.target_bicodec, device=device)
        highest = int(codes.max())
        if highest >= layout.code_size:
            raise ValueError(
                f"{utterance.sample_id}: code {highest} outside the"
                f" {layout.code_size}-entry codebook"
            )
        codec_ids, labels, mask = build_codec_sequence(
            [codes], bos_id=talker.codec_bos_token, pad_id=talker.codec_pad_token
        )
        text_stream = build_text_stream(
            hidden.float(),
            embeds.float(),
            steps=codec_ids.shape[1],
            eos_embed=eos_embed.float(),
            pad_embed=pad_embed.float(),
        ).to(dtype=torch.bfloat16)
        prepared.append(
            {
                "id": utterance.sample_id,
                "codec_ids": codec_ids,
                "labels": labels,
                "mask": mask,
                "text_stream": text_stream,
                "codes": int(codes.numel()),
                "reply_tokens": int(hidden.shape[1]),
                "max_code": highest,
            }
        )
    if not prepared:
        raise SystemExit("no utterances resolved")
    print(f"{len(prepared)} utterances prepared", flush=True)

    sweep = []
    for scale in args.head_scales:
        with torch.no_grad():
            talker.get_input_embeddings().weight[: layout.code_size] = pristine["embed"]
            talker.codec_head.weight[: layout.code_size] = pristine["head"]
        retarget_talker_codebook(talker, seed=args.seed, head_scale=scale)

        rows = []
        for item in prepared:
            with torch.no_grad():
                logits = talker_logits(
                    talker, item["codec_ids"], item["text_stream"], item["mask"], device
                )
                if output_mask is not None:
                    logits = apply_output_mask(logits, output_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                item["labels"].view(-1),
                ignore_index=-100,
            )
            probs = logits.softmax(dim=-1)
            rows.append(
                {
                    "id": item["id"],
                    "codes": item["codes"],
                    "reply_tokens": item["reply_tokens"],
                    "logits": list(logits.shape),
                    "ce": float(loss),
                    "entropy": float(
                        -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1).mean()
                    ),
                    "reserved_mass": float(
                        probs[..., layout.code_size :].sum(dim=-1).mean()
                    ),
                    "eos_mass": float(probs[..., layout.special_ids["eos"]].mean()),
                    "other_reserved_mass": float(
                        (
                            probs[..., layout.code_size :].sum(dim=-1)
                            - probs[..., layout.special_ids["eos"]]
                        )
                        .mean()
                    ),
                }
            )
        summary = {
            "head_scale": scale,
            "mean_ce": sum(r["ce"] for r in rows) / len(rows),
            "mean_entropy": sum(r["entropy"] for r in rows) / len(rows),
            "mean_reserved_mass": sum(r["reserved_mass"] for r in rows) / len(rows),
            "max_reserved_mass": max(r["reserved_mass"] for r in rows),
            "mean_eos_mass": sum(r["eos_mass"] for r in rows) / len(rows),
            "mean_other_reserved_mass": sum(r["other_reserved_mass"] for r in rows)
            / len(rows),
        }
        sweep.append({**summary, "rows": rows})
        print(
            f"  head_scale {scale:<5}  CE {summary['mean_ce']:7.3f}"
            f"  H {summary['mean_entropy']:6.3f}"
            f"  reserved mean {summary['mean_reserved_mass']:.3f}"
            f" max {summary['max_reserved_mass']:.3f}"
            f"  (eos {summary['mean_eos_mass']:.3f},"
            f" other {summary['mean_other_reserved_mass']:.3e})",
            flush=True,
        )

    # Scale-independent: wiring, not weights.
    probe_item = max(prepared, key=lambda item: item["codes"])
    probes = [
        causality_probe(
            talker,
            probe_item["codec_ids"],
            probe_item["text_stream"],
            probe_item["mask"],
            device,
            position=position,
            code_size=layout.code_size,
        )
        for position in (0, probe_item["codes"] // 2, probe_item["codes"] - 2)
        if 0 <= position < probe_item["codes"] - 1
    ]
    leak = max(p["max_delta_at_or_before"] for p in probes)
    moved = min(p["delta_at_perturbed"] for p in probes)
    causal_ok = leak == 0.0 and moved > 0.0
    print(
        f"  causality: max leak at or before the perturbed step {leak:.3e},"
        f" min movement at it {moved:.3e} -> {'PASS' if causal_ok else 'FAIL'}",
        flush=True,
    )

    uniform = math.log(layout.code_size)
    report = {
        "samples": len(prepared),
        "output_mask": None if output_mask is None else int(output_mask.sum().item()),
        "uniform_ce": uniform,
        "vocab": layout.vocab_size,
        "code_size": layout.code_size,
        "max_code_seen": max(item["max_code"] for item in prepared),
        "causal_ok": causal_ok,
        "causal_max_leak": leak,
        "causal_min_movement": moved,
        "sweep": [{k: v for k, v in entry.items() if k != "rows"} for entry in sweep],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"report": report, "sweep": sweep, "probes": probes},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
