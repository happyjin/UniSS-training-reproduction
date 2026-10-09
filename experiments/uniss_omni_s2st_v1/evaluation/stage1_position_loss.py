"""Where in the code stream is the loss? A question about the conditioning.

A four-character Chinese reply carries fifty-odd codes, so the text stream
is exhausted within a handful of steps and every step after that is
conditioned on the *same constant* pad embedding. Whatever the text
contributes has to be carried forward in the KV cache from those first few
steps.

Splitting the stream at the point the text runs out does not answer this,
because that split is confounded with position: the text-conditioned steps
are exactly the early ones, which also have the least autoregressive code
history. A loss curve that falls with offset says only that context helps.

The controlled comparison is the same offsets with the text ablated --
every step fed the constant pad embedding, as the tail already is. If the
two runs score the same where real text was arriving, the Talker is
ignoring it and running as an unconditional code language model, which no
further training would fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from experiments.uniss_omni_s2st_v1.training.stage1_talker_warmup import load_dev
from experiments.uniss_omni_s2st_v1.training.tts_data import tts_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--dev-subset", required=True)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument(
        "--ablate-text",
        action="store_true",
        help="feed the constant pad embedding at every step, as the tail already is",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    if hasattr(model, "token2wav"):
        del model.token2wav
    thinker, talker = model.thinker, model.talker
    layout = read_codec_layout(talker.config)
    retarget_talker_codebook(talker, head_scale=0.1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    talker.load_state_dict(state["talker"], strict=False)
    model.eval()
    device = model.device
    output_mask = valid_output_mask(layout, device=device)
    print(f"checkpoint step {state.get('step')}", flush=True)

    rows = load_dev(args.dev_parquet, args.dev_subset, 600)[: args.samples]
    tokenizer = processor.tokenizer

    in_text: list[float] = []
    after_text: list[float] = []
    by_offset: dict[int, list[float]] = {}

    for row in rows:
        templated = processor.apply_chat_template(
            [tts_prompt(row["text"], row["lang"])],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_text = templated[0] if isinstance(templated, list) else templated
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        reply_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = thinker(
                input_ids=torch.tensor([prompt_ids + reply_ids], device=device),
                output_hidden_states=True, return_dict=True,
            )
            hidden = out.hidden_states[-1][:, len(prompt_ids) :, :]
            reply = torch.tensor([reply_ids], device=device)
            embeds = thinker.get_input_embeddings()(reply)
            eos_embed = thinker.get_input_embeddings()(
                torch.tensor([[talker.text_eos_token]], device=device)
            )
            pad_embed = thinker.get_input_embeddings()(
                torch.tensor([[talker.text_pad_token]], device=device)
            )

            codes = torch.as_tensor(row["codes"], dtype=torch.long, device=device)
            codec_ids, labels, mask = build_codec_sequence(
                [codes], bos_id=talker.codec_bos_token,
                pad_id=talker.codec_pad_token, eos_id=layout.special_ids["eos"],
            )
            steps = codec_ids.shape[1]
            if args.ablate_text:
                stream = pad_embed.float().expand(1, steps, -1).to(dtype=hidden.dtype)
            else:
                stream = build_text_stream(
                    hidden.float(), embeds.float(), steps=steps,
                    eos_embed=eos_embed.float(), pad_embed=pad_embed.float(),
                ).to(dtype=hidden.dtype)
            lm_input = talker.thinker_to_talker_proj(
                talker.get_input_embeddings()(codec_ids) + stream
            )
            position_ids = torch.arange(steps, device=device).view(1, 1, -1).expand(3, 1, -1)
            result = talker.model(
                inputs_embeds=lm_input, attention_mask=mask,
                position_ids=position_ids, return_dict=True,
            )
            logits = apply_output_mask(
                talker.codec_head(result.last_hidden_state), output_mask
            ).float()
            per_token = F.cross_entropy(
                logits[0], labels[0], ignore_index=-100, reduction="none"
            )

        text_length = len(reply_ids)
        values = per_token.tolist()
        for index, value in enumerate(values):
            if labels[0, index].item() == -100:
                continue
            (in_text if index < text_length else after_text).append(value)
            by_offset.setdefault(min(index, 60), []).append(value)

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    report = {
        "checkpoint_step": state.get("step"),
        "ablate_text": args.ablate_text,
        "samples": len(rows),
        "while_text_arriving": {"positions": len(in_text), "mean_ce": mean(in_text)},
        "after_text_exhausted": {"positions": len(after_text), "mean_ce": mean(after_text)},
        "by_offset": {
            str(offset): {"n": len(values), "mean_ce": mean(values)}
            for offset, values in sorted(by_offset.items())
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"  while text arriving: {report['while_text_arriving']['mean_ce']:.3f}"
        f"  ({report['while_text_arriving']['positions']} positions)\n"
        f"  after text exhausted: {report['after_text_exhausted']['mean_ce']:.3f}"
        f"  ({report['after_text_exhausted']['positions']} positions)"
    )
    for offset in (0, 1, 2, 5, 10, 20, 40, 60):
        entry = report["by_offset"].get(str(offset))
        if entry:
            print(f"    offset {offset:>3}: CE {entry['mean_ce']:.3f}  n={entry['n']}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
