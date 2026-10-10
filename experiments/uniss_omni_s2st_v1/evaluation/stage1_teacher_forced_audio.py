"""Decode the model's teacher-forced predictions, to hear what it knows.

Free-running decode collapses after a handful of codes -- top-1 is a few
percent, so the context leaves the training distribution and eos wins.
That says nothing about whether the per-step predictions carry speech.

Feeding gold history and decoding the argmax at every step separates the
two. If this is audible speech, the model has learned the mapping and the
problem is purely that it cannot carry its own context; if it is noise,
the per-step predictions are not yet good enough for any decoding scheme
to rescue.

Writes three files per utterance: the teacher-forced prediction, the gold
codes through the same vocoder, and -- for reference -- how many of the
predictions matched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import (
    build_codec_sequence,
    build_text_stream,
    prefix_labels,
    prefix_mask,
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
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration
    from uniss import UniSSTokenizer

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    if hasattr(model, "token2wav"):
        del model.token2wav
    layout = read_codec_layout(model.talker.config)
    retarget_talker_codebook(model.talker, head_scale=0.1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.talker.load_state_dict(state["talker"], strict=False)
    global_embed = None
    if state.get("global_embed") is not None:
        w = state["global_embed"]
        global_embed = torch.nn.Embedding(w.shape[0], w.shape[1])
        with torch.no_grad():
            global_embed.weight.copy_(w)
        global_embed = global_embed.to(device=model.device, dtype=w.dtype)
    model.eval()
    mask = valid_output_mask(layout, device=model.device)
    vocoder = UniSSTokenizer.from_pretrained(
        args.speech_tokenizer, device=torch.device(args.device)
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    thinker, talker = model.thinker, model.talker
    device = model.device
    report = []

    for index, row in enumerate(
        load_dev(args.dev_parquet, args.dev_subset, 600)[: args.samples], start=1
    ):
        templated = processor.apply_chat_template(
            [tts_prompt(row["text"], row["lang"])],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_text = templated[0] if isinstance(templated, list) else templated
        tok = processor.tokenizer
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        reply_ids = tok(row["text"], add_special_tokens=False)["input_ids"]

        with torch.no_grad():
            out = thinker(
                input_ids=torch.tensor([prompt_ids + reply_ids], device=device),
                output_hidden_states=True, return_dict=True,
            )
            hidden = out.hidden_states[-1][:, len(prompt_ids):, :]
            embeds = thinker.get_input_embeddings()(
                torch.tensor([reply_ids], device=device))
            eos_embed = thinker.get_input_embeddings()(
                torch.tensor([[talker.text_eos_token]], device=device))
            pad_embed = thinker.get_input_embeddings()(
                torch.tensor([[talker.text_pad_token]], device=device))

            codes = torch.as_tensor(row["codes"], dtype=torch.long, device=device)
            codec_ids, labels, codec_mask = build_codec_sequence(
                [codes], bos_id=talker.codec_bos_token,
                pad_id=talker.codec_pad_token, eos_id=layout.special_ids["eos"],
            )
            width = 0 if global_embed is None else 32
            steps = codec_ids.shape[1] + width
            stream = build_text_stream(
                hidden.float(), embeds.float(), steps=steps,
                eos_embed=eos_embed.float(), pad_embed=pad_embed.float(),
            ).to(dtype=hidden.dtype)
            codec_embeds = talker.get_input_embeddings()(codec_ids)
            if width:
                g = torch.as_tensor(row["globals"], dtype=torch.long, device=device)
                codec_embeds = torch.cat(
                    [global_embed(g.view(1, -1)), codec_embeds], dim=1)
                labels = prefix_labels(labels, width)
                codec_mask = prefix_mask(codec_mask, width)
            lm_input = talker.thinker_to_talker_proj(codec_embeds + stream)
            positions = torch.arange(steps, device=device).view(1, 1, -1).expand(3, 1, -1)
            logits = apply_output_mask(
                talker.codec_head(
                    talker.model(
                        inputs_embeds=lm_input, attention_mask=codec_mask,
                        position_ids=positions, return_dict=True,
                    ).last_hidden_state
                ),
                mask,
            )
            predicted = logits.argmax(dim=-1)[0]

        keep = labels[0] != -100
        gold = labels[0][keep]
        pred = predicted[keep]
        # Drop the final eos target: it is not an acoustic code.
        pred_codes = [c for c in pred[:-1].tolist() if c < layout.code_size]
        accuracy = float((pred == gold).float().mean())

        name = f"{index:02d}_{row['lang']}"
        globals_ = list(row["globals"])
        for tag, values in (("教师强制", pred_codes), ("金标", list(row["codes"]))):
            tokens = torch.tensor(
                [*globals_, *values], dtype=torch.long, device=device)
            audio = vocoder.decode(tokens)
            vocoder.save_audio(audio, out_dir / f"{name}_{tag}.wav", sample_rate=16000)
        report.append(
            {
                "name": name,
                "text": row["text"],
                "accuracy": accuracy,
                "predicted_codes": len(pred_codes),
                "gold_codes": int(len(row["codes"])),
            }
        )
        print(
            f"  {name}  top-1 {accuracy*100:5.2f}%  "
            f"预测 {len(pred_codes)} 码 / 金标 {len(row['codes'])}",
            flush=True,
        )

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
