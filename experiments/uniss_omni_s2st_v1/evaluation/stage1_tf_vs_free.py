"""Is the model untrained, or is the generation loop wrong?

Free-running decode emits the same ~50-code cycle for every utterance,
which no partially-trained model does -- a weak model is wrong in
input-dependent ways. The two candidates are the model and the loop, and
teacher forcing separates them: it is the path training actually
optimised, so if its argmax is diverse and input-dependent while
free-running is a fixed cycle, the loop is at fault.

Reports, per utterance: teacher-forced top-1 accuracy against gold, how
many distinct codes each path emits, and whether two different utterances
produce the same output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
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
from experiments.uniss_omni_s2st_v1.evaluation.stage1_generate import generate_codes
from experiments.uniss_omni_s2st_v1.training.stage1_talker_warmup import load_dev
from experiments.uniss_omni_s2st_v1.training.tts_data import tts_prompt


@torch.no_grad()
def teacher_forced(model, processor, row, *, layout, output_mask, device, global_embed):
    """One forward over the gold codes, exactly as the trainer runs it."""
    thinker, talker = model.thinker, model.talker
    templated = processor.apply_chat_template(
        [tts_prompt(row["text"], row["lang"])], add_generation_prompt=True, tokenize=False
    )
    prompt_text = templated[0] if isinstance(templated, list) else templated
    tok = processor.tokenizer
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tok(row["text"], add_special_tokens=False)["input_ids"]

    out = thinker(
        input_ids=torch.tensor([prompt_ids + reply_ids], device=device),
        output_hidden_states=True, return_dict=True,
    )
    hidden = out.hidden_states[-1][:, len(prompt_ids):, :]
    reply = torch.tensor([reply_ids], device=device)
    embeds = thinker.get_input_embeddings()(reply)
    eos_embed = thinker.get_input_embeddings()(
        torch.tensor([[talker.text_eos_token]], device=device))
    pad_embed = thinker.get_input_embeddings()(
        torch.tensor([[talker.text_pad_token]], device=device))

    codes = torch.as_tensor(row["codes"], dtype=torch.long, device=device)
    codec_ids, labels, mask = build_codec_sequence(
        [codes], bos_id=talker.codec_bos_token,
        pad_id=talker.codec_pad_token, eos_id=layout.special_ids["eos"],
    )
    prefix_width = 0 if global_embed is None else 32
    steps = codec_ids.shape[1] + prefix_width
    stream = build_text_stream(
        hidden.float(), embeds.float(), steps=steps,
        eos_embed=eos_embed.float(), pad_embed=pad_embed.float(),
    ).to(dtype=hidden.dtype)

    codec_embeds = talker.get_input_embeddings()(codec_ids)
    if prefix_width:
        g = torch.as_tensor(row["globals"], dtype=torch.long, device=device).view(1, -1)
        codec_embeds = torch.cat([global_embed(g), codec_embeds], dim=1)
        labels = prefix_labels(labels, prefix_width)
        mask = prefix_mask(mask, prefix_width)
    lm_input = talker.thinker_to_talker_proj(codec_embeds + stream)
    position_ids = torch.arange(steps, device=device).view(1, 1, -1).expand(3, 1, -1)
    result = talker.model(
        inputs_embeds=lm_input, attention_mask=mask,
        position_ids=position_ids, return_dict=True,
    )
    logits = apply_output_mask(talker.codec_head(result.last_hidden_state), output_mask)
    predicted = logits.argmax(dim=-1)[0]
    keep = labels[0] != -100
    accuracy = float((predicted[keep] == labels[0][keep]).float().mean())
    return predicted[keep].tolist(), accuracy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--dev-subset", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import Qwen2_5OmniForConditionalGeneration

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
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

    rows = load_dev(args.dev_parquet, args.dev_subset, 600)[: args.samples]
    report = []
    for row in rows:
        tf, acc = teacher_forced(
            model, processor, row, layout=layout, output_mask=mask,
            device=model.device, global_embed=global_embed,
        )
        free = generate_codes(
            model, processor, row["text"], row["lang"],
            layout=layout, output_mask=mask, device=model.device,
            max_steps=args.max_steps, global_embed=global_embed,
            bicodec_global=(
                None if global_embed is None
                else torch.as_tensor(row["globals"], dtype=torch.long)
            ),
        )
        entry = {
            "id": row["id"],
            "gold_len": int(len(row["codes"])),
            "teacher_forced_accuracy": acc,
            "teacher_forced_unique": len(set(tf)),
            "teacher_forced_head": tf[:12],
            "free_running_len": len(free),
            "free_running_unique": len(set(free)),
            "free_running_head": free[:12],
        }
        report.append(entry)
        print(
            f"  {row['id'][:34]:<36} TF 准确率 {acc*100:5.2f}%  "
            f"TF 唯一码 {entry['teacher_forced_unique']:>4}  "
            f"自由生成唯一码 {entry['free_running_unique']:>4}",
            flush=True,
        )

    heads = [tuple(e["free_running_head"]) for e in report]
    identical = len(set(heads)) == 1 and len(heads) > 1
    print(f"\n不同句子的自由生成开头是否完全相同: {identical}")
    if identical:
        print(f"  共同开头: {heads[0]}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"rows": report, "free_running_identical": identical},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
