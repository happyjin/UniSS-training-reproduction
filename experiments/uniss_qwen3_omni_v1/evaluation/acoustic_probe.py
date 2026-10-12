"""The metric Phase D is actually judged on, cheap enough to run during it.

Cross-entropy cannot answer the question this phase asks. Teacher forcing
lets the Talker score well from the code history alone while ignoring the
text stream entirely, which is exactly the failure mode here -- and this
project has the curves to prove the point twice over: stage 1's dev code
CE *rose* from 7.159 to 7.301 while its audio got better, stage 2's *fell*
from 4.432 to 4.108 while its audio got worse.

So this generates. Gold target text through Qwen3's Thinker, the Talker's
own free-running decode, the BiCodec vocoder, Whisper, BLEU. TTS mode
rather than end to end, because the question is whether the Talker can
read the new representation and nothing else.

Measured cost: 7.56 s an utterance on one GPU, so 16 per rank across 8 is
about two minutes plus the ASR. At one probe every 2,000 steps that is
roughly 3% of a run, and it is the only thing that will say whether the
run worked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.uniss_omni_s2st_v1.evaluation.stage1_generate import generate_codes
from experiments.uniss_omni_s2st_v1.training.tts_data import tts_prompt
from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import build_assembly


@torch.no_grad()
def reply_states(assembly, text: str, lang: str):
    """What the Talker reads: Qwen3's states over a TTS reply."""
    templated = assembly.processor.apply_chat_template(
        [tts_prompt(text, lang)], add_generation_prompt=True, tokenize=False
    )
    prompt_text = templated[0] if isinstance(templated, list) else templated
    tok = assembly.tokenizer
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tok(text, add_special_tokens=False)["input_ids"]
    if not reply_ids:
        return None
    ids = torch.tensor([prompt_ids + reply_ids], device=assembly.device)
    out = assembly.thinker(input_ids=ids, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1][:, len(prompt_ids):, :]
    embeds = assembly.thinker.get_input_embeddings()(
        torch.tensor([reply_ids], device=assembly.device))
    return hidden, embeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--talker-checkpoint", required=True)
    ap.add_argument("--phase-d-checkpoint", default=None,
                    help="export carrying trained global_embed and markers")
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--thinker-from", choices=("qwen3", "qwen25"), default="qwen3",
                    help="qwen25 is the control arm: the Talker on the "
                         "representation stage 1 trained it for")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    from uniss import UniSSTokenizer

    rows = []
    for path in args.dev_parquet:
        rows.extend(pq.read_table(path, columns=[
            "id", "translation", "transcription", "src_lang", "tgt_lang",
            "bicodec_global", "target_bicodec", "source_audio_path",
            "reference_audio_path", "synthetic_source", "synthetic_reference",
            "source_audio_duration_seconds", "reference_audio_duration_seconds",
        ]).to_pylist())
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    # A fixed random sample, not the first N: an id-sorted prefix scored
    # 3.29 BLEU high when this project last leaned on one.
    pick = np.random.default_rng(args.seed).permutation(len(rows))[: args.limit]
    rows = [rows[int(i)] for i in sorted(pick)]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} utterances", flush=True)

    a = build_assembly(talker_checkpoint=args.talker_checkpoint, device=args.device,
                       thinker_from=args.thinker_from)
    if args.phase_d_checkpoint:
        state = torch.load(args.phase_d_checkpoint, map_location="cpu",
                           weights_only=False)
        a.talker.load_state_dict(state["talker"], strict=False)
        with torch.no_grad():
            a.global_embed.weight.copy_(
                state["global_embed"].to(a.global_embed.weight.dtype))
            a.markers.eos.copy_(state["markers"]["eos"])
            a.markers.pad.copy_(state["markers"]["pad"])
    a.talker.eval()

    class _Shim:
        """generate_codes reaches for model.thinker and model.talker."""
        def __init__(self, assembly):
            self.thinker = assembly.thinker
            self.talker = assembly.talker
            self.device = assembly.device

    model = _Shim(a)
    vocoder = UniSSTokenizer.from_pretrained(
        args.speech_tokenizer, device=torch.device(args.device))
    out_dir = Path(args.output)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)

    records = []
    for position, row in enumerate(rows):
        text = (row["translation"] or "").strip()
        states = reply_states(a, text, row["tgt_lang"]) if text else None
        codes = []
        if states is not None:
            codes = generate_codes(
                model, a.processor, text, row["tgt_lang"],
                layout=a.layout, output_mask=a.output_mask, device=a.device,
                max_steps=args.max_steps, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                global_embed=a.global_embed,
                bicodec_global=torch.as_tensor(row["bicodec_global"], dtype=torch.long),
                reply_states=states,
                markers=a.markers.embeds(states[0].dtype, a.device),
            )
        index = args.shard + position * args.num_shards
        name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        audio_path = None
        if codes:
            tokens = torch.tensor([*row["bicodec_global"], *codes],
                                  dtype=torch.long, device=a.device)
            vocoder.save_audio(vocoder.decode(tokens),
                               out_dir / "wav" / f"{name}.wav", sample_rate=16000)
            audio_path = str(out_dir / "wav" / f"{name}.wav")
        records.append({
            "index": index, "name": name, "id": row["id"], "mode": "tts",
            "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
            "dataset_name": "CVSS-T",
            "transcription_ref": row["transcription"],
            "translation_ref": row["translation"],
            # TTS mode: the text is given, so the hypothesis is the reference.
            "generated_text_raw": text, "generated_text_clean": text,
            "translated_text": text, "reference_text": row["translation"],
            "source_audio_path": row["source_audio_path"],
            "reference_audio_path": row["reference_audio_path"],
            "source_audio_duration_seconds": row["source_audio_duration_seconds"],
            "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
            "synthetic_source": bool(row["synthetic_source"]),
            "synthetic_reference": bool(row["synthetic_reference"]),
            "codes": len(codes), "gold_codes": len(row["target_bicodec"]),
            "audio_path": audio_path, "error": None,
        })
    with (out_dir / f"probe_{args.shard}.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"-> {out_dir}  {sum(1 for r in records if r['audio_path'])}/{len(records)} 有音频")


if __name__ == "__main__":
    main()
