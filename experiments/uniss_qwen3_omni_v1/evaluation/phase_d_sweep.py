"""Decoding sweep for the end-to-end talker: the thinker once, the talker many.

ASR-BLEU sits far below Text-BLEU for the end-to-end route, and the gap is
almost all substitutions -- the speech has the right length and says the
wrong words. Both routes decode the talker the same way (temperature 0.9,
top-p 0.8, repetition penalty 1.3), settings chosen for the cascade, whose
talker reads cleaner states and is more certain of each code. A less
certain talker turns the same sampling into more misreadings, so the
first question is how much of the gap the decoding itself accounts for.

Everything upstream of the talker is the same under every decoding
setting: the thinker decodes greedily, so its translation and the states
it wrote it from do not change. This runs the 30B thinker once per
utterance and the 0.4B talker once per setting, instead of the whole
pipeline once per setting.

Each setting writes the records phase_d_e2e writes, with ``mode`` set to
the setting's name. That keeps (id, mode) unique across settings -- the
key the ASR and UTMOS scripts deduplicate on when resuming -- and it is
the field text_metrics groups by, so one scoring pass over all settings
returns one BLEU per setting and direction from the project's own scorer.

Records are appended one utterance at a time and an interrupted shard
resumes where it stopped: the generator in phase_d_e2e writes only at the
end, and a shard that dies there loses everything it did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.uniss_omni_s2st_v1.evaluation.stage1_generate import generate_codes
from experiments.uniss_qwen3_omni_v1.evaluation.phase_d_e2e import (
    INSTRUCTION,
    translate_and_state,
)
from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import build_assembly

# phase_d_e2e's settings, which every number so far was decoded with.
BASE_SEED = 20261009
TOP_K = 40
TOP_P = 0.8


def parse_configs(spec: str) -> list[dict]:
    """``name:temperature:repetition_penalty:seed_offset`` separated by commas."""
    out = []
    for item in spec.split(","):
        name, t, r, s = item.strip().split(":")
        out.append({"name": name, "temperature": float(t),
                    "repetition_penalty": float(r), "seed": BASE_SEED + int(s)})
    if len({c["name"] for c in out}) != len(out):
        raise SystemExit("config names must be unique")
    return out


def load(args):
    """The same assembly and checkpoint loading phase_d_e2e does."""
    a = build_assembly(talker_checkpoint=args.talker_checkpoint, device=args.device)
    state = torch.load(args.phase_d_checkpoint, map_location="cpu", weights_only=False)
    if state.get("proj_variant") == "mlp":
        from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import (
            wrap_projection,
        )
        wrap_projection(a.talker)
    if state.get("thinker_lora"):
        from experiments.uniss_omni_s2st_v1.training.stage2_tasks import (
            THINKER_TARGETS,
        )
        from experiments.uniss_streaming_p2st_traj_v1.training import lora
        names = {k.rsplit(".", 1)[0] for k in state["thinker_lora"]}
        layers = lora.inject(
            a.thinker, rank=int(state.get("lora_rank", 32)),
            alpha=float(state.get("lora_alpha", 64.0)),
            targets=THINKER_TARGETS,
            layer_filter=lambda n: any(x.startswith(n + ".") for x in names),
        )
        lora.load_state_dict(layers, state["thinker_lora"])
        a.thinker.requires_grad_(False)
    info = a.talker.load_state_dict(state["talker"], strict=False)
    stray = [k for k in list(info.missing_keys) + list(info.unexpected_keys)
             if "thinker_to_talker_proj" in k]
    if stray:
        raise SystemExit(f"projection keys did not match: {stray}")
    with torch.no_grad():
        a.global_embed.weight.copy_(state["global_embed"].to(a.global_embed.weight.dtype))
        a.markers.eos.copy_(state["markers"]["eos"])
        a.markers.pad.copy_(state["markers"]["pad"])
    a.talker.eval()
    print(f"talker loaded: proj={state.get('proj_variant', 'linear')}, "
          f"missing {len(info.missing_keys)}, unexpected {len(info.unexpected_keys)}",
          flush=True)
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--talker-checkpoint", required=True)
    ap.add_argument("--phase-d-checkpoint", required=True)
    ap.add_argument("--dev-parquet", required=True)
    ap.add_argument("--subset", required=True, help="jsonl with one id per line")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from uniss import UniSSTokenizer

    configs = parse_configs(args.configs)
    wanted = {json.loads(l)["id"] for l in
              Path(args.subset).read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = pq.read_table(args.dev_parquet, columns=[
        "id", "translation", "transcription", "src_lang", "tgt_lang",
        "bicodec_global", "target_bicodec", "source_audio_path",
        "reference_audio_path", "synthetic_source", "synthetic_reference",
        "source_audio_duration_seconds", "reference_audio_duration_seconds",
    ]).to_pylist()
    rows = [r for r in rows if r["id"] in wanted]
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    rows = rows[args.shard :: args.num_shards]

    out_dir = Path(args.output)
    done: dict[str, set[str]] = {}
    for c in configs:
        (out_dir / c["name"] / "wav").mkdir(parents=True, exist_ok=True)
        part = out_dir / c["name"] / f"e2e_{args.shard}.jsonl"
        done[c["name"]] = set()
        if part.exists():
            # A shard killed mid-write leaves a torn last line; that
            # utterance is simply redone.
            for line in part.read_text(encoding="utf-8").splitlines():
                try:
                    done[c["name"]].add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [r for r in rows if any(r["id"] not in done[c["name"]] for c in configs)]
    print(f"{len(rows)} utterances, {len(rows) - len(todo)} already done, "
          f"{len(configs)} settings: {[c['name'] for c in configs]}", flush=True)
    if not todo:
        return

    a = load(args)

    class _Shim:
        def __init__(self, assembly):
            self.thinker = assembly.thinker
            self.talker = assembly.talker
    model = _Shim(a)
    vocoder = UniSSTokenizer.from_pretrained(
        args.speech_tokenizer, device=torch.device(args.device))

    for position, row in enumerate(todo):
        hypothesis, states = translate_and_state(
            a, row["source_audio_path"], row["tgt_lang"], args.max_new_tokens,
            single_pass=True, instruction=INSTRUCTION[row["tgt_lang"]])
        index = rows.index(row) * args.num_shards + args.shard
        name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        for c in configs:
            if row["id"] in done[c["name"]]:
                continue
            codes = []
            if states is not None:
                codes = generate_codes(
                    model, a.processor, hypothesis, row["tgt_lang"],
                    layout=a.layout, output_mask=a.output_mask, device=a.device,
                    max_steps=args.max_steps, temperature=c["temperature"],
                    top_k=TOP_K, top_p=TOP_P,
                    repetition_penalty=c["repetition_penalty"], seed=c["seed"],
                    global_embed=a.global_embed,
                    bicodec_global=torch.as_tensor(row["bicodec_global"], dtype=torch.long),
                    reply_states=states,
                    markers=a.markers.embeds(states[0].dtype, a.device))
            audio_path = None
            if codes:
                wav = out_dir / c["name"] / "wav" / f"{name}.wav"
                tokens = torch.tensor([*row["bicodec_global"], *codes],
                                      dtype=torch.long, device=a.device)
                vocoder.save_audio(vocoder.decode(tokens), wav, sample_rate=16000)
                audio_path = str(wav)
            record = {
                "index": index, "name": name, "id": row["id"], "mode": c["name"],
                "decode_temperature": c["temperature"],
                "decode_repetition_penalty": c["repetition_penalty"],
                "decode_seed": c["seed"],
                "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
                "dataset_name": "CVSS-T",
                "transcription_ref": row["transcription"],
                "translation_ref": row["translation"],
                "generated_text_raw": hypothesis, "generated_text_clean": hypothesis,
                "translated_text": hypothesis, "reference_text": row["translation"],
                "source_audio_path": row["source_audio_path"],
                "reference_audio_path": row["reference_audio_path"],
                "source_audio_duration_seconds": row["source_audio_duration_seconds"],
                "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
                "synthetic_source": bool(row["synthetic_source"]),
                "synthetic_reference": bool(row["synthetic_reference"]),
                "codes": len(codes), "gold_codes": len(row["target_bicodec"]),
                "audio_path": audio_path, "error": None,
            }
            with (out_dir / c["name"] / f"e2e_{args.shard}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if position % 10 == 0:
            print(f"  {position}/{len(todo)}  {hypothesis[:44]}", flush=True)
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
