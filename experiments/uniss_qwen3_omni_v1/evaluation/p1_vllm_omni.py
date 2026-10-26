"""P1-a on vLLM-Omni: the same official pipeline, served by an inference engine.

p1_native_talker.py runs Qwen3-Omni through transformers one utterance at
a time, which spends most of its time on per-step overhead: a full CVSS-T
pass is close to four hours on eight H200s with the GPUs ~18% busy.
vLLM-Omni runs the same three stages -- thinker, talker with its residual
code predictor, code2wav -- as an engine: continuous batching without
padding, CUDA graphs, and the stages pipelined. This writes P1-a records
in p1_native_talker's shape so the scoring harness and p1_compare read
them unchanged; P1-b's BiCodec swap stays where BiCodec runs.

What is held fixed: the prompt (built by the same processor and
instruction), a greedy thinker to the same end-of-turn token, and the
talker's official sampling settings. What is not: the kernels. vLLM's
attention and MoE kernels are not transformers', so some translations
come out differently, and the comparison against the transformers run
says how many.

Runs in the vllm_omni_030 venv, not the conda environments.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
THINKER_EOS = 151645
CODEC_EOS = 2150


def instruction() -> dict:
    """phase_d_e2e's INSTRUCTION, read without importing its training stack."""
    tree = ast.parse((Path(__file__).parent / "phase_d_e2e.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "INSTRUCTION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("INSTRUCTION not found in phase_d_e2e.py")


def record(row, index, hypothesis, audio_path, codes, extra):
    """p1_native_talker.record, field for field."""
    name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
    return {
        "index": index, "name": name, "id": row["id"], "mode": "tts",
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
        "codes": codes, "gold_codes": len(row["target_bicodec"]),
        "audio_path": audio_path, "error": None if audio_path else "no audio",
        **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    help="name:parquet[:subset_jsonl]; without a subset, every row of the parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--speaker", default="Ethan")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--limit", type=int, default=0, help="per direction, for smoke runs")
    ap.add_argument("--deploy-config", default=None, help="vLLM-Omni deploy YAML; default: the package's for this model")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen3OmniMoeProcessor
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    out = Path(args.output)
    text = instruction()
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    work = []
    for spec in args.pair:
        name, parquet, *subset = spec.split(":")
        rows = pq.read_table(parquet, columns=[
            "id", "translation", "transcription", "src_lang", "tgt_lang",
            "bicodec_global", "target_bicodec", "source_audio_path",
            "reference_audio_path", "synthetic_source", "synthetic_reference",
            "source_audio_duration_seconds", "reference_audio_duration_seconds",
        ]).to_pylist()
        if subset and subset[0]:
            wanted = {json.loads(l)["id"] for l in Path(subset[0]).read_text().splitlines() if l.strip()}
            rows = [r for r in rows if r["id"] in wanted]
        rows.sort(key=lambda r: r["id"])
        if args.limit:
            rows = rows[: args.limit]
        (out / "p1a" / name / "wav").mkdir(parents=True, exist_ok=True)
        (out / "p1a" / name / "wav24k").mkdir(parents=True, exist_ok=True)
        part = out / "p1a" / name / f"p1_{args.shard}.jsonl"
        done = set()
        if part.exists():
            for line in part.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        work += [(name, i, r) for i, r in list(enumerate(rows))[args.shard :: args.num_shards] if r["id"] not in done]
    print(f"{len(work)} utterances to do in this shard", flush=True)
    if not work:
        return

    prompts = []
    for name, index, row in work:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": row["source_audio_path"]},
            {"type": "text", "text": text[row["tgt_lang"]]},
        ]}]
        audio, sr = sf.read(row["source_audio_path"], dtype="float32")
        prompts.append({
            "prompt": processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=False)[0],
            "multi_modal_data": {"audio": (audio, sr)},
            "additional_information": {"speaker": [args.speaker.lower()]},
        })

    kwargs = {"model": QWEN3, "async_chunk": False}
    if args.deploy_config:
        kwargs["deploy_config"] = args.deploy_config
    t_load = time.time()
    omni = Omni(**kwargs)
    load_s = time.time() - t_load
    sampling = [
        SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[THINKER_EOS], seed=args.seed),
        # generate's talker defaults; vLLM-Omni applies the codec suppression itself
        SamplingParams(temperature=0.9, top_k=50, top_p=1.0, max_tokens=4096, repetition_penalty=1.05,
                       stop_token_ids=[CODEC_EOS], detokenize=False, seed=args.seed),
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096 * 16, detokenize=True,
                       repetition_penalty=1.1, seed=args.seed),
    ][: omni.num_stages]

    def write(k, text, wav24, elapsed):
        """One utterance's record, as soon as both its text and its speech are in."""
        name, index, row = work[k]
        stem = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        a_path, frames = None, 0
        if wav24 is not None and wav24.numel():
            frames = int(wav24.numel() // 1920)
            sf.write(out / "p1a" / name / "wav24k" / f"{stem}.wav", wav24.numpy(), 24_000)
            a_path = str((out / "p1a" / name / "wav" / f"{stem}.wav").resolve())
            sf.write(a_path, torchaudio.functional.resample(wav24, 24_000, 16_000).numpy(), 16_000)
        rec = record(row, index, text, a_path, frames, {
            "system": "p1a_native_preset_voice_vllm_omni", "speaker": args.speaker.lower(),
            "engine": "vllm-omni 0.30.0", "seed": args.seed,
            # a batch engine has no per-utterance time; this is the time since generation began
            "gen_seconds": round(elapsed, 3),
            "audio_path_24k": str((out / "p1a" / name / "wav24k" / f"{stem}.wav").resolve()) if a_path else None})
        with (out / "p1a" / name / f"p1_{args.shard}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    texts, wavs, written = {}, {}, set()
    t0 = time.time()
    for output in omni.generate(prompts, sampling, py_generator=True, use_tqdm=False):
        k = int(output.request_id.split("_", 1)[0])
        if output.final_output_type == "text":
            texts[k] = output.outputs[0].text.strip()
        elif output.final_output_type == "audio":
            audio = output.outputs[0].multimodal_output["audio"]
            if isinstance(audio, list):
                audio = torch.cat([(a if isinstance(a, torch.Tensor) else torch.tensor(a)).flatten() for a in audio])
            wavs[k] = audio.float().detach().cpu().reshape(-1)
        if k in texts and k in wavs and k not in written:
            write(k, texts.pop(k), wavs.pop(k), time.time() - t0)
            written.add(k)
            if len(written) % 500 == 0:
                print(f"  {len(written)}/{len(work)}  {len(written) / (time.time() - t0):.2f}/s", flush=True)
    wall = time.time() - t0
    omni.close()
    for k in range(len(work)):  # a text with no speech, or the reverse, is still recorded
        if k not in written:
            write(k, texts.get(k, ""), wavs.get(k), wall)
    missing = len(work) - len(written)
    print(f"load {load_s:.0f}s  generate {wall:.0f}s for {len(work)} utterances "
          f"({wall / len(work):.2f}s each, {len(work) / wall:.2f}/s)  incomplete {missing}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
