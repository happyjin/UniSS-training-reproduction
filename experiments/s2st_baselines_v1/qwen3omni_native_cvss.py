"""Baseline (c) of paper/baselines_run_plan_v1.md: Qwen3-Omni-30B-A3B-Instruct NATIVE offline S2ST on CVSS-T test --
its own Thinker translates each whole utterance (greedy) and its own Talker + code2wav speak it (preset voice "Ethan",
the official talker sampling), served by vLLM-Omni. This is experiments/uniss_qwen3_omni_v1/evaluation/p1_vllm_omni.py
(P1-a) on the paper's CVSS-T items (data/cvss_t/si_items_v1/items_<d>.jsonl, 4,897 a direction) instead of a parquet:
same processor-built prompt, same instruction (phase_d_e2e.INSTRUCTION), same sampling; nothing tuned.

Rows are written in si_infer's results shape, so the paper's scorers read them unchanged:
  reports/uniss_si_v4/speech_only.py  ->  evaluation.asr_transcribe (asr_speech.jsonl)  ->  BLEU as
  paper/bayes_simul_s2st_v11/scripts/cvss_table.py;  reports/uniss_si_v4/cvss_table1_metrics.py (A.PCP, SIM-O).

Work = both directions' items in a fixed order (en2zh then zh2en, each sorted by id), shard k takes items[k::n].
Output <out>/cvss_<d>/{wav,wav24k}/<id>.wav and <out>/cvss_<d>/part_<k>of<n>.jsonl (appended; ids already there are
skipped, so a killed shard resumes). Runs in venvs/vllm_omni_030 (put its bin on PATH).

    python -m experiments.s2st_baselines_v1.qwen3omni_native_cvss --out <dir> --shard k --num-shards n \
        --deploy-config experiments/s2st_baselines_v1/qwen3omni_native_split.yaml
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[2]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
ITEMS = REPO / "data/cvss_t/si_items_v1/items_{d}.jsonl"
LANGS = {"en2zh": ("eng", "cmn"), "zh2en": ("cmn", "eng")}
THINKER_EOS = 151645
CODEC_EOS = 2150


def instruction() -> dict:
    """phase_d_e2e's INSTRUCTION (keyed by target language), read without importing its training stack."""
    path = REPO / "experiments/uniss_qwen3_omni_v1/evaluation/phase_d_e2e.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "INSTRUCTION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("INSTRUCTION not found in phase_d_e2e.py")


def all_items() -> list:
    out = []
    for d in ("en2zh", "zh2en"):
        rows = [json.loads(l) for l in open(str(ITEMS).format(d=d), encoding="utf-8") if l.strip()]
        rows.sort(key=lambda r: r["id"])
        out += [dict(r, direction=d) for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--speaker", default="Ethan")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--limit", type=int, default=0, help="first N items of this shard only (smoke)")
    ap.add_argument("--deploy-config", default=None)
    args = ap.parse_args()

    from transformers import Qwen3OmniMoeProcessor
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    out = Path(args.out)
    tag = f"{args.shard}of{args.num_shards}"
    items = all_items()[args.shard::args.num_shards]
    if args.limit:
        items = items[: args.limit]
    done = set()
    for d in LANGS:
        (out / f"cvss_{d}" / "wav").mkdir(parents=True, exist_ok=True)
        (out / f"cvss_{d}" / "wav24k").mkdir(parents=True, exist_ok=True)
        part = out / f"cvss_{d}" / f"part_{tag}.jsonl"
        if part.exists():
            for line in part.read_text(encoding="utf-8").splitlines():
                try:
                    done.add((d, json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    work = [r for r in items if (r["direction"], r["id"]) not in done]
    print(f"shard {tag}: {len(items)} items, {len(work)} to do", flush=True)
    if not work:
        return

    text = instruction()
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    prompts = []
    for r in work:
        tgt = LANGS[r["direction"]][1]
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": r["audio"]},
            {"type": "text", "text": text[tgt]},
        ]}]
        audio, sr = sf.read(r["audio"], dtype="float32")
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
        SamplingParams(temperature=0.9, top_k=50, top_p=1.0, max_tokens=4096, repetition_penalty=1.05,
                       stop_token_ids=[CODEC_EOS], detokenize=False, seed=args.seed),
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096 * 16, detokenize=True,
                       repetition_penalty=1.1, seed=args.seed),
    ][: omni.num_stages]

    def write(k, hyp, wav24, elapsed):
        r = work[k]
        d = r["direction"]
        src, tgt = LANGS[d]
        base = out / f"cvss_{d}"
        stem = r["id"]
        path16, dur = None, 0.0
        if wav24 is not None and wav24.numel():
            sf.write(base / "wav24k" / f"{stem}.wav", wav24.numpy(), 24_000)
            wav16 = torchaudio.functional.resample(wav24, 24_000, 16_000)
            path16 = str((base / "wav" / f"{stem}.wav").resolve())
            sf.write(path16, wav16.numpy(), 16_000)
            dur = round(wav16.numel() / 16_000, 3)
        rec = {"id": r["id"], "mode": "si", "name": r["id"], "src_lang": src, "tgt_lang": tgt,
               "translation_ref": r["translation"], "translated_text": hyp,
               "audio_path": path16, "speech_path": path16, "audio_duration_seconds": dur,
               "source_audio_path": r["audio"], "source_duration_ms": r["dur_ms"],
               "system": "qwen3omni_native_vllm_omni", "speaker": args.speaker.lower(), "seed": args.seed,
               "engine": "vllm-omni 0.30.0", "gen_seconds_since_start": round(elapsed, 3),
               "error": None if path16 else "no audio"}
        with (base / f"part_{tag}.jsonl").open("a", encoding="utf-8") as fh:
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
            if len(written) % 200 == 0:
                print(f"  {len(written)}/{len(work)}  {len(written) / (time.time() - t0):.2f}/s", flush=True)
    wall = time.time() - t0
    omni.close()
    for k in range(len(work)):   # text without speech (or the reverse) is still recorded, as an error row
        if k not in written:
            write(k, texts.get(k, ""), wavs.get(k), wall)
    print(f"load {load_s:.0f}s  generate {wall:.0f}s for {len(work)} items ({len(work) / wall:.2f}/s)  "
          f"incomplete {len(work) - len(written)}", flush=True)
    print(f"QWEN3OMNI_NATIVE_SHARD_DONE {tag} -> {out}", flush=True)


if __name__ == "__main__":
    main()
