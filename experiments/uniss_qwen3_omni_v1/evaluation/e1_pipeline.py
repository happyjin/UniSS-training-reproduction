"""E1: Qwen3-Omni's Thinker and UniSS as one inference chain, untrained.

Source speech in, target speech out, in one process: the Thinker (vLLM)
translates; UniSS's own front end turns the same waveform into GLM-4
content tokens and BiCodec speaker tokens; UniSS (vLLM) speaks the
translation, reading the Thinker's token ids directly -- the two
vocabularies agree on every CVSS-T translation, so nothing is decoded to
text and re-tokenised between them; BiCodec renders the codes in the
source speaker's voice. One sample per sentence.

This is P2 step zero wired as a system rather than assembled from files:
same weights, same prompts, same sampling (repetition penalty 1.3 and the
stop guard). It is not trained, so it is not yet end to end; it is the
chain E2 trains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
UNISS = str(REPO / "pretrained_models/UniSS_full")
LANG = {"eng": "<|eng|>", "cmn": "<|cmn|>"}
THINKER_EOS = 151645
SEMANTIC_0, SEMANTIC_LAST, END_SEMANTIC, END_CONTENT = 155761, 163952, 180392, 180390
COLUMNS = ["id", "translation", "transcription", "src_lang", "tgt_lang", "target_bicodec",
           "source_audio_path", "reference_audio_path", "synthetic_source", "synthetic_reference",
           "source_audio_duration_seconds", "reference_audio_duration_seconds"]


def read_with_soundfile(path, *args, **kwargs):
    """torchaudio.load's (channels, samples) tensor and rate, without FFmpeg.

    UniSS's GLM-4 front end reads its input with torchaudio.load, and the
    torchaudio in this venv decodes through torchcodec, which needs FFmpeg
    shared libraries the host does not have. Every input here is a WAV file.
    """
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    help="name:parquet[:ids_jsonl]; without ids, every row of the parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="per direction, for smoke runs")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--thinker-memory", type=float, default=0.55)
    ap.add_argument("--talker-memory", type=float, default=0.15)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from uniss import UniSSTokenizer

    from experiments.uniss_qwen3_omni_v1.evaluation.uniss_stop_guard import StopGuard

    instruction = load_module("p1v", REPO / "experiments/uniss_qwen3_omni_v1/evaluation/p1_vllm_omni.py").instruction()
    process_input = load_module("uniss_prompt", REPO / "uniss/cli/prompt.py").process_input

    work = []
    for spec in args.pair:
        name, parquet, *ids = spec.split(":")
        rows = pq.read_table(parquet, columns=COLUMNS).to_pylist()
        if ids and ids[0]:
            wanted = {json.loads(l)["id"] for l in Path(ids[0]).read_text().splitlines() if l.strip()}
            rows = [r for r in rows if r["id"] in wanted]
        rows.sort(key=lambda r: r["id"])
        if args.limit:
            rows = rows[: args.limit]
        work += [(name, i, r) for i, r in list(enumerate(rows))[args.shard :: args.num_shards]]
    print(f"{len(work)} utterances in shard {args.shard}", flush=True)
    out = Path(args.output).resolve()
    timing = {}

    # 1. Thinker: translate.
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    thinker = LLM(model=QWEN3, gpu_memory_utilization=args.thinker_memory, max_model_len=8192,
                  limit_mm_per_prompt={"audio": 1}, seed=args.seed)
    prompts = []
    for name, i, row in work:
        wav, sr = sf.read(row["source_audio_path"], dtype="float32")
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": row["source_audio_path"]},
            {"type": "text", "text": instruction[row["tgt_lang"]]}]}]
        text = processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=False)[0]
        prompts.append({"prompt": text, "multi_modal_data": {"audio": (wav, sr)}})
    t0 = time.time()
    said = thinker.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[THINKER_EOS]),
                            use_tqdm=False)
    timing["thinker"] = time.time() - t0
    reply = [[t for t in s.outputs[0].token_ids if t != THINKER_EOS] for s in said]

    # 2. UniSS's front end on the same waveform.
    import torchaudio
    torchaudio.load = read_with_soundfile
    device = torch.device("cuda:0")
    front = UniSSTokenizer.from_pretrained(str(REPO / "pretrained_models/UniSS"), device=device)
    t0 = time.time()
    tokens = [front.tokenize(row["source_audio_path"]) for _, _, row in work]
    timing["front_end"] = time.time() - t0

    # 3. UniSS speaks the Thinker's token ids.
    talker = LLM(model=UNISS, gpu_memory_utilization=args.talker_memory, max_model_len=4096, seed=args.seed,
                 logits_processors=[StopGuard])
    vocab = talker.get_tokenizer()
    requests, params = [], []
    for (name, i, row), ids, (glm, bic) in zip(work, reply, tokens):
        glm = [int(x) for x in (glm.flatten().tolist() if hasattr(glm, "flatten") else glm)]
        bic = [int(x) for x in (bic.flatten().tolist() if hasattr(bic, "flatten") else bic)]
        head = vocab(process_input(glm, bic[:32], "Performance", LANG[row["tgt_lang"]], speed=1.0),
                     add_special_tokens=False)["input_ids"]
        requests.append(TokensPrompt(prompt_token_ids=head + ids + [END_CONTENT]))
        params.append(SamplingParams(
            temperature=0.7, top_p=0.8, repetition_penalty=1.3, max_tokens=1500,
            seed=args.seed + i + (0 if name == "cmn_eng" else 500_000), skip_special_tokens=False,
            extra_args={"stop_guard": {"expected": max(1, round(50 * row["source_audio_duration_seconds"]))}}))
    t0 = time.time()
    spoken = talker.generate(requests, params, use_tqdm=False)
    timing["talker"] = time.time() - t0

    # 4. BiCodec, in the source speaker's voice.
    t0 = time.time()
    files = {}
    for (name, i, row), ids, (glm, bic), res in zip(work, reply, tokens, spoken):
        bic = [int(x) for x in (bic.flatten().tolist() if hasattr(bic, "flatten") else bic)]
        out_ids = list(res.outputs[0].token_ids)
        codes = [t - SEMANTIC_0 for t in out_ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
        (out / name / "wav").mkdir(parents=True, exist_ok=True)
        path = None
        if codes:
            path = str(out / name / "wav" / f"{i:05d}_{row['src_lang']}2{row['tgt_lang']}.wav")
            front.save_audio(front.decode(torch.tensor([*bic[:32], *codes], dtype=torch.long, device=device)),
                             path, sample_rate=16_000)
        translation = vocab.decode(ids, skip_special_tokens=True).strip()
        files.setdefault(name, []).append({
            "index": i, "name": f"{i:05d}_{row['src_lang']}2{row['tgt_lang']}", "id": row["id"], "mode": "tts",
            "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"], "dataset_name": "CVSS-T",
            "transcription_ref": row["transcription"], "translation_ref": row["translation"],
            "generated_text_raw": translation, "generated_text_clean": translation,
            "translated_text": translation, "reference_text": row["translation"],
            "source_audio_path": row["source_audio_path"], "reference_audio_path": row["reference_audio_path"],
            "source_audio_duration_seconds": row["source_audio_duration_seconds"],
            "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
            "synthetic_source": bool(row["synthetic_source"]), "synthetic_reference": bool(row["synthetic_reference"]),
            "codes": len(codes), "gold_codes": len(row["target_bicodec"]), "audio_path": path,
            "error": None if path else "no audio", "system": "e1_qwen3_thinker_uniss_talker",
            "ended": END_SEMANTIC in out_ids, "reply_token_ids": ids,
            "seed": args.seed + i + (0 if name == "cmn_eng" else 500_000)})
    timing["decode"] = time.time() - t0
    for name, recs in files.items():
        with open(out / name / f"part_{args.shard}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    n = len(work)
    print("timing " + "  ".join(f"{k} {v:.1f}s" for k, v in timing.items())
          + f"  total {sum(timing.values()):.1f}s for {n} ({n / sum(timing.values()):.2f}/s)"
          + f"  not ended {sum(not r['ended'] for v in files.values() for r in v)}", flush=True)


if __name__ == "__main__":
    main()
