"""P2 at step zero: the official UniSS speaks Qwen3-Omni's translation, untrained.

P2's talker is the UniSS language model reading the Thinker's translation
tokens. With its gate closed and before any training, that is UniSS in
Performance mode with the translation slot already filled by Qwen3's text:
the two tokenizers give identical ids on all 9,794 CVSS-T translations, so
filling the slot with text is filling it with the Thinker's own tokens.
This measures that starting point -- one sample per sentence, UniSS's own
sampling settings, BiCodec decoded with the source speaker's 32 global
tokens -- on the sentences P1 and the cascade were scored on.

The translations are read from a finished P1 run (they are phase_h's, word
for word), and the source GLM-4 and BiCodec global tokens from the CVSS-T
parquet, so nothing upstream of UniSS is recomputed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

LANG = {"eng": "<|eng|>", "cmn": "<|cmn|>"}
SEMANTIC = re.compile(r"<\|bicodec_semantic_(\d+)\|>")


def record(row, index, translation, audio_path, codes, extra):
    """p1_native_talker.record's shape, so the metric harness reads it unchanged."""
    return {
        "index": index, "name": f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}", "id": row["id"],
        "mode": "tts", "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"], "dataset_name": "CVSS-T",
        "transcription_ref": row["transcription"], "translation_ref": row["translation"],
        "generated_text_raw": translation, "generated_text_clean": translation,
        "translated_text": translation, "reference_text": row["translation"],
        "source_audio_path": row["source_audio_path"], "reference_audio_path": row["reference_audio_path"],
        "source_audio_duration_seconds": row["source_audio_duration_seconds"],
        "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
        "synthetic_source": bool(row["synthetic_source"]), "synthetic_reference": bool(row["synthetic_reference"]),
        "codes": codes, "gold_codes": len(row["target_bicodec"]),
        "audio_path": audio_path, "error": None if audio_path else "no audio", **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    help="name:parquet:ids_jsonl -- rows whose id is listed are spoken")
    ap.add_argument("--translations", required=True, help="a P1 run's p1a directory, holding <name>/results.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="pretrained_models/UniSS_full")
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--limit", type=int, default=0, help="per direction, for smoke runs")
    ap.add_argument("--show", type=int, default=0, help="print this many raw continuations")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--eager", action="store_true",
                    help="no CUDA graphs and no torch.compile: the uniss-eval env's Triton cannot build kernels")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import torch
    from vllm import LLM, SamplingParams
    from uniss import UniSSTokenizer
    from uniss.cli.prompt import process_input

    out = Path(args.output).resolve()
    work = []
    for spec in args.pair:
        name, parquet, ids_path = spec.split(":")
        wanted = [json.loads(l)["id"] for l in Path(ids_path).read_text().splitlines() if l.strip()]
        said = {json.loads(l)["id"]: json.loads(l)["translated_text"]
                for l in open(Path(args.translations) / name / "results.jsonl", encoding="utf-8")}
        rows = {r["id"]: r for r in pq.read_table(parquet).to_pylist() if r["id"] in set(wanted)}
        order = sorted(rows)
        if args.limit:
            order = order[: args.limit]
        (out / name / "wav").mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(order):
            row = rows[k]
            prompt = process_input(row["source_glm"], row["bicodec_global"], "Performance", LANG[row["tgt_lang"]], speed=1.0)
            work.append((name, i, row, said[k], prompt + said[k] + "<|end_content|>"))
    print(f"{len(work)} utterances", flush=True)

    llm = LLM(model=args.model, tokenizer=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              seed=args.seed, max_model_len=4096,
              **({"enforce_eager": True, "compilation_config": 0} if args.eager else {}))
    tok = llm.get_tokenizer()
    tok.chat_template = None
    # configs/uniss.yaml's infer_param, and the example's 1,500-token cap (about 30 s)
    params = [SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.1, max_tokens=1500,
                             seed=args.seed + i, skip_special_tokens=False, spaces_between_special_tokens=False)
              for i in range(len(work))]
    t0 = time.time()
    results = llm.generate([w[4] for w in work], params)
    gen_s = time.time() - t0

    device = torch.device("cuda:0")
    vocoder = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)
    shown, extra_text = 0, 0
    files = {}
    for (name, i, row, translation, _), res in zip(work, results):
        pieces = tok.convert_ids_to_tokens(list(res.outputs[0].token_ids))
        semantic = [int(m.group(1)) for p in pieces if (m := SEMANTIC.fullmatch(p))]
        first = next((j for j, p in enumerate(pieces) if SEMANTIC.fullmatch(p)), len(pieces))
        before = "".join(pieces[:first])
        if re.sub(r"<\|[^|]*\|>", "", before).strip():
            extra_text += 1  # the model wrote words of its own before speaking
        if shown < args.show:
            print(f"--- {name} {row['id']}\n  译文: {translation}\n  续写开头: {''.join(pieces[:first + 6])[:400]}\n"
                  f"  语义 token {len(semantic)} 个，结束于: {''.join(pieces[-3:])}", flush=True)
            shown += 1
        path = None
        if semantic:
            path = str(out / name / "wav" / f"{i:05d}_{row['src_lang']}2{row['tgt_lang']}.wav")
            tokens = torch.tensor([*row["bicodec_global"], *semantic], dtype=torch.long, device=device)
            vocoder.save_audio(vocoder.decode(tokens), path, sample_rate=16_000)
        rec = record(row, i, translation, path, len(semantic), {
            "system": "p2_step0_uniss_reads_qwen3", "model": args.model, "seed": args.seed + i,
            "finish_reason": res.outputs[0].finish_reason, "text_before_speech": before[:200],
            "gen_seconds": round(gen_s / len(work), 4)})
        files.setdefault(name, []).append(rec)
    for name, recs in files.items():
        with open(out / name / "results.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(recs)} 句，无音频 {sum(not r['audio_path'] for r in recs)}，"
              f"截断 {sum(r['finish_reason'] == 'length' for r in recs)}", flush=True)
    print(f"generate {gen_s:.0f}s for {len(work)} ({len(work) / gen_s:.1f}/s)  先写了自己的文字再说的 {extra_text} 句", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
