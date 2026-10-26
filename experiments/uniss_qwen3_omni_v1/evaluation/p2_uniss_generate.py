"""P2 step zero, generation half: UniSS speaks Qwen3-Omni's translation, on vLLM.

p2_step0_uniss_reads.py did this in the uniss-eval environment, where vLLM
0.8.5's Triton cannot build kernels and the model had to run eager. This
half runs in the vllm_omni_030 venv (vLLM 0.30, CUDA graphs) and writes
BiCodec semantic codes only; p2_decode.py turns them into speech where the
BiCodec decoder lives. The prompt is built by UniSS's own process_input,
loaded from its file so the uniss package (and its torch-audio stack) is
not imported here.

--guard applies uniss_stop_guard.StopGuard, which keeps UniSS from running
on past the end of the sentence; one sample per sentence either way.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LANG = {"eng": "<|eng|>", "cmn": "<|cmn|>"}
SEMANTIC_0, SEMANTIC_LAST, END = 155761, 163952, 180392


def process_input():
    spec = importlib.util.spec_from_file_location("uniss_prompt", REPO / "uniss" / "cli" / "prompt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_input


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    help="name:parquet[:ids_jsonl]; without ids, every row of the parquet")
    ap.add_argument("--translations", required=True, help="a P1 run's p1a directory, holding <name>/results.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model", default=str(REPO / "pretrained_models/UniSS_full"))
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--guard", action="store_true")
    # configs/uniss.yaml's infer_param, whose note says lower temperature and a
    # higher repetition penalty may fix "weird output audio"
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--speed-token", type=int, default=None,
                    help="diagnostics only: put <|speed_N|> in the prompt instead of process_input's <|speed_9|>")
    ap.add_argument("--repeat", type=int, default=1, help="diagnostics only: samples per sentence, each its own seed")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams

    build = process_input()
    work = []
    for spec in args.pair:
        name, parquet, *ids = spec.split(":")
        said = {json.loads(l)["id"]: json.loads(l)["translated_text"]
                for l in open(Path(args.translations) / name / "results.jsonl", encoding="utf-8")}
        rows = pq.read_table(parquet, columns=["id", "tgt_lang", "source_glm", "bicodec_global",
                                               "source_audio_duration_seconds"]).to_pylist()
        if ids and ids[0]:
            wanted = {json.loads(l)["id"] for l in Path(ids[0]).read_text().splitlines() if l.strip()}
            rows = [r for r in rows if r["id"] in wanted]
        rows.sort(key=lambda r: r["id"])
        for index, row in list(enumerate(rows))[args.shard :: args.num_shards]:
            prompt = build(row["source_glm"], row["bicodec_global"], "Performance", LANG[row["tgt_lang"]], speed=1.0)
            if args.speed_token is not None:
                prompt = prompt.replace("<|speed_9|>", f"<|speed_{args.speed_token}|>")
            for r in range(args.repeat):
                work.append((name, index, r, row, said[row["id"]], prompt + said[row["id"]] + "<|end_content|>"))
    print(f"{len(work)} requests in shard {args.shard}", flush=True)

    kwargs = {}
    if args.guard:
        from experiments.uniss_qwen3_omni_v1.evaluation.uniss_stop_guard import StopGuard
        kwargs["logits_processors"] = [StopGuard]
    llm = LLM(model=args.model, tokenizer=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, seed=args.seed, **kwargs)
    params = []
    for name, index, r, row, _, _ in work:
        extra = {"stop_guard": {"expected": max(1, round(50 * row["source_audio_duration_seconds"]))}} if args.guard else None
        # the example's 1,500-token cap
        params.append(SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty, max_tokens=1500,
                                     seed=args.seed + index + 1000 * r + (0 if name == "cmn_eng" else 500_000),
                                     skip_special_tokens=False, extra_args=extra))
    t0 = time.time()
    results = llm.generate([w[5] for w in work], params, use_tqdm=False)
    gen_s = time.time() - t0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"codes_{args.shard}.jsonl", "w", encoding="utf-8") as fh:
        for (name, index, r, row, said_text, _), res in zip(work, results):
            ids = list(res.outputs[0].token_ids)
            codes = [t - SEMANTIC_0 for t in ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
            fh.write(json.dumps({
                "name": name, "index": index, "repeat": r, "id": row["id"], "translation": said_text,
                "codes": codes, "ended": END in ids, "finish_reason": res.outputs[0].finish_reason,
                "expected": round(50 * row["source_audio_duration_seconds"]), "guard": args.guard,
                "gen_seconds": round(gen_s / len(work), 4)}, ensure_ascii=False) + "\n")
    print(f"generate {gen_s:.1f}s for {len(work)} ({len(work) / gen_s:.1f}/s)  "
          f"not ended {sum(1 for res in results if END not in res.outputs[0].token_ids)}", flush=True)


if __name__ == "__main__":
    main()
