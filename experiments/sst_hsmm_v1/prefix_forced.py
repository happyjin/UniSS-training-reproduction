"""sst_hsmm_v1 iteration 2 (prereg change 22): prefix-forced full-audio continuation (= the system's final read with a given
committed prefix). Each event {"key", "id", "dir", "audio", "prefix"} -> prompt = e2e/combo head + prefix (the same string the
streaming system feeds at its final read), WHOLE audio, greedy, max 256 tokens, stop <|im_end|>;
output {"key", "translation"} with translation = prefix + sep + continuation (combo_si_thinker's joining rule).
Used (1) report-only on the test set with N's committed prefixes (diagnostic), (2) on open training mixtures to score
counterfactual commit regret (train data only). Shard k of n by line index.
    <vllm python> -m experiments.sst_hsmm_v1.prefix_forced --events <jsonl> --thinker <merged> --shard k --num-shards n --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    rows = [json.loads(l) for i, l in enumerate(open(a.events, encoding="utf-8")) if l.strip() and i % a.num_shards == a.shard]
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    head = {}
    for d, tgt in (("en2zh", "cmn"), ("zh2en", "eng")):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": inference_instruction()[tgt]}]}]
        head[d] = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=a.thinker, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=20261125, tensor_parallel_size=1)
    cache, prompts = {}, []
    for r in rows:
        if r["audio"] not in cache:
            x, sr = sf.read(str(REPO / r["audio"]) if not r["audio"].startswith("/") else r["audio"], dtype="float32", always_2d=True)
            assert sr == 16000, r["audio"]
            cache = {r["audio"]: x.mean(axis=1)}
        prompts.append({"prompt": head[r["dir"]] + r["prefix"], "multi_modal_data": {"audio": (cache[r["audio"]], 16000)}})
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[151645]), use_tqdm=False)
    with open(a.out + ".tmp", "x", encoding="utf-8") as fh:
        for r, o in zip(rows, res):
            cont, p, zh = o.outputs[0].text, r["prefix"], r["dir"] == "en2zh"
            sep = "" if zh or not p or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
            fh.write(json.dumps({"key": r["key"], "translation": (p + sep + cont).strip()}, ensure_ascii=False) + "\n")
    Path(a.out + ".tmp").rename(a.out)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
