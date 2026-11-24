"""sst_hsmm_v1 prereg change 20: offline (full-audio) ceiling of the N translator on the test set (report only).
gen:   wave17 merged_enc_wise05, the e2e_si_thinker prompt head with the WHOLE test audio (no streaming), greedy, max 256
       tokens, stop <|im_end|>; shard k of n of test_robust/data/items_<d>.jsonl by line index -> {"id", "translation"}.
score: per p3_score cell (same meta / EXCL / DROP), text BLEU of the offline translation vs N's streaming translation
       (test_robust/UE thinker blocks) with combo_score.text_compare (same BLEU / unit bootstrap as p3_score's text_bleu).
    <vllm python> -m experiments.sst_hsmm_v1.offline_ceiling gen --direction d --shard k --num-shards n --out <jsonl>
    <qwen3omni python> -m experiments.sst_hsmm_v1.offline_ceiling score --parts 'dir/offline_{d}_*.jsonl' --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
WAVE17 = str(REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05")
TDATA = REPO / "eval_outputs/mist_w3po_v1/test_robust/data"


def gen(a):
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    rows = [json.loads(l) for i, l in enumerate(open(TDATA / f"items_{a.direction}.jsonl", encoding="utf-8"))
            if l.strip() and i % a.num_shards == a.shard]
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    tgt = "cmn" if a.direction == "en2zh" else "eng"
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": inference_instruction()[tgt]}]}]
    head = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=WAVE17, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=20261124, tensor_parallel_size=1)
    prompts = []
    for r in rows:
        x, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, r["audio"]
        prompts.append({"prompt": head, "multi_modal_data": {"audio": (x.mean(axis=1), 16000)}})
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[151645]), use_tqdm=False)
    with open(a.out + ".tmp", "x", encoding="utf-8") as fh:
        for r, o in zip(rows, res):
            fh.write(json.dumps({"id": r["id"], "translation": o.outputs[0].text.strip()}, ensure_ascii=False) + "\n")
    Path(a.out + ".tmp").rename(a.out)
    print("done", len(rows), flush=True)


def score(a):
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    from experiments.sst_hsmm_v1 import p3_score as S
    from experiments.mist_w3po_v1.voicecond.combo_score import LANG, jl, text_compare
    res = {}
    for d in S.DIRS:
        meta = {m["id"]: m for m in jl(S.T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(S.T / "data" / f"items_{d}.jsonl")}
        off = {r["id"]: r["translation"] for p in sorted(glob.glob(a.parts.format(d=d))) for r in jl(p)}
        N = {r["id"]: r["translation"] for r in jl(S.sysd("N")["thinker"](d))}
        common = (set(off) & set(N)) - S.DROP
        for cell in S.CELLS:
            items = sorted(i for i in common if i in meta and f"{meta[i]['set']}/{meta[i]['cond']}" == cell
                           and not (meta[i]["set"] == "obsi" and meta[i]["pair"] in S.EXCL[d]))
            X = {i: {"text": off[i]} for i in items}
            Y = {i: {"text": N[i]} for i in items}
            res[f"{d}/{cell}"] = {"n": len(items), "offline_vs_N": text_compare(meta, ref, items, X, Y, LANG[d], "stream")}
            print(d, cell, res[f"{d}/{cell}"], flush=True)
    json.dump(res, open(a.out, "x"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "score"))
    ap.add_argument("--direction", choices=("en2zh", "zh2en"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--parts", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else score(a)


if __name__ == "__main__":
    main()
