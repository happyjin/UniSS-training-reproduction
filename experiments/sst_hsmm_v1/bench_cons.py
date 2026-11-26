"""sst_hsmm_v1 prereg change 38-0: real-time gate for the K-sample consistency evidence. One live stream at a time (items processed
sequentially, reads in order), per read the wall time of llm.generate for the current request set (raw greedy draft with the committed
prefix, enhanced-view greedy draft, SAP-verifier greedy draft) = pass A, and the same plus one wave17 request with n = K sampled short
continuations = pass B. Commits follow min(LA2, cross-view LCP) so prompts grow realistically; identical in both passes. Prefix caching
off in both passes (conservative, no cross-pass cache advantage). Writes per-read times and a summary JSON.
    <vllm python> -m experiments.sst_hsmm_v1.bench_cons --direction en2zh --n-items 40 --k 6 --max-tokens 24 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, join, units  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave15_si_thinker_cvla import lcp, load_views  # noqa: E402

D = REPO / "eval_outputs/sst_hsmm_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True); ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    zh = a.direction.endswith("zh")
    rows = [json.loads(l) for l in open(D / f"edev/data/items_{a.direction}.jsonl", encoding="utf-8")]
    rows = rows[:: max(1, len(rows) // a.n_items)][: a.n_items]
    items = []
    for r in rows:
        x, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        aud = x.mean(axis=1)[: int(r["dur_ms"]) * 16]
        items.append({"id": r["id"], "wav": r["audio"], "audio": aud, "dur_ms": len(aud) // 16})
    load_views(D / f"edev/views/{a.direction}", items)
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": inference_instruction()["cmn" if zh else "eng"]}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=str(REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"), gpu_memory_utilization=0.6, max_model_len=8192,
              limit_mm_per_prompt={"audio": 1}, seed=20261126, tensor_parallel_size=1, enable_prefix_caching=False,
              enable_lora=True, max_loras=1, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"],
              hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]})
    g = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=1)
    smp = SamplingParams(n=a.k, temperature=a.temp, top_p=0.95, max_tokens=a.max_tokens, stop_token_ids=[THINKER_EOS], seed=7)
    vreq = LoRARequest("sap_verifier", 1, str((D / "v/lora/a1.0000").resolve()))
    res = {"A": [], "B": []}
    for it in items[:2]:   # warm-up (not timed)
        t = min(1000, it["dur_ms"])
        p = {"prompt": head, "multi_modal_data": {"audio": (it["audio"][: t * 16], 16000)}}
        llm.generate([p, p, p, p], [g, g, g, smp], use_tqdm=False, lora_request=[None, None, vreq, None])
    for passname in ("A", "B"):
        for it in items:
            committed, prev, r = [], None, 0
            while True:
                t = min((r + 1) * 1000, it["dur_ms"]); final = t >= it["dur_ms"]
                p0 = {"prompt": head + join(committed, zh), "multi_modal_data": {"audio": (it["audio"][: t * 16], 16000)}}
                if final:
                    ps, sp, lr = [p0], [g], [None]
                else:
                    p1 = {"prompt": p0["prompt"], "multi_modal_data": {"audio": (it["v1"][t], 16000)}}
                    ps, sp, lr = [p0, p1, p0], [g, g, g], [None, None, vreq]
                    if passname == "B":
                        ps, sp, lr = ps + [p0], sp + [smp], lr + [None]
                t0 = time.perf_counter()
                o = llm.generate(ps, sp, use_tqdm=False, lora_request=lr)
                dt = time.perf_counter() - t0
                res[passname].append({"id": it["id"], "t": t, "final": final, "sec": dt, "n_committed": len(committed)})
                cont = o[0].outputs[0].text
                sep = "" if zh or not committed or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
                draft = units(join(committed, zh) + sep + cont, zh)
                if final:
                    break
                c1 = o[1].outputs[0].text
                sep1 = "" if zh or not committed or c1[:1].isspace() or c1[:1] in ",.;:!?)" else " "
                d1 = units(join(committed, zh) + sep1 + c1, zh)
                agreed = lcp(draft, prev) if prev is not None else 0
                cut = min(agreed, lcp(draft, d1))
                if cut > len(committed):
                    committed = committed + draft[len(committed): cut]
                prev = draft; r += 1
    out.mkdir(parents=True)
    with open(out / f"reads_{a.direction}.jsonl", "x") as fh:
        for k, v in res.items():
            for x in v:
                fh.write(json.dumps({"pass": k, **x}) + "\n")
    summ = {}
    for k, v in res.items():
        nf = np.array([x["sec"] for x in v if not x["final"]])
        summ[k] = {"nonfinal_reads": len(nf), "median_ms": float(np.median(nf) * 1000), "p95_ms": float(np.percentile(nf, 95) * 1000),
                   "mean_ms": float(nf.mean() * 1000)}
    a_ = {(x["id"], x["t"]): x["sec"] for x in res["A"] if not x["final"]}
    pairs = [(a_[(x["id"], x["t"])], x["sec"]) for x in res["B"] if not x["final"] and (x["id"], x["t"]) in a_]
    dd = np.array([b - a for a, b in pairs])
    summ["extra"] = {"pairs": len(pairs), "median_ms": float(np.median(dd) * 1000), "p95_ms": float(np.percentile(dd, 95) * 1000)}
    summ["config"] = {"k": a.k, "max_tokens": a.max_tokens, "temp": a.temp, "items": len(items), "prefix_caching": False}
    (out / f"summary_{a.direction}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
