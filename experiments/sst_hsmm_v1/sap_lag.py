"""sst_hsmm_v1 prereg change 37-2: SAP support-lag precheck (GPU part writes per-item records; --summary is CPU).

For every item and every read c = 1..C (t = min(c * 1000, dur_ms), C = ceil(dur_ms / 1000)) a fresh greedy draft from audio[:t]
with no committed prefix, twice: wave17 + SAP LoRA (the verifier) and wave17 alone (reference). Truth: the reference row's
per-token support read g_i (data.py), k(c) = #{i : g_i <= c}. Coverage curves: truth k(c) / n, SAP n_sap(c) / n_sap(C),
wave17 n_w(c) / n_w(C) (token counts of the drafts, EOS excluded). Lag L = sum_{c < C} (truth - draft coverage) in reads
(= mean horizontal distance between the two monotone curves); L+ / L- = the positive / negative parts.
Item -> row: '<cond>__<row id>' (UniST held-out items), else the p4 sample (s, j) of id 's%04d_j%02d' (SAP training items).
    <vllm python> -m experiments.sst_hsmm_v1.sap_lag --items <items.jsonl> --rows <rows.jsonl> --direction en2zh --out <x.jsonl>
    <qwen3omni python> -m experiments.sst_hsmm_v1.sap_lag --summary <dir with *.jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SAP = REPO / "eval_outputs/sst_hsmm_v1/v/lora/a1.0000"
W17 = REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"
SPONT = ("EN_B", "emilia")


def row_of(item_id, samples):
    if "__" in item_id:
        return item_id.split("__", 1)[1]
    s, j = item_id.split("_")
    return samples[(int(s[1:]), int(j[1:]))]


def gen(a):
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    samples = {(x["s"], x["j"]): x["row"] for x in map(json.loads, open(REPO / "eval_outputs/sst_hsmm_v1/p4/samples.jsonl"))}
    rows = {}
    for f in a.rows.split(","):
        rows.update({r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
    items = []
    for r in map(json.loads, open(a.items, encoding="utf-8")):
        rid = row_of(r["id"], samples)
        if rid not in rows or (a.spont and not rid.startswith(SPONT)):
            continue
        items.append({"id": r["id"], "row": rid, "wav": r["audio"], "dur_ms": int(r["dur_ms"])})
        if a.limit and len(items) >= a.limit:
            break
    tgt = "cmn" if a.direction.endswith("zh") else "eng"
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": inference_instruction()[tgt]}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=str(W17), gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1}, seed=20261126,
              tensor_parallel_size=1, enable_lora=True, max_loras=1, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"],
              hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]})
    sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS])
    vreq = LoRARequest("sap_verifier", 1, str(SAP.resolve()))
    jobs = []
    for it in items:
        x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
        assert sr == 16000, it["wav"]
        aud = x.mean(axis=1)[: it["dur_ms"] * 16]
        C = math.ceil(it["dur_ms"] / 1000)
        for c in range(1, C + 1):
            t = min(c * 1000, it["dur_ms"])
            jobs.append((it["id"], c, {"prompt": head, "multi_modal_data": {"audio": (aud[: t * 16], 16000)}}))
    res = {}
    B = 256
    for b0 in range(0, len(jobs), B):
        chunk = jobs[b0: b0 + B]
        ps = [j[2] for j in chunk]
        outs = llm.generate(ps + ps, sp, use_tqdm=False, lora_request=[vreq] * len(ps) + [None] * len(ps))
        for k, (iid, c, _) in enumerate(chunk):
            so, wo = outs[k].outputs[0], outs[len(ps) + k].outputs[0]
            res[(iid, c)] = ([t for t in so.token_ids if t != THINKER_EOS], [t for t in wo.token_ids if t != THINKER_EOS], so.text, wo.text)
        print(f"{min(b0 + B, len(jobs))}/{len(jobs)}", flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            C = math.ceil(it["dur_ms"] / 1000)
            r = rows[it["row"]]
            fh.write(json.dumps({"id": it["id"], "row": it["row"], "C": C, "g": r["g"], "n_ref": len(r["tok"]),
                                 "n_sap": [len(res[(it["id"], c)][0]) for c in range(1, C + 1)],
                                 "n_w17": [len(res[(it["id"], c)][1]) for c in range(1, C + 1)],
                                 "sap_text": [res[(it["id"], c)][2] for c in range(1, C + 1)],
                                 "w17_text": [res[(it["id"], c)][3] for c in range(1, C + 1)]}, ensure_ascii=False) + "\n")


def lag(r, key):
    C, n = r["C"], max(r["n_ref"], 1)
    fin = max(r[key][-1], 1)
    d = [sum(g <= c for g in r["g"]) / n - min(r[key][c - 1] / fin, 1.0) for c in range(1, C)]
    return sum(d), sum(x for x in d if x > 0), sum(x for x in d if x < 0)


def summary(a):
    print("file                         items   SAP L / L+ / L-          wave17 L / L+ / L-")
    for f in sorted(glob.glob(f"{a.summary}/*.jsonl")):
        R = [json.loads(l) for l in open(f, encoding="utf-8")]
        R = [r for r in R if r["C"] >= 2]
        for cond in ("all", "clean", "noisy"):
            S = [r for r in R if cond == "all" or r["id"].startswith(cond + "__")]
            if not S:
                continue
            s = np.array([lag(r, "n_sap") for r in S]); w = np.array([lag(r, "n_w17") for r in S])
            print(f"{Path(f).stem:24s} {cond:5s} {len(S):5d}   {s[:,0].mean():+.3f} / {s[:,1].mean():.3f} / {s[:,2].mean():+.3f}"
                  f"     {w[:,0].mean():+.3f} / {w[:,1].mean():.3f} / {w[:,2].mean():+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items"); ap.add_argument("--rows"); ap.add_argument("--direction"); ap.add_argument("--out")
    ap.add_argument("--spont", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.6); ap.add_argument("--summary")
    a = ap.parse_args()
    summary(a) if a.summary else gen(a)


if __name__ == "__main__":
    main()
