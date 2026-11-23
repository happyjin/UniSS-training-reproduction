"""sst_hsmm_v1 streaming Thinker (reports/sst_hsmm_v1/prereg.zh-CN.md 3.3): the trained (merged) Thinker writes per 1 s read
what the HSMM lets it write. Read r: prompt = wave5 / si_thinker_stream head with the audio heard so far + the text
committed so far; greedy decoding with the HSMMGate logits processor: the wait token's logit gets -g_psi(s, f, started,
c/C) and the request writes the next token iff P(write) = 1 - P'(wait) >= 0.5 (else the wait token, a stop token, ends
the read); s / f / started follow hsmm's dynamics with the mean speech duration per non-punctuation token (p0 summary)
for every written token and s <- max(s - 1, -3) between reads. The last read (audio = the whole source) may not wait and
generates to the end of the turn. Output <out>/blocks_<dir>_w1.jsonl in si_infer's --text-policy file format.
SST_RULE=survival (prereg change 6): the pending token is written when S * (1 - w) <= 0.5, S = its survival (product of
1 - w over the reads it has already waited, reset to 1 after every written token); the read-final wait's 1 - w is
recomputed in the main process from vLLM's raw logprob of the wait token (raw = before logits processors). threshold
(default) = the original rule (S = 1).
    SST_GATE=<gate_final.pt> SST_TOKMS=<ms> <vllm_omni_030 python> -m experiments.sst_hsmm_v1.e2e_si_thinker \
        --thinker <merged dir> --direction en2zh --items-file <jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
WAIT, THINKER_EOS = 151662, 151645
PUNCT = re.compile(r"^[\s，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]+$")
_G = {}
RULE = os.environ.get("SST_RULE", "threshold")
assert RULE in ("threshold", "survival"), RULE


def _gate():
    if "W" not in _G:
        st = torch.load(os.environ["SST_GATE"], map_location="cpu")
        _G["W"] = [st[k].double().numpy() for k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias")]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN3)
        _G["punct"] = {i for i in range(len(tok)) if PUNCT.match(tok.decode([i]) or "x")}
        _G["tok_s"] = float(os.environ["SST_TOKMS"]) / 1000.0
    return _G


def gate_value(s, f, st, cf):
    W1, b1, W2, b2 = _gate()["W"]
    x = np.array([s / 8.0, f, st, cf])
    return float(np.tanh(W1 @ x + b1) @ W2.T + b2)


def advance(s, f, st, toks):
    G = _gate()
    for t in toks:
        p = t in G["punct"]
        s = max(s, 0.0) + (0.0 if p else G["tok_s"])
        s = min(s, 8.0)
        f, st = int(p), 1
    return s, f, st


def p_wait_gated(lp_wait_raw, g):
    """P'(wait) after the gate shift: p e^-g / (1 - p + p e^-g), p = raw P_wait (log-space safe)"""
    a = lp_wait_raw - g
    b = math.log1p(-min(math.exp(lp_wait_raw), 1 - 1e-12))
    m = max(a, b)
    return math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))


def make_hsmm(s0, f0, started, cfrac, final, surv0=1.0):
    def proc(output_ids, logits):
        if final:
            logits[WAIT] = -math.inf
            return logits
        s, f, st = advance(s0, f0, started, output_ids)
        logits[WAIT] = logits[WAIT] - gate_value(s, f, st, cfrac)
        pw = torch.softmax(logits.float(), dim=-1)[WAIT]
        surv = surv0 if len(output_ids) == 0 else 1.0        # survival of the pending token (1 after a write)
        if surv * pw >= 0.5:
            w = logits[WAIT].clone()
            logits[:] = -math.inf
            logits[WAIT] = w
        else:
            logits[WAIT] = -math.inf
        return logits
    return proc


class HSMMGate(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        a = (params.extra_args or {}).get("hsmm")
        return make_hsmm(**a) if a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--items-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    out = Path(a.out) / f"blocks_{a.direction}_w1.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    zh = a.direction.endswith("zh")
    tgt = "cmn" if zh else "eng"
    items = []
    for r in map(json.loads, open(a.items_file, encoding="utf-8")):
        x, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, r["audio"]
        x = x.mean(axis=1)[: int(r["dur_ms"]) * 16]
        items.append({"id": r["id"], "audio": x, "dur_ms": len(x) // 16, "committed": "", "blocks": [], "done": False,
                      "s": 0.0, "f": 0, "st": 0, "surv": 1.0, "reads": []})
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": inference_instruction()[tgt]}]}]
    head = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=a.thinker, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=a.seed, tensor_parallel_size=1,
              logits_processors=["experiments.sst_hsmm_v1.e2e_si_thinker:HSMMGate"])   # FQCN: safe under spawn too
    t0, r = time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"]]
        prompts, params = [], []
        for it in live:
            t = min((r + 1) * 1000, it["dur_ms"])
            final = t >= it["dur_ms"]
            C = math.ceil(it["dur_ms"] / 1000)
            it["t"], it["final"] = t, final
            prompts.append({"prompt": head + it["committed"], "multi_modal_data": {"audio": (it["audio"][: t * 16], 16000)}})
            it["cfrac"] = (r + 1) / C
            params.append(SamplingParams(temperature=0.0, max_tokens=160 if final else 64,
                                         stop_token_ids=[THINKER_EOS] if final else [WAIT, THINKER_EOS],
                                         logprobs=0 if (RULE == "survival" and not final) else None,
                                         extra_args={"hsmm": {"s0": it["s"], "f0": it["f"], "started": it["st"],
                                                              "cfrac": (r + 1) / C, "final": final,
                                                              "surv0": it["surv"] if RULE == "survival" else 1.0}}))
        res = llm.generate(prompts, params, use_tqdm=False)
        for it, rr in zip(live, res):
            o = rr.outputs[0]
            toks = [x for x in o.token_ids if x not in (WAIT, THINKER_EOS)]
            text = o.text
            it["s"], it["f"], it["st"] = advance(it["s"], it["f"], it["st"], toks)
            if RULE == "survival" and not it["final"]:
                ids = list(o.token_ids)
                if ids and ids[-1] == WAIT and o.logprobs:          # the read ended on a wait decision
                    lpw = o.logprobs[-1][WAIT].logprob                # raw (pre-processor) log P_wait
                    pw = p_wait_gated(lpw, gate_value(it["s"], it["f"], it["st"], it["cfrac"]))
                    it["surv"] = (1.0 if toks else it["surv"]) * pw
                else:
                    it["surv"] = 1.0
            it["reads"].append({"t": it["t"], "n_tok": len(toks), **({"surv": round(it["surv"], 4)} if RULE == "survival" else {})})
            if text or it["final"]:
                it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["final"]})
                it["committed"] += text
            if it["final"]:
                it["done"] = True
            else:
                it["s"] = max(it["s"] - 1.0, -3.0)
        r += 1
        print(f"{a.direction} hsmm round {r}: {len(live)} live, {time.time() - t0:.0f}s", flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            fh.write(json.dumps({"id": it["id"], "translation": it["committed"], "blocks": blocks, "reads": it["reads"]},
                                ensure_ascii=False) + "\n")
    print(out, len(items), flush=True)


if __name__ == "__main__":
    main()
