"""sst_hsmm_v1 streaming Thinker (reports/sst_hsmm_v1/prereg.zh-CN.md 3.3): the trained (merged) Thinker writes per 1 s read
what the HSMM lets it write. Read r: prompt = wave5 / si_thinker_stream head with the audio heard so far + the text
committed so far; greedy decoding with the HSMMGate logits processor: the wait token's logit gets -g_psi(s, f, started,
c/C) and the request writes the next token iff P(write) = 1 - P'(wait) >= 0.5 (else the wait token, a stop token, ends
the read); SST_BLOCK_S (prereg change 8) adds the Talker per-block overhead to the state of a read's first token; s / f / started follow hsmm's dynamics with the mean speech duration per non-punctuation token (p0 summary)
for every written token and s <- max(s - 1, -3) between reads. The last read (audio = the whole source) may not wait and
generates to the end of the turn. Output <out>/blocks_<dir>_w1.jsonl in si_infer's --text-policy file format.
SST_RULE=survival (prereg change 6): the pending token is written when S * (1 - w) <= 0.5, S = its survival (product of
1 - w over the reads it has already waited, reset to 1 after every written token); the read-final wait's 1 - w is
recomputed in the main process from vLLM's raw logprob of the wait token (raw = before logits processors). threshold
(default) = the original rule (S = 1).
SST_CTRL=<control.json> (prereg change 11, P6; change 13: "lag": "speech" -> playback-queue lag, updates only after t > L*): Gate2 (multiplier-conditioned, no c/C) + per-item online primal-dual
control of lam_L: after every non-final read, lag = t - W / rho (W = non-punctuation tokens written so far, rho = the
direction's tokens per source second), log lam_L <- clip(log lam_L + eta * (lag - L*), log LAM_LO, log LAM_HI); lam_P fixed.
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
CTRL = json.load(open(os.environ["SST_CTRL"])) if os.environ.get("SST_CTRL") else None
GUARD = os.environ.get("SST_GUARD", "") == "1"   # change 14 (P7): HSMM proposes, local agreement verifies, pause-imminent override
PAUSE_S = 0.7                                     # change 14: s <= 0.7 now -> s - 1 <= -0.3 at the next read = a pause (lattice rule)
TAU = float(os.environ.get("SST_TAU", "0.5"))     # write iff (survival-weighted) P(write) > TAU; 0.5 = prereg rule (post-hoc curve A only)


def _gate():
    if "W" not in _G:
        st = torch.load(os.environ["SST_GATE"], map_location="cpu")
        _G["W"] = [st[k].double().numpy() for k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias")]
        _G["skip"] = st["skip.weight"].double().numpy().reshape(-1) if "skip.weight" in st else None
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN3)
        _G["punct"] = {i for i in range(len(tok)) if PUNCT.match(tok.decode([i]) or "x")}
        _G["tok_s"] = float(os.environ["SST_TOKMS"]) / 1000.0
    return _G


def gate_value(s, f, st, cf, zL=0.0, zP=0.0):
    G = _gate()
    W1, b1, W2, b2 = G["W"]
    if G["skip"] is not None:                                  # Gate2 (change 11): (s, f, started, z_L, z_P), no c/C
        x = np.array([s / 8.0, f, st, zL, zP])
        return float(np.tanh(W1 @ x + b1) @ W2.T + b2 + G["skip"] @ x)
    x = np.array([s / 8.0, f, st, cf])
    return float(np.tanh(W1 @ x + b1) @ W2.T + b2)


BLOCK_S = float(os.environ.get("SST_BLOCK_S", "0.0"))   # prereg change 8 (P5): Talker per-block overhead; 0 = unchanged


def advance(s, f, st, toks):
    """toks = the tokens written so far in THIS read (the first one starts a block: speech at max(s, BLOCK_S))"""
    G = _gate()
    for k, t in enumerate(toks):
        p = t in G["punct"]
        s = max(s, BLOCK_S if k == 0 else 0.0) + (0.0 if p else G["tok_s"])
        s = min(s, 8.0)
        f, st = int(p), 1
    return s, f, st


def lam_args(lgL):
    """change 11: z features of the item's current lam_L and the fixed lam_P"""
    def z(lg):
        lo, hi = math.log(CTRL["lam_lo"]), math.log(CTRL["lam_hi"])
        return (lg - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    return {"zL": z(lgL), "zP": z(math.log(CTRL["lamP"]))}


def p_wait_gated(lp_wait_raw, g):
    """P'(wait) after the gate shift: p e^-g / (1 - p + p e^-g), p = raw P_wait (log-space safe)"""
    a = lp_wait_raw - g
    b = math.log1p(-min(math.exp(lp_wait_raw), 1 - 1e-12))
    m = max(a, b)
    return math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))


def make_hsmm(s0, f0, started, cfrac, final, surv0=1.0, zL=0.0, zP=0.0):
    def proc(output_ids, logits):
        if final:
            logits[WAIT] = -math.inf
            return logits
        s, f, st = advance(s0, f0, started, output_ids)
        logits[WAIT] = logits[WAIT] - gate_value(s, f, st, cfrac, zL, zP)
        pw = torch.softmax(logits.float(), dim=-1)[WAIT]
        surv = surv0 if len(output_ids) == 0 else 1.0        # survival of the pending token (1 after a write)
        if surv * pw >= 1.0 - TAU:
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
    zh_tgt = zh
    tgt = "cmn" if zh else "eng"
    items = []
    for r in map(json.loads, open(a.items_file, encoding="utf-8")):
        x, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, r["audio"]
        x = x.mean(axis=1)[: int(r["dur_ms"]) * 16]
        items.append({"id": r["id"], "audio": x, "dur_ms": len(x) // 16, "committed": "", "blocks": [], "done": False,
                      "s": 0.0, "f": 0, "st": 0, "surv": 1.0, "reads": [],
                      "lgL": math.log(CTRL["lam0"]) if CTRL else 0.0, "W": 0, "seg": [], "pend": 0.0, "draft": []})
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
                                                              "surv0": it["surv"] if RULE == "survival" else 1.0,
                                                              **(lam_args(it["lgL"]) if CTRL else {})}}))
        if GUARD:                                        # change 14: unconstrained greedy draft (WAIT masked) per non-final read
            dl = [it for it in live if not it["final"]]
            dprompts = [pr for it, pr in zip(live, prompts) if not it["final"]]
            dparams = [SamplingParams(temperature=0.0, max_tokens=64, stop_token_ids=[THINKER_EOS],
                                      extra_args={"hsmm": {"s0": 0.0, "f0": 0, "started": 0, "cfrac": 0.0, "final": True}}) for _ in dl]
            res_all = llm.generate(prompts + dprompts, params + dparams, use_tqdm=False)
            res, dres = res_all[: len(prompts)], {id(it): x for it, x in zip(dl, res_all[len(prompts):])}
        else:
            res = llm.generate(prompts, params, use_tqdm=False)
        tkz = llm.get_tokenizer() if GUARD else None
        for it, rr in zip(live, res):
            o = rr.outputs[0]
            toks = [x for x in o.token_ids if x not in (WAIT, THINKER_EOS)]
            text = o.text
            if GUARD and not it["final"]:
                prev = it["draft"]
                k = 0
                while k < len(toks) and k < len(prev) and toks[k] == prev[k]:
                    k += 1
                over = it["st"] == 1 and it["s"] <= PAUSE_S and len(toks) > 0
                if not over and k < len(toks):
                    while k > 0 and ("\ufffd" in tkz.decode(toks[:k]) or (not zh_tgt and not tkz.decode([toks[k]]).startswith((" ", ",", ".", "!", "?", ";", ":")))):
                        k -= 1                               # no partial characters / words
                    toks = toks[:k]
                    text = tkz.decode(toks) if k else ""
                it.setdefault("n_over", 0); it["n_over"] += int(over and k < len(toks))
                d = [x for x in dres[id(it)].outputs[0].token_ids if x not in (WAIT, THINKER_EOS)]
                it["draft"] = d[len(toks):] if d[: len(toks)] == toks else []
            it["s"], it["f"], it["st"] = advance(it["s"], it["f"], it["st"], toks)
            if RULE == "survival" and not it["final"]:
                ids = list(o.token_ids)
                if ids and ids[-1] == WAIT and o.logprobs:          # the read ended on a wait decision
                    lpw = o.logprobs[-1][WAIT].logprob                # raw (pre-processor) log P_wait
                    pw = p_wait_gated(lpw, gate_value(it["s"], it["f"], it["st"], it["cfrac"]))
                    it["surv"] = (1.0 if toks else it["surv"]) * pw
                else:
                    it["surv"] = 1.0
            it["reads"].append({"t": it["t"], "n_tok": len(toks), **({"surv": round(it["surv"], 4)} if RULE == "survival" else {}),
                                **({"lamL": round(math.exp(it["lgL"]), 5)} if CTRL else {})})
            if CTRL and not it["final"]:                         # change 11: online primal-dual update of lam_L
                nw = sum(1 for x in toks if x not in _gate()["punct"])
                it["W"] += nw
                c = CTRL[a.direction]
                if CTRL.get("lag") == "speech":                  # change 13: speech lag over the playback queue
                    if nw:
                        st_ = max(it["t"] + CTRL["ob_ms"], it["pend"]); it["pend"] = st_ + nw * CTRL["tokms"][a.direction]
                        it["seg"].append((st_, it["pend"], nw))
                    Wp = sum(n * min(max((it["t"] - s0) / max(s1 - s0, 1e-9), 0.0), 1.0) for s0, s1, n in it["seg"])
                    lag = it["t"] / 1000.0 - Wp / c["rho"]
                    upd = it["t"] / 1000.0 > c["Lstar"]          # before L* a lag below the setpoint is unavoidable
                else:
                    lag = it["t"] / 1000.0 - it["W"] / c["rho"]
                    upd = True
                if upd:
                    it["lgL"] = min(max(it["lgL"] + CTRL["eta"] * (lag - c["Lstar"]), math.log(CTRL["lam_lo"])), math.log(CTRL["lam_hi"]))
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
            fh.write(json.dumps({"id": it["id"], "translation": it["committed"], "blocks": blocks, "reads": it["reads"], **({"n_over": it.get("n_over", 0)} if GUARD else {})},
                                ensure_ascii=False) + "\n")
    print(out, len(items), flush=True)


if __name__ == "__main__":
    main()
