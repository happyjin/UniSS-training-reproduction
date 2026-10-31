"""QABC's decision rules applied explicitly to the adapter's posterior (paper v10, prereg_v5 section 4).

si_thinker_stream --policy selfwait commits what greedy decoding with a bias b on the wait marker says before the marker
(Proposition 5: unit u+1 survives iff rho(u) > sigma(b - log pi(u)), pi the top content token's share of the non-wait
mass). Here every read instead decodes the draft with the marker barred and asks vLLM for the marker's raw
log-probability at every step (logprob_token_ids; raw = before the bar), which gives the hazard h(u) = P(wait before
unit u+1) and rho(u) = 1 - h(u) for every unit of the draft, and pi(u). A rule then picks how many units to commit:

  p5    the current implementation, replayed from the same numbers: stop at the first token where
        log P(wait) + b > log P(chosen token), b = logit tau(q); commits exactly what --policy selfwait commits
  seq   Theorem 1 exactly: commit units while rho(u) > tau(q)
  seqb  the within-block queue (review M4): unit u+1 waits behind the u units committed before it in this read, so its
        delay from waiting is D(u) = max(0, Delta - q - r u) and its threshold tau(q + r u)
  glob  the global minimiser over prefix lengths a of L(a) = P(S < a) + w sum_{u >= a} P(S >= u), w = 1 - tau(q)
        (Appendix A's loss divided by C_err), smallest a among ties
  recal seq on a Platt-recalibrated hazard, logit h' = alpha logit h + beta (--recal-json {direction: [alpha, beta]})

tau(q) = 1 - (1 - tau0)(max(0, Delta - q) / Delta) capped at 0.9975, tau0 = sigma(--b0), q the committed speech not yet
played (the SDR queue of si_thinker_stream: --rate seconds a unit). The last read commits the rest, as selfwait does.
--dump writes every read (draft units, h, pi, q, the rule's choice); --label adds the whole-sentence continuation of the
same model at every read (one more pass on the full audio), from which the stable length S is the common prefix.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_qabc_rules --direction en2zh --windows 1,2 --rules p5,seq \
        --thinker <merged> --thinker-tag s0 --b0 1 --rate 0.191 --out <dir> [--items-file dev.jsonl] [--dump <dir> --label]
  label mode (another model's whole-sentence continuation for the reads of a dump, e.g. the frozen Thinker):
    python -m ...si_qabc_rules label --direction en2zh --dump-file <dump.jsonl> --thinker <model> --out <labels.jsonl>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, WAIT_MARK, join, units

TAU_MAX = 0.9975


def tau_of(tau0: float, window_s: float, q_s: float) -> float:
    t = 1 - (1 - tau0) * max(0.0, window_s - max(0.0, q_s)) / window_s
    return min(max(t, tau0), TAU_MAX)


def unit_spans(text: str, zh: bool) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S" if zh else r"\S+", text)]


def read_numbers(out, zh: bool) -> dict:
    """Per token: text, raw log P(token), raw log P(wait); per draft unit: hazard h and pi at the token that starts it."""
    toks, lp_tok, lp_wait, starts = [], [], [], []
    pos = 0
    for tid, alts in zip(out.token_ids, out.logprobs or []):
        if tid == THINKER_EOS:
            break
        ch = alts.get(tid)
        w = alts.get(WAIT_MARK)
        text = ch.decoded_token if ch is not None and ch.decoded_token is not None else ""
        toks.append(text)
        lp_tok.append(float(ch.logprob) if ch is not None else -30.0)
        lp_wait.append(float(w.logprob) if w is not None else -30.0)
        starts.append(pos + len(text) - len(text.lstrip()))
        pos += len(text)
    cont = "".join(toks)
    spans = unit_spans(cont, zh)
    at = {}
    for j, s in enumerate(starts):
        at.setdefault(s, j)
    h, pi = [], []
    for a, _ in spans:
        j = at.get(a)
        if j is None:   # no token starts this unit (it begins inside one): no chance to wait before it
            h.append(0.0)
            pi.append(1.0)
        else:
            pw = math.exp(lp_wait[j])
            h.append(pw)
            pi.append(min(1.0, math.exp(lp_tok[j]) / max(1e-12, 1 - pw)))
    return {"toks": toks, "lp_tok": lp_tok, "lp_wait": lp_wait, "cont": cont, "spans": spans, "h": h, "pi": pi}


def choose(rule: str, nums: dict, tau0: float, window_s: float, q_s: float, rate: float, recal=None) -> str:
    """The prefix of the continuation's text this rule commits."""
    cont, spans, h = nums["cont"], nums["spans"], nums["h"]
    tau = tau_of(tau0, window_s, q_s)
    if rule == "p5":
        b = math.log(tau / (1 - tau))
        for j, (lt, lw) in enumerate(zip(nums["lp_tok"], nums["lp_wait"])):
            if lw + b > lt:
                return "".join(nums["toks"][:j])
        return cont
    if rule == "recal":
        a, c = recal
        h = [0.0 if x <= 0 else 1 / (1 + math.exp(-(a * math.log(min(x, 1 - 1e-9) / (1 - min(x, 1 - 1e-9))) + c))) for x in h]
    K = len(spans)
    rho = [1 - x for x in h]
    if rule in ("seq", "recal", "seqb"):
        a = 0
        while a < K and rho[a] > (tau_of(tau0, window_s, q_s + rate * a) if rule == "seqb" else tau):
            a += 1
    elif rule == "glob":
        w = 1 - tau
        surv = np.concatenate([[1.0], np.cumprod(rho)])             # surv[i] = P(S >= i), i = 0..K
        tail = np.concatenate([np.cumsum(surv[:K][::-1])[::-1], [0.0]])   # tail[a] = sum_{u=a}^{K-1} P(S >= u)
        L = 1 - surv + w * tail
        a = int(np.argmin(L))
    else:
        raise ValueError(rule)
    return cont[: spans[a - 1][1]] if a else ""


def load_items(direction: str, items_file: str, limit: int) -> list[dict]:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items
    if items_file:
        items = [{"id": r["id"], "wav": r["audio"], "start": 0, "end": int(r["dur_ms"]), "trg_text": r.get("translation", "")}
                 for r in map(json.loads, open(items_file, encoding="utf-8"))]
    else:
        items = realsi_items(direction)
    if limit and limit < len(items):
        items = [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]
    cache = {}
    for it in items:
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        it["audio"] = cache[it["wav"]][it["start"] * 16: it["end"] * 16]
        it["dur_ms"] = len(it["audio"]) // 16
    return items


def make_head(direction: str) -> str:
    from transformers import Qwen3OmniMoeProcessor
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    tgt = "cmn" if direction.endswith("zh") else "eng"
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": inference_instruction()[tgt]}]}]
    return processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]


def run_rule(llm, items, head, zh, window_ms, rule, tau0, rate, out, dump, label, recal, tag):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logit_bias={WAIT_MARK: -100.0},
                        logprob_token_ids=[WAIT_MARK])
    full_sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logit_bias={WAIT_MARK: -100.0})
    for it in items:
        it.update(committed=[], blocks=[], drafts=[], done=False, busy=0)
    window_s = window_ms / 1000
    recs, t0, r = [], time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"]]
        prompts = []
        for it in live:
            it["t"] = min((r + 1) * window_ms, it["dur_ms"])
            prompts.append({"prompt": head + join(it["committed"], zh),
                            "multi_modal_data": {"audio": (it["audio"][: it["t"] * 16], 16000)}})
        results = llm.generate(prompts, sp, use_tqdm=False)
        fulls = [None] * len(live)
        if label:
            todo = [k for k, it in enumerate(live) if it["t"] < it["dur_ms"]]
            fp = [{"prompt": prompts[k]["prompt"], "multi_modal_data": {"audio": (live[k]["audio"], 16000)}} for k in todo]
            for k, g in zip(todo, llm.generate(fp, full_sp, use_tqdm=False) if fp else []):
                fulls[k] = units(g.outputs[0].text, zh)
        for k, (it, res) in enumerate(zip(live, results)):
            o = res.outputs[0]
            n = len(it["committed"])
            final = it["t"] >= it["dur_ms"]
            nums = read_numbers(o, zh)
            q_s = max(0, it["busy"] - it["t"]) / 1000
            said_text = nums["cont"] if final else choose(rule, nums, tau0, window_s, q_s, rate, recal)
            sep = "" if zh or not it["committed"] or said_text[:1].isspace() or said_text[:1] in ",.;:!?)" else " "
            draft_all = units(join(it["committed"], zh) + sep + nums["cont"], zh)
            it["drafts"].append({"t": it["t"], "draft": join(draft_all, zh)})
            new = units(join(it["committed"], zh) + sep + said_text, zh)[n:] if said_text.strip() else []
            if dump is not None and not final:
                rec = {"id": it["id"], "t": it["t"], "dur_ms": it["dur_ms"], "n_committed": n,
                       "committed": join(it["committed"], zh), "draft_units": units(nums["cont"], zh),
                       "h": [round(x, 6) for x in nums["h"]], "pi": [round(x, 6) for x in nums["pi"]],
                       "q_s": round(q_s, 3), "tau": round(tau_of(tau0, window_s, q_s), 6), "keep": len(new)}
                if fulls[k] is not None:
                    rec["full_units"] = fulls[k]
                recs.append(rec)
            if final:
                it["done"] = True
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"]})
                it["committed"] += new
                if new:
                    it["busy"] = max(it["busy"], it["t"]) + len(new) * rate * 1000
        r += 1
        print(f"{tag} round {r}: {len(live)} sentences, {time.time() - t0:.0f}s", flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 "drafts": it["drafts"]}, ensure_ascii=False) + "\n")
    if dump is not None:
        with open(dump, "x", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)


def label_mode(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--direction", required=True, choices=("en2zh", "zh2en"))
    ap.add_argument("--dump-file", required=True)
    ap.add_argument("--items-file", default="")
    ap.add_argument("--thinker", default=QWEN3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    from vllm import LLM, SamplingParams
    zh = a.direction.endswith("zh")
    items = {it["id"]: it for it in load_items(a.direction, a.items_file, 0)}
    recs = [json.loads(l) for l in open(a.dump_file, encoding="utf-8")]
    head = make_head(a.direction)
    llm = LLM(model=a.thinker, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=20261021, tensor_parallel_size=a.tp)
    sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logit_bias={WAIT_MARK: -100.0})
    prompts = [{"prompt": head + r["committed"], "multi_modal_data": {"audio": (items[r["id"]]["audio"], 16000)}} for r in recs]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    with open(a.out, "x", encoding="utf-8") as fh:
        for r, o in zip(recs, outs):
            fh.write(json.dumps({"id": r["id"], "t": r["t"], "full_units": units(o.outputs[0].text, zh)}, ensure_ascii=False) + "\n")
    print(f"labels: {len(recs)} reads -> {a.out}", flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "label":
        return label_mode(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--rules", required=True, help="comma-separated: p5, seq, seqb, glob, recal; one file each")
    ap.add_argument("--thinker", required=True, help="the merged adapted Thinker")
    ap.add_argument("--thinker-tag", required=True)
    ap.add_argument("--b0", type=float, default=1.0)
    ap.add_argument("--rate", type=float, required=True, help="the SDR queue rate, s per unit (0.191 en2zh, 0.339 zh2en)")
    ap.add_argument("--recal-json", default="", help="recal: {direction: [alpha, beta]}")
    ap.add_argument("--recal-tag", default="", help="recal: names the calibration in the file name")
    ap.add_argument("--items-file", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="", help="write every read's numbers to <dump>/<blocks stem>.jsonl")
    ap.add_argument("--label", action="store_true", help="with --dump: the whole-sentence continuation of the same model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    zh = args.direction.endswith("zh")
    windows = [float(w) for w in args.windows.split(",")]
    rules = args.rules.split(",")
    recal = json.load(open(args.recal_json))[args.direction] if "recal" in rules else None
    runs = []
    for w in windows:
        for rule in rules:
            name = f"blocks_{args.direction}_w{w:g}_qr{rule}{args.recal_tag if rule == 'recal' else ''}" \
                   f"{args.thinker_tag}b{args.b0:g}q{args.rate:g}.jsonl"
            runs.append((w, rule, Path(args.out) / name))
    for _, _, o in runs:
        if o.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.dump:
        Path(args.dump).mkdir(parents=True, exist_ok=True)
    items = load_items(args.direction, args.items_file, args.limit)
    head = make_head(args.direction)
    from vllm import LLM
    llm = LLM(model=args.thinker, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=args.seed, tensor_parallel_size=args.tp)
    tau0 = 1 / (1 + math.exp(-args.b0))
    for w, rule, out in runs:
        dump = Path(args.dump) / f"{out.stem}.jsonl" if args.dump else None
        run_rule(llm, items, head, zh, int(round(w * 1000)), rule, tau0, args.rate, out, dump, args.label, recal,
                 f"{args.direction} {out.stem}")


if __name__ == "__main__":
    main()
