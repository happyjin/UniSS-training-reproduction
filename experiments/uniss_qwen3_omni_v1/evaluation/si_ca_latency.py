"""Computation-aware latency for the Thinker's commit rules: what each read costs when one stream runs alone.

The policy runs (si_thinker_stream) batch every sentence of RealSI together, so their write times
are the moments the source was heard, as if the Thinker answered at once. Here a recorded run is
replayed one sentence at a time, one read at a time, with the request that rule issues at that read
-- one greedy pass (LocalAgreement-2), the same with top-5 log-probabilities and the commit head
(the head), or n samples (the vote) -- and each read is timed. A read can start when its audio has
arrived and the read before it has finished, so its commits land at

    c_r = max(t_r, c_{r-1}) + compute_r

  time   (vLLM venv)  replay a blocks file and time every read           -> timing jsonl
  score  (eval venv)  LAAL with and without the compute, text and speech -> one json line
                      (speech: the whole-system run's blocks re-timed: a block plays at
                       max(c_r + --talker-ms, the end of the block before), its words keep their place)

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_ca_latency time --blocks <blocks.jsonl> --direction zh2en --kind head --head <head.pt> --tau 0.25 --out t.jsonl
    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_ca_latency score --timing t.jsonl --run <system run>/realsi_zh2en
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

CJK = re.compile(r"[一-鿿]|[A-Za-z0-9']+")
EN = re.compile(r"[A-Za-z0-9']+")


def cmd_time(args) -> None:
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, WAIT_MARK, draft_tokens, units
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    zh = args.direction.endswith("zh")
    recs = [json.loads(l) for l in open(args.blocks, encoding="utf-8")]
    if args.limit and args.limit < len(recs):
        recs = [recs[round(i * (len(recs) - 1) / (args.limit - 1))] for i in range(args.limit)]
    items = {it["id"]: it for it in realsi_items(args.direction)}
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": inference_instruction()["cmn" if zh else "eng"]}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=args.thinker or QWEN3, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=args.seed)
    greedy = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS])
    greedy_lp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=5)
    selfwait = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS, WAIT_MARK],
                              logit_bias={WAIT_MARK: args.wait_bias})
    sample = SamplingParams(n=args.n, temperature=0.7, top_p=0.95, max_tokens=160, stop_token_ids=[THINKER_EOS], seed=args.seed)
    scorer = None
    if args.kind == "head":
        from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import commit_units, load_head
        scorer = load_head(args.head)
    cache = {}

    def audio_of(it):
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache.clear()
            cache[it["wav"]] = x.mean(axis=1)
        return cache[it["wav"]][it["start"] * 16: it["end"] * 16]

    def one(prompt, audio, t, final, n_committed, prev_greedy):
        req = {"prompt": prompt, "multi_modal_data": {"audio": (audio[: t * 16], 16000)}}
        t0 = time.perf_counter()
        if final or args.kind == "greedy":
            res = llm.generate([req], greedy, use_tqdm=False)
        elif args.kind == "selfwait":   # the amortised policy: one greedy pass that stops at the wait marker
            res = llm.generate([req], selfwait, use_tqdm=False)
        elif args.kind == "vote":
            res = llm.generate([req], sample, use_tqdm=False)
        else:   # the head: one greedy pass with log-probabilities, then the head on its tokens
            res = llm.generate([req], greedy_lp, use_tqdm=False)
            if args.virtual_ms and t > args.virtual_ms:   # the virtual draft on the audio minus its last virtual_ms
                llm.generate([{"prompt": prompt, "multi_modal_data": {"audio": (audio[: (t - args.virtual_ms) * 16], 16000)}}],
                             greedy, use_tqdm=False)
            o = res[0].outputs[0]
            gu = units(o.text, zh)
            agree = 0
            while agree < min(len(gu), len(prev_greedy)) and gu[agree] == prev_greedy[agree]:
                agree += 1
            commit_units(scorer, draft_tokens(o, o.text, zh), len(gu), t / 1000, n_committed, agree, zh, args.tau,
                         args.decision)
        return (time.perf_counter() - t0) * 1000, res

    warm = recs[0]
    it = items[warm["id"]]
    for _ in range(3):   # compile and cache before anything is timed
        one(head, audio_of(it), min(2000, it["end"] - it["start"]), False, 0, [])
    with open(out, "w", encoding="utf-8") as fh:
        for k, r in enumerate(recs):
            it = items[r["id"]]
            audio = audio_of(it)
            dur = len(audio) // 16
            blocks = r["blocks"]
            reads, prev = [], []
            for d in r["drafts"]:
                t = d["t"]
                committed = "".join(b["text"] for b in blocks if b["write_ms"] < t)
                n_committed = len(units(committed, zh))
                ms, res = one(head + committed, audio, t, t >= dur, n_committed, prev)
                if args.kind == "head":
                    prev = units(res[0].outputs[0].text, zh)
                reads.append({"t": t, "ms": round(ms, 1)})
            fh.write(json.dumps({"id": r["id"], "dur_ms": dur, "reads": reads}) + "\n")
            if k % 20 == 0:
                print(f"{k}/{len(recs)} {r['id']} reads {len(reads)} mean {np.mean([x['ms'] for x in reads]):.0f} ms", flush=True)
    print(f"-> {out}", flush=True)


def ca_times(reads: list[dict]) -> dict[int, float]:
    """read time t -> when its commits land, reads processed one after another."""
    out, c = {}, 0.0
    for x in reads:
        c = max(float(x["t"]), c) + x["ms"]
        out[x["t"]] = c
    return out


def cmd_score(args) -> None:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_extra_metrics import laal, to_timeline
    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units

    timing = {json.loads(l)["id"]: json.loads(l) for l in open(args.timing)}
    run = Path(args.run)
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")}
    words = {json.loads(l)["id"]: json.loads(l)["words"] for l in open(run / "aligned_words.jsonl")}
    text_l, text_ca, sp_l, sp_ca, per_read, lagged = [], [], [], [], [], []
    for sid, tm in timing.items():
        r = recs.get(sid)
        if r is None or not r["blocks"]:
            continue
        ca = ca_times(tm["reads"])
        per_read += [x["ms"] for x in tm["reads"]]
        lagged.append(max(0.0, max(ca.values()) - max(ca)))   # how far the last commit trails its read
        unit = CJK if r["tgt_lang"] == "cmn" else EN
        n_ref = len(units(r["translation_ref"], r["tgt_lang"]))
        src = r["source_duration_ms"]
        d0 = [b["write_ms"] for b in r["blocks"] for _ in unit.findall(b["text"])]
        d1 = [ca.get(b["write_ms"], b["write_ms"]) for b in r["blocks"] for _ in unit.findall(b["text"])]
        if d0:
            text_l.append(laal(d0, src, n_ref))
            text_ca.append(laal(d1, src, n_ref))
        w = words.get(sid)
        if not w:
            continue
        sp_l.append(laal([to_timeline(e, r["blocks"]) for _, e in w], src, n_ref))
        moved, end_prev = [], 0.0
        for b in r["blocks"]:   # the same speech, each block played once its text exists (and the talker has started)
            start = max(ca.get(b["write_ms"], b["write_ms"]) + args.talker_ms, end_prev)
            moved.append({**b, "start_ms": start})
            end_prev = start + b["dur_ms"]
        sp_ca.append(laal([to_timeline(e, moved) for _, e in w], src, n_ref))
    res = {"sentences": len(text_l), "read_ms_mean": round(float(np.mean(per_read)), 1),
           "read_ms_p90": round(float(np.percentile(per_read, 90)), 1),
           "text_laal_s": round(float(np.mean(text_l)) / 1000, 3), "text_ca_laal_s": round(float(np.mean(text_ca)) / 1000, 3),
           "speech_laal_s": round(float(np.mean(sp_l)) / 1000, 3), "speech_ca_laal_s": round(float(np.mean(sp_ca)) / 1000, 3),
           "last_commit_backlog_s": round(float(np.mean(lagged)) / 1000, 3), "talker_ms": args.talker_ms}
    print(json.dumps(res), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("time")
    a.add_argument("--blocks", required=True)
    a.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    a.add_argument("--kind", choices=("greedy", "vote", "head", "selfwait"), required=True)
    a.add_argument("--virtual-ms", type=int, default=0, help="head: also time the virtual draft on the audio minus this")
    a.add_argument("--thinker", default="", help="selfwait: the merged Thinker the amortised policy runs on")
    a.add_argument("--wait-bias", type=float, default=1.0, help="selfwait: logit bias on the wait marker")
    a.add_argument("--n", type=int, default=8)
    a.add_argument("--head", default="")
    a.add_argument("--tau", type=float, default=0.3)
    a.add_argument("--decision", choices=("threshold", "bayes"), default="threshold")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--gpu-mem", type=float, default=0.85)
    a.add_argument("--seed", type=int, default=20261021)
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--timing", required=True)
    b.add_argument("--run", required=True, help="the whole-system run's realsi_<direction> directory")
    b.add_argument("--talker-ms", type=float, default=0.0, help="time from a block's text to its first sound")
    b.add_argument("--out", default="")
    args = ap.parse_args()
    cmd_time(args) if args.cmd == "time" else cmd_score(args)


if __name__ == "__main__":
    main()
