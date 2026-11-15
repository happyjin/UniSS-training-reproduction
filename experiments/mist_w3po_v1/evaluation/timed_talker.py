"""v11 P1-1 (docs/mist_v11_plan.zh-CN.md §3; mock ACL review M5): per-block compute time of the streaming Talker.

Runs uniss_qwen3_omni_v1.evaluation.si_infer unchanged (same flags as the PA speech layer: --backlog-token
--select-k 2 --speed-policy queue ...), with three wrappers:
  * vllm.LLM.generate: a call with several prompts is split into one call per prompt, run one after the other and
    timed one by one, so each block's generation (both candidates, n=2) is measured as a single-stream system would
    pay it (outputs are the same: every request carries its own seed);
  * si_infer.plcrf_round: tags the round r and its owners (items with a block r, in order), times the whole round;
  * si_infer.score_candidates: times the candidate scoring of a round (BiCodec decode of the stream so far + the
    independent CTC scorer over HTTP).
Front-end (source tokenisation, voices) and the final BiCodec decode are reported by si_infer's own "timing" line.

Optional --keep-free-gib G: at the first generate call (the vLLM engine and the CTC scorer already hold their
memory) one uint8 tensor takes all but G GiB of the card's free memory, so no other job is stacked on the card while
it is being measured (released at exit).

Writes --timing-out (jsonl, refuses an existing file):
  {"kind": "gen", "r": r, "owner": <item id or null>, "i": i, "ms": ms, "n_tok": [tokens per candidate]}
  {"kind": "score", "r": r, "n": blocks in the round, "ms": ms}
  {"kind": "round", "r": r, "n": blocks in the round, "ms": ms}

    python -m experiments.mist_w3po_v1.evaluation.timed_talker --timing-out <jsonl> [--keep-free-gib 14] -- <si_infer args>
"""
from __future__ import annotations

import json
import sys
import time

from experiments.mist_w3po_v1.evaluation.timed_reads import split


def wrap_generate(orig, ctx, rec, clock=time.perf_counter):
    def generate(self, prompts, sampling_params=None, *a, **k):
        if ctx.get("reserve"):
            ctx.pop("reserve")()
        ps = prompts if isinstance(prompts, list) else [prompts]
        owners = ctx.get("owners") or []
        outs = []
        for i, p in enumerate(ps):
            sp = sampling_params[i] if isinstance(sampling_params, list) else sampling_params
            t = clock()
            o = orig(self, [p], [sp] if isinstance(sampling_params, list) else sp, *a, **k)
            ms = (clock() - t) * 1000.0
            rec({"kind": "gen", "r": ctx.get("r"), "owner": owners[i] if len(owners) == len(ps) else None, "i": i,
                 "ms": round(ms, 3), "n_tok": [len(c.token_ids) for c in o[0].outputs]})
            outs += o
        return outs

    return generate


def wrap_round(orig, ctx, rec, clock=time.perf_counter):
    def plcrf_round(r, items, *a, **k):
        ctx["r"] = r
        ctx["owners"] = [it["id"] for it in items if r < len(it["blocks"])]
        t = clock()
        try:
            return orig(r, items, *a, **k)
        finally:
            rec({"kind": "round", "r": r, "n": len(ctx["owners"]), "ms": round((clock() - t) * 1000.0, 3)})
            ctx["owners"] = None

    return plcrf_round


def wrap_score(orig, ctx, rec, clock=time.perf_counter):
    def score_candidates(*a, **k):
        t = clock()
        try:
            return orig(*a, **k)
        finally:
            rec({"kind": "score", "r": ctx.get("r"), "n": len(ctx.get("owners") or []),
                 "ms": round((clock() - t) * 1000.0, 3)})

    return score_candidates


def main() -> None:
    argv = sys.argv[1:]
    own_end = argv.index("--") if "--" in argv else len(argv)
    own = argv[:own_end]
    if "--timing-out" not in own:
        raise SystemExit("--timing-out <jsonl> required")
    i = own.index("--timing-out")
    out = own[i + 1]
    keep = 0.0
    if "--keep-free-gib" in own:
        keep = float(own[own.index("--keep-free-gib") + 1])
    rest = argv[own_end + 1:]
    import torch
    import vllm

    from experiments.uniss_qwen3_omni_v1.evaluation import si_infer as SI

    fh = open(out, "x", encoding="utf-8")
    held = []

    def rec(d):
        fh.write(json.dumps(d) + "\n")
        fh.flush()

    def reserve():
        if keep <= 0:
            return
        free, total = torch.cuda.mem_get_info(0)
        n = int(free - keep * 2**30)
        if n > 2**30:
            held.append(torch.empty(n, dtype=torch.uint8, device="cuda:0"))
        print(f"TIMED_TALKER reserve {n / 2**30:.1f} GiB (free was {free / 2**30:.1f} of {total / 2**30:.1f}, keep {keep})",
              flush=True)

    ctx = {"reserve": reserve}
    vllm.LLM.generate = wrap_generate(vllm.LLM.generate, ctx, rec)
    SI.plcrf_round = wrap_round(SI.plcrf_round, ctx, rec)
    SI.score_candidates = wrap_score(SI.score_candidates, ctx, rec)
    sys.argv = ["si_infer"] + rest
    try:
        SI.main()
    finally:
        fh.close()
    print(f"TIMED_TALKER_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
