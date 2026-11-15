"""v11 P1-1 (docs/mist_v11_plan.zh-CN.md §3; mock ACL review M5): per-read wall-clock time of the streaming Thinker.

Runs a decode entry point (crf/decode_crf_ckpt_finalpost for B, crf/decode_crf_ckpt_waitbias for the method) unchanged,
but wraps core.engine.Engine.read as the OUTERMOST wrapper: the wrapper is installed when online_decode.main starts,
i.e. after decode_crf_ckpt_waitbias / online_decode_finalpost have installed theirs, so the measured time contains
every wrapper, the prefix re-encoding of each read and the generation. Each call is timed between
torch.cuda.synchronize() of every visible device. Use with --batch 1 so one call = one read of one window.

Optional --keep-free-gib G: at the first read, after the model is loaded, one uint8 tensor takes all but G GiB of the
card's free memory, so that no other job is stacked on the card while it is being measured (released at exit).

Writes --reads-out (jsonl, refuses an existing file), one line per Engine.read call:
  {"akeys": [<window id>|<t ms>, ...], "ms": wall ms, "n_new": [generated ids per item], "wall": unix time at the end}

    python -m experiments.mist_w3po_v1.evaluation.timed_reads --target experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost \
        --reads-out <jsonl> [--keep-free-gib 14] -- <the target's arguments, incl. --batch 1>
"""
from __future__ import annotations

import importlib
import json
import sys
import time


def split(argv: list[str]) -> tuple[dict, list[str]]:
    if "--" not in argv:
        raise SystemExit("usage: --target <module> --reads-out <jsonl> [--keep-free-gib G] -- <target args>")
    i = argv.index("--")
    own, rest = argv[:i], argv[i + 1:]
    opts = {"target": None, "reads_out": None, "keep_free_gib": 0.0}
    k = 0
    while k < len(own):
        key = own[k].lstrip("-").replace("-", "_")
        if key not in opts:
            raise SystemExit(f"unknown option {own[k]}")
        opts[key] = own[k + 1]
        k += 2
    if not opts["target"] or not opts["reads_out"]:
        raise SystemExit("--target and --reads-out are required")
    opts["keep_free_gib"] = float(opts["keep_free_gib"])
    return opts, rest


def make_timed_read(inner, fh, sync, clock=time.perf_counter, reserve=None):
    """Engine.read wrapper: times inner(self, items, ...) between sync() calls and writes one json line."""
    state = {"first": True}

    def read(self, items, *a, **k):
        if state["first"]:
            state["first"] = False
            if reserve is not None:
                reserve()
        sync()
        t = clock()
        outs = inner(self, items, *a, **k)
        sync()
        ms = (clock() - t) * 1000.0
        fh.write(json.dumps({"akeys": [it.akey for it in items], "ms": round(ms, 3),
                             "n_new": [len(o.ids) for o in outs], "wall": time.time()}) + "\n")
        fh.flush()
        return outs

    return read


def main() -> None:
    opts, rest = split(sys.argv[1:])
    import torch

    from experiments.mist_w3po_v1.core import engine as E
    from experiments.mist_w3po_v1.evaluation import online_decode as OD

    fh = open(opts["reads_out"], "x", encoding="utf-8")
    held = []

    def sync():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def reserve():
        keep = opts["keep_free_gib"]
        if keep <= 0:
            return
        free, total = torch.cuda.mem_get_info(0)
        n = int(free - keep * 2**30)
        if n > 2**30:
            held.append(torch.empty(n, dtype=torch.uint8, device="cuda:0"))
        print(f"TIMED_READS reserve {n / 2**30:.1f} GiB (free was {free / 2**30:.1f} of {total / 2**30:.1f}, keep {keep})",
              flush=True)

    orig_main = OD.main

    def timed_main():
        inner = E.Engine.read
        E.Engine.read = make_timed_read(inner, fh, sync, reserve=reserve)
        try:
            orig_main()
        finally:
            E.Engine.read = inner
            fh.close()

    OD.main = timed_main
    mod = importlib.import_module(opts["target"])
    sys.argv = [opts["target"]] + rest
    mod.main()
    print(f"TIMED_READS_DONE {opts['reads_out']}", flush=True)


if __name__ == "__main__":
    main()
