"""Several G-STAR-SI decoding runs (gstar_si_stream / gstar_si_stream_online) on one card with ONE copy of the Thinker.

A single decoding run keeps the card at ~25 % (the HF MoE decoder is launch-bound) but holds ~69 GB, so two runs do not
fit side by side. Here the parent loads the Thinker once and hands it to one spawned worker per run through CUDA IPC
(torch.multiprocessing shares the weights, nothing is copied); each worker runs the unchanged script's main() with its
own argv, its own batches and its own output, so a run gives what it gives alone. Each run's stdout/stderr goes to its
own log, as the shell redirect of a single job did.

runs json: [{"module": "gstar_si_stream_online" | "gstar_si_stream", "argv": [...], "log": path}], every argv with the
same --thinker as given here. Exit status: 0 only if every run succeeded.

    CUDA_VISIBLE_DEVICES=3 python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_multi --thinker <merged> runs.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import torch
import torch.multiprocessing as mp

PKG = "experiments.uniss_qwen3_omni_v1.evaluation"


def worker(thinker, run: dict, thinker_path: str) -> None:
    fd = os.open(run["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    torch.set_num_threads(int(os.environ.get("GSTAR_THREADS", max(4, (os.cpu_count() or 32) // 8))))   # as load_thinker

    def shared(path):
        assert os.path.realpath(path) == os.path.realpath(thinker_path), (path, thinker_path)
        return thinker

    stream = importlib.import_module(f"{PKG}.gstar_si_stream")
    stream.load_thinker = shared
    mod = importlib.import_module(f"{PKG}.{run['module']}")
    mod.load_thinker = shared   # gstar_si_stream_online imports it by name
    sys.argv = [run["module"]] + run["argv"]
    print(f"GSTAR_MULTI worker pid {os.getpid()}: {run['module']} {' '.join(run['argv'])}", flush=True)
    mod.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinker", required=True)
    ap.add_argument("runs")
    args = ap.parse_args()
    runs = json.load(open(args.runs))
    for r in runs:
        assert r["module"] in ("gstar_si_stream", "gstar_si_stream_online"), r["module"]
        i = r["argv"].index("--thinker")
        assert os.path.realpath(r["argv"][i + 1]) == os.path.realpath(args.thinker), r["argv"]

    from experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream import load_thinker

    t0 = time.time()
    thinker = load_thinker(args.thinker)
    print(f"thinker loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    ctx = mp.get_context("spawn")
    procs = []
    for r in runs:
        p = ctx.Process(target=worker, args=(thinker, r, args.thinker))
        p.start()
        procs.append((r, p))
        print(f"started pid {p.pid}: {r['log']}", flush=True)
    bad = 0
    for r, p in procs:
        p.join()
        print(f"exit {p.exitcode}: {r['log']} ({time.time() - t0:.0f}s)", flush=True)
        bad += p.exitcode != 0
    print(f"GSTAR_MULTI_DONE ok={len(runs) - bad} fail={bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
