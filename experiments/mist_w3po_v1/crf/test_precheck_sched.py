"""Tests of the sharding additions to crf/precheck_sched.py (CPU only, real sched_v1 windows, random log-probs):

  (a) parse_shard: "K/N" parsing and its errors
  (b) shard_select on the real 500-window sample (same seed): for N in 1, 2, 3, 7, 8 the shards are disjoint, their
      union is exactly the unsharded sample, each keeps the sample order, sizes differ by at most 1; N = 1 is identity
  (c) load_scored: one dir -> its own order; the complete shards (any order of dirs) -> the unsharded sample order and
      the right logprobs dir per window; a window in two dirs -> error; an incomplete set -> PARTIAL, concatenated
  (d) end to end through main(): --analysis-only on one unsharded scored dir vs on its 3 shard dirs (comma-separated)
      gives identical report.json and per_window_*.jsonl; the unsharded windows.json records no new options

Temporary dirs go under $TMPDIR and are left in place (no deletion).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_sched
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_sched as P

SEED = 20261109


def _expect_exit(fn, *a):
    try:
        fn(*a)
    except SystemExit:
        return
    raise AssertionError(f"{fn.__name__}{a} did not exit")


def test_parse_shard():
    assert P.parse_shard("") is None
    assert P.parse_shard("0/1") == (0, 1)
    assert P.parse_shard("7/8") == (7, 8)
    for bad in ("8/8", "-1/8", "1/0", "3", "a/b", "1/2/3"):
        _expect_exit(P.parse_shard, bad)


def test_shard_union_disjoint():
    full = P.load_records(P.SCHED, 500, SEED)
    ids = [r["id"] for r in full]
    assert len(ids) == 500 and len(set(ids)) == 500
    assert [r["id"] for r in P.shard_select(full, 0, 1)] == ids
    for N in (2, 3, 7, 8):
        shards = [[r["id"] for r in P.shard_select(full, k, N)] for k in range(N)]
        flat = [i for s in shards for i in s]
        assert len(flat) == len(set(flat)) == 500, N                       # disjoint
        assert set(flat) == set(ids), N                                    # union = the sample
        for s in shards:                                                   # sample order kept
            pos = [ids.index(i) for i in s]
            assert pos == sorted(pos)
        assert max(map(len, shards)) - min(map(len, shards)) <= 1
        inter = [shards[i % N][i // N] for i in range(500)]                # round robin = the sample
        assert inter == ids, N
    # the same sample is drawn again (the shard runs re-draw it independently)
    assert [r["id"] for r in P.load_records(P.SCHED, 500, SEED)] == ids


def _write_scored(root: Path, name: str, ids: list[str], lps: dict, args: dict) -> Path:
    d = root / name
    (d / "logprobs").mkdir(parents=True)
    (d / "windows.json").write_text(json.dumps({"sched": P.SCHED, "ids": ids, "plan": {}, "args": args}))
    for i in ids:
        np.savez(d / "logprobs" / f"{i}.npz", tok_lp=lps[i][0], end_lp=lps[i][1])
    return d


def _fake_scored(n: int, seed: int, N: int, tag: str):
    root = Path(tempfile.mkdtemp(prefix=f"test_precheck_{tag}_"))          # under $TMPDIR; never deleted (no rm)
    recs = P.load_records(P.SCHED, n, seed)
    rng = np.random.default_rng(seed)
    lps = {r["id"]: (-rng.exponential(1.0, (r["n_read"], r["n_tok"])),
                     np.log(rng.uniform(1e-3, 1, (r["n_read"], r["n_tok"] + 1)))) for r in recs}
    ids = [r["id"] for r in recs]
    full = _write_scored(root, "full", ids, lps, {"n": n, "seed": seed})
    shards = [_write_scored(root, f"shard_{k}of{N}", [r["id"] for r in P.shard_select(recs, k, N)], lps,
                            {"n": n, "seed": seed, "shard": f"{k}/{N}", "score_only": True}) for k in range(N)]
    return root, ids, full, shards


def test_load_scored():
    root, ids, full, shards = _fake_scored(24, 7, 3, "load")
    sched, got, lp_of = P.load_scored([str(full)])
    assert sched == P.SCHED and got == ids and all(lp_of[i] == full / "logprobs" for i in ids)
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        sched, got, lp_of = P.load_scored([str(shards[k]) for k in order])
        assert got == ids, order
        for k, d in enumerate(shards):
            for i in json.loads((d / "windows.json").read_text())["ids"]:
                assert lp_of[i] == d / "logprobs"
    _expect_exit(P.load_scored, [str(shards[0]), str(shards[1]), str(shards[0])])      # a window twice
    _expect_exit(P.load_scored, [str(full), str(shards[0])])                            # overlap with the full dir
    sched, got, _ = P.load_scored([str(shards[2]), str(shards[0])])                     # incomplete: concatenated
    a, c = [json.loads((shards[k] / "windows.json").read_text())["ids"] for k in (2, 0)]
    assert got == a + c


def test_main_merged_equals_unsharded():
    root, ids, full, shards = _fake_scored(24, 11, 3, "main")
    env = dict(os.environ)
    outs = {}
    for name, src in (("one", str(full)), ("merged", ",".join(str(d) for d in reversed(shards)))):
        o = root / f"analysis_{name}"
        r = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.crf.precheck_sched", "--analysis-only",
                            src, "--out", str(o), "--samples", "4", "--lattice-batch", "16"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0 and "SCHED_PRECHECK_DONE" in r.stdout, r.stderr[-3000:]
        outs[name] = o
    a, b = outs["one"], outs["merged"]
    assert json.loads((a / "report.json").read_text()) == json.loads((b / "report.json").read_text())
    for mode in ("any", "chunk"):
        assert (a / f"per_window_{mode}.jsonl").read_bytes() == (b / f"per_window_{mode}.jsonl").read_bytes()
    wa = json.loads((a / "windows.json").read_text())
    assert wa["ids"] == ids and json.loads((b / "windows.json").read_text())["ids"] == ids
    assert "shard" not in wa["args"] and "score_only" not in wa["args"]
    # the existing refuse-to-overwrite still holds
    r = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.crf.precheck_sched", "--analysis-only",
                        str(full), "--out", str(a)], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "拒绝覆盖" in (r.stdout + r.stderr)


def test_uniform_end():
    """--uniform-end = the analysis on log-probs with end_lp zeroed (written to disk by hand), and is recorded."""
    root, ids, full, shards = _fake_scored(16, 13, 2, "uniform")
    zero = root / "full_end_zero"
    (zero / "logprobs").mkdir(parents=True)
    w = json.loads((full / "windows.json").read_text())
    (zero / "windows.json").write_text(json.dumps(w))
    for i in ids:
        z = np.load(full / "logprobs" / f"{i}.npz")
        np.savez(zero / "logprobs" / f"{i}.npz", tok_lp=z["tok_lp"], end_lp=np.zeros_like(z["end_lp"]))
    outs = {}
    for name, src, extra in (("flag", ",".join(map(str, shards)), ["--uniform-end"]), ("byhand", str(zero), [])):
        o = root / f"analysis_{name}"
        r = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.crf.precheck_sched", "--analysis-only",
                            src, "--out", str(o), "--samples", "4", "--lattice-batch", "16", *extra],
                           capture_output=True, text=True, env=dict(os.environ))
        assert r.returncode == 0 and "SCHED_PRECHECK_DONE" in r.stdout, r.stderr[-3000:]
        outs[name] = o
    a = json.loads((outs["flag"] / "report.json").read_text())
    b = json.loads((outs["byhand"] / "report.json").read_text())
    for mode in ("any", "chunk"):
        assert a[mode].pop("end_prior").startswith("uniform") and "end_prior" not in b[mode]
    assert a == b
    assert json.loads((outs["flag"] / "windows.json").read_text())["args"]["uniform_end"] is True
    assert "uniform_end" not in json.loads((outs["byhand"] / "windows.json").read_text())["args"]


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")
