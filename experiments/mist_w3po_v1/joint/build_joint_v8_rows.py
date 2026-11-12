#!/usr/bin/env python
"""Rows of the joint bidirectional v8 (stage 1 of reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md): the two
separate v8 Simul SFT runs' training samples, in their own order, interleaved 32 + 32 per global batch of 64.

Each separate run (run_thinker_gstar.sh -> pretrain_thinker_gstar.GStarRows, --gstar-seed 20261026, 3000 iterations x
GBS 64 = 192,000 samples) consumed order[:192000] with order = concat(rng.permutation(n) for each pass),
rng = numpy default_rng(seed) (the zh->en run thinker_gstar_s2dl_zh2en_3k_20261030 on e2e_gstar_stream_zh2en_v1, the
en->zh run thinker_gstar_s2dl_20261029b on e2e_gstar_stream_v1). Output line 64 t + j (t = 0 .. steps - 1) is
  j <  32: zh->en sample 32 t + j
  j >= 32: en->zh sample 32 t + j - 32
copied verbatim (the input line, byte for byte). Read with --gstar-sequential (GBS 64, TRAIN_ITERS = steps), global batch t
of the joint run holds exactly zh->en samples [32t, 32t + 32) and en->zh samples [32t, 32t + 32), so after 6000 steps
each direction has seen exactly its separate run's 192,000 samples in the same order. A manifest (<out>.manifest.json)
holds counts, input sha256 and the per-direction sample indices' sha256. Refuses to overwrite.

    python -m experiments.mist_w3po_v1.joint.build_joint_v8_rows --a <zh2en rows> --b <en2zh rows> --out <rows.jsonl> \\
        [--seed 20261026] [--per-dir 192000] [--block 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def gstar_order(n_rows: int, n_samples: int, seed: int) -> np.ndarray:
    """pretrain_thinker_gstar.GStarRows' sample order (non-sequential), first n_samples."""
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(n_rows) for _ in range(math.ceil(n_samples / n_rows))])
    return order[:n_samples]


def interleave(oa: np.ndarray, ob: np.ndarray, block: int) -> list[tuple[int, int]]:
    """[(source 0|1, row index)] : block of a, block of b, ... (len(oa) == len(ob), a multiple of block)."""
    if len(oa) != len(ob) or len(oa) % block:
        raise ValueError("per-direction sample counts must be equal and a multiple of the block")
    out = []
    for t in range(len(oa) // block):
        out += [(0, int(i)) for i in oa[t * block:(t + 1) * block]]
        out += [(1, int(i)) for i in ob[t * block:(t + 1) * block]]
    return out


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="zh->en stream rows (first half of each batch)")
    ap.add_argument("--b", required=True, help="en->zh stream rows (second half)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--per-dir", type=int, default=192000)
    ap.add_argument("--block", type=int, default=32)
    a = ap.parse_args(argv)
    out, man = Path(a.out), Path(a.out + ".manifest.json")
    if out.exists() or man.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    la = open(a.a, "rb").read().splitlines(keepends=True)
    lb = open(a.b, "rb").read().splitlines(keepends=True)
    oa, ob = gstar_order(len(la), a.per_dir, a.seed), gstar_order(len(lb), a.per_dir, a.seed)
    seq = interleave(oa, ob, a.block)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as fh:
        for src, i in seq:
            ln = (la, lb)[src][i]
            fh.write(ln if ln.endswith(b"\n") else ln + b"\n")
    rep = {"a": a.a, "b": a.b, "rows_a": len(la), "rows_b": len(lb), "seed": a.seed, "per_dir": a.per_dir,
           "block": a.block, "lines": len(seq), "steps_at_gbs": len(seq) // (2 * a.block),
           "sha256_a": sha(a.a), "sha256_b": sha(a.b), "sha256_out": sha(str(out)),
           "order_a_sha256": hashlib.sha256(oa.astype(np.int64).tobytes()).hexdigest(),
           "order_b_sha256": hashlib.sha256(ob.astype(np.int64).tobytes()).hexdigest()}
    with open(man, "x") as fh:
        fh.write(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
