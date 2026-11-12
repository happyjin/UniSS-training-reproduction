"""Tests of joint/build_joint_v8_rows.py and pretrain_thinker_gstar --gstar-sequential, CPU only, synthetic rows:
  (a) gstar_order = pretrain_thinker_gstar.GStarRows(sequential=False).order[:n] (the separate v8 runs' sample order)
  (b) GStarRows(sequential=True) over the built file: global batch t (samples [64t, 64t + 64)) = zh->en samples
      [32t, 32t + 32) then en->zh samples [32t, 32t + 32) of the two separate orders, rows copied byte for byte
  (c) default GStarRows (sequential unset) is the pre-change permutation formula; refuses to overwrite

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.joint.test_build_joint_v8_rows
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.joint import build_joint_v8_rows as B
from experiments.uniss_qwen3_omni_v1.training.pretrain_thinker_gstar import GStarRows


def rows(path: Path, n: int, d: str) -> list[str]:
    ls = [json.dumps({"id": f"{d}_{i}", "dir": d, "x": "é" * (i % 3)}, ensure_ascii=False) for i in range(n)]
    path.write_text("\n".join(ls) + "\n", encoding="utf-8")
    return ls


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        la, lb = rows(d / "a.jsonl", 37, "zh2en"), rows(d / "b.jsonl", 29, "en2zh")
        per, steps = 96, 3   # 3 steps x 32 per direction (> 1 pass of each file: permutations concatenated)
        for p, n in ((d / "a.jsonl", 37), (d / "b.jsonl", 29)):
            g = GStarRows(str(p), per, 20261026, "x", 0.0, False)
            assert np.array_equal(B.gstar_order(n, per, 20261026), g.order[:per])
            rng = np.random.default_rng(20261026)   # (c) the pre-change formula
            ref = np.concatenate([rng.permutation(n) for _ in range(math.ceil(per / n))])
            assert np.array_equal(g.order, ref)
        B.main(["--a", str(d / "a.jsonl"), "--b", str(d / "b.jsonl"), "--out", str(d / "j.jsonl"), "--per-dir", str(per)])
        oa, ob = B.gstar_order(37, per, 20261026), B.gstar_order(29, per, 20261026)
        j = GStarRows(str(d / "j.jsonl"), steps * 64, 20261026, "x", 0.0, False, sequential=True)
        assert len(j.rows) == steps * 64 and np.array_equal(j.order, np.arange(steps * 64))
        out = (d / "j.jsonl").read_text(encoding="utf-8").splitlines()
        for t in range(steps):
            for k in range(32):
                assert out[64 * t + k] == la[oa[32 * t + k]]
                assert out[64 * t + 32 + k] == lb[ob[32 * t + k]]
                assert j.rows[int(j.order[64 * t + k])]["dir"] == "zh2en"
                assert j.rows[int(j.order[64 * t + 32 + k])]["dir"] == "en2zh"
        man = json.loads((d / "j.jsonl.manifest.json").read_text())
        assert man["lines"] == steps * 64 and man["steps_at_gbs"] == steps
        try:
            B.main(["--a", str(d / "a.jsonl"), "--b", str(d / "b.jsonl"), "--out", str(d / "j.jsonl"), "--per-dir", str(per)])
            raise AssertionError("overwrote")
        except SystemExit:
            pass
    print("ALL 3 PASSED (separate orders reproduced, joint batches 32 + 32 in order, default loader unchanged)")


if __name__ == "__main__":
    main()
