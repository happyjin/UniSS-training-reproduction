"""Test of the additive --split option of crf/build_sched_windows.py (CPU):

  (a) default (train): the first 1000 windows rebuilt now are byte-identical to eval_outputs/.../sched_v1 shards
      000-001, and summary.json records no new option
  (b) --split validation on 600 dev windows: every window has v8 validation rows and all of them match the
      reconstruction (0 mismatches, 0 unknown reads), no bound violations, split recorded

Outputs go under $TMPDIR and are left in place (no deletion).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_build_sched_windows
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCHED = Path("eval_outputs/mist_w3po_v1/crf/sched_v1")
V8_VAL = "data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_validation.jsonl"


def _build(out: Path, *extra: str) -> dict:
    r = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.crf.build_sched_windows", "--out", str(out),
                        "--workers", "2", *extra], capture_output=True, text=True)
    assert r.returncode == 0 and "SCHED_WINDOWS_DONE" in r.stdout, r.stderr[-3000:]
    return json.loads((out / "summary.json").read_text())


def test_default_train_identical():
    root = Path(tempfile.mkdtemp(prefix="test_sched_build_train_"))      # under $TMPDIR; never deleted (no rm)
    summ = _build(root / "out", "--limit", "1000")
    for k in (0, 1):
        f = f"shard_{k:03d}.jsonl"
        assert (root / "out" / f).read_bytes() == (SCHED / f).read_bytes(), f
    assert not (root / "out" / "shard_002.jsonl").exists()
    assert "split" not in summ["args"]
    st = summ["stats"]
    assert st["windows"] == 1000 and st.get("v8_rows_mismatch", 0) == 0 and st["v8_rows_match"] > 0


def test_validation_split():
    root = Path(tempfile.mkdtemp(prefix="test_sched_build_val_"))        # under $TMPDIR; never deleted (no rm)
    summ = _build(root / "out", "--split", "validation", "--v8-rows", V8_VAL, "--limit", "600")
    st = summ["stats"]
    assert summ["args"]["split"] == "validation"
    assert st["windows"] == 600 and st["v8_windows_with_rows"] == 600, st
    assert st.get("v8_rows_mismatch", 0) == 0 and st.get("v8_rows_unknown_read", 0) == 0 and st["v8_rows_match"] > 0
    assert st["viol_lo_tokens"] == 0 and st["viol_hi_tokens"] == 0
    recs = [json.loads(l) for l in open(root / "out" / "shard_000.jsonl", encoding="utf-8")]
    assert all("/validation/" in r["audio"] for r in recs)


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")
