"""Test of the additive en2zh options of crf/build_sched_windows.py (CPU):

  (a) defaults unchanged: test_build_sched_windows.test_default_train_identical (first 1000 zh2en windows byte-identical
      to eval_outputs/.../sched_v1 shards 000-001; summary.json records no new option)
  (b) en2zh on the first 400 en2zh train windows: every window has v8 read rows and all of them match the
      reconstruction (0 mismatches, 0 unknown reads), every window's chunks equal its own stream_target (pieces and
      commit times), no bound violation, the en2zh options are recorded, unit lengths are characters

Outputs go under $TMPDIR and are left in place (no deletion).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_build_sched_windows_en2zh
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from experiments.mist_w3po_v1.crf import test_build_sched_windows as T

SRC = "data/ami_multispk_si_v1/e2e_gstar_stream_v1"
EN2ZH = ["--src", SRC, "--v8-rows", f"{SRC}/rows_train.jsonl", "--text-key", "zh", "--slot-key", "",
         "--words-manual", "data/ami_multispk_si_v1/annotations/ami_public_manual_1.6.2.zip", "--piece-strip", "0",
         "--units", "char"]


def test_en2zh():
    root = Path(tempfile.mkdtemp(prefix="test_sched_build_en2zh_"))      # under $TMPDIR; never deleted (no rm)
    summ = T._build(root / "out", "--limit", "400", *EN2ZH)
    st = summ["stats"]
    assert st["windows"] == 400 and st["v8_windows_with_rows"] == 400, st
    assert st.get("v8_rows_mismatch", 0) == 0 and st.get("v8_rows_unknown_read", 0) == 0 and st["v8_rows_match"] > 0, st
    assert st["stream_target_match"] == 400 and st.get("stream_target_mismatch", 0) == 0, st
    assert st["viol_lo_tokens"] == 0 and st["viol_hi_tokens"] == 0, st
    a = summ["args"]
    assert a["text_key"] == "zh" and a["slot_key"] == "" and a["piece_strip"] == 0 and a["units"] == "char", a
    rec = json.loads(open(root / "out" / "shard_000.jsonl", encoding="utf-8").readline())
    u = rec["units"][0]
    assert isinstance(u["words"], int) and u["words"] >= 1


def main() -> None:
    T.test_default_train_identical()
    print("default (zh2en) byte-identical: ok")
    test_en2zh()
    print("en2zh: ok")


if __name__ == "__main__":
    main()
