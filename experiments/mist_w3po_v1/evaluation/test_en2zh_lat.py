"""Tests of evaluation/zh_units.py and evaluation/en2zh_lat.py (CPU):

  (a) zh_units: one token per CJK character / Latin word, punctuation attached and not counted, idempotent,
      len(charify(s).split()) == n_units(s)
  (b) plumbing: en2zh_lat with the test switches --dr zh2en --units word (verbatim copies, zh2en universe and refs)
      reproduces the stored zh2en latency json of A'rl + PL-EM seed 1 (C2 windows) field by field ("unit",
      "unit_end_parts_online", "stream", "check", "segment_block_mismatch", "load")
  (c) en2zh mode on a toy: charified copies have one token per unit; the run / talker paths in the tool's output point
      at <work>

Work dirs go under $TMPDIR and are left in place (no deletion).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.evaluation.test_en2zh_lat
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.zh_units import charify, n_units, units

DEC = "eval_outputs/mist_w3po_v1/decode"
SPD = "eval_outputs/mist_w3po_v1/speech"
NAME = "dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem"
STORED = f"eval_outputs/mist_w3po_v1/detector_v1/chunk_table_v1/old/lat_{NAME}.json"
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
REF = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"


def test_units():
    s = "“好的，”我们用 PowerPoint 吧。3.5 版本"
    assert units(s) == ["好", "的", "我", "们", "用", "PowerPoint", "吧", "3.5", "版", "本"], units(s)
    c = charify(s)
    assert c == "“好 的，” 我 们 用 PowerPoint 吧。 3.5 版 本", c
    assert len(c.split()) == n_units(s) == 10
    assert charify(c) == c
    assert charify("。") == "。" and charify("") == "" and n_units("。！") == 0
    assert charify("it's fine, okay") == "it's fine, okay"


def test_plumbing():
    root = Path(tempfile.mkdtemp(prefix="test_en2zh_lat_"))      # under $TMPDIR; never deleted (no rm)
    out = root / "lat.json"
    r = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.en2zh_lat", "--ref-win", REF,
                        "--work", str(root / "work"), "--dr", "zh2en", "--units", "word", "--run", f"{DEC}/{NAME}",
                        "--talker", f"{SPD}/{NAME}/talker", "--win", WIN, "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    a, b = json.load(open(out)), json.load(open(STORED))
    for k in ("unit", "unit_end_parts_online", "stream", "check", "segment_block_mismatch", "load"):
        assert a[k] == b[k], k
    assert a["run"] == str(root / "work" / "run") and a["talker"] == str(root / "work" / "talker")


def test_charify_copies():
    from experiments.mist_w3po_v1.evaluation import en2zh_lat as E

    root = Path(tempfile.mkdtemp(prefix="test_en2zh_copies_"))
    src = root / "hyp_src.jsonl"
    src.write_text(json.dumps({"id": "w", "segments": [{"spk": 1, "text": "我们开会，好吗？"}],
                               "blocks": [{"spk": 1, "text": "我们开会，好吗？"}]}, ensure_ascii=False) + "\n")
    E.charify_hyp(src, root / "hyp.jsonl")
    h = json.loads((root / "hyp.jsonl").read_text())
    assert h["segments"][0]["text"] == h["blocks"][0]["text"] == "我 们 开 会， 好 吗？"


def main() -> None:
    test_units()
    print("(a) zh_units: ok")
    test_charify_copies()
    print("(c) charified copies: ok")
    test_plumbing()
    print("(b) plumbing reproduces the stored zh2en json: ok")


if __name__ == "__main__":
    main()
