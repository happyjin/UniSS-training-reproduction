"""Tests of speech_latency_lit --text-only (CPU, read-only on existing dev5 outputs):
  (a) the default (speech) run reproduces the stored talker_crf precheck output lat_aprule.json key for key
      (run / talker paths aside), i.e. the default path is unchanged;
  (b) --text-only on the same decode needs no talker; its text_laal_online covers every unit with text (n >= the
      speech run's n) and equals the speech run's on the units the talker covered (here: identical when n is equal).

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.evaluation.test_speech_latency_lit_textonly
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

RUN = "eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule"
TALKER = "eval_outputs/mist_w3po_v1/speech/plcrf_dev5_cands/talker"
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
STORED = "eval_outputs/mist_w3po_v1/talker_crf/precheck_dev5/lat_aprule.json"


def run(args: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "lat.json"
        subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.speech_latency_lit", *args,
                        "--out", str(out)], check=True, capture_output=True)
        return json.loads(out.read_text())


def main() -> None:
    ref = json.loads(Path(STORED).read_text())
    a = run(["--run", RUN, "--talker", TALKER, "--win", WIN])
    for k in ref:
        if k in ("run", "talker"):
            continue
        assert a[k] == ref[k], k
    print("PASS (a) default output == stored lat_aprule.json", flush=True)
    b = run(["--run", RUN, "--text-only", "--win", WIN])
    ta, tb = a["unit"]["text_laal_online"], b["unit"]["text_laal_online"]
    for reg in ("all", "ov", "1spk"):
        assert tb[reg]["n"] >= ta[reg]["n"], reg
        if tb[reg]["n"] == ta[reg]["n"]:
            assert tb[reg] == ta[reg], reg
    assert b["stream"] == {} and b["unit_end_parts_online"] == {}
    print(f"PASS (b) text-only text LAAL {tb['all']} (speech run {ta['all']})", flush=True)
    try:
        run(["--run", RUN, "--win", WIN])
        raise AssertionError("missing --talker accepted")
    except subprocess.CalledProcessError:
        pass
    print("ALL 2 PASSED")


if __name__ == "__main__":
    main()
