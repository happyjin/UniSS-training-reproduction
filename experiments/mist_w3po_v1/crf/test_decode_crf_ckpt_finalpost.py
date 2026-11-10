"""Unit tests for crf/decode_crf_ckpt_finalpost.py (CPU; online_decode.main, the merged-model loader and apply_ckpt are
stubbed, so no model is loaded):

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_decode_crf_ckpt_finalpost

  off           no flag / '--final-read-post off': online_decode.main gets the argv without --crf-ckpt, Engine.read is
                not patched, Engine.from_merged applies the checkpoint (decode_crf_ckpt.main unchanged)
  close_detect  online_decode.main gets the same plain argv; Engine.read wraps the original; a final item reaches the
                original read with P zeroed from final_cut_ms(t, close_detect) // 80 and full=True, a non-final item is
                passed as the same object; the checkpoint is applied as well
  errors        bad mode, missing --crf-ckpt, --lora with --crf-ckpt -> SystemExit
Fixture files (an empty checkpoint dir, a 1-line windows jsonl) go to a new tempfile.mkdtemp dir; nothing is deleted.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.crf import decode_crf_ckpt as dc
from experiments.mist_w3po_v1.crf import decode_crf_ckpt_finalpost as dcf
from experiments.mist_w3po_v1.evaluation import online_decode as od
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond, name


def fixtures():
    d = Path(tempfile.mkdtemp(prefix="test_dcf_"))
    ck = d / "ckpt" / "step_00001"
    ck.mkdir(parents=True)
    for f in ("lora.pt", "extra.pt", "DONE"):
        (ck / f).write_bytes(b"")
    win = d / "windows.jsonl"
    win.write_text(json.dumps({"id": "wA", "close_detect_ms": 5000}) + "\n", encoding="utf-8")
    return ck, win


def run_main(argv: list[str], read_stub=None) -> dict:
    """fp/dc patches are class attributes of E.Engine: save the raw descriptors and restore them afterwards."""
    saved = (vars(E.Engine)["read"], vars(E.Engine)["from_merged"], od.main, dc.apply_ckpt)
    got = {"applied": []}
    E.Engine.from_merged = classmethod(lambda cls, path, direction, max_gib=40: SimpleNamespace(path=path))
    dc.apply_ckpt = lambda eng, ck: got["applied"].append((eng.path, str(ck))) or {"ckpt": str(ck)}
    if read_stub is not None:
        E.Engine.read = read_stub
    got["read_before"] = vars(E.Engine)["read"]

    def fake_od_main():
        got["argv"] = list(sys.argv)
        got["read"] = vars(E.Engine)["read"]
        got["eng"] = E.Engine.from_merged("T", "zh2en", 120)

    od.main = fake_od_main
    sys.argv = ["x"] + argv
    try:
        dcf.main()
    finally:
        E.Engine.read, E.Engine.from_merged, od.main, dc.apply_ckpt = saved
    return got


def main():
    ck, win = fixtures()
    base = ["--thinker", "T", "--windows", str(win), "--out", "o.jsonl", "--policy", "wait", "--loop-guard-k", "6"]
    orig_read, orig_fm = vars(E.Engine)["read"], vars(E.Engine)["from_merged"]

    # --- off
    for label, argv in (("no flag", ["--crf-ckpt", str(ck)] + base),
                        ("'--final-read-post off'", base[:2] + ["--final-read-post", "off", "--crf-ckpt", str(ck)] + base[2:])):
        g = run_main(argv)
        check(f"off ({label}): online_decode.main sees the plain argv", g["argv"] == ["x"] + base)
        check(f"off ({label}): Engine.read not patched", g["read"] is g["read_before"] is orig_read)
        check(f"off ({label}): checkpoint applied once to the loaded engine", g["applied"] == [("T", str(ck))]
              and g["eng"].path == "T")
    check("class attributes restored", vars(E.Engine)["read"] is orig_read and vars(E.Engine)["from_merged"] is orig_fm)

    # --- close_detect
    seen = []

    def read_stub(self, items, *a, **k):
        seen.append((items, a, k))
        return "r"

    n0 = fp.STATS["final"]
    g = run_main(["--final-read-post", "close_detect", "--crf-ckpt", str(ck)] + base, read_stub=read_stub)
    check("close_detect: online_decode.main sees the plain argv", g["argv"] == ["x"] + base)
    check("close_detect: Engine.read wraps the original", g["read"] is not read_stub
          and getattr(g["read"], "__wrapped__", None) is read_stub)
    check("close_detect: checkpoint applied once", g["applied"] == [("T", str(ck))])
    rng = np.random.default_rng(0)
    post = (rng.random((15, 8)) + 0.5).astype(np.float32)
    wav = np.zeros(1150 * 16, np.float32)
    it_nf = E.ReadItem(wav=wav, post=post, t_ms=1000, final=False, committed=[], akey="wA|1000")
    it_f = E.ReadItem(wav=wav, post=post, t_ms=1150, final=True, committed=[11], akey="wA|1150")
    r = g["read"](None, [it_nf, it_f], max_new=4)
    items, a, k = seen[-1]
    cut = fp.final_cut_ms(1150, 5000)
    check("close_detect: cut = t for a late close (1150 ms)", cut == 1150)
    check("close_detect: original read called once with the same extra args", r == "r" and len(seen) == 1
          and k == {"max_new": 4} and a == ())
    check("close_detect: non-final item passed as the same object", items[0] is it_nf)
    f = items[1]
    check("close_detect: final item P zeroed from cut // 80, kept before",
          np.array_equal(f.post[: cut // 80], post[: cut // 80]) and not f.post[cut // 80:].any())
    check("close_detect: final item full=True, everything else unchanged",
          f.full and f.wav is wav and f.t_ms == 1150 and f.final and f.committed == [11] and f.akey == "wA|1150")
    check("close_detect: the caller's item and P untouched", not it_f.full and np.array_equal(it_f.post, post))
    check("close_detect: stats count the final read", fp.STATS["final"] == n0 + 1)
    check("class attributes restored", vars(E.Engine)["read"] is orig_read and vars(E.Engine)["from_merged"] is orig_fm)

    # --- errors
    for label, argv in (("bad mode", ["--final-read-post", "full", "--crf-ckpt", str(ck)] + base),
                        ("missing --crf-ckpt", ["--final-read-post", "close_detect"] + base),
                        ("--lora with --crf-ckpt", ["--crf-ckpt", str(ck), "--lora", "L"] + base)):
        try:
            run_main(argv)
            check(f"{label} rejected", False)
        except SystemExit:
            check(f"{label} rejected", True)
    check("class attributes restored", vars(E.Engine)["read"] is orig_read and vars(E.Engine)["from_merged"] is orig_fm)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
