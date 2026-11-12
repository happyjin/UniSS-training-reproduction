"""Unit tests for crf/decode_crf_ckpt_specclose.py (CPU; online_decode.main, the merged-model loader and apply_ckpt are
stubbed as in test_decode_crf_ckpt_finalpost, so no model is loaded):

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_decode_crf_ckpt_specclose

  field     online_decode.main gets the plain argv; Engine.read wraps the original; a final item reaches it with P zeroed
            from max(t - 1040, min(t, final_post_cut_ms)) // 80 and full=True; a non-final item is the same object
  errors    a --final-read-post flag, a window without final_post_cut_ms, missing --crf-ckpt -> SystemExit
Fixture files go to a new tempfile.mkdtemp dir; nothing is deleted.
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
from experiments.mist_w3po_v1.crf import decode_crf_ckpt_specclose as dcs
from experiments.mist_w3po_v1.evaluation import online_decode as od
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond, name


def fixtures():
    d = Path(tempfile.mkdtemp(prefix="test_dcs_"))
    ck = d / "ckpt" / "step_00001"
    ck.mkdir(parents=True)
    for f in ("lora.pt", "extra.pt", "DONE"):
        (ck / f).write_bytes(b"")
    win = d / "windows.jsonl"
    win.write_text(json.dumps({"id": "wA", "close_detect_ms": 5000, "final_post_cut_ms": 720}) + "\n"
                   + json.dumps({"id": "wB", "close_detect_ms": 5000, "final_post_cut_ms": 4000}) + "\n", encoding="utf-8")
    bad = d / "bad.jsonl"
    bad.write_text(json.dumps({"id": "wA", "close_detect_ms": 5000}) + "\n", encoding="utf-8")
    return ck, win, bad


def run_main(argv: list[str], read_stub=None) -> dict:
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
        dcs.main()
    finally:
        E.Engine.read, E.Engine.from_merged, od.main, dc.apply_ckpt = saved
    return got


def main():
    ck, win, bad = fixtures()
    base = ["--thinker", "T", "--windows", str(win), "--out", "o.jsonl", "--policy", "wait", "--loop-guard-k", "6"]
    orig_read, orig_fm = vars(E.Engine)["read"], vars(E.Engine)["from_merged"]
    seen = []

    def read_stub(self, items, *a, **k):
        seen.append((items, a, k))
        return "r"

    g = run_main(["--crf-ckpt", str(ck)] + base, read_stub=read_stub)
    check("field: online_decode.main sees the plain argv", g["argv"] == ["x"] + base)
    check("field: Engine.read wraps the original", getattr(g["read"], "__wrapped__", None) is read_stub)
    check("field: checkpoint applied once", g["applied"] == [("T", str(ck))])
    rng = np.random.default_rng(0)
    post = (rng.random((30, 8)) + 0.5).astype(np.float32)
    wav = np.zeros(2300 * 16, np.float32)
    nf = E.ReadItem(wav=wav, post=post, t_ms=1000, final=False, committed=[], akey="wA|1000")
    fa = E.ReadItem(wav=wav, post=post, t_ms=2300, final=True, committed=[11], akey="wA|2300")
    fb = E.ReadItem(wav=wav, post=post, t_ms=2300, final=True, committed=[], akey="wB|2300")
    g["read"](None, [nf, fa, fb], max_new=4)
    items, a, k = seen[-1]
    check("field: non-final item passed as the same object", items[0] is nf)
    cut_a = max(0, 2300 - E.LOOKAHEAD_MS, min(2300, 720))      # 1260: never earlier than the engine's own t - 1.04 s
    cut_b = max(0, 2300 - E.LOOKAHEAD_MS, min(2300, 4000))     # 2300: every in-window frame
    check("field: final cut = max(t - 1040, min(t, final_post_cut_ms))",
          fp.final_cut_ms(2300, 720 + E.LOOKAHEAD_MS) == cut_a == 1260 and fp.final_cut_ms(2300, 4000 + E.LOOKAHEAD_MS) == cut_b)
    for it, cut in ((items[1], cut_a), (items[2], cut_b)):
        check(f"field: final item {it.akey} P zeroed from {cut // 80}, kept before, full=True",
              np.array_equal(it.post[: cut // 80], post[: cut // 80]) and not it.post[cut // 80:].any() and it.full)
    check("field: caller's items untouched", not fa.full and np.array_equal(fa.post, post))
    check("class attributes restored", vars(E.Engine)["read"] is orig_read and vars(E.Engine)["from_merged"] is orig_fm)
    for label, argv in (("--final-read-post flag", ["--final-read-post", "close_detect", "--crf-ckpt", str(ck)] + base),
                        ("window without final_post_cut_ms", ["--crf-ckpt", str(ck)] + base[:3] + [str(bad)] + base[4:]),
                        ("missing --crf-ckpt", base)):
        try:
            run_main(argv)
            check(f"{label} rejected", False)
        except SystemExit:
            check(f"{label} rejected", True)
    check("class attributes restored", vars(E.Engine)["read"] is orig_read and vars(E.Engine)["from_merged"] is orig_fm)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
