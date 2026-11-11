"""Unit tests for silence_share (S1): runs and kinds on synthetic blocks, measure on a fake talker dir, retime."""
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import silence_share as S


def tone(ms, amp=0.3, f=220.0):
    t = np.arange(ms * 16) / 16000.0
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def quiet(ms, amp=1e-4):
    return np.full(ms * 16, amp, dtype=np.float32)   # -80 dBFS: 20*log10(1e-4)


def test_runs():
    x = np.concatenate([quiet(200), tone(300), quiet(100), tone(200), quiet(400), tone(100), quiet(160)])
    r = S.silence_runs(x)
    assert r == [(0, 200, "lead"), (800, 1200, "internal"), (1300, 1460, "trail")]   # the 100 ms gap is < 150 ms
    assert S.new_dur(1460, r) == 1460 - 50 - 250 - 10
    # -39 dB below the peak is not silent
    assert S.silence_runs(np.concatenate([tone(200), tone(300, amp=0.3 * 10 ** (-39 / 20))])) == []
    assert S.silence_runs(np.concatenate([tone(200), tone(300, amp=0.3 * 10 ** (-41 / 20))])) == [(200, 500, "trail")]
    z = np.zeros(16 * 300, dtype=np.float32)
    assert S.block_runs(z) == [(0, 300, "whole")] and S.new_dur(300, S.block_runs(z)) == 150


def test_measure_retime():
    with tempfile.TemporaryDirectory() as d:
        import soundfile as sf

        b0 = np.concatenate([tone(400), quiet(300)])          # trail 300
        b1 = np.concatenate([quiet(200), tone(500)])          # lead 200
        sf.write(f"{d}/a_speech.wav", np.concatenate([b0, b1]), 16000)
        rows = [{"id": "W__S1", "speech_path": f"{d}/a_speech.wav", "error": None,
                 "blocks": [{"dur_ms": 700, "write_ms": 0, "start_ms": 0}, {"dur_ms": 700, "write_ms": 0, "start_ms": 700}]},
                {"id": "W__S2", "error": "x", "blocks": []}]
        Path(f"{d}/t").mkdir()
        Path(f"{d}/t/results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        m = S.measure(f"{d}/t")
        assert m["blocks"] == 2 and m["speech_s"] == 1.4
        assert m["kinds"]["trail"]["total_s"] == 0.3 and m["kinds"]["lead"]["total_s"] == 0.2
        assert abs(m["all"]["share"] - 0.5 / 1.4) < 1e-3 and m["compressible_s"] == 0.2
        st = S.retime(f"{d}/t", m, f"{d}/o")
        assert st == {"blocks": 2, "cut_s": 0.2}
        out = [json.loads(l) for l in open(f"{d}/o/results.jsonl")]
        assert [b["dur_ms"] for b in out[0]["blocks"]] == [550, 650] and [b["dur_ms_orig"] for b in out[0]["blocks"]] == [700, 700]
        assert out[1] == rows[1]
        try:
            S.retime(f"{d}/t", m, f"{d}/o")
            raise AssertionError("overwrote")
        except SystemExit:
            pass


if __name__ == "__main__":
    test_runs()
    test_measure_retime()
    print("ok")
