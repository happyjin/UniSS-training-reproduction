"""CPU tests of stack47_watchdog.py: iteration-time parsing, the slow rule (n consecutive points over baseline x 1.4),
the error pattern, the decode / training cmdline split.

    python experiments/mist_w3po_v1/scripts/test_stack47_watchdog.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack47_watchdog as W  # noqa: E402

L = (" [2026-11-13 06:05:30.589539] iteration     3260/   15000 | consumed samples:       208640 | elapsed time per "
     "iteration (ms): 4793.8 | throughput per GPU (TFLOP/s/GPU): 4.9 | learning rate: 8.490517E-05 |")


def test_parse():
    m = W.IT_RE.search(L)
    assert m and m.group(1) == "3260" and float(m.group(2)) == 4793.8
    assert not W.ERR_RE.search(L)
    assert W.ERR_RE.search("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert W.ERR_RE.search("Traceback (most recent call last):")


def test_slow():
    b = 4650.0
    assert not W.slow_trip([(1, 7000), (2, 7000)], b, 0.4, 3)
    assert W.slow_trip([(1, 4700), (2, 6600), (3, 6700), (4, 6520)], b, 0.4, 3)
    assert not W.slow_trip([(2, 6600), (3, 6400), (4, 6700)], b, 0.4, 3)   # not consecutive
    assert not W.slow_trip([(2, 6510), (3, 6510), (4, 6510)], b, 0.4, 3)   # equal to the threshold: not over


def test_cmd_split():
    assert W.is_decode("python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost --crf-ckpt x")
    assert not W.is_decode("python -u experiments/uniss_qwen3_omni_v1/training/pretrain_thinker_gstar.py --x")
    assert not W.is_decode("python -m experiments.mist_w3po_v1.crf.heldout_loss --base x")


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
