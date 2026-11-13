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


def test_min_free_flag():
    src = Path(W.__file__).read_text()
    assert '"--min-free-gib", type=float, default=0.0' in src
    assert "mem[c][1] - mem[c][0] < a.min_free_gib * 1024" in src   # default 0: never true (used <= total)


def test_warmup_flag():
    src = Path(W.__file__).read_text()
    assert '"--warmup-s", type=float, default=0.0' in src and "if a.warmup_s > 0:" in src
    import stack_cards as S
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "s"
        f.write_text("")
        assert S.read_cards(str(f)) == set()          # empty file: stack mode on, no card yet
        f.write_text("5 6 7\n")
        assert S.read_cards(str(f)) == {5, 6, 7}



def test_phys():
    # 修正 4: --phys counts the held-out loss jobs as stacked jobs; never a CRF / Megatron training
    assert W.is_decode("python -m experiments.mist_w3po_v1.crf.heldout_loss --base x", True)
    assert not W.is_decode("python -m experiments.mist_w3po_v1.crf.heldout_loss --base x")
    assert not W.is_decode("python -u experiments/mist_w3po_v1/crf/train_crf_sft.py --objective serial_v2p", True)
    assert W.is_decode("python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost --crf-ckpt x", True)
    src = Path(W.__file__).read_text()
    assert '"--phys", action="store_true"' in src and '"--file-cards", nargs="+", type=int, default=None' in src



def test_pair_flag():
    # 修正 5: --pair writes "phys pair"; without --pair the token string is exactly the old one
    src = (Path(__file__).resolve().parent / "stack47_watchdog.py").read_text()
    assert '"--pair", action="store_true"' in src and 'raise SystemExit("--pair needs --phys")' in src
    assert 'tok = (" phys" if a.phys else "") + (" pair" if a.pair else "")' in src
    for phys, pair, want in ((False, False, ""), (True, False, " phys"), (True, True, " phys pair")):
        assert (" phys" if phys else "") + (" pair" if pair else "") == want

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
