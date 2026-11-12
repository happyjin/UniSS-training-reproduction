"""CPU test: thinker_t_eval_orch --plain-thinker (v8 baseline decode) vs the default (bit-identical)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thinker_t_eval_orch as T  # noqa: E402

CK = "eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z/ckpt/step_03000"


def cmds(ckpt):
    return [j.cmd(1) for j in T.build_jobs("r", "n", ckpt, 1.0, "none", [])]


def test_default_unchanged_and_plain():
    T.PLAIN = False
    base = cmds(CK)
    assert all("--crf-ckpt" in c for c in base if "online_decode" in c or "decode_crf" in c)
    T.PLAIN = True
    try:
        plain = cmds(CK)
    finally:
        T.PLAIN = False
    assert cmds(CK) == base
    d0 = [c for c in plain if "online_decode_finalpost" in c]
    assert len(d0) == 8
    for c, b in zip(d0, [c for c in base if "decode_crf_ckpt_finalpost" in c]):
        assert "--crf-ckpt" not in c and "--final-read-post close_detect --thinker" in c
        assert c.replace("experiments.mist_w3po_v1.evaluation.online_decode_finalpost --final-read-post close_detect ",
                         "X ") == b.replace(f"experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost --crf-ckpt {CK} "
                                            "--final-read-post close_detect ", "X ")
    assert [c for c in plain if "online_decode" not in c] == [c for c in base if "decode_crf" not in c]


if __name__ == "__main__":
    test_default_unchanged_and_plain()
    print("ok")
