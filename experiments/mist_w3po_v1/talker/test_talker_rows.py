"""Unit tests for talker_rows: the sequences equal si_infer.build_seq's, PL-EM rows / weights, cover pick."""
import numpy as np

from experiments.mist_w3po_v1.talker import talker_rows as T
from experiments.uniss_qwen3_omni_v1.evaluation import si_infer as S


def cand(o, v, n, lp, nll, cover, closing=180395):
    return {"o": o, "v": v, "codes": list(range(n)), "closing": closing, "lp": lp, "ntok": n + 1, "nll": nll,
            "cover": cover, "forced": False}


def dump_item():
    sp = T.SPEED_0
    b0 = {"write_ms": 1000, "text": "hello there friend", "final": False, "glm": [1, 2, 3], "glm_end": 2, "ids": [10, 11, 12],
          "speed_base": sp + 11, "speed": sp + 12, "backlog_bin": 0, "chosen": 1,
          "cands": [cand(0, sp + 12, 50, -100, 0.5, 1.0), cand(0, sp + 12, 40, -90, 0.3, 1.0),
                    cand(2, sp + 14, 30, -80, 2.0, 0.5), cand(2, sp + 14, 35, -85, 0.4, 1.0)]}
    b1 = {"write_ms": 1200, "text": "go", "final": True, "glm": [1, 2, 3, 4], "glm_end": 4, "ids": [13],
          "speed_base": sp + 2, "speed": sp + 8, "backlog_bin": 1, "chosen": 0,
          "cands": [cand(0, sp + 8, 20, -50, 0.2, 1.0, 180392), cand(0, sp + 8, 25, -55, 1.2, 0.0, 180392),
                    cand(2, sp + 10, 15, -40, 0.9, 1.0, 180392), cand(2, sp + 10, 18, -45, None, None, None)]}
    return {"id": "M_on_0000__S1", "voice": list(range(32)), "dur_ms": 1500, "blocks": [b0, b1]}


def as_infer(it):
    """the si_infer item the dump came from (history = the chosen candidates)."""
    bl = []
    for b in it["blocks"]:
        c = b["cands"][b["chosen"]]
        bl.append({**b, "look_ids": [], "codes": c["codes"], "closing": T.closing(c, b["final"])})
    return {"voice": it["voice"], "blocks": bl}


def test_seq_equals_infer():
    it = dump_item()
    inf = as_infer(it)
    for r in (0, 1):
        for v in (T.SPEED_0 + 5, T.SPEED_0 + 9):
            for backlog in (False, True):
                assert T.full_seq(it, r, v, backlog) == S.build_seq(inf, r, T.LANG_ENG, backlog, cur_speed=v)


def test_plem_rows():
    it = dump_item()
    W = {"M_on_0000": {"open_detect_ms": 900, "close_detect_ms": 1600, "dur_ms": 1500}}
    blocks, stats = T.plem_rows([it], W)
    assert stats["blocks"] == 2 and stats["aprule_latency_s"] > 0 and stats["lambda"] >= 0
    for bk in blocks:
        assert abs(sum(c["w"] for c in bk["cands"]) - 1) < 1e-9 and abs(sum(bk["qv"]) - 1) < 1e-9
        for c in bk["cands"]:
            assert c["own"][bk["spd_pos"]] == T.LANG_ENG and c["own"][bk["spd_pos"] - 1] == T.TASK_TTS
            assert c["own"][bk["spd_pos"] + 1] == c["ref"][-2 if bk["r"] == 1 else -3]
            assert len(c["own"]) - len(c["ref"]) == (bk["r"] + 1)   # one backlog token per block (the speed tokens stay)
    # the low-NLL candidate gets the most weight in block 0
    b0 = blocks[0]
    assert max(b0["cands"], key=lambda c: c["w"])["y"][:3] == [T.SEMANTIC_0 + i for i in range(3)]


def test_cover_pick_and_align():
    it = dump_item()
    assert T.cover_pick(it["blocks"][0]) == 1     # both offset-0 cover 1; NLL 0.3 < 0.5
    assert T.cover_pick(it["blocks"][1]) == 0
    blocks, st = T.align_rows([it])
    assert st["align_pos"] == 3 and st["align_neg"] == 1
    r0 = blocks[0]["rows"]
    assert [x["ce"] for x in r0] == [0, 1]
    x = r0[0]
    assert x["seq"][x["text_start"] - 1] == T.START_CONTENT and x["seq"][x["text_start"]: x["text_start"] + x["M"]] == [10, 11, 12]


if __name__ == "__main__":
    test_seq_equals_infer()
    test_plem_rows()
    test_cover_pick_and_align()
    print("ok")
