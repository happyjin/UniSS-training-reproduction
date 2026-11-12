"""Unit test of talker_rows --lang cmn --units char (en2zh, CPU): with the module set to cmn every sequence equals
si_infer.build_seq with LANG_TOKEN["cmn"] (the sampling run's own sequences), the PL-EM rows carry the cmn token at the
TTS head, W_n counts zh_units; the defaults (eng / word) give the unchanged rows (test_talker_rows passes as well).

    PYTHONPATH=. python -m experiments.mist_w3po_v1.talker.test_talker_rows_cmn
"""
from experiments.mist_w3po_v1.talker import talker_rows as T
from experiments.mist_w3po_v1.talker import test_talker_rows as TT
from experiments.uniss_qwen3_omni_v1.evaluation import si_infer as S
from experiments.uniss_qwen3_omni_v1.training.e2_batch import LANG_TOKEN

W = {"M_on_0000": {"open_detect_ms": 900, "close_detect_ms": 1600, "dur_ms": 1500}}


def zh_item():
    it = TT.dump_item()
    it["blocks"][0]["text"] = "我们，开个会吧。"     # 6 units
    it["blocks"][1]["text"] = "用PPT"               # 2 units
    return it


def main() -> None:
    assert T.LANG == T.LANG_ENG and T.UNITS == "word" and T.LANG_CMN == LANG_TOKEN["cmn"]
    it = zh_item()
    base_rows, base_stats = T.plem_rows([TT.dump_item()], W)
    st_word = T.stream_of(it, W["M_on_0000"])
    try:
        T.LANG, T.UNITS = T.LANG_CMN, "char"
        inf = TT.as_infer(it)
        for r in (0, 1):
            for v in (T.SPEED_0 + 5, T.SPEED_0 + 9):
                for backlog in (False, True):
                    assert T.full_seq(it, r, v, backlog) == S.build_seq(inf, r, LANG_TOKEN["cmn"], backlog, cur_speed=v)
                    assert T.LANG_ENG not in T.full_seq(it, r, v, backlog)
        st = T.stream_of(it, W["M_on_0000"])
        assert list(st.words) == [6, 2], st.words
        rows, stats = T.plem_rows([it], W)
        for bk in rows:
            for c in bk["cands"]:
                assert c["own"][bk["spd_pos"]] == T.LANG_CMN and c["ref"].count(T.LANG_ENG) == 0
    finally:
        T.LANG, T.UNITS = T.LANG_ENG, "word"
    assert list(st_word.words) == [len(b["text"].split()) for b in it["blocks"]]
    rows2, stats2 = T.plem_rows([TT.dump_item()], W)
    assert stats2 == base_stats and [b["cands"] for b in rows2] == [b["cands"] for b in base_rows]
    print("ok")


if __name__ == "__main__":
    main()
