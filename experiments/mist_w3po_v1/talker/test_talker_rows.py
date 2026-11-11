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


W1 = {"M_on_0000": {"open_detect_ms": 900, "close_detect_ms": 1600, "dur_ms": 1500}}
PLEM_GOLDEN = "b7c999d52cdd32ae17b6ffe8396140dec1646affca2410f8b1243e252a74cad1"   # plem_rows before the v2 additions


def test_plem_default_bit_identical():
    """talker_crf_prereg_v2's additions leave the default mode's rows bit-identical (hash taken before the change)."""
    import hashlib
    import json

    blocks, stats = T.plem_rows([dump_item()], W1)
    s = json.dumps({"blocks": blocks, "stats": stats}, sort_keys=True)
    assert hashlib.sha256(s.encode()).hexdigest() == PLEM_GOLDEN


def test_block_rows_reproduces_plem():
    it = dump_item()
    blocks, stats = T.plem_rows([it], W1)
    st = T.stream_of(it, W1["M_on_0000"])
    beta = st.forward_backward(stats["lambda"])[3]
    states = st.path_states([b["chosen"] for b in it["blocks"]])
    for bk in blocks:
        q = st.conditional(bk["r"], states[bk["r"]], stats["lambda"], beta)
        assert T.block_rows(it, bk["r"], q) == bk


def test_bon_rows():
    it = dump_item()
    blocks, st = T.bon_rows([it])
    assert st["blocks"] == 2 and st["weights"] == "bon"
    for bk in blocks:
        b = it["blocks"][bk["r"]]
        x = np.asarray([T.feat(c) for c in b["cands"]])
        q = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        keep = q >= T.Q_MIN
        assert len(bk["cands"]) == keep.sum()
        assert np.allclose([c["w"] for c in bk["cands"]], q[keep] / q[keep].sum(), atol=1e-12)
        qv = np.zeros(T.N_SPEED)
        for c, w in zip(b["cands"], q):
            qv[c["v"] - T.SPEED_0] += w
        assert np.allclose(bk["qv"], qv, atol=1e-12)
        # same sequences as PL-EM's rows (the dump's per-stream backlog token)
        assert bk["cands"][0]["own"][: bk["spd_pos"]] == T.block_rows(it, bk["r"], np.ones(4) / 4)["cands"][0]["own"][: bk["spd_pos"]]


def meeting_items():
    """two speakers of one window and one of a later window, one meeting; A' path durations make a shared backlog."""
    sp = T.SPEED_0
    a = dump_item()
    b = dump_item()
    b["id"] = "M_on_0000__S2"
    b["blocks"][0]["write_ms"] = 1100
    b["blocks"][1]["write_ms"] = 1300
    c = dump_item()
    c["id"] = "M_on_0001__S1"
    c["blocks"] = c["blocks"][:1]
    c["blocks"][0]["final"] = True
    c["blocks"][0]["write_ms"] = 400
    for it in (a, b, c):
        for x in it["blocks"]:
            x["speed_base"] = sp + 5
    W = {"M_on_0000": {"meeting": "M", "start_s": 10.0, "open_detect_ms": 900, "close_detect_ms": 1600, "dur_ms": 1500},
         "M_on_0001": {"meeting": "M", "start_s": 12.0, "open_detect_ms": 300, "close_detect_ms": 900, "dur_ms": 800}}
    return [a, b, c], W


def test_meeting_rows():
    from experiments.mist_w3po_v1.talker import shared_channel as SC

    items, W = meeting_items()
    blocks, st = T.plem_meeting_rows(items, W)
    assert st["meetings"] == 1 and st["chain_len_max"] == 5 and abs(st["budget_s"] - (st["aprule_latency_1ch_s"] - 1.0)) < 1e-9
    assert st["blocks"] == 5
    bl = SC.shared_backlogs(items, W, T.chosen_durs(items))
    # FIFO: S1 block 0 at 10.9 s (window 0: max(1000, 900) = 1000 -> 11.0 s), ...; the backlog token is the shared one
    for bk in blocks:
        n = [it["id"] for it in items].index(bk["id"])
        r = bk["r"]
        for c in bk["cands"]:
            assert c["own"][bk["spd_pos"] - 1] == T.TASK_TTS
            own_bins = [c["own"][i + 1] for i in range(len(c["own"]) - 1) if c["own"][i] == items[n]["blocks"][0]["speed_base"]
                        and c["own"][i + 1] in T.BACKLOG_IDS]
            assert own_bins == [T.BACKLOG_IDS[SC.backlog_bin(bl[(n, k)])] for k in range(r + 1)]
        assert abs(sum(c["w"] for c in bk["cands"]) - 1) < 1e-9 and abs(sum(bk["qv"]) - 1) < 1e-9
    assert max(bl.values()) > 0   # the shared channel queues (per stream it would not for the second speaker)


MEETING_GOLDEN = "5af25abb55dd5d22b8fd9b838392ea6c917b185ee27c1f3bb4e8afdb02d6de16"   # plem_meeting_rows before v3


def test_meeting_default_bit_identical():
    """talker_crf_prereg_v3's additions leave the --scope meeting default rows bit-identical (hash taken before)."""
    import hashlib
    import json

    items, W = meeting_items()
    blocks, stats = T.plem_meeting_rows(items, W)
    s = json.dumps({"blocks": blocks, "stats": stats}, sort_keys=True)
    assert hashlib.sha256(s.encode()).hexdigest() == MEETING_GOLDEN


def test_meeting_budget_ref():
    """E2' (talker_crf_prereg_v3): budget = the reference path's proxy on the meeting lattice - drop."""
    items, W = meeting_items()
    ap = {it["id"]: [b["chosen"] for b in it["blocks"]] for it in items}
    # the A' path as the reference with the default drop reproduces the default lattice exactly
    c0, s0, l0, st0 = T.meeting_lattice(items, W)
    c1, s1, l1, st1 = T.meeting_lattice(items, W, ref_paths=ap)
    assert l0 == l1 and st1["ref_latency_1ch_s"] == st0["aprule_latency_1ch_s"] and st1["budget_s"] == st0["budget_s"]
    assert {k: v for k, v in st1.items() if k in st0} == st0
    # another reference path and drop: budget = its proxy - drop, lambda meets it (or hits a bound)
    ref = {it["id"]: [0 for _ in it["blocks"]] for it in items}
    ch, ss, lam, st = T.meeting_lattice(items, W, drop=0.5, ref_paths=ref)
    words = sum(sum(x.words) for x in ss)
    lat = sum(x.path_latency([ref[items[n]["id"]][r] for n, r in c]) for c, x in zip(ch.values(), ss)) / words
    assert abs(st["ref_latency_1ch_s"] - round(lat, 4)) < 1e-12 and abs(st["budget_s"] - round(lat - 0.5, 4)) < 1e-12
    assert st["budget_ref"] == "stream_viterbi" and st["drop_1ch_s"] == 0.5
    e = T.PL.corpus_latency(ss, lam)
    assert lam in (0.0, 50.0) or abs(e - (lat - 0.5)) < 1e-3
    # stream_viterbi_paths = each stream's own Viterbi under the per-stream lattice
    paths, rst = T.stream_viterbi_paths(items, W)
    streams, lam_s, _ = T.lattice(items, W)
    assert rst["stream_lambda"] == lam_s
    assert paths == {it["id"]: x.viterbi(lam_s) for it, x in zip(items, streams)}
    # rows with the reference: same construction, weights from the new lambda
    blocks, stb = T.plem_meeting_rows(items, W, ref_paths=paths, drop=0.5)
    assert stb["budget_ref"] == "stream_viterbi" and stb["blocks"] == 5
    for bk in blocks:
        assert abs(sum(c["w"] for c in bk["cands"]) - 1) < 1e-9


if __name__ == "__main__":
    test_seq_equals_infer()
    test_plem_rows()
    test_cover_pick_and_align()
    test_plem_default_bit_identical()
    test_block_rows_reproduces_plem()
    test_bon_rows()
    test_meeting_rows()
    test_meeting_default_bit_identical()
    test_meeting_budget_ref()
    print("ok")
