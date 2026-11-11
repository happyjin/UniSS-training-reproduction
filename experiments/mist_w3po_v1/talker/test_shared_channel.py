"""Unit tests for the shared output channel (talker_crf_prereg_v2 experiment 2): FIFO order = speech_latency_lit's
online_1ch schedule, the meeting shards, and si_infer --backlog-scope meeting (plcrf_meeting_loop with a fake talker):
every block's backlog token is the shared channel's backlog from the durations synthesised before it."""
import types
from collections import defaultdict

import numpy as np

from experiments.mist_w3po_v1.talker import shared_channel as SC
from experiments.uniss_qwen3_omni_v1.evaluation import si_infer as S


def wins():
    return {"M_on_0000": {"id": "M_on_0000", "meeting": "M", "start_s": 10.0, "open_detect_ms": 900, "close_detect_ms": 1600,
                          "dur_ms": 1500},
            "M_on_0001": {"id": "M_on_0001", "meeting": "M", "start_s": 11.2, "open_detect_ms": 300, "close_detect_ms": 900,
                          "dur_ms": 800},
            "N_on_0000": {"id": "N_on_0000", "meeting": "N", "start_s": 0.0, "open_detect_ms": 100, "close_detect_ms": 700,
                          "dur_ms": 3000}}


def lit_1ch(items, W, durs):
    """speech_latency_lit's online_1ch FIFO (its loop, restated): -> (item index, block index) -> ch_start - avail (ms)."""
    blk, bym = {}, defaultdict(list)
    for n, it in enumerate(items):
        w = W[it["id"].split("__S")[0]]
        for i, b in enumerate(it["blocks"]):
            c = max(b["write_ms"], w["open_detect_ms"])
            if b["write_ms"] >= w["dur_ms"]:
                c = max(c, w["close_detect_ms"])
            av = w["start_s"] + c / 1000
            blk[(it["id"], i)] = {"avail": av, "dur": durs[(n, i)] / 1000, "n": n}
            bym[w["meeting"]].append((av, (it["id"], i)))
    out = {}
    for m, L in bym.items():
        prev = -1e9
        for av, kk in sorted(L, key=lambda x: (x[0], x[1][0], x[1][1])):
            b = blk[kk]
            st = max(av, prev)
            prev = st + b["dur"]
            out[(b["n"], kk[1])] = (st - av) * 1000
    return out


def items_meeting(rng):
    out = []
    for wid, spk, writes in (("M_on_0000", 1, [1000, 1200, 1500]), ("M_on_0000", 2, [950, 1500]),
                             ("M_on_0001", 1, [500, 800]), ("N_on_0000", 3, [400, 900, 3000])):
        out.append({"id": f"{wid}__S{spk}", "blocks": [{"write_ms": w, "text": "a b c", "final": i == len(writes) - 1}
                                                       for i, w in enumerate(writes)]})
    return out


def test_fifo_matches_latency_lit():
    rng = np.random.default_rng(0)
    W = wins()
    items = items_meeting(rng)
    durs = {(n, r): int(rng.integers(5, 200)) * 20 for n, it in enumerate(items) for r in range(len(it["blocks"]))}
    ref = lit_1ch(items, W, durs)
    got = SC.shared_backlogs(items, W, durs)
    assert set(got) == set(ref)
    for k in ref:
        assert abs(got[k] - ref[k]) < 1e-6, (k, got[k], ref[k])
    q = SC.meeting_queues(items, W)
    for L in q.values():   # a stream's own blocks stay in order
        seen = defaultdict(lambda: -1)
        for _, n, r in L:
            assert r == seen[n] + 1
            seen[n] = r


def test_meeting_shards():
    W = wins()
    items = items_meeting(None)
    sh = SC.meeting_shard_of(items, W, 2)
    assert sorted(sh) == ["M", "N"] and set(sh.values()) == {0, 1}
    assert SC.meeting_shard_of(items, W, 8) == SC.meeting_shard_of(list(reversed(items)), W, 8)
    assert SC.backlog_bin(0) == 0 and SC.backlog_bin(499) == 0 and SC.backlog_bin(500) == 1 and SC.backlog_bin(9e9) == 7


class FakeOut:
    def __init__(self, ids, lp):
        self.token_ids, self.cumulative_logprob = ids, lp


class FakeTalker:
    """returns n outputs per request; code count from the request's seed; records the prompts."""

    def __init__(self):
        self.prompts = []

    def generate(self, reqs, params, use_tqdm=False):
        res = []
        for rq, p in zip(reqs, params):
            self.prompts.append(rq["prompt_token_ids"])
            outs = []
            for j in range(p.n):
                L = 10 + (p.seed * 7 + j * 13) % 90
                outs.append(FakeOut([S.SEMANTIC_0 + 5] * L + [S.WAIT_READ], -1.0 * L))
            res.append(types.SimpleNamespace(outputs=outs))
        return res


def fake_params(**k):
    return types.SimpleNamespace(**k)


def infer_item(it, rng):
    sp = 180337
    bl = []
    for i, b in enumerate(it["blocks"]):
        bl.append({**b, "glm": list(range(20)), "glm_end": 5 * (i + 1), "ids": [10 + i, 11], "look_ids": [],
                   "speed_base": sp + 9, "speed": sp + 9})
    return {"id": it["id"], "voice": list(range(32)), "blocks": bl}


def test_meeting_loop_shared_backlog_token():
    rng = np.random.default_rng(1)
    W = wins()
    items = [infer_item(it, rng) for it in items_meeting(rng)]
    args = types.SimpleNamespace(speed_policy="queue", queue_shift_max=6, queue_shift_per_s=3.0, final_shift=2,
                                 self_speed=False, cand_k=4, select_k=2, seed=20261021, temperature=0.7, history_ban=8,
                                 backlog_token=True, scorer_url="", cand_cover=False)
    orig = S.score_candidates
    S.score_candidates = lambda cands, owners, r, *a, **k: [[(float(len(c)), None) for c in cl] for cl in cands]
    try:
        talker = FakeTalker()
        S.plcrf_meeting_loop(items, W, talker, None, None, None, args, 180373, "eng", [],
                             fake_params, None)
    finally:
        S.score_candidates = orig
    durs = {(n, r): len(b["codes"]) * 20 for n, it in enumerate(items) for r, b in enumerate(it["blocks"])}
    ref = lit_1ch(items, W, durs)
    for (n, r), v in ref.items():
        b = items[n]["blocks"][r]
        assert abs(b["backlog_ms_shared"] - v) < 0.06, (n, r, b["backlog_ms_shared"], v)
        assert b["backlog_bin"] == SC.backlog_bin(v)
        # b1: the shorter of the two samples (fake NLL = length)
        assert b["closing"] == S.WAIT_READ and b["select_nll"] == min(b["cand_nll"])
    # each prompt's last backlog token is the current block's shared bin
    for n, it in enumerate(items):
        for r, b in enumerate(it["blocks"]):
            seq = S.build_seq(it, r, 180373, True, cur_speed=b["speed"])
            assert seq[[i for i, t in enumerate(seq) if t in S.BACKLOG_IDS][-1]] == S.BACKLOG_IDS[b["backlog_bin"]]
    assert max(v for v in ref.values()) > 500   # the shared channel queues across speakers


if __name__ == "__main__":
    test_fifo_matches_latency_lit()
    test_meeting_shards()
    test_meeting_loop_shared_backlog_token()
    print("ok")
