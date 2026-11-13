"""CPU tests of stack_cards.py (--stack-cards of thinker_t_eval_orch.py): shard-decode matching, the order gate,
eligibility (one decode per card, 135 GiB by real use), the plan split on a fake orchestrator (other jobs planned as
before; shard decodes only on the stack cards; FILE absent -> exactly the original plan), and the default CLI path
(no --stack-cards: nothing applied).

    python experiments/mist_w3po_v1/scripts/test_stack_cards.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack_cards as S  # noqa: E402


class J:
    def __init__(self, jid, kind="dec", prio=1, order=0):
        self.id, self.kind, self.prio, self.order = jid, kind, prio, order
        self.state, self.gpu, self.card = "wait", kind != "cpu", None


class FO:
    LOCK = "/dev/null"
    NEED = {"dec": 64.0}
    pend: list = []

    @staticmethod
    def ledger_pending(_):
        return FO.pend

    @staticmethod
    def ledger_append(*a, **k):
        pass


class O:
    def __init__(self, jobs):
        self.jobs, self.ledger_owner, self.ledger, self.logs = jobs, "joint_x", None, []
        self.planned = None

    def plan(self, snap, ready):
        self.planned = list(ready)
        return [(j, 0) for j in ready]

    def launch_gpu(self, plan):
        pass

    def log(self, m):
        self.logs.append(m)


def snap(used):
    return {"cards": {c: {"used": u, "total": 140.4} for c, u in used.items()}}


def test_match():
    assert S.is_shard_decode("s1z_dev5_x_s03000__dec3")
    assert not S.is_shard_decode("s1z_dev5_x_s03000__dec_heldout")
    assert not S.is_shard_decode("s1z_dev5_x_s03000__score")


def test_gate():
    t = 1000.0
    w = {"a": {"rank": 3000, "n": 4, "t": t}, "b": {"rank": 2500, "n": 4, "t": t}, "me": {"rank": 500, "n": 4, "t": t}}
    assert S.gate_open(w, "me", 500, "newest", t) == (False, "a")
    assert S.gate_open(w, "a", 3000, "newest", t)[0]
    assert S.gate_open(w, "me", 500, "oldest", t)[0]
    assert not S.gate_open(w, "a", 3000, "oldest", t)[0]
    w["ref"] = {"rank": 1e9, "n": 1, "t": t}
    assert not S.gate_open(w, "a", 3000, "newest", t)[0] and not S.gate_open(w, "me", 500, "oldest", t)[0]
    w["ref"]["t"] = t - 200   # stale
    assert S.gate_open(w, "a", 3000, "newest", t)[0]
    w["a"]["n"] = 0           # nothing ready
    assert S.gate_open(w, "b", 2500, "newest", t)[0]
    assert S.gate_open({"z": {"rank": 3000, "n": 4, "t": t}}, "e", 3000, "newest", t)[0]   # equal rank: both go
    assert not S.gate_open({"x": {"rank": 500, "n": 1, "t": t}}, "ap", -1e9, "oldest", t)[0]


def test_eligible():
    tr = {4: [(1, 70.3, "train")]}
    assert S.eligible(4, tr, [], set(), 140.4, 70.3) == (True, "")
    assert not S.eligible(4, tr, [], set(), 140.4, 71.5)[0]          # 135 - 71.5 < 64
    assert not S.eligible(4, {4: tr[4] + [(2, 62.0, "other")]}, [], set(), 140.4, 70.3)[0]
    assert not S.eligible(4, tr, [{"card": 4}], set(), 140.4, 70.3)[0]
    assert not S.eligible(4, tr, [], {4}, 140.4, 70.3)[0]
    assert S.eligible(4, {4: tr[4] + [(3, 1.0, "placeholder")]}, [], set(), 140.4, 70.3)[0]


def test_wants_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "wants.jsonl"
        for n in (3, 2):
            S.append_line(p, {"owner": "a", "rank": 1, "n": n, "t": 1})
        S.append_line(p, {"owner": "b", "rank": 2, "n": 1, "t": 1})
        w = S.read_wants(p)
        assert w["a"]["n"] == 2 and w["b"]["n"] == 1


def run_plan(stack_text, procs, group_wants=None, rank=3000.0):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stack"
        if stack_text is not None:
            f.write_text(stack_text)
        g = Path(d) / "g"
        g.mkdir()
        for r in group_wants or []:
            S.append_line(g / "wants.jsonl", r)
        jobs = [J("n__dec0", order=1), J("n__dec1", order=2), J("n__dec_heldout", order=3), J("n__talk0", "talk", 2, 4)]
        o = O(jobs)
        orig = S.card_procs
        S.card_procs = lambda: procs
        try:
            S.apply_stack_cards(o, str(f), rank, str(g), FO, "led")
            out = o.plan(snap({0: 10.0, 4: 70.3, 5: 70.0, 6: 70.5}), list(jobs))
        finally:
            S.card_procs = orig
        return o, out


def test_plan_split():
    procs = {4: [(1, 70.3, "train")], 5: [(2, 70.0, "train"), (9, 62.0, "other")], 6: [(3, 70.5, "train")]}
    o, out = run_plan("4 5 6\n", procs)
    ids = {j.id for j in o.planned}
    assert ids == {"n__dec_heldout", "n__talk0"}                    # the original plan never sees shard decodes
    st = [(j.id, c) for j, c in out if j.id.startswith("n__dec") and j.id != "n__dec_heldout"]
    assert sorted(c for _, c in st) == [4, 6] and len(st) == 2      # card 5 has a decode; one per card
    # FILE absent -> exactly the original plan with every ready job
    o2, out2 = run_plan(None, procs)
    assert len(o2.planned) == 4 and out2 == [(j, 0) for j in o2.planned]
    # a live higher-ranked member with ready decodes -> no stack placement now
    t = time.time()
    o3, out3 = run_plan("4 5 6\n", procs, [{"owner": "other", "rank": 3500, "n": 2, "t": t}])
    assert all(c == 0 for _, c in out3)


def run_plan_phys(stack_text, procs, used):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stack"
        f.write_text(stack_text)
        g = Path(d) / "g"
        g.mkdir()
        jobs = [J("n__dec0", order=1), J("n__dec1", order=2), J("n__dec_heldout", order=3), J("n__talk0", "talk", 2, 4)]
        o = O(jobs)
        orig = S.card_procs
        S.card_procs = lambda *a: procs
        try:
            S.apply_stack_cards(o, str(f), 3000.0, str(g), FO, "led")
            out = o.plan(snap(used), list(jobs))
        finally:
            S.card_procs = orig
        return o, out


def test_phys_mode():
    # 修正 4: token "phys" -> limit total - 1.5 by real use, need 67 (shard) / 71 (held-out), held-out stackable
    assert S.job_need("n__dec0", False) == 64.0 and S.job_need("n__dec_heldout", False) == 64.0
    assert S.job_need("n__dec0", True) == 67.0 and S.job_need("n__dec_heldout", True) == 71.0
    assert S.stackable("n__dec_heldout", True) and not S.stackable("n__dec_heldout", False)
    tr = {0: [(1, 71.0, "train")]}
    assert S.eligible(0, tr, [], set(), 140.4, 71.0, 67.0, float("inf"), 1.5)[0]       # 140.4 - 1.5 - 71 = 67.9
    assert not S.eligible(0, tr, [], set(), 140.4, 72.0, 67.0, float("inf"), 1.5)[0]   # 66.9 < 67
    assert not S.eligible(0, tr, [], set(), 140.4, 71.5)[0]                            # default 135 cap unchanged
    procs = {c: [(c + 1, 0.0, "train")] for c in range(8)}
    used = {0: 60.0, 1: 71.0, 2: 72.0, 3: 66.0, 4: 80.0, 5: 80.0, 6: 80.0, 7: 80.0}
    o, out = run_plan_phys("0 1 2 3 4 5 6 7 phys\n", procs, used)
    assert {j.id for j in o.planned} == {"n__talk0"}                    # held-out goes the stack path too
    st = {j.id: c for j, c in out if j.id.startswith("n__dec")}
    # card 0 holds the talker of the original plan (own job); free: card 3 72.9, card 1 67.9, card 2 66.9 (no), 4-7 no;
    # greedy in job order, most free first; the held-out (71) finds no card left
    assert st == {"n__dec0": 3, "n__dec1": 1}, st
    o, out = run_plan_phys("0 1 2 3 4 5 6 7 phys\n", procs, {0: 60.0, 1: 60.0, 2: 60.0, 3: 60.0, 4: 90.0, 5: 90.0,
                                                             6: 90.0, 7: 90.0})
    st = {j.id: c for j, c in out if j.id.startswith("n__dec")}
    assert st == {"n__dec0": 3, "n__dec1": 2, "n__dec_heldout": 1}, st
    # a CRF training counts as the training only in phys mode
    assert S.PHYS_TRAIN_PAT == ("pretrain_", "train_crf_sft.py") and S.TRAIN_PAT == ("pretrain_",)


def test_cli_default_unchanged():
    src = (Path(__file__).resolve().parent / "thinker_t_eval_orch.py").read_text()
    assert "    if a.stack_cards:\n" in src and 'ap.add_argument("--stack-cards", default=""' in src
    # the module is imported and applied only inside "if a.stack_cards:" (default "")
    i = src.index("    if a.stack_cards:\n")
    assert "import stack_cards" not in src[:i] and src[i:].index("import stack_cards") < 200


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print("ok", k)
    print(f"{n}/{n} passed")
