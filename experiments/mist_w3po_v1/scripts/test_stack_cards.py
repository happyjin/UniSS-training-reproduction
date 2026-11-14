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
    assert S.PHYS_TRAIN_PAT == ("pretrain_", "train_crf_sft.py", "train_v2p_lat.py") and S.TRAIN_PAT == ("pretrain_",)



class JC(J):
    def __init__(self, jid, cmd, **k):
        super().__init__(jid, **k)
        self.cmd, self.sid = cmd, 123


def test_pair_mode():
    # 修正 5: tokens "phys pair" -> a shard decode on two cards (need PAIR_NEED_DEC each), held-out not stackable
    assert S.read_pair is not None and S.PAIR_NEED_DEC == 44.0 and S.PAIR_MAX_GIB == 34
    assert S.PAIR_HIST_S == 30 and S.HIST_S == 180                     # PHYS / default history unchanged
    assert S.stackable("n__dec0", True, True) and not S.stackable("n__dec_heldout", True, True)
    assert S.stackable("n__dec_heldout", True)                          # PHYS without pair unchanged
    procs = {c: [(c + 1, 0.0, "train")] for c in range(8)}
    used = {0: 60.0, 1: 86.0, 2: 85.0, 3: 84.0, 4: 97.0, 5: 86.5, 6: 83.0, 7: 85.5}
    # free = 138.9 - used: 1 52.9, 2 53.9, 3 54.9, 4 41.9 (no), 5 52.4, 6 55.9, 7 53.4; card 0 has the talker (own)
    o, out = run_plan_phys("0 1 2 3 4 5 6 7 phys pair\n", procs, used)
    assert {j.id for j in o.planned} == {"n__dec_heldout", "n__talk0"}  # held-out planned as without stacking
    st = {j.id: c for j, c in out if j.id in ("n__dec0", "n__dec1")}
    assert st == {"n__dec0": 3, "n__dec1": 2}, st                       # pairs (3, 6) and (2, 7) by free size
    assert S.pick_pairs({1: 52.9, 2: 53.9, 3: 54.9, 5: 52.4, 6: 55.9, 7: 53.4}, 2) == [(3, 6), (2, 7)]
    assert S.pick_pairs({1: 52.9, 4: 41.9}, 1) == []                   # one eligible card: no pair
    assert S.pick_pairs({c: 50.0 for c in range(8)}, 9) == [(6, 7), (4, 5), (2, 3), (0, 1)]
    # without "phys", "pair" alone does nothing
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "s"
        f.write_text("0 1 pair\n")
        assert not S.read_pair(str(f))
        f.write_text("0 1 phys pair\n")
        assert S.read_pair(str(f)) and S.read_phys(str(f))


def test_pair_cmd():
    base = lambda att: f"export A=1; python -m x --batch 16 --max-gib 120 --shard 0 > log_a{att}"
    c = S.pair_cmd(base, 3, 6)
    assert c(2) == "export CUDA_VISIBLE_DEVICES=3,6; export A=1; python -m x --batch 16 --max-gib 34 --shard 0 > log_a2"
    assert S.pair_cmd(lambda att: "python -m heldout --mem-cap-gib 70", 0, 1) is None


def test_pair_launch():
    # the launch wraps the command (two cards), writes two ledger lines; a later plain launch gets the original back
    base = lambda att: f"python -m x --max-gib 120 a{att}"
    jobs = [JC("n__dec0", base, order=1), JC("n__dec1", base, order=2)]
    led = []
    started = []

    class FO2(FO):
        @staticmethod
        def ledger_append(path, rec):
            led.append(rec)

    class O2(O):
        ledger = "led"

        def snapshot(self):
            return snap(used)

        def start(self, j, c):
            started.append((j.id, c, j.cmd(1)))
            j.state, j.card = "run", c

        def launch_gpu(self, plan):
            self.plain = [(j.id, c, j.cmd(1)) for j, c in plan]

    used = {c: 85.0 for c in range(8)}
    procs = {c: [(c + 1, 85.0, "train")] for c in range(8)}
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "stack"
        f.write_text("0 1 2 3 4 5 6 7 phys pair\n")
        o = O2(jobs)
        o.ledger = "led"
        orig = S.card_procs
        S.card_procs = lambda *a: procs
        try:
            S.apply_stack_cards(o, str(f), 3000.0, None, FO2, "led")
            pl = o.plan(snap(used), list(jobs))
            o.launch_gpu(pl)
            assert len(started) == 1                                    # one stacked start per round
            jid, c, cmd = started[0]
            assert jid == "n__dec0" and cmd.startswith(f"export CUDA_VISIBLE_DEVICES={c},") and "--max-gib 34" in cmd
            c2 = int(cmd.split(",")[1].split(";")[0])
            assert c2 != c and {r["card"] for r in led} == {c, c2} and {r["id"] for r in led} == {"n__dec0", "n__dec0#pair"}
            # stack file withdrawn: dec0 planned plainly again -> original command
            jobs[0].state = "wait"
            f.rename(Path(d) / "gone")
            o.launch_gpu([(jobs[0], 5)])
            assert o.plain == [("n__dec0", 5, "python -m x --max-gib 120 a1")]
        finally:
            S.card_procs = orig


def test_pair_history():
    # PAIR mode: a card freed 60 s ago (decode ended) is eligible again; PHYS mode still sees the 180 s maximum
    procs = {c: [(c + 1, 0.0, "train")] for c in range(4)}
    with tempfile.TemporaryDirectory() as d:
        for text, want in (("0 1 2 3 phys pair\n", 1), ("0 1 2 3 phys\n", 0)):
            f = Path(d) / "stack"
            f.write_text(text)
            jobs = [J("n__dec0", order=1)]
            o = O(jobs)
            orig, ot = S.card_procs, S.time.time
            S.card_procs = lambda *a: procs
            try:
                S.apply_stack_cards(o, str(f), 3000.0, None, FO, "led")
                now = [1000.0]
                S.time.time = lambda: now[0]
                o.plan(snap({0: 120.0, 1: 120.0, 2: 120.0, 3: 120.0}), [])      # decodes running: 120 GiB
                now[0] = 1060.0
                out = o.plan(snap({0: 84.0, 1: 84.0, 2: 120.0, 3: 120.0}), list(jobs))
            finally:
                S.card_procs, S.time.time = orig, ot
            assert len([x for x in out if x[0].id == "n__dec0"]) == want, (text, out)

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
