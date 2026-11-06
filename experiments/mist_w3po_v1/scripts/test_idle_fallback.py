"""Simulation tests of the idle-fallback switching (no GPU, no launcher is run):
decide() over a timeline, queue_view() on a synthetic queue directory, and the hand-over order of cardctl
(to_workload / to_keeper) with the process operations replaced by recorders.

    python experiments/mist_w3po_v1/scripts/test_idle_fallback.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardctl  # noqa: E402
from idle_fallback import IDLE_S, START_S, decide, queue_view  # noqa: E402


def test_decide():
    t = 1000.0
    # busy card never switches
    assert decide(t + 5000, t + 5000, True, True, False, False, 0) == "keep"
    # idle < 10 min: keep the keeper
    assert decide(t + IDLE_S - 1, t, False, True, False, False, 0) == "keep"
    # idle >= 10 min: switch
    assert decide(t + IDLE_S, t, False, True, False, False, 0) == "to_workload"
    # no keeper and no workload: keeper back at once
    assert decide(t, t, False, False, False, False, 0) == "start_keeper"
    assert decide(t, t, True, False, False, False, 0) == "start_keeper"
    # workload running: leave it (gpuq takes it back itself)
    assert decide(t, t, True, False, True, True, 999) == "keep"
    # workload whose worker never started (foreign process on the card): back to the keeper after START_S
    assert decide(t, t, False, False, True, False, START_S - 1) == "keep"
    assert decide(t, t, False, False, True, False, START_S + 1) == "to_keeper"


def test_queue_view():
    if True:   # mkdtemp, never deleted (no-rm rule); lives under the work dir
        q = Path(tempfile.mkdtemp(prefix="test_idle_fallback_", dir="/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1"))
        (q / "jobs").mkdir()
        (q / "state").mkdir()
        jobs = [
            {"id": "a", "mem_gb": 66, "deps": []},                                  # running on 2
            {"id": "b", "mem_gb": 38, "deps": [], "only": [4, 5, 6, 7], "cards": 2},  # runnable on 4..7
            {"id": "c", "mem_gb": 66, "deps": ["a"]},                               # waits for a: not runnable
            {"id": "d", "mem_gb": 0, "deps": []},                                   # CPU: ignored
            {"id": "e", "mem_gb": 20, "deps": [], "only": [0]},                     # done
        ]
        for i, j in enumerate(jobs):
            (q / "jobs" / f"{i:03d}_{j['id']}.json").write_text(json.dumps(j))
        (q / "state" / "a.started").write_text(json.dumps({"time": 0, "cards": [2]}))
        (q / "state" / "e.started").write_text(json.dumps({"time": 0, "cards": [0]}))
        (q / "state" / "e.rc").write_text("0\n")
        running, runnable = queue_view(q)
        assert running == {2}, running
        assert runnable == {4, 5, 6, 7}, runnable


class Rec(cardctl.Card):
    """Card with the process operations recorded instead of executed."""

    def __init__(self, keeper: int, workload: int, sup_survives: bool = True):
        self.c, self.ops = 0, []
        self.k, self.w, self.sup_survives = keeper, workload, sup_survives

    def keeper_pid(self):
        return self.k

    def workload_pid(self):
        return self.w

    def start_keeper(self, wait_s=90):
        self.ops.append("start_keeper")
        self.k = 11
        return 11

    def stop_keeper(self, wait_s=30):
        self.ops.append("stop_keeper")
        self.k = 0
        return True

    def start_workload(self):
        self.ops.append("start_workload")
        self.w = 22
        return 22

    def stop_workload(self, wait_s=40):
        self.ops.append("stop_workload")
        self.w = 0
        return True


def test_handover_order(monkey_alive=True):
    orig_alive, orig_sleep = cardctl.alive, cardctl.time.sleep
    cardctl.time.sleep = lambda s: None
    try:
        cardctl.alive = lambda pid: True
        c = Rec(keeper=11, workload=0)
        assert c.to_workload() and c.ops == ["start_workload", "stop_keeper"], c.ops   # supervisor waits first
        c = Rec(keeper=0, workload=22)
        assert c.to_keeper() and c.ops == ["start_keeper", "stop_workload"], c.ops      # keeper up before release
        cardctl.alive = lambda pid: False   # supervisor died at once (e.g. bad args): keeper stays
        c = Rec(keeper=11, workload=0)
        assert not c.to_workload() and c.ops == ["start_workload"], c.ops
    finally:
        cardctl.alive, cardctl.time.sleep = orig_alive, orig_sleep


if __name__ == "__main__":
    test_decide()
    test_queue_view()
    test_handover_order()
    print("IDLE_FALLBACK_TESTS_OK")
