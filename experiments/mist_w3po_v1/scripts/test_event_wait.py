"""Unit test for event_wait.drift_state (pure function, CPU)."""
from experiments.mist_w3po_v1.scripts.event_wait import drift_state


def row(s, segs=4.0, sp=0.0, rep=0.001, kl=0.01):
    return {"step": s, "segs_per_traj": segs, "spurious_per_traj": sp, "rep_mean": rep, "kl": kl}


def test():
    dia = {s: row(s) for s in range(1, 51)}
    ok = {s: row(s) for s in range(1, 30)}
    assert drift_state(ok, dia)[0] is False
    # 4 consecutive segs violations -> no stop; 5 -> stop at step 14
    m = {s: row(s, segs=4.5 if 10 <= s <= 13 else 4.0) for s in range(1, 20)}
    assert drift_state(m, dia)[0] is False
    m = {s: row(s, segs=4.5 if 10 <= s <= 14 else 4.0) for s in range(1, 20)}
    hit, s, _ = drift_state(m, dia)
    assert hit and s == 14
    # violations before step 10 do not count
    m = {s: row(s, kl=0.1 if s <= 13 else 0.01) for s in range(1, 20)}
    assert drift_state(m, dia)[0] is False
    # rep_mean branch of C2; conditions counted separately (alternating -> no stop)
    m = {s: row(s, rep=0.004) for s in range(10, 15)}
    assert drift_state(m, dia)[0] is True
    m = {s: (row(s, kl=0.1) if s % 2 else row(s, rep=0.004)) for s in range(10, 20)}
    assert drift_state(m, dia)[0] is False
    print("test_event_wait OK")


if __name__ == "__main__":
    test()
