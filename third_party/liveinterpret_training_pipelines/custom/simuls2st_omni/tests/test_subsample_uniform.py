"""Uniform hour/count subsample is seed-stable and not a jsonl prefix."""

from simuls2st_omni.train.datasets.subsample import subsample_uniform


def test_hour_budget_stops_near_target():
    records = [{"id": i, "sec": 10.0} for i in range(100)]
    picked = subsample_uniform(
        records,
        max_hours=100 / 3600,
        seed=0,
        duration_sec_fn=lambda r: r["sec"],
    )
    assert abs(sum(r["sec"] for r in picked) - 100.0) < 10.0
    assert {r["id"] for r in picked} != set(range(len(picked)))


def test_same_seed_same_ids():
    records = [{"id": i, "sec": 1.0} for i in range(50)]
    a = subsample_uniform(
        records, max_samples=10, seed=7, duration_sec_fn=lambda r: r["sec"]
    )
    b = subsample_uniform(
        records, max_samples=10, seed=7, duration_sec_fn=lambda r: r["sec"]
    )
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_under_budget_keeps_all():
    records = [{"id": i, "sec": 1.0} for i in range(5)]
    picked = subsample_uniform(
        records,
        max_hours=1.0,
        seed=1,
        duration_sec_fn=lambda r: r["sec"],
    )
    assert len(picked) == 5
