"""Unit tests for the shared pending ledger of finalpost_dev_orch (2026-11-11): append / pending / visible / TTL, the
free-plan subtraction of other owners' pending memory, the per-card vLLM gap; default (no ledger) unchanged."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalpost_dev_orch as fo  # noqa: E402


def test_ledger():
    with tempfile.TemporaryDirectory() as d:
        L = f"{d}/locks/pending.jsonl"
        assert fo.ledger_pending(L) == []
        fo.ledger_append(L, {"owner": "a", "id": "j1", "card": 4, "mib": 24576, "vllm": True, "state": "pending"})
        fo.ledger_append(L, {"owner": "b", "id": "j1", "card": 5, "mib": 40000, "vllm": False, "state": "pending"})
        fo.ledger_append(L, {"owner": "a", "id": "j2", "card": 6, "mib": 1000, "vllm": False, "state": "pending"})
        fo.ledger_append(L, {"owner": "a", "id": "j2", "card": 6, "state": "visible"})
        p = fo.ledger_pending(L)
        assert sorted((r["owner"], r["id"]) for r in p) == [("a", "j1"), ("b", "j1")]
        assert fo.ledger_pending(L, t=fo.now() + fo.LEDGER_TTL_S + 1) == []
        Path(L).open("a").write("not json\n")
        assert len(fo.ledger_pending(L)) == 2

        o = fo.Orch.__new__(fo.Orch)
        o.jobs, o.by_id = [], {}
        snap = {"cards": {c: {"used": 30.0, "total": 140.0} for c in (4, 5, 6)}, "per_job": {},
                "other_plan": {c: 30.0 for c in (4, 5, 6)}}
        base = o.free_plan(snap)                       # default: no ledger, unchanged
        o.ledger, o.ledger_owner = L, "a"
        led = o.free_plan(snap)
        assert led[4] == base[4] and abs(led[5] - (base[5] - 40000 / 1024)) < 1e-9 and led[6] == base[6]
        assert o.ledger_vllm_busy(4) and not o.ledger_vllm_busy(5) and not o.ledger_vllm_busy(6)
        o.ledger = None
        assert not o.ledger_vllm_busy(4) and fo.Orch.ledger is None


if __name__ == "__main__":
    test_ledger()
    print("ok")
