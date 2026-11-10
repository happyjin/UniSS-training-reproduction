"""Unit tests for talker_crf_orch.py (job lists, exclude file, process tree); no GPU, nothing launched."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import talker_crf_orch as t  # noqa: E402


def test_traindec_jobs():
    jobs = t.build_traindec()
    ids = [j.id for j in jobs]
    assert len(ids) == 10 and len(set(ids)) == 10
    dec = [j for j in jobs if j.kind == "dec"]
    assert len(dec) == 8 and all(not j.deps for j in dec)
    cmd = dec[0].cmd(1)
    assert t.WIN_TRAIN in cmd and t.CKPT in cmd and "--shard-by meeting" in cmd and "--num-shards 8" in cmd
    merge = next(j for j in jobs if j.id.endswith("__merge"))
    assert sorted(merge.deps) == sorted(j.id for j in dec)
    sprep = next(j for j in jobs if j.id.endswith("__sprep"))
    assert sprep.deps == [merge.id] and t.WIN_TRAIN in sprep.cmd(1) and "drop_long_streams.py" in sprep.cmd(1)


def test_excluded_and_tree():
    with tempfile.TemporaryDirectory() as d:
        o = t.YieldOrch.__new__(t.YieldOrch)
        o.run = Path(d)
        assert o.excluded() == set()
        (Path(d) / "exclude_cards").write_text("0 1\n3\n")
        assert o.excluded() == {0, 1, 3}
    assert os.getpid() in t.tree(os.getpid())


if __name__ == "__main__":
    test_traindec_jobs()
    test_excluded_and_tree()
    print("ok")
