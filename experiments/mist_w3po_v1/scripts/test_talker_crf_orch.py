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


def test_sample_jobs():
    jobs = t.build_sample("RUN")
    by = {j.id: j for j in jobs}
    assert {f"scorer{i}" for i in range(4)} <= set(by)
    assert all(by[f"scorer{i}"].kind == "asr" and "RUN/scorers_stop" in by[f"scorer{i}"].cmd(1) for i in range(4))
    for pre in ("dev5", "train"):
        for k in range(8):
            c = by[f"{pre}__talk{k}"].cmd(1)
            assert "--cand-speeds 0,2 --cand-k 4 --dump-cands --cand-cover" in c and "--speed-policy queue" in c
            assert "--select-k 2" in c and "--speed-shift 2" in c and f"127.0.0.1:{t.SCORER_PORT0 + k % 4}/score" in c
            assert f"--shard {k} --num-shards 8" in c and "--seed" not in c
    assert "dev5_hyp" in by["dev5__sscore"].deps and t.WIN_DEV5 in by["dev5__sscore"].cmd(1)
    assert not any(j.id.startswith("train__asr") for j in jobs)


if __name__ == "__main__":
    test_sample_jobs()
    print("ok sample")
