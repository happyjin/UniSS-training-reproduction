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
        g0 = t.GLOBAL_EXCLUDE
        t.GLOBAL_EXCLUDE = f"{d}/global_ex"
        assert o.excluded() == {0, 1, 3}
        (Path(d) / "global_ex").write_text(" 5 6")
        assert o.excluded() == {0, 1, 3, 5, 6}
        t.GLOBAL_EXCLUDE = g0
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


def test_precheck_jobs():
    jobs = t.build_precheck("RUN")
    by = {j.id: j for j in jobs}
    assert {"pre_select", "render_lattice", "render_cover", "pre_summary"} <= set(by)
    for sel in ("lattice", "cover"):
        assert by[f"render_{sel}"].kind == "asr" and f"sel_{sel}.json" in by[f"render_{sel}"].cmd(1)
        assert all(by[f"{sel}__asrprep{k}"].deps == [f"render_{sel}"] for k in range(8))
        assert f"{sel}_hyp" in by[f"{sel}__sscore"].deps
    s = by["pre_summary"].cmd(1)
    for k in ("aprule", "lattice", "cover"):
        assert f"--lat-{k}" in s and f"--gap-{k}" in s and f"--dec-{k}" in s
    assert len(by["pre_summary"].deps) == 8


if __name__ == "__main__":
    test_precheck_jobs()
    print("ok precheck")


def test_eval_jobs():
    for scheme, tags in (("plem", ("k1", "c1b1s2")), ("align", ("k1gate", "c1b1s2"))):
        jobs = t.build_eval("RUN", scheme, "EXP", 3.5, ref_quality=True)
        by = {j.id: j for j in jobs}
        assert all(f"scorer{i}" in by for i in range(4))
        for tag in tags:
            n1 = f"{t.A_PRIME}__{scheme}_{tag}"
            c = by[f"{n1}__talk0"].cmd(1)
            c2 = by[f"{n1}_tts2__talk0"].cmd(1)
            assert "--model EXP/uniss" in c and t.EVAL_COMMON in c and "--seed" not in c and "--seed 20261109" in c2
            assert f"{t.SPD}/{t.A_PRIME}/talker_in/items.jsonl" in c
            if tag == "k1":
                assert "--backlog-token --self-speed" in c and "--select-k" not in c and "scorer-url" not in c
            if tag == "k1gate":
                assert "--cover-gate EXP/added.pt --cover-kappa 3.5" in c and "--select-k" not in c
            if tag == "c1b1s2":
                assert "--select-k 2" in c and "scorer-url" in c
                assert ("--backlog-token" in c) == (scheme == "plem")
            g = by[f"gate_{scheme}_{tag}"].cmd(1)
            assert f"--ref {t.DEC}/{t.A_PRIME} {t.DEC}/{t.A_PRIME}_tts2" in g
        assert "aprime__qual" in by and "aprime_tts2__qual" in by


if __name__ == "__main__":
    test_eval_jobs()
    print("ok eval")
