"""Unit tests for talker_crf_v3_orch.py (E2' job list); no GPU, nothing launched."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import talker_crf_v3_orch as v3  # noqa: E402


def test_jobs():
    jobs = v3.build_v3("RUN")
    by = {j.id: j for j in jobs}
    assert len(by) == len(jobs)
    assert all(d in by for j in jobs for d in j.deps)
    assert v3.TC.SCORER_PORT0 == 18801 and "--port 18801" in by["scorer0"].cmd(1) and "RUN/scorers_stop" in by["scorer0"].cmd(1)
    sel = by["pre3_select"].cmd(1)
    assert "--budget-ref stream_viterbi --drop-1ch 0.5" in sel and "precheck_dev5/sel_lattice.json" in sel and v3.PRE3 in sel
    assert "meeting3_hyp" in by["meeting3__sscore"].deps and v3.TC.WIN_DEV5 in by["meeting3__sscore"].cmd(1)
    for n in v3.names():
        for k in range(8):
            c = by[f"{n}__talk{k}"].cmd(1)
            assert f"{v3.TC.EVAL_COMMON} --backlog-token --select-k 2 --backlog-scope meeting --backlog-win {v3.TC.WIN_DEV}" in c
            assert f"127.0.0.1:{18801 + k % 4}/score" in c and "export_plemm2)/uniss" in c and f"--shard {k} --num-shards 8" in c
            assert ("--seed 20261109" in c) == n.endswith("_tts2")
    assert v3.FLAGS == v3.V2.FLAGS["plemm"]
    assert "--ref eval_outputs/mist_w3po_v1/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06__plem_c1b1s2 " in by["gate_vs_plem_plemm2"].cmd(1)
    assert v3.V2_DONE in by["summary_v4"].cmd(1) and "eval_summary_v3" in by["summary_v4"].cmd(1)
    assert set(by["summary_v4"].deps) >= {"gate_plemm2", "gate_vs_plem_plemm2", "pre3_summary"}
    small = {j.id for j in jobs if j.small}
    assert small == {f"scorer{i}" for i in range(4)} | {"render_meeting3"} | {f"{n}__qual" for n in v3.names()}
    assert v3.SMALL_WAIT_S == 25 and v3.fo.Orch.small_wait_s is None and not v3.fo.Job("x", "asr", 0, [], None).small


if __name__ == "__main__":
    test_jobs()
    print("ok")
