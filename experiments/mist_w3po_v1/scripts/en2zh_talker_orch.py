#!/usr/bin/env python
"""En->Zh PL-EM talker data (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §4): talker_crf_orch.py's
traindec and sample stages (talker_crf_prereg_v1 §0 / §3: the zh->en recipe), on the En->Zh train subset, with the
commands rewritten by en2zh_orch.fix (decode --thinker en2zh v8 --dir en2zh, talker prep ihm_mix / nemotron_lowlat /
--sep '', si_infer --direction en2zh, si_ctc_scorer --lang zh, MODELSCOPE_CACHE) and checked the same way.

  traindec  the selected Thinker checkpoint (--ckpt) decodes the train16 windows (windows_en2zh_c2g06_train16_v1, 8
            shards by meeting; the A' decode flags) -> merged hyp.jsonl (no scoring on train) -> talker prep + drop of
            streams > 400 blocks -> speech/<name>/talker_in
  sample    4 Chinese CTC scorer servers + 8 talker shards on that talker input: T1c, the A' talker settings c1+b1+s2
            (queue speed, --speed-shift 2, b1 k 2 = the A' rule path) with --cand-speeds 0,2 --cand-k 4 --dump-cands
            --cand-cover (8 candidates a block, every one CTC-scored) -> speech/<name>/talker_cands/cands_*.jsonl.
            Only the train part of talker_crf_orch's sample stage (its dev5 part fed the zh->en precheck, not run here).
Placement / yielding / exclude files: talker_crf_orch.YieldOrch (as the zh->en run).

    python en2zh_talker_orch.py --stage traindec|sample --run <new run dir> --name <train decode name> \\
        --ckpt <Thinker ckpt> [--traindec-run <traindec run dir, sample stage>] [--scorer-port0 18861] [--dry] [--resume]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en2zh_orch as E  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import talker_crf_orch as tc  # noqa: E402

WIN_TRAIN = "eval_outputs/mist_w3po_v1/data/windows_en2zh_c2g06_train16_v1/windows_train_online.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["traindec", "sample"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--traindec-run", default="")
    ap.add_argument("--scorer-port0", type=int, default=18861)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.chdir(fo.REPO)
    tc.WIN_TRAIN, tc.TRAIN_NAME, tc.CKPT = WIN_TRAIN, a.name, a.ckpt
    tc.SCORER_PORT0 = a.scorer_port0
    if a.stage == "sample":
        if not a.traindec_run:
            raise SystemExit("--traindec-run required for the sample stage")
        tc.TRAINDEC_RUN = a.traindec_run
    run = str(Path(a.run))
    S = {"win": WIN_TRAIN, "ref": E.REF_DEV}
    if a.stage == "traindec":
        jobs = tc.build_traindec()
        dirs = [f"{fo.DEC}/{a.name}", f"{fo.SPD}/{a.name}"]
    else:
        jobs = [j for j in tc.build_sample(run) if j.id.startswith(("scorer", "train__talk"))]
        assert len(jobs) == tc.N_SCORERS + 8, [j.id for j in jobs]
        dirs = [f"{fo.SPD}/{a.name}/talker_cands"]
    for j in jobs:
        j.deps = [d for d in j.deps if d in {x.id for x in jobs}]
        j.cmd = (lambda att, f=j.cmd: E.fix(f(att), S))
        c = j.cmd(1)
        if ".si_infer " in c:
            assert "--cand-speeds 0,2 --cand-k 4 --dump-cands --cand-cover" in c and f"{fo.SPD}/{a.name}/talker_in/" in c, c
    if a.dry:
        for j in jobs:
            print(j.prio, j.kind, j.id, "deps", len(j.deps), "|", j.cmd(1)[len(fo.ENV):][:900])
        print(len(jobs), "jobs; existing new dirs:", [d for d in dirs if Path(d).exists()] or "none")
        return
    if a.stage == "traindec" and not Path(a.ckpt, "DONE").exists():
        raise SystemExit(f"checkpoint not done: {a.ckpt}")
    tc.cp.CANDS = {}
    r = Path(run)
    if a.resume:
        o = tc.YieldOrch(r, jobs)
        o.adopt()
    else:
        if r.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {r}")
        ex = [d for d in dirs if Path(d).exists()]
        if ex:
            raise SystemExit(f"拒绝覆盖已有输出: {ex}")
        r.mkdir(parents=True)
        o = tc.YieldOrch(r, jobs)
    o.spread = True
    o.log(f"en2zh talker stage {a.stage}; {len(jobs)} jobs; ckpt {a.ckpt}; name {a.name}")
    try:
        o.loop()
    except Exception as e:  # noqa: BLE001
        import traceback
        o.log("CRASH " + traceback.format_exc().replace("\n", " | "))
        o.alert(f"ORCHESTRATOR CRASH {e!r}; running jobs continue")
        raise


if __name__ == "__main__":
    main()
