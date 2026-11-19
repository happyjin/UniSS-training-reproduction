#!/usr/bin/env python
"""Idea 2 prereg v3 §C (OOD-3 no-harm): joint_orch.py on a meeting SUBSET of the AMI / AliMeeting test sets, exactly as
the B T1c rows of joint_test.sh run them (joint_orch --dir zh2en --set test|ali / --dir en2zh --set test, --speech-from,
--talker t1c, --window 1, --cap135), with only the set definitions swapped in this process (like game_orch.py):
  windows -> data/open_voicecond_v1/idea2/ood3/<ss>_win.jsonl, reference windows -> <ss>_ref.jsonl
(final_test_orch.SETS[...]["win"/"ref"], joint_orch.ZH_WIN[...], en2zh_orch.SETS["test"]). With --voice-override <table>
every talk job runs voicecond/idea1_si_infer_vo.py (sha1-checked token override), exactly as idea1_game_orch.py does;
without it the run is the B arm on the same subset chain.

    python experiments/mist_w3po_v1/voicecond/idea2_ood3_orch.py --subset ami|en|ali [--voice-override <tokens.jsonl>] \
        <joint_orch.py arguments: --dir ... --set ... --base ... --run ... --name ... --speech-from idea2_ood3_<ss>_txt ...>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/mist_w3po_v1/scripts"))
OD = "data/open_voicecond_v1/idea2/ood3"
OLD = "-m experiments.uniss_qwen3_omni_v1.evaluation.si_infer "
NEW = "-m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override {} "


def take(argv, flag):
    if flag not in argv:
        return argv, None
    i = argv.index(flag)
    v = argv[i + 1]
    return argv[:i] + argv[i + 2:], v


def main() -> None:
    argv, ss = take(sys.argv[1:], "--subset")
    argv, table = take(argv, "--voice-override")
    assert ss in ("ami", "en", "ali"), "--subset ami|en|ali"
    d, s = argv[argv.index("--dir") + 1], argv[argv.index("--set") + 1]
    assert (ss, d, s) in (("ami", "zh2en", "test"), ("en", "en2zh", "test"), ("ali", "zh2en", "ali")), (ss, d, s)
    import en2zh_orch as E
    import final_test_orch as F
    import joint_orch as J
    win, ref = f"{OD}/{ss}_win.jsonl", f"{OD}/{ss}_ref.jsonl"
    for p in (win, ref):
        assert (REPO / p).exists(), p
    if ss == "ami":
        F.SETS["ami"] = dict(F.SETS["ami"], win=win, ref=ref)
        J.ZH_WIN["test"] = win
    elif ss == "ali":
        F.SETS["ali"] = dict(F.SETS["ali"], win=win, ref=ref)
        J.ZH_WIN["ali"] = win
    else:
        E.SETS["test"] = dict(E.SETS["test"], win=win, ref=ref)
    print(f"[idea2 ood3] subset {ss}: win {win} ref {ref}", flush=True)
    if table:
        table = str(Path(table).resolve())
        assert Path(table).is_file(), table
        import finalpost_dev_orch as fo
        import talker_crf_orch as tc
        made = []
        oj = fo.Job.__init__

        def job_init(self, *a, **k):
            oj(self, *a, **k)
            made.append(self)
        fo.Job.__init__ = job_init

        def swap(f):
            def cmd(att):
                c = f(att)
                assert c.count(OLD) == 1, c[:300]
                return c.replace(OLD, NEW.format(table))
            return cmd
        oi = tc.YieldOrch.__init__

        def orch_init(self, run, jobs, *a, **k):
            n = 0
            for j in jobs:
                if j.kind == "talk":
                    j.cmd = swap(j.cmd)
                    j.cmd(1)
                    n += 1
            assert n > 0, "no talk jobs"
            print(f"[idea2 ood3] voice-override swapped into {n} talk jobs ({table})", flush=True)
            oi(self, run, jobs, *a, **k)
        tc.YieldOrch.__init__ = orch_init
    J.main(argv)
    if table and "--dry" in argv:
        talks = [j for j in made if j.kind == "talk"]
        for j in talks[:2]:
            print("[idea2 ood3 dry]", j.id, "|", swap(j.cmd)(1)[:600])
        print(f"[idea2 ood3 dry] {len(talks)} talk jobs would run idea1_si_infer_vo")


if __name__ == "__main__":
    main()
