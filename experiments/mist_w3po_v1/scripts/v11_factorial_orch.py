#!/usr/bin/env python
"""v11 P0-1 (docs/mist_v11_plan.zh-CN.md §3; reports/mist_w3po_v1/v11_p01_factorial_prereg.zh-CN.md): joint_orch.py,
unchanged, speaking an existing (read-merged) text with one speech-layer variant of the PA factorial ablation. The
variant is expressed by rewriting the si_infer talker command that the wrapped orchestrator builds (the same hook as
joint_orch_wb.py: joint_orch.check is applied to every command, after the per-direction / test-set fixes):

  t1c        --talker t1c verbatim (fixed rate, one sample, base Talker T1c)                      [R1 on rl text]
  c1s2       --talker c1b1s2 minus the CTC selection: " --select-k 2", its " --scorer-url ..." and the wait for the
             scorer port removed; no scorer servers are started (queue rate rule + --speed-shift 2, one sample) [R2]
  c1b1s2     --talker c1b1s2 verbatim (rule rate + best-of-2 CTC selection = the v10 Table 6 rule layer)   [R3]
  c1b1s<N>   c1b1s2 with "--speed-shift 2" -> "--speed-shift N" (rate-matched rule)                         [R3']
  plemk1     --talker plem minus the CTC selection (PA Talker, backlog token, queue rate rule + shift 2, one
             sample); no scorer servers                                                                    [R4]
(R0 = B with t1c on the unmerged text and R5 = B + rl + PA Talker with selection are the existing E2 runs.)

    python v11_factorial_orch.py --variant <v> <joint_orch.py arguments without --talker> [--dry]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_orch as J  # noqa: E402
import talker_crf_orch as tc  # noqa: E402

WAIT = re.compile(r"for i in \$\(seq 1 180\); do \(echo > /dev/tcp/127\.0\.0\.1/\d+\) 2>/dev/null && break; sleep 5; done; ")
URL = re.compile(r" --scorer-url http://127\.0\.0\.1:\d+/score")


def parse(v: str) -> tuple[str, str, int]:
    """variant -> (wrapped --talker, rewrite kind, speed shift)"""
    if v == "t1c":
        return "t1c", "", 0
    if v == "c1s2":
        return "c1b1s2", "nosel", 0
    if v == "c1b1s2":
        return "c1b1s2", "", 0
    m = re.fullmatch(r"c1b1s(\d+)", v)
    if m:
        return "c1b1s2", "shift", int(m.group(1))
    if v == "plemk1":
        return "plem", "nosel", 0
    raise SystemExit(f"unknown --variant {v}")


def rewrite(cmd: str, kind: str, shift: int) -> str:
    if ".si_infer " not in cmd or not kind:
        return cmd
    if kind == "nosel":
        assert cmd.count(" --select-k 2") == 1 and len(URL.findall(cmd)) == 1 and len(WAIT.findall(cmd)) == 1, cmd
        cmd = WAIT.sub("", URL.sub("", cmd.replace(" --select-k 2", "", 1)))
        assert "--select-k" not in cmd and "--scorer-url" not in cmd and "/dev/tcp/" not in cmd, cmd
    elif kind == "shift":
        assert cmd.count("--speed-shift 2 ") == 1 and "--speed-policy queue" in cmd, cmd
        cmd = cmd.replace("--speed-shift 2 ", f"--speed-shift {shift} ", 1)
    return cmd


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--variant" not in argv or "--talker" in argv:
        raise SystemExit("--variant <v> required; --talker is set from the variant")
    i = argv.index("--variant")
    talker, kind, shift = parse(argv[i + 1])
    argv = argv[:i] + argv[i + 2:] + ["--talker", talker]
    if kind == "nosel":
        tc.scorer_jobs = lambda add, run: []   # no talker asks a scorer
    orig = J.check
    J.check = lambda cmd, base: rewrite(orig(cmd, base), kind, shift)
    J.main(argv)


if __name__ == "__main__":
    main()
