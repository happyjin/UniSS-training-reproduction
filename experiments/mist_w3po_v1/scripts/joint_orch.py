#!/usr/bin/env python
"""Evaluation orchestrator of the joint bidirectional run (b) (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md
§2.3, §3, §7): the existing per-direction orchestrators, unchanged, with ONLY the Thinker base swapped:

  --dir zh2en --set dev|dev5   thinker_t_eval_orch.py (the zh->en A' dev chain; dev5 = the c2g06 dev windows of the 5
                               precheck meetings), --win set here
  --dir zh2en --set test|ali   final_test_orch.py --set ami|ali (the zh->en test chain), --win set here
  --dir en2zh --set dev|dev5|test   en2zh_orch.py --set ... (the En->Zh chain; --heldout / --plem-export / --seeds pass
                               through)
  --base <merged Thinker dir>  every "--thinker <the direction's separate v8>" becomes "--thinker <base>" (decode and,
                               for en2zh, held-out); checked on every command: the decode commands hold the base, no
                               command holds a separate v8 path
  --heldout <dev sched dir>    zh2en: one more GPU job (kind dec, id <name>__dec_heldout), en2zh_orch's held-out job
                               with --dir zh2en: crf/heldout_loss.py --base <base> --ckpt <ckpt> --sched <dir> --dir zh2en
                               -> crf/t_eval/<name>/heldout_<name>.json (recorded; stage-2 tie-break only)
Ledger owner "joint_<run name>". Everything else (decode flags, placement, --cap135, talker chains, scoring, latency) is
the wrapped orchestrator's.

    python joint_orch.py --dir zh2en|en2zh --set dev|dev5|test|ali --base <dir> [--heldout <dir>] \\
        <thinker_t_eval_orch.py arguments> [--dry]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en2zh_orch as E  # noqa: E402
import final_test_orch as F  # noqa: E402
import finalpost_dev_orch as fo  # noqa: E402
import thinker_t_eval_orch as T  # noqa: E402

WD = "eval_outputs/mist_w3po_v1/data"
ZH_WIN = {"dev": f"{WD}/windows_zh2en_c2g06_v1/windows_dev_online.jsonl",
          "dev5": f"{WD}/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl",
          "test": F.SETS["ami"]["win"], "ali": F.SETS["ali"]["win"]}
SEPARATE = (fo.THINKER, E.THINKER)   # the separate v8 Thinkers (zh2en_3k, 20261029b)


def take(argv: list[str], key: str):
    if key not in argv:
        return argv, None
    i = argv.index(key)
    return argv[:i] + argv[i + 2:], argv[i + 1]


def check(cmd: str, base: str) -> str:
    bad = [p for p in SEPARATE if p in cmd]
    assert not bad, (bad, cmd)
    if "decode_crf_ckpt" in cmd or "online_decode" in cmd or "heldout_loss" in cmd:
        assert f"--thinker {base} " in cmd or f"--base {base} " in cmd, cmd
    return cmd


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, d = take(argv, "--dir")
    argv, key = take(argv, "--set")
    argv, base = take(argv, "--base")
    argv, held = take(argv, "--heldout")
    if d not in ("zh2en", "en2zh") or not base:
        raise SystemExit("--dir zh2en|en2zh and --base <merged Thinker> required")
    if (d == "zh2en" and key not in ZH_WIN) or (d == "en2zh" and key not in E.SETS):
        raise SystemExit(f"--set {key} not valid for --dir {d}")
    base = base.rstrip("/")
    if base in SEPARATE:
        raise SystemExit("--base is a separate v8 Thinker: use the per-direction orchestrator")
    if not Path(fo.REPO, base, "sd_proj.pt").exists() and "--dry" not in argv:
        raise SystemExit(f"--base {base}: no sd_proj.pt (not a merged G-STAR-SI Thinker)")
    orig_main = T.main

    def t_main():
        T.OWNER = "joint"   # after en2zh_orch.main's T.OWNER = "en2zh"
        orig_main()

    T.main = t_main
    if d == "en2zh":
        E.THINKER = base    # en2zh_orch.fix swaps fo.THINKER -> E.THINKER (decode) and its held-out job reads E.THINKER
        orig_fix = E.fix
        E.fix = lambda cmd, S, test=False: check(orig_fix(cmd, S, test), base)
        if held:
            argv += ["--heldout", held]
        sys.argv = [sys.argv[0], "--set", key] + argv
        E.main()
        return
    win = ZH_WIN[key]
    if "--win" in argv and argv[argv.index("--win") + 1] != win:
        raise SystemExit(f"--win must be {win} for --dir zh2en --set {key}")
    if "--win" not in argv:
        argv += ["--win", win]
    if held and not Path(fo.REPO, held, "summary.json").exists():
        raise SystemExit(f"--heldout: no sched dir {held}")
    orig = T.build_jobs

    def build_jobs(run, name, ckpt, *a, **k):
        jobs = orig(run, name, ckpt, *a, **k)
        if held:
            O = f"eval_outputs/mist_w3po_v1/crf/t_eval/{name}"
            jobs.append(fo.Job(f"{name}__dec_heldout", "dec", 1, [],
                               lambda att: fo.ENV + f"mkdir -p {O} && [ -e {O}/heldout_{name}.json ] || {fo.PY} -m "
                                                    f"experiments.mist_w3po_v1.crf.heldout_loss --base {fo.THINKER} "
                                                    f"--ckpt {ckpt} --sched {held} --dir zh2en --out "
                                                    f"{O}/heldout_{name}.json --mem-cap-gib 70 > {O}/heldout.log 2>&1",
                               retries=1))
            jobs[-1].order = len(jobs)
        for j in jobs:
            j.cmd = (lambda att, f=j.cmd: check(f(att).replace(f"--thinker {fo.THINKER} ", f"--thinker {base} ")
                                                .replace(f"--base {fo.THINKER} ", f"--base {base} "), base))
            j.cmd(1)   # check every command now
        return jobs

    T.build_jobs = build_jobs
    if key in ("test", "ali"):
        sys.argv = [sys.argv[0], "--set", "ami" if key == "test" else "ali"] + argv
        F.main()   # wraps build_jobs once more: the test-set swaps and checks
    else:
        sys.argv = [sys.argv[0]] + argv
        T.main()


if __name__ == "__main__":
    main()
