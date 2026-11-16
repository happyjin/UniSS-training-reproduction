#!/usr/bin/env python
"""E9 cascade orchestrator (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3, §4): joint_orch.py --
the orchestrator that ran B on the three test sets (joint_test.sh, BASE_TALKER=t1c) -- unchanged, with two command
swaps applied LAST (after joint_orch / final_test_orch / en2zh_orch built and checked every command):

  decode  "-m experiments.mist_w3po_v1.evaluation.online_decode_finalpost "   (thinker_t_eval_orch --plain-thinker)
          -> "-m experiments.mist_w3po_v1.evaluation.online_decode_finalpost_zerosd "   (speaker stream zeroed)
  prep    "-m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker_v6 prep "
          -> "-m experiments.mist_w3po_v1.evaluation.gstar_talker_v6_unk prep "   (accepts the relabelled "unk")
Everything else is B's: decode flags (finalpost close_detect, policy wait, lag 2000, ctx 0, batch 16, --max-gib 120,
loop guard, 1.0 s reads, 8 shards by meeting), --cap135 placement, the T1c talker (--speed-policy fixed --history-ban 8,
8 shards per seed, seeds default / 20261109), ASR, score_run. One placement change: the per-card planning cap is
129 GiB instead of 135 (CAP_TOTAL; this run's rule used + estimate + 6 GiB <= 135 GiB), so decodes run one per card.
Cards: pass --exclude 0 1 2 3 (another agent's cards). Checks: --base is a v8 speaker-blind model; a decode
command is the zero-sd module on --base and never a CRF checkpoint; no command keeps the unswapped modules. Ledger
owner "e9_<run name>".

    python e9_orch.py <joint_orch.py arguments: --dir --set --base --run --name --ckpt - [--plain-thinker] ...> [--dry]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalpost_dev_orch as fo  # noqa: E402
import joint_orch as J  # noqa: E402
import thinker_t_eval_orch as T  # noqa: E402

CAP_TOTAL = 129.0   # GPU rule of this run: used + estimate + 6 GiB <= 135 GiB per card (fo.CAP_TOTAL 135 -> 129); with
                    # a decode really holding ~66.5 GiB this is one decode per card (cap135 would plan two)

DEC_OLD = "-m experiments.mist_w3po_v1.evaluation.online_decode_finalpost "
DEC_NEW = "-m experiments.mist_w3po_v1.evaluation.online_decode_finalpost_zerosd "
PREP_OLD = "-m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker_v6 prep "
PREP_NEW = "-m experiments.mist_w3po_v1.evaluation.gstar_talker_v6_unk prep "
BLIND = ("eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_zh2en_v8/thinker_merged",
         "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_en2zh_v8/thinker_merged")


def swap(cmd: str, base: str) -> str:
    assert "decode_crf_ckpt" not in cmd, cmd
    if DEC_OLD in cmd:
        cmd = cmd.replace(DEC_OLD, DEC_NEW)
        assert f"--thinker {base} " in cmd and "--final-read-post close_detect " in cmd, cmd
    if PREP_OLD in cmd:
        cmd = cmd.replace(PREP_OLD, PREP_NEW)
    assert DEC_OLD not in cmd and PREP_OLD not in cmd, cmd
    return cmd


def main() -> None:
    argv = sys.argv[1:]
    if "--base" not in argv:
        raise SystemExit("--base <speaker-blind model> required")
    base = argv[argv.index("--base") + 1].rstrip("/")
    if base not in BLIND:
        raise SystemExit(f"--base must be one of {BLIND}")
    if "--exclude" not in argv or argv[argv.index("--exclude") + 1: argv.index("--exclude") + 5] != ["0", "1", "2", "3"]:
        raise SystemExit("--exclude 0 1 2 3 required (cards 0-3 belong to another agent)")
    if "--speech-from" not in argv and "--plain-thinker" not in argv:
        raise SystemExit("a decode needs --plain-thinker --ckpt - (no CRF checkpoint)")
    orig_main = T.main

    def e9_main():
        T.OWNER = "e9"
        fo.CAP_TOTAL = CAP_TOTAL   # read by free_plan at every tick (apply_cap135 leaves it alone)
        inner = T.build_jobs   # the outermost wrapper by now (final_test_orch / en2zh_orch around joint_orch's)

        def build_jobs(*a, **k):
            jobs = inner(*a, **k)
            for j in jobs:
                j.cmd = (lambda att, f=j.cmd: swap(f(att), base))
                j.cmd(1)   # check every command now
            return jobs

        T.build_jobs = build_jobs
        orig_main()

    T.main = e9_main
    sys.argv = [sys.argv[0]] + argv
    J.main(argv)


if __name__ == "__main__":
    main()
