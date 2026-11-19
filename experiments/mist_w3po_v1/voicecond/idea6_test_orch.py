#!/usr/bin/env python
"""Idea 6 full-system test (prereg "完整系统测试预登记"): the M2 orchestrators, unchanged, with no card excluded
(user 2026-11-19: "8块卡都可以叠加跑"). The voice tables of the two arms come from idea6_test_tokens.py.
    python experiments/mist_w3po_v1/voicecond/idea6_test_orch.py bench <scripts/voicecond_bench_orch.py args incl. --exclude none>
    python experiments/mist_w3po_v1/voicecond/idea6_test_orch.py ood3  <voicecond/idea2_ood3_orch.py args, no --exclude>
bench: voicecond_bench_orch insists on an --exclude value; "none" passes its check and is stripped before joint_orch
parses the arguments (as idea2_e2e_orch.py). The shared exclusion file (sync_train_exclude.sh) still applies.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiments/mist_w3po_v1/scripts"))


def throughput_overrides() -> None:
    """2026-11-19 ~16:10 Prague (user: "尽可能每个gpu叠加多个测试任务"): optional env overrides of the placement constants
    of finalpost_dev_orch, applied in this process only (unset = unchanged):
      I6_LAUNCH_GAP_TALK / I6_LAUNCH_GAP_ASR  s between two launches of one round inside gpu_launch.lock (5 / 10); with 18
                                              orchestrators the lock waits reached 750 s and the cards drained between rounds
      I6_NEED_TALK                            GiB planned per talk job (24; measured 19-20: parent ~4 + EngineCore ~15)
      I6_OTHER_WINDOW_S                       s over which other processes' memory is maxed for planning (180)"""
    import os
    import finalpost_dev_orch as fo
    ch = {}
    if os.environ.get("I6_LAUNCH_GAP_TALK"):
        fo.LAUNCH_GAP_S["talk"] = ch["gap_talk"] = int(os.environ["I6_LAUNCH_GAP_TALK"])
    if os.environ.get("I6_LAUNCH_GAP_ASR"):
        fo.LAUNCH_GAP_S["asr"] = ch["gap_asr"] = int(os.environ["I6_LAUNCH_GAP_ASR"])
    if os.environ.get("I6_NEED_TALK"):
        fo.NEED["talk"] = ch["need_talk"] = float(os.environ["I6_NEED_TALK"])
    if os.environ.get("I6_OTHER_WINDOW_S"):
        fo.OTHER_WINDOW_S = ch["other_window_s"] = int(os.environ["I6_OTHER_WINDOW_S"])
    # 16:20 Prague: an ASR shard often stays below VISIBLE["asr"] = 12 GiB (4.9-6.5 GiB seen), so its pending ledger entry
    # (17.5 GiB) never gets a "visible" line and is charged to that card for every other owner until LEDGER_TTL_S (300 s);
    # 4-5 such phantom entries per card blocked the empty cards
    if os.environ.get("I6_VISIBLE_ASR"):
        fo.VISIBLE["asr"] = ch["visible_asr"] = float(os.environ["I6_VISIBLE_ASR"])
    if os.environ.get("I6_LEDGER_TTL_S"):
        fo.LEDGER_TTL_S = ch["ledger_ttl_s"] = int(os.environ["I6_LEDGER_TTL_S"])
    if os.environ.get("I6_MAX_CPU"):
        fo.MAX_CPU = ch["max_cpu"] = int(os.environ["I6_MAX_CPU"])
    if ch:
        print(f"[idea6_test_orch] throughput overrides {ch}", flush=True)


def main() -> None:
    which, argv = sys.argv[1], sys.argv[2:]
    throughput_overrides()
    if which == "bench":
        import voicecond_bench_orch as V
        assert "--exclude" in argv and argv[argv.index("--exclude") + 1] == "none", "use --exclude none"
        oj = V.J.main

        def jmain(args):
            i = args.index("--exclude")
            assert args[i + 1] == "none"
            return oj(args[:i] + args[i + 2:])
        V.J.main = jmain
        sys.argv = [V.__file__] + argv
        V.main()
    elif which == "ood3":
        import experiments.mist_w3po_v1.voicecond.idea2_ood3_orch as O
        assert "--voice-override" in argv and "--exclude" not in argv
        sys.argv = [O.__file__] + argv
        O.main()
    else:
        raise SystemExit("bench|ood3")


if __name__ == "__main__":
    main()
