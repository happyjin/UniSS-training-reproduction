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


def main() -> None:
    which, argv = sys.argv[1], sys.argv[2:]
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
