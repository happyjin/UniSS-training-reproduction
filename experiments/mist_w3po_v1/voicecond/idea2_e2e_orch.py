#!/usr/bin/env python
"""Idea 2 e2e test (prereg addendum 2026-11-19): the M2 orchestrators with the e2e Talker export on every talk job.
    python experiments/mist_w3po_v1/voicecond/idea2_e2e_orch.py bench --talker-model <export>/uniss <scripts/voicecond_bench_orch.py args>
    python experiments/mist_w3po_v1/voicecond/idea2_e2e_orch.py ood3  --talker-model <export>/uniss <voicecond/idea2_ood3_orch.py args>
Only the voice-override module string of the wrapped orchestrator is swapped for voicecond/idea2_e2e_talker.py (which
in turn runs the M2 override module with --model replaced); every other job, flag, placement and check is the M2 one.
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
    i = argv.index("--talker-model")
    model = argv[i + 1]
    del argv[i:i + 2]
    assert (Path(model) / "model.safetensors").exists(), model
    hook = f"-m experiments.mist_w3po_v1.voicecond.idea2_e2e_talker --talker-model {model} "
    if which == "bench":
        import voicecond_bench_orch as V
        V.HOOK = hook + "--via bench "
        if "--exclude" in argv and argv[argv.index("--exclude") + 1] == "none":
            # user 2026-11-19: evals may stack on training cards -> no card excluded; voicecond_bench_orch insists on an
            # --exclude value, so "none" passes its check and is stripped before joint_orch parses the arguments
            oj = V.J.main

            def jmain(args):
                i = args.index("--exclude")
                assert args[i + 1] == "none"
                return oj(args[:i] + args[i + 2:])
            V.J.main = jmain
        sys.argv = [V.__file__] + argv
        print(f"[idea2 e2e orch] bench hook -> {V.HOOK}", flush=True)
        V.main()
    elif which == "ood3":
        import experiments.mist_w3po_v1.voicecond.idea2_ood3_orch as O
        assert "--voice-override" in argv, "ood3: the e2e arm needs --voice-override <token table>"
        O.NEW = hook + "--via vo --voice-override {} "
        sys.argv = [O.__file__] + argv
        print(f"[idea2 e2e orch] ood3 talk command -> {O.NEW}", flush=True)
        O.main()
    else:
        raise SystemExit("bench|ood3")


if __name__ == "__main__":
    main()
