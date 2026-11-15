#!/usr/bin/env python
"""joint_orch.py with another Zh->En PL-EM export for --talker plem (paper v11 P2, the fluency-constrained PA Talker,
reports/mist_w3po_v1/v11_p2_fluency_pa_prereg.zh-CN.md). thinker_t_eval_orch reads its module constant PLEM (the
original Zh->En PL-EM export) when it builds the plem talker commands; this wrapper sets it to --plem-x before
joint_orch.main runs. Zh->En only (En->Zh already takes --plem-export). Everything else is joint_orch's (placement,
--cap135, ledger, scorers, latency). Additive: joint_orch.py / thinker_t_eval_orch.py are not changed.

    python joint_orch_plemx.py --plem-x <export dir with uniss/model.safetensors> <joint_orch.py arguments, --dir zh2en>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_orch as J  # noqa: E402


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, x = J.take(argv, "--plem-x")
    if not x:
        raise SystemExit("--plem-x <export> required")
    if "--dir" not in argv or argv[argv.index("--dir") + 1] != "zh2en":
        raise SystemExit("--plem-x is for --dir zh2en (En->Zh: joint_orch.py --plem-export)")
    if "--talker" not in argv or argv[argv.index("--talker") + 1] != "plem":
        raise SystemExit("--plem-x needs --talker plem")
    x = x.rstrip("/")
    if not Path(J.fo.REPO, x, "uniss", "model.safetensors").exists():
        raise SystemExit(f"--plem-x {x}: no uniss/model.safetensors")
    J.T.PLEM = x
    sys.argv = [sys.argv[0]] + argv
    J.main(argv)


if __name__ == "__main__":
    main()
