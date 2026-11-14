#!/usr/bin/env python
"""joint_orch.py with plan A's inference-time write bias (user 2026-11-14 ~10:20 Prague): every decode command that
joint_orch builds with crf/decode_crf_ckpt_finalpost gets crf/decode_crf_ckpt_waitbias --wait-bias <b> instead
(same arguments otherwise). joint_orch.check is wrapped (it is applied to every command), so placement, --cap135, the
ledger, talker chains, scoring and latency are joint_orch's. A command that holds decode_crf_ckpt but not the
finalpost module is refused (the bias would silently not apply). Additive: joint_orch.py is not changed.

    python joint_orch_wb.py --wait-bias <b> <joint_orch.py arguments>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_orch as J  # noqa: E402

OLD = "-m experiments.mist_w3po_v1.crf.decode_crf_ckpt_finalpost "
NEW = "-m experiments.mist_w3po_v1.crf.decode_crf_ckpt_waitbias --wait-bias {b} "


def rewrite(cmd: str, bias: float) -> str:
    if "decode_crf_ckpt" in cmd:
        if OLD not in cmd:
            raise SystemExit(f"decode command without {OLD.strip()}: {cmd[:300]}")
        cmd = cmd.replace(OLD, NEW.format(b=bias))
    return cmd


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--wait-bias" not in argv:
        raise SystemExit("--wait-bias <float> required")
    i = argv.index("--wait-bias")
    bias = float(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
    orig = J.check
    J.check = lambda cmd, base: rewrite(orig(cmd, base), bias)
    J.main(argv)


if __name__ == "__main__":
    main()
