"""Run a metric_headroom-based dev tool (speech_latency_lit, speech_gap_decomp, ...) against another reference set:
sets metric_headroom.REF_WIN (the scored reference windows) and metric_headroom.DR (the universe / tokenisation set:
zh2en or ali) before the tool runs; nothing else changes (the tool's own arguments, incl. --win, pass through).
Additive: no existing tool is changed (final test, reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md).

    python -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win <ref windows jsonl> --dr zh2en|ali \\
        experiments.mist_w3po_v1.evaluation.speech_latency_lit --run ... --talker ... --win ... --out ...
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 5 or argv[0] != "--ref-win" or argv[2] != "--dr" or argv[3] not in ("zh2en", "ali"):
        raise SystemExit(__doc__)
    ref, dr, mod, rest = argv[1], argv[3], argv[4], argv[5:]
    from experiments.mist_w3po_v1.evaluation import metric_headroom as M
    if not (M.REPO / ref).exists():
        raise SystemExit(f"reference windows missing: {ref}")
    M.REF_WIN, M.DR = ref, dr
    print(f"[on_refset] REF_WIN={ref} DR={dr} -> {mod}", file=sys.stderr, flush=True)
    sys.argv = [mod] + rest
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
