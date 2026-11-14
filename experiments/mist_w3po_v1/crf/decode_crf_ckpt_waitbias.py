"""crf/decode_crf_ckpt_finalpost.py with an inference-time write bias (plan A, user 2026-11-14 ~10:20 Prague;
reports/mist_w3po_v1/joint_baseline_vs_method_20261114.zh-CN.md "解决方案" A): every Engine.read call gets
wait_bias += --wait-bias (the engine's own argument: added to the <wait> logit at every generated position of every
item before the bar; a final read / barred item still has <wait> at -1e9, so only non-final reads change). A negative
value makes <wait> less likely, i.e. the Thinker writes earlier / more per read. Nothing else changes: the argv without
--wait-bias goes to decode_crf_ckpt_finalpost.main unchanged (--final-read-post, --crf-ckpt, online_decode arguments).

  --wait-bias 0   bit-identical to decode_crf_ckpt_finalpost (the wrapper is not installed)

    python -m experiments.mist_w3po_v1.crf.decode_crf_ckpt_waitbias --wait-bias -2 --crf-ckpt <ckpt> \
        --final-read-post close_detect --thinker <merged> --windows <jsonl> --out <new jsonl> <online_decode args>
"""
from __future__ import annotations

import sys

from experiments.mist_w3po_v1.core import engine as E

STATS = {"reads": 0}


def take_bias(argv: list[str]) -> tuple[float, list[str]]:
    """-> (bias, argv without it); accepts '--wait-bias X' and '--wait-bias=X'; required."""
    out, bias, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--wait-bias":
            bias = float(argv[i + 1])
            i += 2
            continue
        if a.startswith("--wait-bias="):
            bias = float(a.split("=", 1)[1])
            i += 1
            continue
        out.append(a)
        i += 1
    if bias is None:
        raise SystemExit("--wait-bias <float> is required")
    return bias, out


def install(bias: float) -> None:
    orig = E.Engine.read

    def read(self, items, *a, wait_bias: float = 0.0, **k):
        STATS["reads"] += 1
        return orig(self, items, *a, wait_bias=wait_bias + bias, **k)

    read.__wrapped_waitbias__ = bias
    E.Engine.read = read


def main() -> None:
    bias, rest = take_bias(sys.argv[1:])
    if bias != 0.0:
        install(bias)
    print(f"wait bias {bias} (non-final reads; <wait> logit += bias)", flush=True)
    sys.argv = [sys.argv[0]] + rest
    from experiments.mist_w3po_v1.crf import decode_crf_ckpt_finalpost as dfp
    dfp.main()
    print(f"WAITBIAS_STATS bias={bias} read_calls={STATS['reads']}", flush=True)


if __name__ == "__main__":
    main()
