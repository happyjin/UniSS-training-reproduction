"""Entry point of plan B (crf/serial_v2p_lat.py): install the latency-budget posterior (V2P_LAT_RATIO, required, in
[0, 1]) into serial_v2p_train, then run crf/train_crf_sft.py as __main__ with the same argv (torch.distributed.run
starts this file instead of train_crf_sft.py; every rank installs the patch itself). --objective must be serial_v2p.

    V2P_LAT_RATIO=0.5 python -m torch.distributed.run ... experiments/mist_w3po_v1/crf/train_v2p_lat.py <train_crf_sft args>
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    if "--objective" not in sys.argv or sys.argv[sys.argv.index("--objective") + 1] != "serial_v2p":
        raise SystemExit("train_v2p_lat: --objective serial_v2p required")
    ratio = float(os.environ["V2P_LAT_RATIO"])
    from experiments.mist_w3po_v1.crf import serial_v2p_lat as L
    L.install(ratio)
    print(f"V2P_LAT installed: ratio {ratio}", flush=True)
    script = str(Path(__file__).resolve().parent / "train_crf_sft.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
