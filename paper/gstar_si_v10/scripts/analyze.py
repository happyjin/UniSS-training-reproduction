"""Error analysis for the paper's error table with the slot map each system's targets were built with (paper v4).

gstar_analyze's default slot map is the En->Zh offline one; the v3 runs used it for every system, which mislabels the
Zh->En slots and the low-latency (streaming) slots. Here: En->Zh offline e2e_gstar_v1, En->Zh streaming
e2e_gstar_stream_v1, Zh->En e2e_gstar_zh2en_v1 with the preset of the run (offline / lowlat). Writes
eval_outputs/.../gstar_v5_20261030/analysis/<dir>_<name>.json (never the runs' own analysis.json; v5 reruns all: the
Zh->En simultaneous system is now the 3,000-step model).

    python scripts/analyze.py   (from the paper folder)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems import D, DIRS, HEAD, STREAMING, V5, hyp_dir, windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
SYSTEMS = ("O2-nosd", "O2", "O2-oracle", "O2_LA2_d1", HEAD)


def slot_map(dr: str, name: str) -> tuple[Path, str]:
    stream = name in STREAMING
    if dr == "zh2en":
        return D / "e2e_gstar_zh2en_v1/slot_map.json", "lowlat" if stream else "offline"
    return D / ("e2e_gstar_stream_v1" if stream else "e2e_gstar_v1") / "slot_map.json", "lowlat" if stream else "offline"


def main() -> None:
    out = V5 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for dr in DIRS:
        for k in SYSTEMS:
            d = hyp_dir(dr, k)
            f = out / f"{dr}_{k}.json"
            if d is None or not (d / "hyp.jsonl").exists() or f.exists():
                continue
            sm, preset = slot_map(dr, k)
            subprocess.run([PYE, "-m", "experiments.uniss_qwen3_omni_v1.evaluation.gstar_analyze", "--windows",
                            str(windows(dr, k)), "--hyp", str(d / "hyp.jsonl"), "--slot-map", str(sm), "--preset", preset,
                            "--out", str(f)], cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
                           check=True, stdout=subprocess.DEVNULL)
            print("wrote", f.name)


if __name__ == "__main__":
    main()
