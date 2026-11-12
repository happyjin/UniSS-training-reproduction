"""Talker tasks (talker_worker_hy.py format) for every registered grid point, direction and talker seed
(plan §4: seeds default 20261021 / 1 / 2). Refuses an existing tasks file.

    python experiments/baselines_hysimulmt_v1/make_tasks.py eval_outputs/baselines_hysimulmt_v1/tasks.jsonl
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hy_realsi import GRID  # noqa: E402

E = REPO / "eval_outputs/baselines_hysimulmt_v1"
rows = []
for seed in (0, 1, 2):           # seed-major order: every point gets its first seed before any gets a third
    for point in GRID:
        for d in ("en2zh", "zh2en"):
            w = GRID[point][0] / 1000   # --window is a label only (text policy file)
            name = point + (f"-seed{seed}" if seed else "")
            rows.append({"key": f"hy_{point}_{d}_s{seed}", "blocks": str(E / "thinker" / point / f"blocks_{d}.jsonl"),
                         "direction": d, "window": w, "seed": seed,
                         "out": str(E / "talker" / f"system_{name}_m{w:g}" / f"realsi_{d}")})
with open(sys.argv[1], "x", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in rows)
print(len(rows), "tasks")
