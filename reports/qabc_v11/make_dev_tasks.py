"""prereg_v6 step 1: the throughput-control choice on the spontaneous dev (non-test long audio).

The dev pseudo-talks (cont_queue.pseudo_talks) carry LA2's and A0's (s0, b0 = 1, SDR) sentence-level blocks of v4
(si_v4/devsp_la2, si_v4/devsp_s0) at read windows 0.5/1/2 s; every block gets the tracked queue and shift of each
control in cont_queue.CONTROLS (gamma = GAMMA0); the Talker speaks every file with talker seed 0. Writes annotated
blocks to qabc_v11/devcont/blocks/ and talker_worker.py tasks to <tasks.jsonl>; refuses to overwrite.

    PYTHONPATH=. python reports/qabc_v11/make_dev_tasks.py eval_outputs/uniss_qwen3_omni_v1/qabc_v11/devcont/tasks.jsonl
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cont_queue as cq  # noqa: E402

O = cq.REPO / "eval_outputs/uniss_qwen3_omni_v1"
V4 = O / "si_v4"
DEV = O / "qabc_v11/devcont"
WINDOWS = ("0.5", "1", "2")


def src(system, d, w):
    return (V4 / f"devsp_la2/blocks_{d}_w{w}.jsonl" if system == "la2"
            else V4 / f"devsp_s0/blocks_{d}_w{w}_sws0b1q{cq.RATE[d]:g}.jsonl")


def main():
    tasks_path = Path(sys.argv[1])
    (DEV / "blocks").mkdir(parents=True, exist_ok=True)
    tasks = []
    for d in ("en2zh", "zh2en"):
        talks = cq.pseudo_talks(d)
        for system in ("la2", "a0"):
            for w in WINDOWS:
                base = cq.read_blocks(src(system, d, w))
                for ctrl in cq.CONTROLS:
                    b = copy.deepcopy(base)
                    cq.carry(b, talks, d, ctrl, cq.GAMMA0)
                    out = DEV / "blocks" / f"{system}_{d}_w{w}_{ctrl}.jsonl"
                    n = cq.write_blocks(src(system, d, w), b, out)
                    key = f"dev_{system}_{d}_m{w}_{ctrl}"
                    tasks.append({"key": key, "blocks": str(out), "direction": d, "window": float(w), "seed": 0,
                                  "out": str(DEV / "speech" / f"{system}_{d}_m{w}_{ctrl}"), "items": str(cq.SPONT[d]),
                                  "speed_policy": "tracked"})
                    print(key, n)
    with open(tasks_path, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(t) + "\n" for t in tasks)
    print(len(tasks), "tasks ->", tasks_path)


if __name__ == "__main__":
    main()
