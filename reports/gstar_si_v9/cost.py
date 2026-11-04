"""Paper v9 P1-8 (review M11): cost. Training: wall-clock between the first and last logged Megatron iteration of each
training run x its GPUs (logs/multispk/**/thinker_gstar_*.log; world_size from the logged arguments). Inference: the
queued-read simulation of listener_latency_v8.py (median single-stream read time measured in v6 on 2 H200 cards,
batch 1): GPU-hours per meeting-hour = RTF x 2 cards; plus the read-time medians themselves. Every read re-encodes the
whole window prefix (no key-value cache), so a read's cost grows with the time since the window opened.
Writes <out> (refuses to overwrite).

    python3 reports/gstar_si_v9/cost.py --out eval_outputs/uniss_qwen3_omni_v1/gstar_v9_20261104/cost.json
"""
import argparse
import glob
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "eval_outputs/uniss_qwen3_omni_v1"
RUNS = {"MIST (offline)": "thinker_gstar_o2_20261029b", "MIST-Simul En-Zh": "thinker_gstar_s2dl_20261029b",
        "MIST-Simul Zh-En": "thinker_gstar_s2dl_zh2en_3k_20261030",
        "speaker-blind Simul En-Zh (C1)": "thinker_gstar_s2dlnosd_en2zh_v8",
        "speaker-blind Simul Zh-En (C1)": "thinker_gstar_s2dlnosd_zh2en_v8"}
TS = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+\] iteration\s+(\d+)/\s*(\d+)")


def train(run):
    logs = [p for p in glob.glob(str(REPO / "logs/multispk/**/*.log"), recursive=True) if Path(p).stem == run]
    if not logs:
        return None
    first = last = None
    ws = None
    for ln in open(logs[0], errors="ignore"):
        m = TS.search(ln)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            first = first or (t, int(m.group(2)))
            last = (t, int(m.group(2)), int(m.group(3)))
        elif ws is None and re.match(r"\s+world_size \.+ (\d+)", ln):
            ws = int(re.match(r"\s+world_size \.+ (\d+)", ln).group(1))
    if not first or not last or not ws:
        return None
    h = (last[0] - first[0]).total_seconds() / 3600
    per_it = h / max(last[1] - first[1], 1)
    return {"gpus": ws, "iterations": last[2], "hours": round(per_it * last[2], 2),
            "gpu_hours": round(per_it * last[2] * ws, 1), "log": str(Path(logs[0]).relative_to(REPO))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {"training": {k: train(v) for k, v in RUNS.items()}, "inference": {}}
    L = EV / "gstar_v8_20261103/latency"
    for st in ("en2zh", "zh2en", "ali"):
        for s in ("S2dl", "O2_LA2"):
            f = L / f"{st}_{s}_single_queue.json"
            if f.exists():
                r = json.load(open(f))
                res["inference"][f"{st}_{s}"] = {"read_ms_median": r["read_ms"], "rtf_simulated": r["rtf_mean"],
                                                 "gpu_hours_per_meeting_hour": round(2 * r["rtf_mean"], 2), "cards": 2}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
