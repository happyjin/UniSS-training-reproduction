"""Deadline-label streaming rows with mixed read intervals (paper v4, the reviewer's multi-interval request): the chunks
and commit times that build_ami_gstar_stream_rows.py stored in each training window (stream_target), read every Delta
seconds with Delta drawn per window from --windows, so that one model sees the read times of several intervals.
Rows, mixing (whole-window offline rows --offline-share, single-speaker rows --sd-share) and the empty-read rule are as
in build_ami_gstar_stream_rows.py; only the read times change. En->Zh (the stored windows of e2e_gstar_stream_v1).

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_stream_rows_multid \
        --out data/ami_multispk_si_v1/e2e_gstar_stream_multid_v1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

LOOKAHEAD_MS = 1040


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", default="data/ami_multispk_si_v1/e2e_gstar_stream_v1")
    ap.add_argument("--offline", default="data/ami_multispk_si_v1/e2e_gstar_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="0.5,1,2")
    ap.add_argument("--keep-empty", type=float, default=0.5)
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--sd-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    st, out = Path(args.stream), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    deltas = [round(float(x) * 1000) for x in args.windows.split(",")]
    stats = defaultdict(lambda: defaultdict(int))
    rows = {"train": [], "validation": []}
    for split in rows:
        for w in map(json.loads, open(st / f"windows_{split}_ref.jsonl", encoding="utf-8")):
            tg = w["stream_target"]
            chunks, dur, preset = tg["chunks"], w["dur_ms"], tg["preset"]
            win_ms = rng.choice(deltas)
            stats[split][f"windows_d{win_ms}"] += 1
            reads = list(range(win_ms, dur, win_ms)) + [dur]
            prev = 0
            for t in reads:
                last = t >= dur
                done = [c for c in chunks if c["commit_ms"] <= prev]
                new = [c for c in chunks if c["commit_ms"] > prev and (last or c["commit_ms"] <= t)]
                target = "".join(c["piece"] for c in new)
                if target or last or rng.random() < args.keep_empty:
                    rows[split].append({"id": f"{w['id']}|{preset}|d{win_ms}|{t}", "audio": w["audio"], "dir": "en2zh",
                                        "t_ms": t, "dur_ms": dur, "post": w["post"][preset],
                                        "post_until_ms": max(0, t - LOOKAHEAD_MS),
                                        "prefix": "".join(c["piece"] for c in done), "target": target,
                                        "end": "eos" if last else "wait", "kind": "read", "inst": "gstar",
                                        "preset": preset})
                    stats[split]["reads"] += 1
                    stats[split]["empty_reads"] += not target
                prev = t
    off_rows = [json.loads(l) for l in open(Path(args.offline) / "rows_train.jsonl", encoding="utf-8")]
    gstar_off = [r for r in off_rows if r.get("inst") == "gstar"]
    sd_off = [r for r in off_rows if r.get("inst") == "sd"]
    n = len(rows["train"])
    rows["train"] += rng.sample(gstar_off, min(len(gstar_off), round(args.offline_share * n)))
    rows["train"] += rng.sample(sd_off, min(len(sd_off), round(args.sd_share * n)))
    out.mkdir(parents=True)
    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(out / f"rows_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    summary = {sp: dict(v) for sp, v in stats.items()}
    summary["rows"] = {sp: len(rs) for sp, rs in rows.items()}
    summary["rows"]["train_read"] = n
    summary["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
