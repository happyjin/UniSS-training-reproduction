"""Post-hoc check of an si_infer --backlog-scope meeting run (talker_crf_prereg_v2 §2.3, E2 / E2'): every block's backlog
token (results' backlog_bin) against the shared channel's backlog recomputed from the run's own block durations
(shared_channel.shared_backlogs: speech_latency_lit's FIFO one-channel schedule), with both duration readings
(codes x 20 ms and dur_ms); also the largest |backlog_ms_shared - recomputed| where the run logged it. CPU.

    python -m experiments.mist_w3po_v1.talker.check_shared_bins <talker dir> <dev windows jsonl> [--out <json>]
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from experiments.mist_w3po_v1.talker import shared_channel as SC


def check(talker: str, win: str) -> dict:
    W = {w["id"]: w for w in map(json.loads, open(win, encoding="utf-8"))}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{talker}/part_*.jsonl")) for l in open(f, encoding="utf-8")]
    items = [{"id": r["id"], "blocks": r["blocks"]} for r in rows if r.get("blocks") and not r.get("error")]
    res = {"talker": talker, "items": len(items), "blocks": sum(len(it["blocks"]) for it in items)}
    for name, durs in (("codes_x20", {(n, i): b["codes"] * 20 for n, it in enumerate(items) for i, b in enumerate(it["blocks"])}),
                       ("dur_ms", {(n, i): b["dur_ms"] for n, it in enumerate(items) for i, b in enumerate(it["blocks"])})):
        bl = SC.shared_backlogs(items, W, durs)
        c, dmax, nlog = Counter(), 0.0, 0
        for (n, i), v in bl.items():
            b = items[n]["blocks"][i]
            c["match" if SC.backlog_bin(v) == b.get("backlog_bin") else "mismatch"] += 1
            if "backlog_ms_shared" in b:
                nlog += 1
                dmax = max(dmax, abs(b["backlog_ms_shared"] - v))
        res[name] = {"bin_match": c["match"], "bin_mismatch": c["mismatch"], "logged": nlog,
                     "max_abs_logged_minus_recomputed_ms": round(dmax, 2)}
    res["bin_hist"] = sorted(Counter(b.get("backlog_bin") for it in items for b in it["blocks"]).items(),
                             key=lambda x: (x[0] is None, x[0]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("talker")
    ap.add_argument("win")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = check(a.talker, a.win)
    if a.out:
        if Path(a.out).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
