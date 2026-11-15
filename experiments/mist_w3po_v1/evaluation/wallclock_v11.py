"""v11 P1-1 (docs/mist_v11_plan.zh-CN.md §3; mock ACL review M5): computation-aware replay of whole meetings from
measured compute times (timed_reads.py, timed_talker.py), for one system and one test set.

Model (no other approximation than the ones listed):
  Thinker   one decoder stream per meeting (batch 1). Read t of window w is due at w.start + t, clamped to the
            window's opening (open_detect_ms) and, for its last read, to the moment its end is known
            (close_detect_ms) -- the chunk protocol's release rule, as reports/gstar_si_v8/b6_wallclock.py.
            start = max(due, end of the previous read), end = start + measured ms. A read's commit is its end
            (computation-free: its due time).
  Talker    one Talker worker per meeting. Block (item, r) is released at the commit of the read that wrote it
            (window, write_ms). FIFO by release; block r of an item also waits for block r-1 of the same item (its
            prompt holds the codes spoken so far). Cost = its own generation (both candidates) + an equal share of
            its round's candidate scoring + an equal share of generations not attributable to one block (reruns)
            + front-end and final BiCodec decode spread over blocks (front end per block, decode per second of
            speech). Computation-free: ready = due release.
  Playback  layout "per-speaker": one channel per (meeting, diarizer slot); "single": one channel per meeting.
            A block plays from max(ready, end of the previous block on its channel) for codes x 20 ms.
Outputs, per layout, for computation-aware ("ca") and computation-free ("cf") timelines and their per-block
difference: block lag = play end - source end of the block's text (EndOffset-like), first-word lag = play start -
source start, playback backlog = play start - ready; p50 / p90 / mean / max, and the slope of the ca-cf lag over
meeting time (s per minute; > 0 means the system falls further behind as the meeting goes on). Thinker / Talker
real-time factor = total compute / meeting span (first due .. last due), unprocessed-read queue (reads due but not
finished, sampled at every due time), read queueing delay, share of reads above the 1 s cadence.

    python -m experiments.mist_w3po_v1.evaluation.wallclock_v11 --windows <online windows> --meetings m1,m2 \
        --reads <timed reads jsonl> --talker-in <dir: items.jsonl, blocks.jsonl> --talker-out <si_infer output dir> \
        --talker-timing <timed_talker jsonl> --talker-log <si_infer log> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CADENCE_MS = 1000
CODE_MS = 20


def read_due(w: dict, t_ms: float) -> float:
    x = max(t_ms, w["open_detect_ms"])
    if t_ms >= w["dur_ms"]:
        x = max(x, w["close_detect_ms"])
    return w["start_s"] + x / 1000.0


def stats(xs) -> dict:
    a = np.asarray([x for x in xs if x is not None and not math.isnan(x)], dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def thinker_queue(reads: list[dict]) -> dict:
    """reads: {"key", "due", "ms"} of one meeting. Returns per key commit times and the queue summary."""
    order = sorted(range(len(reads)), key=lambda i: (reads[i]["due"], i))
    prev, commit, waits, ends = -math.inf, {}, [], []
    for i in order:
        r = reads[i]
        s = max(r["due"], prev)
        e = s + r["ms"] / 1000.0
        prev = e
        commit[r["key"]] = e
        waits.append(s - r["due"])
        ends.append(e)
    dues = sorted(r["due"] for r in reads)
    ends_sorted = sorted(ends)
    qlen = []
    for d in dues:   # reads due by d and not finished by d
        due_n = int(np.searchsorted(dues, d, side="right"))
        done_n = int(np.searchsorted(ends_sorted, d, side="right"))
        qlen.append(due_n - done_n)
    span = (dues[-1] - dues[0]) if len(dues) > 1 else float("nan")
    tot = sum(r["ms"] for r in reads) / 1000.0
    return {"commit": commit,
            "summary": {"reads": len(reads), "compute_s": tot, "span_s": span, "rtf": tot / span if span else None,
                        "read_ms": stats([r["ms"] for r in reads]),
                        "over_cadence": float(np.mean([r["ms"] > CADENCE_MS for r in reads])),
                        "queue_wait_s": stats(waits), "queue_reads": stats(qlen),
                        "drift_s": float(ends[-1] - max(r["due"] for r in reads)) if reads else None}}


def talker_costs(timing: list[dict], blocks: list[dict], front_s: float, decode_s: float) -> dict:
    """(item id, r) -> compute seconds."""
    gen = defaultdict(float)
    loose = defaultdict(float)   # r -> generations without a single owner
    score = defaultdict(float)
    nround = {}
    for d in timing:
        if d["kind"] == "gen":
            if d.get("owner") is not None:
                gen[(d["owner"], d["r"])] += d["ms"] / 1000.0
            else:
                loose[d["r"]] += d["ms"] / 1000.0
        elif d["kind"] == "score":
            score[d["r"]] += d["ms"] / 1000.0
        elif d["kind"] == "round":
            nround[d["r"]] = d["n"]
    n_blocks = max(1, len(blocks))
    speech = sum(b["dur_s"] for b in blocks) or 1.0
    out = {}
    for b in blocks:
        k = (b["item"], b["r"])
        n = max(1, nround.get(b["r"], 1))
        out[k] = (gen.get(k, 0.0) + (score[b["r"]] + loose[b["r"]]) / n + front_s / n_blocks
                  + decode_s * b["dur_s"] / speech)
    return out


def replay(blocks: list[dict], cost: dict, layout: str) -> list[dict]:
    """blocks: {"item", "r", "meeting", "slot", "rel_ca", "rel_cf", "dur_s", "src_start", "src_end"}. Returns the
    blocks with ready / play start / play end for "ca" and "cf"."""
    res = [dict(b) for b in blocks]
    by_m = defaultdict(list)
    for b in res:
        by_m[b["meeting"]].append(b)
    for m, bs in by_m.items():
        # Talker worker, computation-aware
        free, done = -math.inf, {}
        for b in sorted(bs, key=lambda x: (x["rel_ca"], x["item"], x["r"])):
            s = max(b["rel_ca"], free, done.get((b["item"], b["r"] - 1), -math.inf))
            b["ready_ca"] = s + cost.get((b["item"], b["r"]), 0.0)
            free = b["ready_ca"]
            done[(b["item"], b["r"])] = b["ready_ca"]
        for b in bs:
            b["ready_cf"] = b["rel_cf"]
        for mode in ("ca", "cf"):
            chan_end = defaultdict(lambda: -math.inf)
            for b in sorted(bs, key=lambda x: (x[f"rel_{mode}"], x["item"], x["r"])):
                ch = (m, b["slot"]) if layout == "per-speaker" else (m,)
                s = max(b[f"ready_{mode}"], chan_end[ch])
                b[f"start_{mode}"], b[f"end_{mode}"] = s, s + b["dur_s"]
                chan_end[ch] = b[f"end_{mode}"]
    return res


def summarise(res: list[dict]) -> dict:
    out = {}
    for mode in ("ca", "cf"):
        out[mode] = {"end_lag": stats([b[f"end_{mode}"] - b["src_end"] for b in res]),
                     "start_lag": stats([b[f"start_{mode}"] - b["src_start"] for b in res]),
                     "backlog": stats([b[f"start_{mode}"] - b[f"ready_{mode}"] for b in res])}
    diff = [(b["src_end"], b["end_ca"] - b["end_cf"]) for b in res]
    out["ca_minus_cf_end"] = stats([d for _, d in diff])
    out["ca_minus_cf_start"] = stats([b["start_ca"] - b["start_cf"] for b in res])
    if len(diff) > 2:
        x = np.array([t for t, _ in diff]) / 60.0
        y = np.array([d for _, d in diff])
        out["ca_minus_cf_slope_s_per_min"] = float(np.polyfit(x, y, 1)[0])
    return out


def parse_timing_line(log_text: str) -> tuple[float, float]:
    m = re.findall(r"^timing (.*)$", log_text, flags=re.M)
    if not m:
        raise SystemExit("no 'timing' line in the talker log")
    kv = dict(re.findall(r"(\w+) ([0-9.]+)s", m[-1]))
    return float(kv.get("front_end", 0.0)), float(kv.get("decode", 0.0))


def load(args) -> dict:
    meetings = set(args.meetings.split(","))
    W = {}
    for l in open(args.windows, encoding="utf-8"):
        w = json.loads(l)
        if w["meeting"] in meetings:
            W[w["id"]] = w
    reads_by_m = defaultdict(list)
    for f in args.reads:
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            if len(d["akeys"]) != 1:
                raise SystemExit(f"batched read in {f}: use --batch 1")
            wid, t = d["akeys"][0].rsplit("|", 1)
            if wid in W:
                reads_by_m[W[wid]["meeting"]].append({"key": (wid, int(t)), "due": read_due(W[wid], int(t)),
                                                      "ms": d["ms"]})
    tin = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.talker_in) / "blocks.jsonl", encoding="utf-8")}
    tout = {}
    for f in sorted(Path(args.talker_out).glob("part_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            tout[r["id"]] = r
    timing = [json.loads(l) for l in open(args.talker_timing, encoding="utf-8")]
    front_s, decode_s = parse_timing_line(Path(args.talker_log).read_text(encoding="utf-8", errors="replace"))
    return {"W": W, "reads": reads_by_m, "tin": tin, "tout": tout, "timing": timing, "front_s": front_s,
            "decode_s": decode_s}


def run(D: dict) -> dict:
    W = D["W"]
    th = {m: thinker_queue(rs) for m, rs in D["reads"].items()}
    commit = {k: v for m in th for k, v in th[m]["commit"].items()}
    blocks, missing = [], 0
    for iid, rec in D["tout"].items():
        wid, slot = iid.rsplit("__", 1)
        if wid not in W:
            continue
        w = W[wid]
        src = D["tin"].get(iid, {}).get("blocks", [])
        for r, b in enumerate(rec["blocks"]):
            key = (wid, int(b["write_ms"]))
            if key not in commit:
                missing += 1
                continue
            sb = src[r] if r < len(src) else {}
            blocks.append({"item": iid, "r": r, "meeting": w["meeting"], "slot": slot, "rel_ca": commit[key],
                           "rel_cf": read_due(w, b["write_ms"]), "dur_s": b["codes"] * CODE_MS / 1000.0,
                           "src_start": w["start_s"] + sb.get("start_s", 0.0),
                           "src_end": w["start_s"] + sb.get("end_s", b["write_ms"] / 1000.0)})
    cost = talker_costs(D["timing"], blocks, D["front_s"], D["decode_s"])
    span = sum(th[m]["summary"]["span_s"] for m in th)
    talk_s = sum(cost.values())
    out = {"meetings": sorted(th), "thinker": {m: th[m]["summary"] for m in th},
           "thinker_rtf_all": sum(th[m]["summary"]["compute_s"] for m in th) / span,
           "talker": {"blocks": len(blocks), "blocks_missing_read": missing, "compute_s": talk_s, "rtf": talk_s / span,
                      "block_s": stats(list(cost.values())), "front_end_s": D["front_s"], "decode_s": D["decode_s"],
                      "gen_ms": stats([d["ms"] for d in D["timing"] if d["kind"] == "gen"]),
                      "score_ms": stats([d["ms"] for d in D["timing"] if d["kind"] == "score"])},
           "layouts": {}}
    for lay in ("per-speaker", "single"):
        out["layouts"][lay] = summarise(replay(blocks, cost, lay))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--meetings", required=True)
    ap.add_argument("--reads", nargs="+", required=True)
    ap.add_argument("--talker-in", required=True)
    ap.add_argument("--talker-out", required=True)
    ap.add_argument("--talker-timing", required=True)
    ap.add_argument("--talker-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {args.out}")
    res = run(load(args))
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"thinker_rtf": res["thinker_rtf_all"], "talker_rtf": res["talker"]["rtf"],
                      "end_lag_ca_minus_cf_p50": res["layouts"]["per-speaker"]["ca_minus_cf_end"].get("p50")}))


if __name__ == "__main__":
    main()
