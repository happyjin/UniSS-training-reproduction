"""CPU audit of the FLAD anchors an RL run used (2026-11-07: pilot anchor_R 0.07 / 0.17 / 0.19 at steps 6 / 12 / 13).

For each step of [--from-step, --to-step) it rebuilds the trainer's window stream (Sampler(seed), --windows-per-step),
finds every anchor record of each window in every source (the run's anchors/rank*.jsonl, in file order, and the
--anchor-cache dirs' *.jsonl), and reports
  * per step: anchor R recomputed from the records (as the trainer's metric: mean over groups of the mean anchor R),
    the share of the step's windows that are "hard", the windows' mean unit count and duration
  * disagreements: windows with >1 record whose committed ids differ (cached vs in-trainer), with both R values
  * misalignment: records whose read times / ctx do not match the window (read_times(dur), gold_ctx(seed, ctx-max))
  * the lowest-R windows of the anomalous steps (id, units, dur, anchor R per lag, policy-free text)

    python -m experiments.mist_w3po_v1.rl.check_anchor_rewards --run eval_outputs/mist_w3po_v1/rl/pilot_w3po_v8 \
        --anchor-cache <dir> --to-step 13 --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.core.engine import read_times
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.rollout import Read, Traj, load_window, parse
from experiments.mist_w3po_v1.rl.train_w3po import Sampler, gold_ctx, hard


def records(run: Path, caches: list[str]) -> dict:
    """id -> [(source, line no, record)] in the trainer's load order (own rank file first, then caches)."""
    out = defaultdict(list)
    files = sorted((run / "anchors").glob("rank*.jsonl")) + sorted(
        f for d in caches if d for f in Path(d).glob("*.jsonl"))
    for f in files:
        for n, l in enumerate(open(f, encoding="utf-8")):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            out[r["id"]].append((str(f), n, r))
    return out


def anchor_trajs(rec: dict, w: dict, tok) -> list[Traj]:
    lst = []
    for t in rec["trajs"]:
        tr = Traj(w["id"], t["kind"], rec["ctx"])
        for rd in t["reads"]:
            n = len(rd["ids"])
            tr.reads.append(Read(rd["t"], rd["final"], rd["before"], rd["ids"], [False] * n, [False] * n,
                                 [0.0] * n, [0.0] * n, [0.0] * n, rd["akey"]))
        parse(tr, tok, w["dur_ms"] / 1000)
        lst.append(tr)
    return lst


def anchor_R(rec: dict, w: dict, tok, units, post) -> list[float]:
    dur = w["dur_ms"] / 1000
    return [S.credit(tr.segs, units, dur, tr.wait_reads, [rd.t_ms / 1000 for rd in tr.reads], post=post).R
            for tr in anchor_trajs(rec, w, tok)]


def misaligned(rec: dict, w: dict, ctx: str) -> list[str]:
    bad = []
    times = read_times(w["dur_ms"])
    if rec["ctx"] != ctx:
        bad.append("ctx")
    for t in rec["trajs"]:
        ts = [rd["t"] for rd in t["reads"]]
        if ts != list(times):
            bad.append(f"{t['kind']}:times")
        for rd in t["reads"]:
            if not rd["akey"].startswith(w["id"] + "|") or rd["akey"] != f"{w['id']}|{rd['t']}":
                bad.append(f"{t['kind']}:akey")
                break
        bef = []
        for rd in t["reads"]:
            if rd["before"] != bef:
                bad.append(f"{t['kind']}:before")
                break
            bef = bef + [i for i in rd["ids"] if i not in (S_EOS, S_WAIT)]
    return bad


S_EOS = S_WAIT = None


def main() -> None:
    global S_EOS, S_WAIT
    from experiments.mist_w3po_v1.core.engine import EOS_ID, WAIT_ID
    S_EOS, S_WAIT = EOS_ID, WAIT_ID
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--anchor-cache", default="")
    ap.add_argument("--from-step", type=int, default=0)
    ap.add_argument("--to-step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worst", type=int, default=6)
    a = ap.parse_args()
    run = Path(a.run)
    args = json.loads((run / "args.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args["base"])
    W = [json.loads(l) for l in open(args["windows"], encoding="utf-8")]
    W = [w for w in W if w.get("utts") and w["dur_ms"] <= args["max_dur_ms"]]
    sampler = Sampler(W, args["seed"])
    sess = json.loads((Path(args["ref_src"]) / "slot_map.json").read_text())["lowlat"]
    en = {r["record_id"]: r["en"] for r in map(json.loads, open(args["en_norm"], encoding="utf-8"))}
    recs = records(run, a.anchor_cache.split(","))
    metrics = {}
    for l in open(run / "metrics.jsonl", encoding="utf-8"):
        m = json.loads(l)
        metrics[m["step"]] = m          # last occurrence wins (resumes repeat steps)
    n = args["windows_per_step"]
    steps = {}
    disagree = []
    mis = []
    meet_cache = {}
    for step in range(a.from_step, a.to_step):
        ws = sampler.take(step, n)
        per = []
        for w in ws:
            m = w["meeting"]
            if m not in meet_cache:
                us = []
                for u in map(json.loads, open(Path(args["utts"]) / f"{m}.jsonl", encoding="utf-8")):
                    if u["hes"] or not u.get("placed", True) or any(f.startswith("excluded:") for f in u.get("flags", [])):
                        continue
                    ref = en.get(u["record_id"], "").strip()
                    if ref and u["speaker_id"] in sess.get(m, {}):
                        us.append((u["zh_end"], u["speaker_id"], ref))
                meet_cache[m] = sorted(us)
            ctx = gold_ctx(w, meet_cache, sess, args["seed"], args["ctx_max"])
            rs = recs.get(w["id"], [])
            units = S.units_of(w["utts"])
            _, post = load_window(w)
            Rs = []
            for src, ln, r in rs:
                Rs.append((src, ln, anchor_R(r, w, tok, units, post)))
                bad = misaligned(r, w, ctx)
                if bad:
                    mis.append({"step": step + 1, "id": w["id"], "src": src, "line": ln, "bad": bad})
            if len(rs) > 1:
                ids0 = [[rd["ids"] for rd in t["reads"]] for t in rs[0][2]["trajs"]]
                for (src, ln, r), (_, _, R) in zip(rs[1:], Rs[1:]):
                    ids1 = [[rd["ids"] for rd in t["reads"]] for t in r["trajs"]]
                    if ids1 != ids0:
                        disagree.append({"step": step + 1, "id": w["id"], "a": [rs[0][0], rs[0][1], Rs[0][2]],
                                         "b": [src, ln, R]})
            per.append({"id": w["id"], "hard": hard(w), "units": len(units), "dur": w["dur_ms"] / 1000,
                        "n_rec": len(rs), "src0": Path(rs[0][0]).name if rs else None,
                        "R": Rs[0][2] if Rs else None,
                        "text": [" | ".join(s["text"] for s in tr.segs) for tr in anchor_trajs(rs[0][2], w, tok)]
                        if rs else None,
                        "ref": [u["ref"] for u in units]})
        have = [p for p in per if p["R"] is not None]
        rec = {"step": step + 1, "n": len(per), "n_with_anchor": len(have),
               "anchor_R_recomputed": float(np.mean([np.mean(p["R"]) for p in have])) if have else None,
               "anchor_R_logged": metrics.get(step + 1, {}).get("anchor_R"),
               "R_logged": metrics.get(step + 1, {}).get("R"),
               "units_mean": float(np.mean([p["units"] for p in per])),
               "dur_mean": float(np.mean([p["dur"] for p in per])),
               "srcs": sorted({Path(p["src0"]).stem.rsplit("_", 1)[0] if p["src0"] else "none" for p in per}),
               "src_counts": {k: sum(1 for p in per if (p["src0"] or "none") == k) for k in {p["src0"] or "none" for p in per}},
               "zero_R_windows": sum(1 for p in have if max(p["R"]) <= 0.0),
               "neg_R_windows": sum(1 for p in have if np.mean(p["R"]) < 0.0),
               "worst": sorted(have, key=lambda p: float(np.mean(p["R"])))[: a.worst]}
        steps[step + 1] = rec
        print(json.dumps({k: v for k, v in rec.items() if k != "worst"}), flush=True)
    res = {"steps": steps, "disagree": disagree, "misaligned": mis}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"disagreeing records: {len(disagree)}, misaligned records: {len(mis)}")
    for d in disagree[:20]:
        print(json.dumps(d))
    for d in mis[:20]:
        print(json.dumps(d))


if __name__ == "__main__":
    main()
