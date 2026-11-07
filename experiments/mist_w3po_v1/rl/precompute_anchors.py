"""Precompute the FLAD fixed-lag anchors (rollout.anchors, greedy, base model = the RL reference policy with the adapters
off) for the windows an RL run will sample, so the trainer does not spend ~150 s per step on them (2026-11-07 pilot:
anchors 150-200 s of a 470-650 s step). The window stream is the trainer's own (Sampler(seed), --windows-per-step,
--oversample, steps [--from-step, --steps)), so the anchors are exactly those the trainer would compute; the gold
context string is the trainer's gold_ctx(seed, --ctx-max). Output records have the trainer's anchor-cache format, one
file per shard: <out>/part_<k>.jsonl (appended per batch; a restart skips ids already there). The trainer reads them
with --anchor-cache <out>.

    python -m experiments.mist_w3po_v1.rl.precompute_anchors --base <merged> --out <dir> --steps 300 \
        --windows-per-step 64 --ctx-max 0 --shard k --num-shards n
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from experiments.mist_w3po_v1.core.engine import Engine
from experiments.mist_w3po_v1.rl.rollout import anchors as run_anchors
from experiments.mist_w3po_v1.rl.train_w3po import Sampler, gold_ctx


def window_stream(a) -> list[dict]:
    W = [json.loads(l) for l in open(a.windows, encoding="utf-8")]
    W = [w for w in W if w.get("utts") and w["dur_ms"] <= a.max_dur_ms]
    s = Sampler(W, a.seed)
    n = math.ceil(a.windows_per_step * a.oversample / a.world) * a.world
    seen, out = set(), []
    for step in range(a.from_step, a.steps):
        for w in s.take(step, n):
            if w["id"] not in seen:
                seen.add(w["id"])
                out.append(w)
    return out


def meeting_refs(a, meetings) -> tuple[dict, dict]:
    sess = json.loads((Path(a.ref_src) / "slot_map.json").read_text())["lowlat"]
    en = {r["record_id"]: r["en"] for r in map(json.loads, open(a.en_norm, encoding="utf-8"))}
    meet_utts = {}
    for m in meetings:
        us = []
        for u in map(json.loads, open(Path(a.utts) / f"{m}.jsonl", encoding="utf-8")):
            if u["hes"] or not u.get("placed", True) or any(f.startswith("excluded:") for f in u.get("flags", [])):
                continue
            ref = en.get(u["record_id"], "").strip()
            if ref and u["speaker_id"] in sess.get(m, {}):
                us.append((u["zh_end"], u["speaker_id"], ref))
        meet_utts[m] = sorted(us)
    return meet_utts, sess


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_train_diar.jsonl")
    ap.add_argument("--dir", default="zh2en")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--from-step", type=int, default=0)
    ap.add_argument("--windows-per-step", type=int, default=64)
    ap.add_argument("--oversample", type=float, default=1.0)
    ap.add_argument("--world", type=int, default=4, help="trainer processes (only rounds the per-step count)")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--ctx-max", type=int, default=2)
    ap.add_argument("--lags", default="1000,2000,3000")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--max-dur-ms", type=int, default=20500)
    ap.add_argument("--batch", type=int, default=8, help="windows per anchors() call")
    ap.add_argument("--max-gib", type=int, default=120)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--ref-src", default="data/ami_multispk_si_v1/e2e_gstar_zh2en_v1")
    ap.add_argument("--utts", default="data/ami_multispk_si_v1/zh_mix_v8/utts")
    ap.add_argument("--en-norm", default="data/ami_multispk_si_v1/en_norm_v1.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ws = window_stream(a)[a.shard::a.num_shards]
    done = set()
    for f in out.glob("part_*.jsonl"):
        for l in open(f, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [w for w in ws if w["id"] not in done]
    print(f"{len(ws)} windows in shard {a.shard}/{a.num_shards}, {len(todo)} to do", flush=True)
    meet_utts, sess = meeting_refs(a, {w["meeting"] for w in todo})
    eng = Engine.from_merged(a.base, a.dir, a.max_gib)
    eng.thinker.eval()
    lags = [int(x) for x in a.lags.split(",")]
    t0 = time.time()
    with open(out / f"part_{a.shard}.jsonl", "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            ctxs = {w["id"]: gold_ctx(w, meet_utts, sess, a.seed, a.ctx_max) for w in chunk}
            got = run_anchors(eng, chunk, ctxs, lags, max_new=a.max_new, cache={})
            for w in chunk:
                rec = {"id": w["id"], "ctx": ctxs[w["id"]], "trajs": [
                    {"kind": tr.kind, "reads": [{"t": rd.t_ms, "final": rd.final, "before": rd.before, "ids": rd.ids,
                                                 "akey": rd.akey} for rd in tr.reads]} for tr in got[w["id"]]]}
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{i + len(chunk)}/{len(todo)} windows, {time.time() - t0:.0f}s", flush=True)
    print(f"ANCHORS_DONE {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
