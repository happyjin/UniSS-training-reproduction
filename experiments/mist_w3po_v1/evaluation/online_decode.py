"""W3PO online decoding (plan §4 S0-4, §6): gstar_si_stream_online_v6s.py's --policy wait / lag on whole-meeting online
windows, with the cross-window context of S0-4 and a per-read log.

  --context N   every window's prompt carries the last N committed segments of the previous window of the same
                meeting (engine.ctx_string); windows of one meeting are then decoded in order, one "lane" per meeting,
                lanes batched together. N = 0 reproduces v6s (lanes then take windows from one queue). The previous
                window's segments are those of its complete output (its final read); in the online system that read
                happens at most 0.24 s after the next window's first read (20 s cap case), a deviation logged in
                reports/mist_w3po_v1/LOG.zh-CN.md.
  --policy wait  generate until <wait> (last read: <eos>), commit everything; lag: generate to the end every read,
                commit complete segments ending <= t - --lag-ms (the same-model fixed-lag baseline / FLAD anchor).
  read_log      per read: t, the committed text added ("new"); with the window's ctx, enough to rebuild every
                read's prompt (DAgger rows, S0-2).
Output lines as v6s ({"id", "raw", "translation", "segments", "blocks", "reads"}) + "read_log", "ctx". Each window is
written when it finishes; --resume-from <files> skips windows already decoded there (and seeds each meeting's
context from them), so an interrupted job continues into a new --out file. Refuses an existing --out.

    python -m experiments.mist_w3po_v1.evaluation.online_decode --thinker <merged> --windows <jsonl> --out <jsonl> \
        --policy wait --context 2 --dir zh2en --shard k --num-shards n
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.mist_w3po_v1.core.engine import (
    EOS_ID, WAIT_ID, Engine, Parser, ReadItem, ctx_string, lag_cut)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=["wait", "lag"], default="wait")
    ap.add_argument("--lag-ms", type=int, default=2000)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--dir", default="zh2en", choices=["en2zh", "zh2en"])
    ap.add_argument("--post-key", default="lowlat")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--max-gib", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-by", choices=["meeting", "window"], default="meeting")
    ap.add_argument("--meetings", default="", help="comma list: only these meetings")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume-from", nargs="*", default=[])
    ap.add_argument("--lora", default="", help="an RL adapter (train_w3po ckpt lora.pt), folded into --thinker")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--loop-guard-k", type=int, default=0,
                    help="decode-time loop guard (evaluation/loop_guard.py): cut a read's block after the first copy of a "
                         "pattern repeated >= k times in a row; 0 = off (default, output unchanged)")
    ap.add_argument("--loop-guard-maxp", type=int, default=12, help="longest repeated pattern (tokens) the guard checks")
    ap.add_argument("--budget-tps", type=float, default=0.0,
                    help="with > 0: a window's committed ids are capped at budget-tps * audio s + --budget-slack; 0 = off")
    ap.add_argument("--budget-slack", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    W = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    if args.meetings:
        keep = set(args.meetings.split(","))
        W = [w for w in W if w["meeting"] in keep]
    if args.shard_by == "meeting":
        ms = sorted({w["meeting"] for w in W})
        mine = set(ms[args.shard:: args.num_shards])
        W = [w for w in W if w["meeting"] in mine]
    else:
        W = W[args.shard:: args.num_shards]
    if args.limit:
        W = W[: args.limit]
    done_rec = {}
    for f in args.resume_from:
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            done_rec[r["id"]] = r
    by_m = defaultdict(list)
    for w in W:
        by_m[w["meeting"]].append(w)
    for m in by_m:
        by_m[m].sort(key=lambda w: w["start_s"])
    last_segs = {}   # meeting -> segments of its latest finished window
    todo = defaultdict(list)
    for m, ws in by_m.items():
        for w in ws:
            if w["id"] in done_rec:
                last_segs[m] = done_rec[w["id"]]["segments"]
            else:
                todo[m].append(w)
    n_todo = sum(len(v) for v in todo.values())
    print(f"{n_todo} windows to decode ({len(done_rec)} resumed), {len(todo)} meetings", flush=True)

    eng = Engine.from_merged(args.thinker, args.dir, args.max_gib)
    if args.lora:   # W3PO adapter: inject, load, fold into the weights (lora.merge)
        import re as _re

        from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import ATTN
        from experiments.uniss_streaming_p2st_traj_v1.training import lora as _lora

        keep = lambda parent: _re.search(r"(?:^|\.)model\.layers\.(\d+)\.", parent + ".") is not None  # noqa: E731
        ad = _lora.inject(eng.thinker, rank=args.lora_rank, alpha=args.lora_alpha, targets=ATTN, layer_filter=keep)
        _lora.load_state_dict(ad, torch.load(args.lora, map_location="cpu"))
        _lora.merge(ad)
        print(f"lora {args.lora}: {len(ad)} adapters folded", flush=True)
    tok = eng.tok
    period = round(args.window * 1000)

    def start(w):
        x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
        assert sr == 16000
        ctx = ctx_string(last_segs.get(w["meeting"], []), args.context)
        return {"w": w, "wav": x.mean(1), "post": np.load(w["post"][args.post_key]).astype(np.float32),
                "committed": [], "parser": Parser(tok), "blocks": [], "dur": int(round(len(x) / 16)), "r": 0,
                "ctx": ctx, "log": []}

    # lanes: per meeting with context (sequential windows), else one shared queue
    if args.context > 0:
        queues = [todo[m] for m in sorted(todo)]
    else:
        flat = [w for m in sorted(todo) for w in todo[m]]
        queues = [flat]
    active = []
    t0, n_done = time.time(), 0
    with open(out, "x", encoding="utf-8") as fh:
        while True:
            # fill lanes: one state per meeting queue with context, up to --batch from the shared queue without
            busy = {id(s["q"]) for s in active}
            for q in queues:
                if len(active) >= args.batch:
                    break
                if args.context > 0:
                    if q and id(q) not in busy:
                        s = start(q.pop(0))
                        s["q"] = q
                        active.append(s)
                else:
                    while q and len(active) < args.batch:
                        s = start(q.pop(0))
                        s["q"] = q
                        active.append(s)
            if not active:
                break
            items = []
            for s in active:
                s["r"] += 1
                s["t"] = min(s["r"] * period, s["dur"])
                items.append(ReadItem(wav=s["wav"], post=s["post"], t_ms=s["t"], final=s["t"] >= s["dur"],
                                      committed=s["committed"], ctx=s["ctx"], akey=f"{s['w']['id']}|{s['t']}",
                                      bar_wait=args.policy == "lag",
                                      stop_ms=s["t"] - args.lag_ms if args.policy == "lag" else None))
            outs = eng.read(items, max_new=args.max_new)
            finished = []
            for s, it, o in zip(active, items, outs):
                body = [i for i in o.ids if i not in (EOS_ID, WAIT_ID)]
                if it.final or args.policy == "wait":
                    new = body
                else:
                    new = body[: lag_cut(body, s["t"], args.lag_ms)]
                why = ""
                if args.loop_guard_k > 0 or args.budget_tps > 0:
                    from experiments.mist_w3po_v1.evaluation.loop_guard import guard
                    budget = int(args.budget_tps * s["dur"] / 1000) + args.budget_slack if args.budget_tps > 0 else None
                    new, why = guard(new, s["committed"], args.loop_guard_k, args.loop_guard_maxp, budget)
                s["committed"] = s["committed"] + new
                s["log"].append({"t": s["t"], "new": tok.decode(new, skip_special_tokens=False)})
                if why:
                    s["log"][-1]["guard"] = why
                s["blocks"] += s["parser"].feed(new, s["t"])
                if it.final:
                    s["parser"].close(s["dur"] / 1000, s["t"])
                    finished.append(s)
            for s in finished:
                active.remove(s)
                w = s["w"]
                last_segs[w["meeting"]] = s["parser"].segments
                fh.write(json.dumps({"id": w["id"], "raw": tok.decode(s["committed"], skip_special_tokens=False),
                                     "translation": "".join(x["text"] for x in s["parser"].segments),
                                     "segments": s["parser"].segments, "blocks": s["blocks"], "reads": s["r"],
                                     "read_log": s["log"], "ctx": s["ctx"]}, ensure_ascii=False) + "\n")
                n_done += 1
            if finished:
                fh.flush()
                if n_done % 50 < len(finished):
                    print(f"{n_done}/{n_todo} windows, {time.time() - t0:.0f}s", flush=True)
    print(f"W3PO_ONLINE_DONE {out} {n_done}", flush=True)


if __name__ == "__main__":
    main()
