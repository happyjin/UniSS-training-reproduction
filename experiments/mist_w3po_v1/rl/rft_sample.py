"""Rejection-sampling data for RFT / RAFT (plan branch B2, 2026-11-07): for training windows of the RL stream, the base
model's greedy wait-policy decode and G sampled decodes (rl/rollout.rollout, the RL sampler: temp, temp_when), each
scored with the training reward (rl/stamp.credit); one record per window with every member's R, latency, spurious and
duplicate counts, and the read blocks (t, before, ids) of the greedy decode and of the best sample, so that
rl/build_rft_rows.py can select (best R >= greedy R, no more duplicates / spurious) and write SFT rows.

Windows: rl/precompute_anchors.window_stream (Sampler(seed), --windows-per-step, steps [--from-step, --steps)), sharded
--shard/--num-shards; appended per batch to <out>/part_<shard>.jsonl, a restart skips ids already there.

    python -m experiments.mist_w3po_v1.rl.rft_sample --base <merged> --out <dir> --from-step 100 --steps 163 \
        --shard k --num-shards n [--G 8 --temp 0.8 --temp-when 1.0 --ctx-max 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.mist_w3po_v1.core.engine import Engine
from experiments.mist_w3po_v1.rl import stamp as S
from experiments.mist_w3po_v1.rl.precompute_anchors import meeting_refs, window_stream
from experiments.mist_w3po_v1.rl.rollout import load_window, rollout
from experiments.mist_w3po_v1.rl.train_w3po import gold_ctx


def score(tr, w, units, post) -> dict:
    dur = w["dur_ms"] / 1000
    c = S.credit(tr.segs, units, dur, tr.wait_reads, [rd.t_ms / 1000 for rd in tr.reads], post=post)
    return {"R": c.R, "lat": c.lat, "spurious": c.spurious, "dup": sum(1 for d in c.delta if d < 0), "n_segs": len(tr.segs)}


def reads_of(tr) -> list[dict]:
    return [{"t": rd.t_ms, "final": rd.final, "before": rd.before, "ids": rd.ids} for rd in tr.reads]


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
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--ctx-max", type=int, default=0)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--temp-when", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--max-dur-ms", type=int, default=20500)
    ap.add_argument("--batch", type=int, default=8, help="windows per rollout call")
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
    gen = torch.Generator(device=eng.dev)
    t0 = time.time()
    with open(out / f"part_{a.shard}.jsonl", "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            ctxs = {w["id"]: gold_ctx(w, meet_utts, sess, a.seed, a.ctx_max) for w in chunk}
            cache = {}
            greedy = rollout(eng, chunk, ctxs, 1, temp=0.0, temp_when=None, gen=None, max_new=a.max_new, cache=cache)
            gen.manual_seed(a.seed * 7919 + a.shard * 100003 + i)
            samp = rollout(eng, chunk, ctxs, a.G, temp=a.temp, temp_when=a.temp_when, gen=gen, max_new=a.max_new,
                           cache=cache)
            for w in chunk:
                units = S.units_of(w["utts"])
                _, post = load_window(w)
                g = greedy[w["id"]][0]
                sg = score(g, w, units, post)
                ss = [score(tr, w, units, post) for tr in samp[w["id"]]]
                b = max(range(len(ss)), key=lambda k: ss[k]["R"])
                fh.write(json.dumps({"id": w["id"], "ctx": ctxs[w["id"]], "n_units": len(units), "greedy": sg,
                                     "samples": ss, "best": b, "greedy_reads": reads_of(g),
                                     "best_reads": reads_of(samp[w["id"]][b])}) + "\n")
            fh.flush()
            print(f"{i + len(chunk)}/{len(todo)} windows, {time.time() - t0:.0f}s", flush=True)
    print(f"RFT_SAMPLE_DONE {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
