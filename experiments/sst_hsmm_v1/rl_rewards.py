"""sst_hsmm_v1 DTC-RL (change 41): fork rollouts -> policy-gradient samples with group-relative advantages.

Inputs: rollout dirs (run_<d>_* with blocks_<d>_w1.jsonl = greedy base rows and forks_<d>.jsonl = sampled clones, from
hsc11 / hsc12 --fork-reads F --fork-k K) and the items files (reference "translation", audio, dur_ms).
Per stream: p0rl_analyze.stream_stats (proxy D2 s, gap s, sentence BLEU, chrF, zero-commit share, log score S).
Reward R = - w_d2 * d2 - w_gap * gap + w_bleu * bleu / 100 + w_s * S   (weights from the command line; S is the per-position
mean log score, I1's information-design term; gap is I2's rebuffering term).
Group = the K clones of one (item, fork read); A_k = R_k - mean_k(R) (no std normalisation: rl_plan §2.1 lesson 1);
groups whose rewards are all equal (|A| < 1e-6) are dropped (DAPO dynamic sampling). One sample per clone:
{id, dir, audio, t, prefix, tok, lp, adv, R, d2, gap, bleu, S}. Also writes <out>.stats.json (means of the greedy base streams
and of the clones per direction: the dual update and the run log read these). Refuses to overwrite.
    python -m experiments.sst_hsmm_v1.rl_rewards --root <rollout dir> --data <dir with items_<d>.jsonl> --out <samples.jsonl> \
        --w-d2 1 --w-gap 0 --w-bleu 40 --w-s 0
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.p0rl_analyze import stream_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-d2", type=float, default=1.0)
    ap.add_argument("--w-gap", type=float, default=0.0)
    ap.add_argument("--w-bleu", type=float, default=40.0)
    ap.add_argument("--w-s", type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    import sacrebleu
    rows_out, stats = [], {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        lang = "cmn" if zh else "eng"
        tk = "zh" if zh else "13a"
        sb = lambda h, r: sacrebleu.sentence_bleu(h, [r], tokenize=tk).score  # noqa: E731
        sc = lambda h, r: sacrebleu.sentence_chrf(h, [r]).score  # noqa: E731
        items = {r["id"]: r for r in map(json.loads, open(Path(a.data) / f"items_{d}.jsonl", encoding="utf-8"))}
        base, forks = {}, defaultdict(list)
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"blocks_{d}_w1.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                base[r["id"]] = r
        for f in sorted(glob.glob(str(Path(a.root) / f"run_{d}_*" / f"forks_{d}.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                forks[(r["id"], r["fork_t"])].append(r)
        if not base:
            continue
        R = lambda x: -a.w_d2 * x["d2"] - a.w_gap * x["gap"] + a.w_bleu * x["bleu"] / 100.0 + (  # noqa: E731
            a.w_s * x["S"] if a.w_s and not math.isnan(x["S"]) else 0.0)
        bst = [stream_stats(r, items[i], zh, lang, 0, sb, sc) for i, r in base.items()]
        kept = dropped = 0
        cst = []
        for (iid, ft), fr in sorted(forks.items()):
            it = items[iid]
            ss = [(r, stream_stats(r, it, zh, lang, ft, sb, sc)) for r in sorted(fr, key=lambda x: x["fork_k"])]
            cst += [x for _, x in ss]
            rew = np.array([R(x) for _, x in ss])
            adv = rew - rew.mean()
            if np.all(np.abs(adv) < 1e-6):
                dropped += 1
                continue
            kept += 1
            for (r, x), A, Rk in zip(ss, adv, rew):
                f0 = r["drafts"][0]
                assert f0.get("forked") and f0["t"] == ft and "fork_tok" in f0, (iid, ft)
                rows_out.append({"id": iid, "dir": d, "audio": it["audio"], "t": ft, "prefix": f0["fork_prefix"],
                                 "tok": f0["fork_tok"], "lp": f0["fork_lp"], "adv": float(A), "R": float(Rk),
                                 "d2": x["d2"], "gap": x["gap"], "bleu": x["bleu"], "S": x["S"], "k": r["fork_k"]})
        mean = lambda w, k: float(np.nanmean([x[k] for x in w])) if w else None  # noqa: E731
        stats[d] = {"groups_kept": kept, "groups_dropped": dropped,
                    "base": {k: mean(bst, k) for k in ("bleu", "chrf", "d2", "gap", "pauses", "zero", "S")},
                    "clones": {k: mean(cst, k) for k in ("bleu", "chrf", "d2", "gap", "pauses", "zero", "S")},
                    "n_base": len(bst)}
        print(d, json.dumps(stats[d]), flush=True)
    stats["weights"] = {"w_d2": a.w_d2, "w_gap": a.w_gap, "w_bleu": a.w_bleu, "w_s": a.w_s}
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(str(out) + ".stats.json", "x", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
