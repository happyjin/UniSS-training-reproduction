"""Long-form streaming S2TT scored as SimulEval's StreamLAAL (Papi et al. 2024): the whole talk's output, each target unit
stamped with the moment it was committed on the talk's clock, is resegmented onto the reference segments by a minimum
edit distance alignment (what mwerSegmenter does), then each reference segment gets BLEU text and an LAAL computed on
its own clock (delays minus the segment's offset, its duration as the source length, its reference length in units --
characters for Chinese); StreamLAAL is the mean over segments. Computation-unaware, as the paper's Fig. 5.

  score     --refs refs.jsonl (talk, seg, offset_ms, dur_ms, ref) --items items.jsonl (piece id, talk, offset_ms) --blocks
            <si_thinker_stream output> [...]  ->  one json line per blocks file {bleu, stream_laal_s, ...}
  compare   a paired bootstrap over reference segments of the best BLEU within StreamLAAL budgets, per family

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_streamlaal score --refs r.jsonl --items i.jsonl --blocks 'dir/*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def zh_units(text: str) -> list[str]:
    return [c for c in text if not c.isspace()]


def align(hyp: list[str], ref: list[str]) -> list[int]:
    """For each hypothesis unit, the reference position it is aligned to (Levenshtein; insertions take the position of
    the last aligned reference unit before them)."""
    n, m = len(hyp), len(ref)
    if n == 0:
        return []
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    ref_arr = np.asarray(ref, dtype=object)
    B = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0 diag, 1 up (hyp insertion), 2 left (ref deletion)
    for i in range(1, n + 1):
        sub = D[i - 1, :-1] + (ref_arr != hyp[i - 1])
        up = D[i - 1, 1:] + 1
        best = np.minimum(sub, up)
        Bi = np.where(sub <= up, 0, 1)
        row = np.empty(m + 1, dtype=np.int32)
        row[0] = i
        # left moves need a running scan: row[j] = min(best[j-1], row[j-1] + 1)
        for j in range(1, m + 1):
            v = best[j - 1]
            if row[j - 1] + 1 < v:
                row[j] = row[j - 1] + 1
                B[i, j] = 2
            else:
                row[j] = v
                B[i, j] = Bi[j - 1]
        D[i] = row
    pos = [0] * n
    i, j = n, m
    while i > 0:
        if j > 0 and B[i, j] == 0:
            pos[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif j == 0 or B[i, j] == 1:
            pos[i - 1] = -1
            i -= 1
        else:
            j -= 1
    last = 0
    for k in range(n):
        if pos[k] < 0:
            pos[k] = last
        else:
            last = pos[k]
    return pos


def laal(delays, src, ref_len):
    if not delays:
        return 0.0
    if delays[0] > src:
        return delays[0]
    gamma = max(len(delays), ref_len) / src
    total, tau = 0.0, 0
    for i, d in enumerate(delays):
        total += d - i / gamma
        tau = i + 1
        if d >= src:
            break
    return total / tau


def resegment(refs: list[dict], items: dict, blocks_file: str):
    """Per reference segment: the hypothesis text and its units' delays on the segment's clock."""
    talk_units = defaultdict(list)   # talk -> [(time_ms, unit)]
    for line in open(blocks_file, encoding="utf-8"):
        r = json.loads(line)
        it = items[r["id"]]
        for b in r["blocks"]:
            for u in zh_units(b["text"]):
                talk_units[it["talk"]].append((it["offset_ms"] + b["write_ms"], u))
    by_talk = defaultdict(list)
    for k, s in enumerate(refs):
        by_talk[s["talk"]].append(k)
    hyp_seg = [[] for _ in refs]
    for talk, ks in by_talk.items():
        hu = sorted(talk_units.get(talk, []), key=lambda x: x[0])
        ref_units, owner = [], []
        for k in ks:
            ru = zh_units(refs[k]["ref"])
            ref_units += ru
            owner += [k] * len(ru)
        pos = align([u for _, u in hu], ref_units)
        for (t, u), p in zip(hu, pos):
            hyp_seg[owner[p] if owner else ks[0]].append((t, u))
    return hyp_seg


def score_file(refs, items, f):
    from evaluation.text_metrics import corpus_bleu

    hyp_seg = resegment(refs, items, f)
    hyps = ["".join(u for _, u in hs) for hs in hyp_seg]
    bleu = corpus_bleu(hyps, [s["ref"] for s in refs], language="cmn")["score"]
    lags = []
    for s, hs in zip(refs, hyp_seg):
        delays = sorted(t - s["offset_ms"] for t, _ in hs)
        lags.append(laal(delays, s["dur_ms"], len(zh_units(s["ref"]))) if delays else s["dur_ms"])
    return {"blocks": Path(f).name, "bleu": round(bleu, 2), "stream_laal_s": round(float(np.mean(lags)) / 1000, 3),
            "segments": len(refs), "empty_segments": sum(1 for h in hyps if not h)}, hyps, lags


_CTX = {}


def _one(f):
    res, hyps, lags = score_file(_CTX["refs"], _CTX["items"], f)
    if _CTX["dump"]:
        out = Path(_CTX["dump"]) / (Path(f).stem + ".seg.json")
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"hyps": hyps, "laal_ms": lags}, ensure_ascii=False), encoding="utf-8")
    return res


def cmd_score(args):
    from multiprocessing import Pool

    _CTX.update(refs=[json.loads(l) for l in open(args.refs, encoding="utf-8")],
                items={json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8")}, dump=args.dump)
    files = sorted({f for pat in args.blocks for f in glob.glob(pat)})
    with Pool(min(args.procs, max(1, len(files)))) as pool:   # one alignment per file and talk: CPU-bound, in parallel
        for res in pool.imap(_one, files):
            print(json.dumps(res, ensure_ascii=False), flush=True)


def cmd_compare(args):
    """Families of resegmented runs (from score --dump): best BLEU within each budget, paired bootstrap over segments."""
    from sacrebleu.metrics import BLEU
    from evaluation.text_metrics import normalize_for_bleu

    refs = [json.loads(l) for l in open(args.refs, encoding="utf-8")]
    bleu = BLEU(tokenize="zh")
    R = [normalize_for_bleu(s["ref"], "cmn") for s in refs]
    fams = {}
    for spec in args.family:
        name, pats = spec.split("=", 1)
        runs = []
        for f in sorted({p for pat in pats.split(",") for p in glob.glob(pat)}):
            d = json.load(open(f, encoding="utf-8"))
            H = [normalize_for_bleu(h, "cmn") for h in d["hyps"]]
            runs.append((Path(f).name, np.asarray(bleu._extract_corpus_statistics(H, [R]), dtype=np.float64),
                         np.asarray(d["laal_ms"], dtype=np.float64) / 1000))
        fams[name] = runs
    budgets = [float(x) for x in args.budgets.split(",")]
    ranges = []
    for spec in args.summary:
        lo, hi = map(float, spec.split(","))
        grid = [round(v, 3) for v in np.arange(lo, hi + 1e-9, 0.05)]
        ranges.append((spec, len(budgets), len(budgets) + len(grid)))
        budgets += grid

    def scores(idx):
        out = {}
        for name, runs in fams.items():
            pts = [(float(l[idx].mean()), bleu._compute_score_from_stats(list(st[idx].sum(0))).score) for _, st, l in runs]
            out[name] = [max([b for l, b in pts if l <= x], default=np.nan) for x in budgets]
        return out

    full = scores(np.arange(len(refs)))
    rng = np.random.default_rng(0)
    draws = [scores(rng.integers(0, len(refs), len(refs))) for _ in range(args.n)]
    for spec, a, b in ranges:
        row = {"range_s": spec}
        for n in fams:
            if n == args.ref:
                continue
            fd = np.nanmean([full[n][j] - full[args.ref][j] for j in range(a, b)])
            d = np.asarray([np.nanmean([dr[n][j] - dr[args.ref][j] for j in range(a, b)]) for dr in draws])
            d = d[~np.isnan(d)]
            row[f"{n}-{args.ref}"] = {"mean_diff": round(float(fd), 3), "ci95": [round(float(np.percentile(d, 2.5)), 3),
                                      round(float(np.percentile(d, 97.5)), 3)], "p": round(float((d <= 0).mean()), 4)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for j, x in enumerate(budgets[: ranges[0][1] if ranges else len(budgets)]):
        row = {"budget_s": x, "scores": {n: (None if np.isnan(full[n][j]) else round(full[n][j], 2)) for n in fams}}
        for n in fams:
            if n == args.ref or np.isnan(full[n][j]) or np.isnan(full[args.ref][j]):
                continue
            d = np.asarray([dr[n][j] - dr[args.ref][j] for dr in draws])
            d = d[~np.isnan(d)]
            row[f"{n}-{args.ref}"] = {"diff": round(full[n][j] - full[args.ref][j], 2), "p": round(float((d <= 0).mean()), 4)}
        print(json.dumps(row, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--refs", required=True)
    s.add_argument("--items", required=True)
    s.add_argument("--blocks", action="append", required=True)
    s.add_argument("--dump", default="", help="write each run's resegmented hypotheses and segment LAALs here")
    s.add_argument("--procs", type=int, default=32)
    c = sub.add_parser("compare")
    c.add_argument("--refs", required=True)
    c.add_argument("--family", action="append", required=True, help="NAME=glob of *.seg.json")
    c.add_argument("--ref", required=True)
    c.add_argument("--budgets", required=True)
    c.add_argument("--summary", action="append", default=[])
    c.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    {"score": cmd_score, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
