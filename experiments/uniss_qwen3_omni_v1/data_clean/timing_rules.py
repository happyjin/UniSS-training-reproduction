"""Candidate read/write timing rules, each as a per-word support time, plus the scores that pick one.

Every rule gives each target word the moment its source evidence has been heard
(the support); simul_segments.py --support turns a support into segments, write
chunks and a timeline, the cut points being the same safe pauses for all rules.

  omni      the latest end among the word's OmniAlign links (lexical evidence);
  argmax    Hibiki's contextual alignment: the source unit whose arrival raises the
            word's log-probability under the Thinker most (despiked);
  suff      the earliest prefix after which the word's log-probability stays within
            --tol of its value under the whole source -- the Thinker can already
            say it, and does not change its mind later;
  hybrid    suff, but never before the word's strong OmniAlign links (confidence
            >= --strong) have been heard: a content word is not said before the
            source word it translates.

Scores (per rule, after segmentation, in score_rules()):
  predictable   share of words whose log-probability at the moment they are
                written (the prefix heard by their segment's write time) is within
                --tol of the final one -- they would not be said wrongly;
  early         share of words written before their strong OmniAlign source ended;
  latency       first write, end lag, mean word lag behind the source timeline;
  flow          silence ratio and long gaps, from the timeline.

    python timing_rules.py supports --meta ... --keys ... --ctx ... --omni ... --out supports.parquet
    python timing_rules.py score --segments seg_<rule>.parquet --supports supports.parquet --rule <rule> ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RULES = ("omni", "argmax", "suff", "hybrid")


def per_word(r: dict, ctx: dict | None, omni: list | None, tol: float, strong: float) -> dict:
    src = json.loads(r["source_words"])
    n = len(json.loads(r["target_words"]))
    end = [w["end_ms"] for w in src]
    K = len(src)
    out = {k: [None] * n for k in RULES}
    strong_end = [None] * n
    if omni is not None:
        for l in omni:
            j, u = l["target_index"], l["source_index"]
            if 0 <= j < n and 0 <= u < K:
                out["omni"][j] = max(out["omni"][j] or 0, end[u])
                if l.get("confidence", 1.0) >= strong:
                    strong_end[j] = max(strong_end[j] or 0, end[u])
    curves, bounds = None, list(range(K + 1))
    if ctx is not None:
        for j, a in enumerate(json.loads(ctx["context_align"])):
            if a is not None and 0 <= a < K:
                out["argmax"][j] = end[a]
        curves = json.loads(ctx["curves"]) if ctx.get("curves") else None
        bounds = json.loads(ctx["bounds"]) if ctx.get("bounds") else bounds   # prefix k holds bounds[k] units
        for j, c in enumerate(curves or []):
            if c is None or len(c) != len(bounds):
                continue
            final, k = c[-1], len(c) - 1
            while k > 0 and c[k - 1] >= final - tol:
                k -= 1
            out["suff"][j] = end[bounds[k] - 1] if k > 0 else 0
            out["hybrid"][j] = max(out["suff"][j], strong_end[j] or 0)
    return {"key": r["key"], **{f"support_{k}": json.dumps(v) for k, v in out.items()},
            "strong_end": json.dumps(strong_end), "curves": json.dumps(curves) if curves is not None else None,
            "source_end": json.dumps(end), "bounds": json.dumps(bounds)}


def supports(args) -> None:
    keys = set(pq.read_table(args.keys, columns=["key"]).column("key").to_pylist())
    ctx = {}
    for f in (sorted(Path(args.ctx).glob("*.parquet")) if Path(args.ctx).is_dir() else [Path(args.ctx)]) if args.ctx else []:
        ctx.update({r["key"]: r for r in pq.read_table(f).to_pylist()})   # a file, or a directory of shards
    omni = {}
    for f in sorted(Path(args.omni).glob("*.parquet")) if Path(args.omni).is_dir() else [Path(args.omni)]:
        omni.update({r["key"]: json.loads(r["omni_links"]) for r in pq.read_table(f).to_pylist()})
    rows = []
    for p in sorted(Path(args.meta).glob("*.parquet")):
        for r in pq.read_table(p, columns=["key", "tgt_lang", "source_words", "target_words", "a68_present"]).to_pylist():
            if r["key"] in keys and r["a68_present"] and (r["key"] in ctx or not args.ctx):
                # per target language when given: the pilot's prefix check chose x4 into Chinese, x8 into English
                tol = {"cmn": args.tol_cmn, "eng": args.tol_eng}.get(r["tgt_lang"]) or args.tol
                rows.append(per_word(r, ctx.get(r["key"]), omni.get(r["key"]), tol, args.strong))
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{len(rows)} rows, rules {RULES} -> {args.out}", flush=True)


def score(args) -> dict:
    seg = {r["key"]: r for r in pq.read_table(args.segments).to_pylist()}
    have = set(pq.read_schema(args.supports).names)
    cols = [c for c in ("key", "strong_end", "curves", "source_end", "bounds") if c in have]   # bounds: newer files only
    sup = {r["key"]: r for r in pq.read_table(args.supports, columns=cols).to_pylist()}
    res = {}
    for key, s in seg.items():
        lang = s["tgt_lang"]
        acc = res.setdefault(lang, {"pred": [], "early": [], "first": [], "lag": [], "sr": [], "gaps": []})
        sp = sup.get(key)
        segs = json.loads(s["segments"])
        curves = json.loads(sp["curves"]) if sp and sp["curves"] else None
        strong = json.loads(sp["strong_end"]) if sp else None
        ends = json.loads(sp["source_end"]) if sp else None
        bounds = json.loads(sp["bounds"]) if sp and sp.get("bounds") else None
        for g in segs:
            write_ms = g["chunk"] * args.chunk_ms
            # the longest prefix fully heard by the write time (prefixes as the curves were taken)
            heard = 0
            if ends and bounds:
                while heard + 1 < len(bounds) and ends[bounds[heard + 1] - 1] <= write_ms:
                    heard += 1
            for j in range(g["w0"], g["w1"]):
                if curves and curves[j] is not None and len(curves[j]) > heard:
                    acc["pred"].append(curves[j][heard] >= curves[j][-1] - args.tol)
                if strong and strong[j] is not None:
                    acc["early"].append(write_ms < strong[j])
        acc["first"].append(s["first_write_ms"]); acc["lag"].append(s["end_lag_ms"])
        acc["sr"].append(s["silence_ratio"]); acc["gaps"].append(s["long_gaps"])
    out = {}
    for lang, a in res.items():
        out[lang] = {"predictable": float(np.mean(a["pred"])) if a["pred"] else math.nan,
                     "early": float(np.mean(a["early"])) if a["early"] else math.nan,
                     "first_write_s": float(np.median(a["first"])) / 1000, "end_lag_s": float(np.median(a["lag"])) / 1000,
                     "silence_p90": float(np.percentile(a["sr"], 90)), "long_gaps": float(np.mean(a["gaps"]))}
    print(json.dumps({"rule": args.rule, **out}, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("supports")
    a.add_argument("--meta", required=True); a.add_argument("--keys", required=True)
    a.add_argument("--ctx", default=""); a.add_argument("--omni", required=True); a.add_argument("--out", required=True)
    a.add_argument("--tol", type=float, default=math.log(2)); a.add_argument("--strong", type=float, default=0.9)
    a.add_argument("--tol-cmn", type=float, default=None, help="tolerance for Chinese targets (default --tol)")
    a.add_argument("--tol-eng", type=float, default=None, help="tolerance for English targets (default --tol)")
    b = sub.add_parser("score")
    b.add_argument("--segments", required=True); b.add_argument("--supports", required=True)
    b.add_argument("--rule", required=True); b.add_argument("--chunk-ms", type=int, default=500)
    b.add_argument("--tol", type=float, default=math.log(2))
    args = ap.parse_args()
    if args.cmd == "supports":
        if Path(args.out).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {args.out}")
        supports(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
