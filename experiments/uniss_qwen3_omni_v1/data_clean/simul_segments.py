"""Plan 2: read/write timing data -- when each piece of the target may be spoken.

Per row (timestamps only; no audio is cut or copied):

  1. support: each target word's earliest moment, from the source words it
     is aligned to -- the latest end time among them (mBERT links from the
     a68 parts, or --context, the Qwen3 contextual alignment); unaligned
     words take the next aligned word's support (backward fill), and
     tau_j = max(tau_{j-1}, support_j + delta) keeps it monotonic.
  2. safe cut points: a cut may only fall in a real pause of the target
     speech -- energy below the utterance's peak by --quiet-db for at least
     --min-pause-ms between two words -- and it falls at the pause's middle,
     so the piece before keeps its decay and the piece after its onset. A
     word boundary in connected speech is not a cut point.
  3. segments: words grouped greedily, closed at the first cut point once
     the piece reaches --min-seg-ms; a segment may be spoken at the tau of
     its last word, and owns the codes between its cut points (20 ms each).
  4. source chunks: 1 s; a segment is written in the first chunk whose end
     reaches its tau, the chunks between are reads.
  5. timeline: segments played in order, each from max(its tau, the end of
     the one before); the silence this leaves is the data's silence ratio,
     and how far the end lags the source.
  6. checks: energy at each cut and the decay into it; inversion rate of
     the alignment (NIR).

    python simul_segments.py --meta data/unist_clean_v1/meta --scores ... --keys ... --out segments.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

FRAME_MS = 10
CODE_MS = 20


def energy_db(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    hop = sr * FRAME_MS // 1000
    n = len(wav) // hop
    rms = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10)
    return 20 * np.log10(rms)


def pauses(db: np.ndarray, quiet_db: float, min_ms: int) -> list[tuple[int, int]]:
    """Quiet runs, in ms, at least min_ms long."""
    quiet = db < (db.max() - quiet_db)
    out, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * FRAME_MS >= min_ms:
                out.append((start * FRAME_MS, i * FRAME_MS))
            start = None
    return out


def nir(links: list[tuple[int, int]]) -> float:
    """Normalised inversion rate of the source order read in target order."""
    seq = [s for _, s in sorted(links)]
    n = len(seq)
    if n < 2:
        return 0.0
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
    return inv / (n * (n - 1) / 2)


def plan_row(r: dict, args) -> dict:
    src_words = json.loads(r["source_words"])
    tgt_words = json.loads(r["target_words"])
    n_t = len(tgt_words)
    if args.context:
        aligned = {j: s for j, s in enumerate(json.loads(r["context_align"])) if s is not None}
        links = [(j, s) for j, s in aligned.items()]
    elif args.links:
        links = [(l["target_index"], l["source_index"]) for l in json.loads(r["omni_links"])
                 if l.get("confidence", 1.0) >= args.min_link_conf]
    else:
        links = [(l["target_index"], l["source_index"]) for l in json.loads(r["bilingual_links"])]
    support = [None] * n_t
    if args.support:   # a rule's per-word support, from timing_rules.py
        given = json.loads(r["support_json"])
        support = [given[j] if j < len(given) else None for j in range(n_t)]
        links = [(j, 0) for j in range(n_t) if support[j] is not None]
    for j, s in ([] if args.support else links):
        if 0 <= j < n_t and 0 <= s < len(src_words):
            end = src_words[s]["end_ms"]
            support[j] = end if support[j] is None else max(support[j], end)
    nxt = r["source_duration_ms"]
    for j in range(n_t - 1, -1, -1):          # backward fill
        if support[j] is None:
            support[j] = nxt
        nxt = support[j]
    tau, run = [], 0
    for j in range(n_t):
        run = max(run, support[j] + args.delta_ms)
        tau.append(min(run, r["source_duration_ms"] + args.delta_ms))

    db = energy_db(r["target_audio"])
    quiet = pauses(db, args.quiet_db, args.min_pause_ms)
    total_ms = len(r["target_bicodec"]) * CODE_MS
    cuts = {}   # boundary after word j -> (cut_ms, pause_ms)
    for j in range(n_t - 1):
        lo, hi = tgt_words[j]["end_ms"] - args.slack_ms, tgt_words[j + 1]["start_ms"] + args.slack_ms
        best = None
        for a, b in quiet:
            if b > lo and a < hi and (best is None or b - a > best[1] - best[0]):
                best = (a, b)
        if best:
            cuts[j] = ((best[0] + best[1]) // 2, best[1] - best[0], best[0])

    segments, start_j, start_ms = [], 0, 0
    for j in range(n_t):
        last = j == n_t - 1
        cut = cuts.get(j)
        if not last and (cut is None or cut[0] - start_ms < args.min_seg_ms):
            continue
        end_ms = total_ms if last else cut[0]
        segments.append({"w0": start_j, "w1": j + 1, "tau_ms": int(tau[j]),
                         "code0": int(round(start_ms / CODE_MS)), "code1": int(round(end_ms / CODE_MS)),
                         "cut_ms": None if last else int(end_ms), "pause_ms": None if last else int(cut[1]),
                         "pause_start_ms": None if last else int(cut[2]),
                         "cut_class": None if last else (1 if cut[1] >= args.class1_ms else 2)})
        start_j, start_ms = j + 1, end_ms
    segments[-1]["code1"] = len(r["target_bicodec"])

    # checks at each cut: how far below the peak it sits (+-20 ms, inside the
    # pause), and how the speech ends into the pause -- the time energy takes
    # to fall from 10 dB under the peak to the quiet threshold. A fall of
    # 20 ms or less is abrupt: a stop closure, or speech cut off, not a decay.
    peak, cut_depth, falls = db.max(), [], []
    for s in segments[:-1]:
        i = s["cut_ms"] // FRAME_MS
        win = db[max(0, i - 2) : i + 3]
        cut_depth.append(float(peak - win.max()) if win.size else 0.0)
        k = s["pause_start_ms"] // FRAME_MS
        fall = 0
        while k - fall - 1 >= 0 and db[k - fall - 1] < peak - 10 and fall < 50:
            fall += 1
        s["fall_ms"] = fall * FRAME_MS
        falls.append(fall * FRAME_MS)

    # timeline: source chunks; play each segment from max(its write time, previous end)
    end_prev, gaps, starts, delays = None, [], [], []
    for s in segments:
        write_ms = math.ceil(s["tau_ms"] / args.chunk_ms) * args.chunk_ms
        s["chunk"] = int(write_ms // args.chunk_ms)
        begin = write_ms if end_prev is None else max(write_ms, end_prev)
        if end_prev is not None:
            gaps.append(begin - end_prev)
        starts.append(begin)
        seg_start_ms = s["code0"] * CODE_MS
        for j in range(s["w0"], s["w1"]):   # when each word is heard, against when its source ended
            delays.append(begin + max(0, tgt_words[j]["start_ms"] - seg_start_ms) - support[j])
        end_prev = begin + (s["code1"] - s["code0"]) * CODE_MS
    span = end_prev - starts[0]
    return {
        "key": r["key"], "tgt_lang": r["tgt_lang"], "n_words": n_t, "n_segments": len(segments),
        "segments": json.dumps(segments, ensure_ascii=False),
        "seg_ms_mean": float(np.mean([(s["code1"] - s["code0"]) * CODE_MS for s in segments])),
        "silence_ratio": float(sum(gaps) / span) if span > 0 else 0.0,
        "long_gaps": int(sum(g > 500 for g in gaps)), "end_lag_ms": int(end_prev - r["source_duration_ms"]),
        "first_write_ms": int(starts[0]), "nir": nir(links),
        "mean_word_delay_ms": float(np.mean(delays)), "source_ms": int(r["source_duration_ms"]),
        "cut_depth_min_db": float(min(cut_depth)) if cut_depth else None,
        "abrupt_cuts": int(sum(f <= 20 for f in falls)), "cuts": len(falls),
        "unaligned_words": int(sum(1 for j in range(n_t) if j not in {a for a, _ in links})),
        "class1_cuts": int(sum(s["cut_class"] == 1 for s in segments)), "class2_cuts": int(sum(s["cut_class"] == 2 for s in segments)),
    }


def run(job):
    path, keys, args, extra = job
    cols = ["key", "tgt_lang", "source_words", "target_words", "bilingual_links", "target_audio",
            "target_bicodec", "source_duration_ms", "a68_present"]
    out = []
    for r in pq.read_table(path, columns=cols).to_pylist():
        if r["key"] not in keys or not r["a68_present"]:
            continue
        if args.context or args.links or args.support:
            if r["key"] not in extra:
                continue
            r["support_json" if args.support else "context_align" if args.context else "omni_links"] = extra[r["key"]]
        try:
            out.append(plan_row(r, args))
        except Exception as e:   # a row that cannot be planned is reported, not fatal
            out.append({"key": r["key"], "error": f"{type(e).__name__}: {e}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--keys", required=True, help="parquet with the key column of the rows to plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", default="", help="contextual alignment parquet (key, context_align json)")
    ap.add_argument("--links", default="", help="word links parquet (key, omni_links json) instead of a68's mBERT links")
    ap.add_argument("--min-link-conf", type=float, default=0.0)
    ap.add_argument("--support", default="", help="timing_rules.py supports parquet; with --rule, use that rule's support")
    ap.add_argument("--rule", default="", help="omni | argmax | suff | hybrid")
    ap.add_argument("--delta-ms", type=int, default=300)
    ap.add_argument("--quiet-db", type=float, default=40.0)
    ap.add_argument("--min-pause-ms", type=int, default=80)
    ap.add_argument("--class1-ms", type=int, default=150)
    ap.add_argument("--slack-ms", type=int, default=40)
    ap.add_argument("--min-seg-ms", type=int, default=800)
    ap.add_argument("--chunk-ms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=60)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {args.out}")
    keys = set(pq.read_table(args.keys, columns=["key"]).column("key").to_pylist())
    extra = {}
    if args.context:
        extra = {r["key"]: r["context_align"] for r in pq.read_table(args.context, columns=["key", "context_align"]).to_pylist()}
    elif args.links:
        extra = {r["key"]: r["omni_links"] for r in pq.read_table(args.links).to_pylist()}
    elif args.support:
        col = f"support_{args.rule}"
        extra = {r["key"]: r[col] for r in pq.read_table(args.support, columns=["key", col]).to_pylist()}
    jobs = [(str(p), keys, args, extra) for p in sorted(Path(args.meta).glob("*.parquet"))]
    rows = []
    with mp.Pool(min(args.jobs, len(jobs))) as pool:
        for part in pool.imap_unordered(run, jobs):
            rows += part
    errors = [r for r in rows if "error" in r]
    good = [r for r in rows if "error" not in r]
    pq.write_table(pa.Table.from_pylist(good), args.out)
    print(f"{len(good)} rows planned, {len(errors)} errors" + (f"; e.g. {errors[0]}" if errors else ""), flush=True)


if __name__ == "__main__":
    main()
