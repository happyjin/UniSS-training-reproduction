"""The SI trainer's index over data/unist_simul_v1: which rows, and what a batch needs of each.

Nothing is re-timed or re-cut. A row of the read/write data is kept when its
timeline is usable for simultaneous training, and stored as flat arrays the
dataloader reads without JSON:

  * quality, every tier: no abrupt cut, silence ratio <= 0.3, at most one long
    gap, OmniAlign inversion rate <= 0.3 (recomputed from word_links; the data's
    nir column was measured on support times, where it is always 0), end lag
    within the 90th percentile of its target language;
  * tier A2: source >= 3 s and at least two writes before the source ends
    (under the data's own rule, 500 ms chunks); A1: one such write; B: the
    rest -- short sources, or everything written after the end;
  * read: the source comes from a read-speech corpus (CommonVoice, LibriSpeech,
    VCTK), down-weighted for English sources by the dataset, not here.

Per row: its row in the clean training index (codes, voice, source GLM and the
translation come from there, as Plan 1 read them), the tier, the target
language, the source duration; per target word, where it starts in the
translation (so a segment's text is a slice of it), its support time under
each rule and when it is said in the target speech (so a block may also end
between two words, its codes cut there); per segment, its last word and its
last code. Rows whose words
cannot all be found in the translation are dropped and counted.

    python build_si_train_index.py --out data/unist_simul_train_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RULES = ("omni", "argmax", "suff", "hybrid")
READ = {"CommonVoice_EN", "LibriSpeech", "VCTK"}
TIERS = {"A2": 0, "A1": 1, "B": 2}
COLS = ["key", "tgt_lang", "dataset", "source_parquet", "source_row_index", "source_duration_ms", "translation",
        "target_words", "word_links", "segments", "silence_ratio", "long_gaps", "abrupt_cuts", "end_lag_ms",
        *[f"support_{r}_ms" for r in RULES]]


def _norm(ch: str) -> str:
    ch = unicodedata.normalize("NFKC", ch).lower()
    return ch if ch.isalnum() else ""


def char_starts(text: str, words: list[str]) -> list[int] | None:
    """Where each word starts in text, matching letters and digits only, in order."""
    stream, where = [], []
    for i, ch in enumerate(text):
        for c in _norm(ch):
            stream.append(c)
            where.append(i)
    flat = "".join(stream)
    out, at = [], 0
    for w in words:
        key = "".join(_norm(c) for c in w)
        if not key:
            out.append(where[at] if at < len(where) else len(text))
            continue
        p = flat.find(key, at, at + len(key) + 40)
        if p < 0:
            return None
        out.append(where[p])
        at = p + len(key)
    return out


def nir(links: list[dict]) -> float:
    seq = [s for _, s in sorted((l["target_index"], l["source_index"]) for l in links)]
    n = len(seq)
    if n < 2:
        return 0.0
    return sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j]) / (n * (n - 1) / 2)


def scan(path: str) -> list[dict]:
    out = []
    for r in pq.read_table(path, columns=COLS).to_pylist():
        words = json.loads(r["target_words"])
        segs = json.loads(r["segments"])
        starts = char_starts(r["translation"], [w["text"] for w in words])
        src = r["source_duration_ms"]
        out.append({
            "key": r["key"], "lang": r["tgt_lang"], "read": r["dataset"] in READ, "src_ms": src,
            "parquet": r["source_parquet"], "row": r["source_row_index"], "starts": starts,
            "sup": [[(-1 if v is None else int(v)) for v in json.loads(r[f"support_{k}_ms"])] for k in RULES],
            "seg_w1": [g["w1"] for g in segs], "seg_code1": [g["code1"] for g in segs],
            "early": sum(g["chunk"] * 500 < src for g in segs), "nir": nir(json.loads(r["word_links"] or "[]")),
            "ok": (r["abrupt_cuts"] == 0 and r["silence_ratio"] <= 0.3 and r["long_gaps"] <= 1),
            "lag": r["end_lag_ms"], "n": len(words),
            "w_start_ms": [int(w["start_ms"]) for w in words], "w_end_ms": [int(w["end_ms"]) for w in words]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simul", default="data/unist_simul_v1")
    ap.add_argument("--clean-index", default="data/unist_clean_v1/audio_index_clean_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nir", type=float, default=0.3)
    ap.add_argument("--lag-pct", type=float, default=90.0)
    ap.add_argument("--jobs", type=int, default=22)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")

    from experiments.uniss_qwen3_omni_v1.training.audio_index import AudioIndex
    clean = AudioIndex(args.clean_index)
    where = {}
    for i in range(len(clean)):   # first occurrence: code-switched rows are repeated in the index
        where.setdefault((clean._fields["parquet"][i], int(clean.row_index[i])), i)

    with Pool(args.jobs) as p:
        rows = [r for part in p.map(scan, sorted(glob.glob(f"{args.simul}/rows/*.parquet"))) for r in part]
    lag_cap = {lang: float(np.percentile([r["lag"] for r in rows if r["lang"] == lang], args.lag_pct))
               for lang in ("cmn", "eng")}
    dropped = {"unmatched_words": 0, "not_in_clean_index": 0, "quality": 0}
    kept = []
    for r in rows:
        if r["starts"] is None:
            dropped["unmatched_words"] += 1
            continue
        if (r["parquet"], r["row"]) not in where:
            dropped["not_in_clean_index"] += 1
            continue
        if not (r["ok"] and r["nir"] <= args.max_nir and r["lag"] <= lag_cap[r["lang"]]):
            dropped["quality"] += 1
            continue
        r["tier"] = TIERS["A2" if r["src_ms"] >= 3000 and r["early"] >= 2 else
                          "A1" if r["src_ms"] >= 3000 and r["early"] >= 1 else "B"]
        r["clean"] = where[(r["parquet"], r["row"])]
        kept.append(r)

    out.mkdir(parents=True)
    def flat(name, lists, dtype):
        np.save(out / f"{name}_values.npy", np.concatenate([np.asarray(x, dtype=dtype) for x in lists]))
        np.save(out / f"{name}_offsets.npy", np.concatenate([[0], np.cumsum([len(x) for x in lists])]).astype(np.int64))
    np.save(out / "clean_idx.npy", np.asarray([r["clean"] for r in kept], dtype=np.int64))
    np.save(out / "tier.npy", np.asarray([r["tier"] for r in kept], dtype=np.int8))
    np.save(out / "tgt_eng.npy", np.asarray([r["lang"] == "eng" for r in kept], dtype=bool))
    np.save(out / "read.npy", np.asarray([r["read"] for r in kept], dtype=bool))
    np.save(out / "src_ms.npy", np.asarray([r["src_ms"] for r in kept], dtype=np.int32))
    flat("word_start", [r["starts"] for r in kept], np.int32)
    for k, rule in enumerate(RULES):
        flat(f"support_{rule}", [r["sup"][k] for r in kept], np.int32)
    flat("word_start_ms", [r["w_start_ms"] for r in kept], np.int32)   # target speech: where each word is said
    flat("word_end_ms", [r["w_end_ms"] for r in kept], np.int32)
    flat("seg_w1", [r["seg_w1"] for r in kept], np.int32)
    flat("seg_code1", [r["seg_code1"] for r in kept], np.int32)
    (out / "keys.txt").write_text("\n".join(r["key"] for r in kept) + "\n", encoding="utf-8")
    counts = {lang: {t: sum(1 for r in kept if r["lang"] == lang and r["tier"] == v) for t, v in TIERS.items()}
              for lang in ("cmn", "eng")}
    read_share = {lang: round(float(np.mean([r["read"] for r in kept if r["lang"] == lang and r["tier"] < 2])), 3)
                  for lang in ("cmn", "eng")}
    meta = {"rows": len(kept), "scanned": len(rows), "dropped": dropped, "tiers_by_target": counts,
            "read_share_in_A": read_share, "lag_cap_ms": lag_cap, "max_nir": args.max_nir, "rules": RULES,
            "simul": args.simul, "clean_index": args.clean_index}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "README.md").write_text(__doc__, encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
