"""Plan 1's training index: the scored rows that pass, in AudioIndex's format.

The rows are taken from audio_index_full (same uri, texts, parquet, row), so
the trainer reads them exactly as before; the index adds each row's source
duration (source_ms.npy, so a run without the Thinker decodes no audio) and
the source GLM re-tokenised from the local audio (teacher_glm_*.npy), which is
what inference sees -- UniST's released tokens agree with it on about 42%.

Selection (the pilot's distributions set the bounds; see --help):
  * the target says the translation: CER/WER <= --max-tgt-err, the
    transcript in the target's script, no repeated 8-code run;
  * code-switched Chinese targets: every Latin word heard;
  * the source is the utterance that was translated: error <= --max-src-err;
  * speaking rate and target/source duration inside the 1st-99th percentile
    of each target language;
  * passing code-switched rows appear --codeswitch-weight times.

    python build_clean_index.py --scores data/unist_clean_v1/scores.parquet \
        --out data/unist_clean_v1/audio_index_clean_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FULL = Path("data/unist_omni_v1/audio_index_full")


def blob(path: Path, name: str) -> list[str]:
    return bytes(np.load(path / f"{name}.npy", mmap_mode="r")).decode("utf-8").split("\x00")


def save_blob(path: Path, name: str, parts: list[str]) -> None:
    np.save(path / f"{name}.npy", np.frombuffer("\x00".join(parts).encode("utf-8"), dtype=np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--meta", default="data/unist_clean_v1/meta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tgt-err", type=float, default=0.05)
    ap.add_argument("--max-src-err", type=float, default=0.30)
    ap.add_argument("--codeswitch-weight", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")

    s = pq.read_table(args.scores).to_pandas()
    bounds = {}
    for lang, g in s.groupby("tgt_lang"):
        bounds[lang] = {k: (float(g[k].quantile(0.01)), float(g[k].quantile(0.99))) for k in ("rate", "dur_ratio")}
    ok = ((s.tgt_err <= args.max_tgt_err) & s.tgt_script_ok & ~s.loop8 & (s.src_err <= args.max_src_err)
          & ((s.latin_words == 0) | (s.latin_missed == 0)))
    for lang, b in bounds.items():
        m = s.tgt_lang == lang
        for k, (lo, hi) in b.items():
            ok &= ~m | s[k].between(lo, hi)
    kept = s[ok]
    reasons = {
        "tgt_err": int((s.tgt_err > args.max_tgt_err).sum()), "script": int((~s.tgt_script_ok).sum()),
        "loop8": int(s.loop8.sum()), "src_err": int((s.src_err > args.max_src_err).sum()),
        "latin_missed": int(((s.latin_words > 0) & (s.latin_missed > 0)).sum()),
    }
    print(f"{len(s)} scored, {len(kept)} kept ({len(kept) / len(s):.1%}); dropped by {reasons}", flush=True)

    meta_cols = ["key", "source_parquet", "source_row_index", "source_duration_ms", "teacher_source_glm"]
    meta = {r["key"]: r for f in sorted(Path(args.meta).glob("*.parquet"))
            for r in pq.read_table(f, columns=meta_cols).to_pylist()}
    full_parquet, full_rows = blob(FULL, "parquet"), np.load(FULL / "row_index.npy")
    position = {(p, int(r)): i for i, (p, r) in enumerate(zip(full_parquet, full_rows))}
    fields = {n: blob(FULL, n) for n in ("uri", "src_text", "tgt_text", "langs")}

    order, missing = [], 0
    for r in kept.itertuples():
        m = meta[r.key]
        i = position.get((m["source_parquet"], int(m["source_row_index"])))
        if i is None:
            missing += 1
            continue
        times = args.codeswitch_weight if r.latin_words > 0 else 1
        order += [(i, m)] * times
    if missing:
        print(f"{missing} kept rows are not in audio_index_full (no local source audio); left out", flush=True)

    out.mkdir(parents=True)
    for name in ("uri", "src_text", "tgt_text", "langs"):
        save_blob(out, name, [fields[name][i] for i, _ in order])
    save_blob(out, "parquet", [full_parquet[i] for i, _ in order])
    np.save(out / "row_index.npy", np.asarray([full_rows[i] for i, _ in order], dtype=full_rows.dtype))
    np.save(out / "source_ms.npy", np.asarray([m["source_duration_ms"] for _, m in order], dtype=np.int32))
    glm = [m["teacher_source_glm"] for _, m in order]
    np.save(out / "teacher_glm_offsets.npy", np.concatenate([[0], np.cumsum([len(g) for g in glm])]).astype(np.int64))
    np.save(out / "teacher_glm_values.npy", np.concatenate([np.asarray(g, dtype=np.int32) for g in glm]))
    info = {"rows": len(order), "unique_rows": len(kept) - missing, "scored": len(s), "dropped_by": reasons,
            "thresholds": {"max_tgt_err": args.max_tgt_err, "max_src_err": args.max_src_err,
                           "codeswitch_weight": args.codeswitch_weight, "bounds": bounds},
            "codeswitch_rows": int((kept.latin_words > 0).sum()), "source": str(FULL), "scores": args.scores,
            "task": "speech_to_speech_translation"}
    (out / "meta.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "thresholds"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
