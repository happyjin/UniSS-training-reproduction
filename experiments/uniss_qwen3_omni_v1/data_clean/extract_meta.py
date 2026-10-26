"""One compact parquet per chunk of the 1.5M decoded-target rows, for both cleaning plans.

The rows are simul_uniss_subsecond_v2's a45 parts: UniST train-00000..00014,
the first 100k rows of each, with the target BiCodec codes decoded to FLAC by
UniSS's BiCodec and Qwen3-ForcedAligner word times on both sides. Its a68
parts add, for the rows that passed there (98.7%), word grouping, mBERT
bilingual links and each target word's source support time. The manifests are
16 GB of JSON with the GLM and code lists inline; this keeps what the
cleaning reads and joins a68 onto a45 by formal_input_index.

    python extract_meta.py --out data/unist_clean_v1/meta [--jobs 30]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("data/processed/simul_uniss_subsecond_v2/formal_15shard_v1")


def one(job: tuple[str, str, str]) -> str:
    shard, chunk, out = job
    target = Path(out) / f"{shard}_{chunk}.parquet"
    if target.exists():
        return f"{target} exists, kept"
    a68 = {}
    a68_path = ROOT / "a68_parts" / shard / chunk / "formal_manifest.jsonl"
    if a68_path.exists():
        with open(a68_path, "rb") as fh:
            for line in fh:
                r = orjson.loads(line)
                a68[r["formal_input_index"]] = (
                    orjson.dumps(r["target_words"]).decode(), orjson.dumps(r["bilingual_links"]).decode(),
                    orjson.dumps(r["target_support"]).decode(), bool(r.get("formal_a68_pass")))
    cols = {k: [] for k in (
        "key", "shard", "chunk", "line", "id", "src_lang", "tgt_lang", "transcription", "translation",
        "source_audio", "target_audio", "source_duration_ms", "target_duration_ms", "source_parquet",
        "source_row_index", "dataset", "target_bicodec", "bicodec_global", "teacher_source_glm",
        "source_words", "target_chars", "a45_pass", "a68_present", "a68_pass", "target_words",
        "bilingual_links", "target_support")}
    with open(ROOT / "a45_parts" / shard / chunk / "a45_manifest.jsonl", "rb") as fh:
        for line_no, line in enumerate(fh):
            r = orjson.loads(line)
            extra = a68.get(r["formal_input_index"])
            values = {
                "key": f"{shard}/{chunk}/{line_no}", "shard": shard, "chunk": chunk, "line": line_no,
                "id": r["id"], "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                "transcription": r["transcription"], "translation": r["translation"],
                "source_audio": r["source_audio"], "target_audio": r["target_audio"],
                "source_duration_ms": r["source_duration_ms"], "target_duration_ms": r["target_duration_ms"],
                "source_parquet": Path(r["source_parquet"]).name, "source_row_index": r["source_row_index"],
                "dataset": r["id"].rsplit("_", 1)[0], "target_bicodec": r["target_bicodec"],
                "bicodec_global": r["bicodec_global"], "teacher_source_glm": r["teacher_source_glm"],
                "source_words": orjson.dumps(r["source_words"]).decode(),
                "target_chars": orjson.dumps(r["target_words"]).decode(),
                "a45_pass": bool(r.get("formal_a45_pass")), "a68_present": extra is not None,
                "a68_pass": bool(extra and extra[3]),
                "target_words": extra[0] if extra else None, "bilingual_links": extra[1] if extra else None,
                "target_support": extra[2] if extra else None,
            }
            for k, v in values.items():
                cols[k].append(v)
    Path(out).mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    tmp.rename(target)
    return f"{target}: {len(cols['key'])} rows, a68 {sum(cols['a68_present'])}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    jobs = [(s.name, c.name, args.out) for s in sorted((ROOT / "a45_parts").iterdir())
            for c in sorted(s.iterdir()) if (c / "a45_manifest.jsonl").exists()]
    with mp.Pool(min(args.jobs, len(jobs))) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
