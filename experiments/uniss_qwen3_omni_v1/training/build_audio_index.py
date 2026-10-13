"""Turn the manifests into an index that loads in seconds, not minutes.

read_audio_rows parses every manifest line to build its list. At 969,266
rows that was tolerable; at 19,186,799 it is not, and each of eight ranks
pays it separately at process start -- measured at sixteen minutes and
still going, with the GPUs at 0%.

So the scan happens once and lands in flat numpy arrays, the same shape
the TTS cache already uses. A rank then memory-maps them and holds no
copy of the corpus.

Only the fields the S2ST and TTS tasks read are kept: the audio URI, the
two texts, the language pair, and where the codes live. Anything else in
a manifest line stays in the manifest.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from experiments.uniss_omni_s2st_v1.data.wds_resolver import (
    available_shards,
    parse_wds_uri,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--task", default="speech_to_speech_translation")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    paths = sorted(p for pat in args.manifests for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no manifests matched {args.manifests}")
    present = available_shards(args.audio_root)
    print(f"{len(paths)} manifests, {len(present)} shards present", flush=True)

    uri: list[str] = []
    src_text: list[str] = []
    tgt_text: list[str] = []
    langs: list[str] = []
    parquet: list[str] = []
    row_index: list[int] = []
    total = kept = 0
    started = time.perf_counter()
    for number, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                record = json.loads(line)
                if args.task and record.get("task") != args.task:
                    continue
                audios = record.get("audios") or []
                if not audios or parse_wds_uri(audios[0]).shard not in present:
                    continue
                provenance = record.get("provenance") or {}
                uri.append(audios[0])
                src_text.append(record.get("source_transcript") or "")
                tgt_text.append(record.get("target_text") or "")
                langs.append(f"{record['source_lang']}|{record['target_lang']}")
                parquet.append(provenance.get("source_parquet", ""))
                row_index.append(int(provenance.get("source_row_index", -1)))
                kept += 1
        if number % 20 == 0 or number == len(paths):
            rate = total / max(1e-9, time.perf_counter() - started)
            print(f"  {number}/{len(paths)}  {total:,} 行  保留 {kept:,}  "
                  f"{rate:,.0f} 行/s", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Flat utf-8 blobs with offsets rather than an object array: a rank
    # memory-maps these and decodes only the rows it draws.
    def save_strings(name: str, values: list[str]) -> None:
        blob = "\x00".join(values).encode("utf-8")
        np.save(out / f"{name}.npy", np.frombuffer(blob, dtype=np.uint8))
    for name, values in (("uri", uri), ("src_text", src_text),
                         ("tgt_text", tgt_text), ("langs", langs),
                         ("parquet", parquet)):
        save_strings(name, values)
    np.save(out / "row_index.npy", np.asarray(row_index, dtype=np.int64))
    meta = {"rows": kept, "scanned": total, "manifests": len(paths),
            "task": args.task, "built_seconds": round(time.perf_counter() - started, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
