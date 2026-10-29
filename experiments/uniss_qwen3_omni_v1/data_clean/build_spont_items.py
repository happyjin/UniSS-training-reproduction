"""Spontaneous single-speaker audio for the ABC labels (single-speaker paper v4, candidate A2): the oracle labels need no
reference (they are the frozen Thinker's own whole-sentence continuation), only audio closer to real speech than UniST's.

  en2zh  GigaSpeech (podcast and YouTube; no audiobooks): training segments from the XL train parquet shards, dev
         segments from GigaSpeech test
  zh2en  WenetSpeech (YouTube and podcast, subset M, confidence >= 0.95): training segments from train, dev segments
         from WenetSpeech DEV
Kept: 3-15 s; at least 6 English words / 6 Chinese characters; no <SIL>/<MUSIC>/<NOISE>/<OTHER> segment (GigaSpeech);
mostly Chinese characters (WenetSpeech); at most 2 segments a source recording, so many speakers and topics. Audio is
written as 16 kHz mono wav. Items: {id, audio, dur_ms, transcript} (dev items get their silver translation later).
Neither corpus is one of our test sets (RealSI, ACL60/60, AliMeeting). audit.json counts what each filter dropped.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_spont_items --out data/si_spont_v1
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

RAW = "/mnt/speech-h200-fsx/data/data/raw"
GIGA = f"{RAW}/dataset=gigaspeech/source_version=hf_speechcolab/parquet-data/xl"
WENET = f"{RAW}/dataset=wenetspeech/source_version=openslr_slr121"
GARBAGE = re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>")
PUNCT = {"<COMMA>": ",", "<PERIOD>": ".", "<QUESTIONMARK>": "?", "<EXCLAMATIONPOINT>": "!"}
HAN = re.compile(r"[一-鿿]")


def giga_text(t: str) -> str:
    for k, v in PUNCT.items():
        t = t.replace(" " + k, v).replace(k, v)
    t = t.lower().strip()
    return re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t).replace(" i ", " I ")


def giga(files, n, rng, audit, tag):
    import pyarrow.parquet as pq
    cand = []
    for f in files:
        p = pq.ParquetFile(f)
        meta = p.read(columns=["segment_id", "audio_id", "source", "begin_time", "end_time", "text"]).to_pylist()
        for i, r in enumerate(meta):
            dur = r["end_time"] - r["begin_time"]
            if r["source"] not in (1, 2):
                audit[f"{tag}:audiobook"] += 1
            elif not 3.0 <= dur <= 15.0:
                audit[f"{tag}:duration"] += 1
            elif GARBAGE.search(r["text"]):
                audit[f"{tag}:garbage_tag"] += 1
            elif len(giga_text(r["text"]).split()) < 6:
                audit[f"{tag}:short_text"] += 1
            else:
                cand.append((f, i, r))
    rng.shuffle(cand)
    per, keep = defaultdict(int), []
    for c in cand:
        if per[c[2]["audio_id"]] < 2:
            per[c[2]["audio_id"]] += 1
            keep.append(c)
        if len(keep) == n:
            break
    return keep


def write_giga(keep, out, tag):
    import pyarrow.parquet as pq
    rows = []
    by_file = defaultdict(list)
    for f, i, r in keep:
        by_file[f].append((i, r))
    for f, lst in by_file.items():   # row group by row group: a whole file's audio column is too large for one array
        pf = pq.ParquetFile(f)
        starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
        groups = defaultdict(list)
        for i, r in lst:
            g = int(np.searchsorted(starts, i, side="right") - 1)
            groups[g].append((i - int(starts[g]), r))
        for g, members in groups.items():
            col = pf.read_row_group(g, columns=["audio"]).column("audio")
            for i, r in members:
                x, sr = sf.read(io.BytesIO(col[i].as_py()["bytes"]), dtype="float32", always_2d=True)
                assert sr == 16000, sr
                path = out / "audio" / tag / f"{r['segment_id']}.wav"
                sf.write(path, x.mean(1), 16000)
                rows.append({"id": f"giga_{r['segment_id']}", "audio": str(path.resolve()), "dur_ms": int(len(x) / 16),
                             "transcript": giga_text(r["text"])})
    return rows


def wenet(n_train, n_dev, rng, audit):
    data = json.load(open(f"{WENET}/WenetSpeech.json", encoding="utf-8"))
    cand = {"train": [], "dev": []}
    for a in data["audios"]:
        split = "dev" if a["path"].startswith("audio/dev") else "train" if a["path"].startswith("audio/train") else None
        if split is None:
            continue
        if split == "train" and a.get("source") not in ("youtube", "podcast"):
            audit["zh:source"] += len(a["segments"])
            continue
        segs = []
        for s in a["segments"]:
            dur = s["end_time"] - s["begin_time"]
            ok_sub = ("DEV" in s["subsets"]) if split == "dev" else ("M" in s["subsets"])
            if not ok_sub:
                audit[f"zh_{split}:subset"] += 1
            elif s.get("confidence", 0) < 0.95:
                audit[f"zh_{split}:confidence"] += 1
            elif not 3.0 <= dur <= 15.0:
                audit[f"zh_{split}:duration"] += 1
            elif len(HAN.findall(s["text"])) < 6 or len(HAN.findall(s["text"])) < 0.8 * len(s["text"].replace(" ", "")):
                audit[f"zh_{split}:text"] += 1
            else:
                segs.append(s)
        rng.shuffle(segs)
        cand[split] += [(a["path"], s) for s in segs[:2]]
    del data
    for k in cand:
        rng.shuffle(cand[k])
    return cand["train"][:n_train], cand["dev"][:n_dev]


def write_wenet(keep, out, tag):
    rows = []
    for path, s in keep:
        with sf.SoundFile(f"{WENET}/{path}") as fh:
            sr = fh.samplerate
            fh.seek(int(s["begin_time"] * sr))
            x = fh.read(int((s["end_time"] - s["begin_time"]) * sr), dtype="float32", always_2d=True).mean(1)
        if sr != 16000:
            from scipy.signal import resample_poly
            x = resample_poly(x, 16000, sr).astype(np.float32)
        p = out / "audio" / tag / f"{s['sid']}.wav"
        sf.write(p, x, 16000)
        rows.append({"id": f"wenet_{s['sid']}", "audio": str(p.resolve()), "dur_ms": int(len(x) / 16), "transcript": s["text"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--giga-shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    for t in ("en_train", "en_dev", "zh_train", "zh_dev"):
        (out / "audio" / t).mkdir(parents=True)
    rng = random.Random(args.seed)
    audit = Counter()
    train_files = sorted(glob.glob(f"{GIGA}/train-*.parquet"))
    pick = [train_files[i] for i in sorted(rng.sample(range(len(train_files)), args.giga_shards))]
    en_tr = write_giga(giga(pick, args.n_train, rng, audit, "en_train"), out, "en_train")
    en_dev = write_giga(giga(sorted(glob.glob(f"{GIGA}/test-*.parquet")), args.n_dev, rng, audit, "en_dev"), out, "en_dev")
    zh_tr_k, zh_dev_k = wenet(args.n_train, args.n_dev, rng, audit)
    zh_tr = write_wenet(zh_tr_k, out, "zh_train")
    zh_dev = write_wenet(zh_dev_k, out, "zh_dev")
    for name, rows in (("items_en2zh", en_tr), ("dev_en2zh", en_dev), ("items_zh2en", zh_tr), ("dev_zh2en", zh_dev)):
        with open(out / f"{name}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    summary = {"dropped": dict(audit), "kept": {"en_train": len(en_tr), "en_dev": len(en_dev), "zh_train": len(zh_tr),
                                                "zh_dev": len(zh_dev)},
               "hours": {k: round(sum(r["dur_ms"] for r in v) / 3.6e6, 2) for k, v in
                         (("en_train", en_tr), ("en_dev", en_dev), ("zh_train", zh_tr), ("zh_dev", zh_dev))},
               "giga_shards": [Path(f).name for f in pick]}
    (out / "audit.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("SPONT_DONE", json.dumps(summary["kept"]), json.dumps(summary["hours"]), flush=True)


if __name__ == "__main__":
    main()
