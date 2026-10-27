"""Build the LJSpeech manifest (handbook P0) with official VITS splits plus a held-out calibration split.

The calibration split (T3) is carved from the official training list by crc32, so no system ever
trains on it. Output: <out>/manifest.jsonl, one row per utterance.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text import clean_text, split_units  # noqa: E402


def read_split(path):
  return [line.split("|")[0].split("/")[-1].replace(".wav", "") for line in open(path) if line.strip()]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ljs", required=True)
  ap.add_argument("--filelists", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--calib-mod", type=int, default=31)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  manifest = out / "manifest.jsonl"
  if manifest.exists():
    raise FileExistsError(f"refusing to overwrite {manifest}")
  fl = Path(args.filelists)
  split = {}
  for name in ["train", "val", "test"]:
    for uid in read_split(fl / f"ljs_audio_text_{name}_filelist.txt"):
      split[uid] = name
  for uid, name in list(split.items()):
    if name == "train" and zlib.crc32(uid.encode()) % args.calib_mod == 0:
      split[uid] = "calib"
  rows = []
  for line in open(Path(args.ljs) / "metadata.csv", encoding="utf-8"):
    uid, _, norm = line.rstrip("\n").split("|")
    if uid not in split:
      continue
    wav = Path(args.ljs) / "wavs" / f"{uid}.wav"
    info = sf.info(str(wav))
    cleaned = clean_text(norm)
    rows.append({"utt_id": uid, "split": split[uid], "group_id": uid[:5], "audio_path": str(wav),
                 "sample_rate": info.samplerate, "duration_s": round(info.duration, 4), "text": norm,
                 "clean": cleaned, "units": split_units(cleaned), "speaker_id": "ljs", "language": "en",
                 "license": "public-domain (US)", "is_synthetic": False})
  with open(manifest, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  counts = {}
  for r in rows:
    counts[r["split"]] = counts.get(r["split"], 0) + 1
  print(json.dumps({"rows": len(rows), "splits": counts}))


if __name__ == "__main__":
  main()
