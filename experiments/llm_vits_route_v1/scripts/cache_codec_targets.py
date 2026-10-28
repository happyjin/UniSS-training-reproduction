"""Stage A caches for the pretrained-codec schemes (tts_schemes environment).

  --what s3   VoxCPM2 AudioVAE posterior mu/logvar, 25 Hz x 64 (handbook S3.4; the strict VAE keeps logvar)
  --what s4   Qwen3-TTS-Tokenizer-12Hz codes, 12.5 Hz x 16 codebooks (handbook S4.3)
Writes <out>/shard<k>of<n>.npz (utt_id -> array; S3 stores mu and logvar stacked as float16 [2, 64, T_A]).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TOK12 = "/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/7dd38ad4e9bad454aae9cd937d0cd577604fe229"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--what", required=True, choices=["s3", "s4"])
  ap.add_argument("--out", required=True)
  ap.add_argument("--shard", type=int, default=0)
  ap.add_argument("--num-shards", type=int, default=1)
  ap.add_argument("--batch", type=int, default=16)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  dst = out / f"shard{args.shard}of{args.num_shards}.npz"
  if dst.exists():
    raise FileExistsError(dst)
  rows = [json.loads(l) for l in open(ROOT / "data/manifest_v2.jsonl")][args.shard::args.num_shards]
  res = {}
  if args.what == "s3":
    from audiovae import encode_stats, load_audiovae
    vae = load_audiovae("cuda")
    for n, r in enumerate(rows):
      w = torch.from_numpy(sf.read(r["audio_path"], dtype="float32")[0])[None].cuda()
      mu, lv = encode_stats(vae, w)
      res[r["utt_id"]] = torch.cat([mu, lv], 0).half().cpu().numpy()
      if n % 500 == 0:
        print(f"{n}/{len(rows)}", flush=True)
  else:
    from qwen_tts import Qwen3TTSTokenizer
    tok = Qwen3TTSTokenizer.from_pretrained(TOK12, device_map="cuda:0")
    for b in range(0, len(rows), args.batch):
      chunk = rows[b:b + args.batch]
      wavs = [sf.read(r["audio_path"], dtype="float32")[0] for r in chunk]
      enc = tok.encode(wavs, sr=22050)
      for r, c in zip(chunk, enc.audio_codes):
        res[r["utt_id"]] = c.cpu().numpy().astype(np.int16)
      if b % 800 == 0:
        print(f"{b}/{len(rows)}", flush=True)
  np.savez(dst, **res)
  print(f"DONE {len(res)} -> {dst}", flush=True)


if __name__ == "__main__":
  main()
