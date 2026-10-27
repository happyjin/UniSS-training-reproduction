"""Audio metrics for a directory of synthesized wavs: Whisper-large-v3 WER (Whisper English normaliser on both
sides), UTMOS22-strong, and per-word mean log-F0 error against the real recording (words located by the committed
durations in the synthesis and by the d* alignment in the recording; frames voiced in both only).

Writes <out>/audio_metrics.json and per_utt.jsonl.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from stream_common import token_is_word, unit_token_spans  # noqa: E402

os.environ.setdefault("TORCH_HOME", "/opt/dlami/nvme/jasonleeeli/cache/torch")
HOP, SR = 256, 22050


def edit_distance(a, b):
  d = list(range(len(b) + 1))
  for i in range(1, len(a) + 1):
    prev, d[0] = d[0], i
    for j in range(1, len(b) + 1):
      cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
      prev, d[j] = d[j], cur
  return d[-1]


def word_logf0(wav, unit_frames, word_masks):
  """Mean log-F0 over the word tokens of each unit; unit_frames: list of per-token frame counts per unit."""
  f0 = librosa.yin(wav, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
  voiced = librosa.feature.rms(y=wav, frame_length=1024, hop_length=HOP)[0] > 0.01
  n = min(len(f0), len(voiced))
  out, pos = [], 0
  for frames, wm in zip(unit_frames, word_masks):
    idx = []
    for c, w in zip(frames, wm):
      if w:
        idx.extend(range(pos, pos + int(c)))
      pos += int(c)
    idx = [t for t in idx if t < n and voiced[t]]
    out.append(float(np.mean(np.log(f0[idx]))) if len(idx) >= 3 else None)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("--align-dir", required=True)
  ap.add_argument("--manifest", default=str(ROOT / "data/manifest_v2.jsonl"))
  ap.add_argument("--whisper", default="openai/whisper-large-v3")
  ap.add_argument("--no-f0", action="store_true")
  args = ap.parse_args()
  run = Path(args.run)
  if (run / "audio_metrics.json").exists():
    raise FileExistsError(run / "audio_metrics.json")
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
  proc = WhisperProcessor.from_pretrained(args.whisper)
  asr = WhisperForConditionalGeneration.from_pretrained(args.whisper, torch_dtype=torch.float16).cuda().eval()
  norm = EnglishTextNormalizer(proc.tokenizer.english_spelling_normalizer)
  utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda().eval()
  rows = {r["utt_id"]: r for r in map(json.loads, open(args.manifest))}
  durs = np.load(Path(args.align_dir) / "durations.npz")
  commits = {}
  for f in glob.glob(str(run / "commits*.jsonl")):
    for d in map(json.loads, open(f)):
      commits[d["utt_id"]] = d
  wavs = sorted(glob.glob(str(run / "wavs" / "*.wav")))
  per, errs, words, moss, f0e = [], 0, 0, [], []
  for b in range(0, len(wavs), 16):
    batch = wavs[b:b + 16]
    auds = [sf.read(w, dtype="float32")[0] for w in batch]
    a16 = [librosa.resample(a, orig_sr=SR, target_sr=16000) for a in auds]
    feats = proc(a16, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
      gen = asr.generate(feats.input_features.cuda().half(), attention_mask=feats.attention_mask.cuda(),
                         language="en", task="transcribe")
    hyps = proc.batch_decode(gen, skip_special_tokens=True)
    for w, a, hyp in zip(batch, auds, hyps):
      uid = Path(w).stem
      r = rows[uid]
      ref_n, hyp_n = norm(r["text"]).split(), norm(hyp).split()
      e = edit_distance(ref_n, hyp_n)
      errs, words = errs + e, words + len(ref_n)
      with torch.no_grad():
        mos = float(utmos(torch.from_numpy(a)[None].cuda(), SR).item())
      moss.append(mos)
      rec = {"utt_id": uid, "hyp": hyp, "wer": e / max(1, len(ref_n)), "utmos": mos}
      if not args.no_f0 and uid in commits:
        isw = token_is_word(r["clean"])
        spans = unit_token_spans(r)
        wmask = [isw[s:t] for s, t in spans]
        syn = word_logf0(a, [u["dur"] for u in commits[uid]["units"]], wmask)
        gt_wav = sf.read(r["audio_path"], dtype="float32")[0]
        gt = word_logf0(gt_wav, [durs[uid][s:t] for s, t in spans], wmask)
        diffs = [abs(x - y) for x, y in zip(syn, gt) if x is not None and y is not None]
        rec["f0_err"] = float(np.mean(diffs)) if diffs else None
        f0e.extend(diffs)
      per.append(rec)
  out = {"n": len(per), "wer": errs / max(1, words), "utmos": float(np.mean(moss)),
         "f0_logmae": float(np.mean(f0e)) if f0e else None}
  with open(run / "per_utt_audio.jsonl", "w") as f:
    for p in per:
      f.write(json.dumps(p) + "\n")
  json.dump(out, open(run / "audio_metrics.json", "w"), indent=2)
  print(json.dumps(out), flush=True)


if __name__ == "__main__":
  main()
