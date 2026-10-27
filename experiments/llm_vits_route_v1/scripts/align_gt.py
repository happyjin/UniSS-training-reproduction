"""Frozen duration labels d* (handbook 2.3): MAS of the E0 backbone on every ground-truth recording.

Writes <out>/durations.npz (utt_id -> int16 token durations, sum == spectrogram frames) and
<out>/unit_prosody.jsonl with, per unit, word frames (chars and blanks of letters/apostrophes/hyphens) and
pause frames (space, punctuation and their blanks): the measurable targets Y used by every system.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
import monotonic_align  # noqa: E402
from mel_processing import spectrogram_torch  # noqa: E402
from stream_common import load_backbone, unit_word_pause  # noqa: E402
from text import intersperse, text_to_ids  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--config", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--batch", type=int, default=32)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  if (out / "durations.npz").exists():
    raise FileExistsError(out / "durations.npz")
  hps = json.load(open(args.config))
  dd = hps["data"]
  net = load_backbone(args.config, args.ckpt).cuda().eval()
  rows = [json.loads(l) for l in open(args.manifest)]
  durs = {}
  with open(out / "unit_prosody.jsonl", "w") as fu:
    for b in range(0, len(rows), args.batch):
      chunk = rows[b:b + args.batch]
      ids = [torch.LongTensor(intersperse(text_to_ids(r["clean"]))) for r in chunk]
      wavs = [torch.from_numpy(sf.read(r["audio_path"], dtype="float32")[0]) for r in chunk]
      xl = torch.LongTensor([len(i) for i in ids]).cuda()
      wl = torch.LongTensor([len(w) for w in wavs])
      x = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True).cuda()
      w = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).cuda()
      with torch.no_grad():
        spec = spectrogram_torch(w, dd["filter_length"], dd["sampling_rate"], dd["hop_length"], dd["win_length"])
        yl = (wl // dd["hop_length"]).cuda()
        spec = spec[:, :, :int(yl.max())]
        h, m_p, logs_p, x_mask = net.enc_p(x, xl)
        z, m_q, logs_q, y_mask = net.enc_q(spec, yl)
        z_p = net.flow(m_q * y_mask, y_mask)   # posterior mean: labels must not depend on a noise draw
        s_p_sq_r = torch.exp(-2 * logs_p)
        neg = (torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
               + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
               + torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
               + torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True))
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = monotonic_align.maximum_path(neg, attn_mask.squeeze(1))
        d = attn.sum(1).long().cpu().numpy()
      for k, r in enumerate(chunk):
        dk = d[k, :int(xl[k])].astype(np.int16)
        assert dk.sum() == int(yl[k]), r["utt_id"]
        durs[r["utt_id"]] = dk
        fu.write(json.dumps({"utt_id": r["utt_id"], "units": unit_word_pause(r, dk)}) + "\n")
      print(f"{b + len(chunk)}/{len(rows)}", flush=True)
  np.savez(out / "durations.npz", **durs)


if __name__ == "__main__":
  main()
