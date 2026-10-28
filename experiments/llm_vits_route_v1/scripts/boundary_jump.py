"""How abrupt the seams between committed units are: on the log-mel spectrogram (hop 256 at 22.05 kHz, the latent
clock), the mean absolute change between adjacent frames at the first frame of every unit after the first, divided by
the same change averaged over all frames of the utterance (1 = a seam is as smooth as speech is on average; natural
speech also changes more at word starts, so the recording, cut at its d* unit boundaries, is the reference).

  python boundary_jump.py name=<run dir> ... [--gt]
Run dirs hold wavs/ and commits*.jsonl (unit durations in frames). --gt adds the recordings with d* boundaries.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from eval_metrics import load_runs  # noqa: E402
from stream_common import unit_token_spans  # noqa: E402

HOP, SR = 256, 22050


def jump(wav, bounds):
    mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=1024, hop_length=HOP, n_mels=80, center=False)
    lm = np.log(mel + 1e-5)
    d = np.abs(np.diff(lm, axis=1)).mean(0)                 # d[t-1] = change from frame t-1 to t
    b = [t - 1 for t in bounds if 1 <= t - 1 < len(d)]
    return float(d[b].mean()) if b else np.nan, float(d.mean()), len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    rows = {r["utt_id"]: r for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl"))}
    out = {}
    specs = [s.split("=", 1) for s in args.runs]
    if args.gt:
        specs.append(("recording", None))
    for name, d in specs:
        num, den, nb = [], [], 0
        if d is None:
            durs = dict(np.load(ROOT / "runs/final/align/durations.npz"))
            ids = [k for k, r in rows.items() if r["split"] == args.split]
            for uid in ids:
                spans = unit_token_spans(rows[uid])
                cum = np.cumsum(durs[uid])
                bounds = [int(cum[a - 1]) for a, _ in spans[1:] if a > 0]
                wav, _ = sf.read(rows[uid]["audio_path"], dtype="float32")
                j, m, n = jump(wav, bounds)
                if n:
                    num.append(j * n); den.append(m * n); nb += n
        else:
            runs = load_runs(Path(d))
            for uid, rec in runs.items():
                us = sorted(rec["units"], key=lambda u: u["i"])
                cum = np.cumsum([sum(u["dur"]) for u in us])
                bounds = [int(c) for c in cum[:-1]]
                wav, _ = sf.read(Path(d) / "wavs" / f"{uid}.wav", dtype="float32")
                j, m, n = jump(wav, bounds)
                if n:
                    num.append(j * n); den.append(m * n); nb += n
        out[name] = {"boundary_jump_ratio": round(float(np.sum(num) / np.sum(den)), 4), "boundaries": nb}
        print(name, json.dumps(out[name]), flush=True)


if __name__ == "__main__":
    main()
