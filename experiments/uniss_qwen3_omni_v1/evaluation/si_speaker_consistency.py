"""How steady the output voice is across the pieces of one long recording: WavLM speaker-verification embeddings of
each piece's speech (si_infer *_speech.wav), cosine to the recording's mean voice and between consecutive pieces.

    python si_speaker_consistency.py --infer <si_infer output dir> [--infer ...] --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infer", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).to(args.device).eval()
    report = {}
    for d in args.infer:
        recs = [json.loads(l) for l in open(Path(d) / "results.jsonl", encoding="utf-8")]
        embs = []
        for r in recs:
            if r.get("error") or not r.get("speech_path"):
                continue
            x, sr = sf.read(r["speech_path"], dtype="float32")
            if len(x) < sr * 0.5:
                continue
            inp = fe(x, sampling_rate=sr, return_tensors="pt").to(args.device)
            with torch.no_grad():
                e = model(**inp).embeddings[0]
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        E = np.stack(embs)
        mean = E.mean(0)
        mean /= np.linalg.norm(mean)
        to_mean = E @ mean
        consec = np.sum(E[1:] * E[:-1], axis=1)
        report[d] = {"pieces": len(E), "to_mean": round(float(to_mean.mean()), 4), "to_mean_p10": round(float(np.percentile(to_mean, 10)), 4),
                     "consecutive": round(float(consec.mean()), 4), "consecutive_p10": round(float(np.percentile(consec, 10)), 4)}
        print(d, json.dumps(report[d]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
