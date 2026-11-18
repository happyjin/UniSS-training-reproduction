"""Voice-texture measures for reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md, on paired talker runs
(speech/<run>/talker/results.jsonl speech_path): over speech frames (as game_muffle_analysis.spectral) the median
spectral flatness 300-4000 Hz (higher = noisier, less harmonic), mean spectral contrast of librosa's 6 octave bands
(lower = flatter peaks / valleys, smeared formants) and the harmonic share of librosa HPSS (harmonic / (harmonic +
percussive) energy; lower = less periodic voicing). Prints arm medians and paired differences.
    python -m experiments.mist_w3po_v1.evaluation.game_voice_texture <runA>:<runB> ...
"""
import json, sys, multiprocessing as mp
import numpy as np
import librosa
from pathlib import Path
from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import read16
SPD = Path(__file__).resolve().parents[3] / "eval_outputs/mist_w3po_v1/speech"


def feats(p):
    x = read16(p)
    if len(x) < 4000:
        return None
    S = np.abs(librosa.stft(x, n_fft=512, hop_length=160)) ** 2
    e = 10 * np.log10(S.sum(0) + 1e-12)
    sp = e > np.percentile(e, 95) - 25
    f = librosa.fft_frequencies(sr=16000, n_fft=512)
    band = (f >= 300) & (f <= 4000)
    flat = librosa.feature.spectral_flatness(S=S[band][:, sp], power=1.0)[0]
    con = librosa.feature.spectral_contrast(S=np.sqrt(S[:, sp]), sr=16000, n_fft=512, fmin=125.0, n_bands=5)
    h, pc = librosa.effects.hpss(x)
    return {"flatness": float(np.median(flat)), "contrast": float(con[:5].mean()),
            "harm_share": float((h ** 2).sum() / ((h ** 2).sum() + (pc ** 2).sum() + 1e-12))}


def job(a):
    i, p = a
    try:
        return i, feats(p)
    except Exception:
        return i, None


if __name__ == "__main__":
    for pair in sys.argv[1:]:
        A, B = pair.split(":")
        R = {}
        for run in (A, B):
            L = [json.loads(l) for l in open(SPD / run / "talker/results.jsonl")]
            with mp.get_context("fork").Pool(40) as pool:
                R[run] = {i: v for i, v in pool.map(job, [(r["id"], r["speech_path"]) for r in L if r.get("speech_path")]) if v}
        ids = sorted(set(R[A]) & set(R[B]))
        print(f"{B} vs {A}: n={len(ids)}")
        for k in ("flatness", "contrast", "harm_share"):
            x = np.array([R[A][i][k] for i in ids]); y = np.array([R[B][i][k] for i in ids]); d = y - x
            rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            print(f"  {k:10} {A[-1]} {np.median(x):.4f}  {B[-1]} {np.median(y):.4f}  mean diff {d.mean():+.4f} "
                  f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  B>A in {np.mean(d > 0) * 100:.0f}%")
