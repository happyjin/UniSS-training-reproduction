"""Experiment A (SV-anchored voice code; reports/mist_w3po_v1/robust_a_svanchor_prereg.zh-CN.md): shared, numpy-only
definitions used by both venvs (nemo_diar for TitaNet / ECAPA2, vllm_omni_030 for BiCodec):
  * training pairs = the Idea 1 M2 / V1 control-arm data (data/open_voicecond_v1/idea1/train_en + train_zh, 6 s clips,
    order = concatenation as in idea1_e2e_train.py);
  * ONE prompt-span crop per pair, drawn once with seed 20261119: length L ~ U[1.5, 4.5] s, start ~ U[0, 6 s - L]
    (the deploy prompt is the first max(first write, 1 s) of the source, 1.5-4 s for 10-90 % of S1 dev prompts);
    the same crop is used for the noisy y (TitaNet input, phi_V1 input) and the clean s (target u* = phi_0(s crop));
  * monitor pairs (record only) = the 200 ID-1 diag dev pairs V1's training log used, cropped the same way (seed + 1);
  * S1 dev rows = the wave3 V1 table rows (prompt files wave3_combos/V/tokens/prompts/<d>/<key>.wav, sha1 checked);
    clean-twin prompts of noisy rows = oracle_voice/tokens/twin_prompts/<d>/<key>.wav.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SEED = 20261119
SR = 16000
CLIP = 96000
LMIN, LMAX = int(1.5 * SR), int(4.5 * SR)
I = REPO / "data/open_voicecond_v1/idea1"
TRAIN = [I / "train_en", I / "train_zh"]
DIAG = [I / "diag_en", I / "diag_zh"]
VT = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/tokens"
ORT = REPO / "eval_outputs/mist_w3po_v1/oracle_voice/tokens"
V1 = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
DIRS = ("en2zh", "zh2en")


def crops(n: int, seed: int) -> np.ndarray:
    """(n, 2) int64 [start, length] within a 6 s clip"""
    rng = np.random.default_rng(seed)
    L = rng.integers(LMIN, LMAX + 1, size=n)
    s = (rng.random(n) * (CLIP - L + 1)).astype(np.int64)
    return np.stack([s, L], 1).astype(np.int64)


def train_crops() -> np.ndarray:
    n = sum(np.load(d / "y.npy", mmap_mode="r").shape[0] for d in TRAIN)
    return crops(n, SEED)


def train_iter(kind: str):
    """yield (j, cropped float64 waveform) over the training pairs for kind in {y, s}"""
    C = train_crops()
    j = 0
    for d in TRAIN:
        A = np.load(d / f"{kind}.npy", mmap_mode="r")
        for k in range(A.shape[0]):
            s, L = C[j]
            yield j, np.asarray(A[k, s:s + L], dtype=np.float64)
            j += 1


def mon_rows():
    """the 200 ID-1 diag dev pairs of idea1_e2e_train.py's monitor: list of (dir, id)"""
    out = []
    for d in DIAG:
        rows = [r for r in map(json.loads, open(d / "manifest.jsonl", encoding="utf-8")) if r["split"] == "dev"]
        out += [(d, r["id"]) for r in rows[:: max(1, len(rows) // 100)][:100]]
    return out


def mon_iter(kind: str):
    import soundfile as sf
    rows = mon_rows()
    C = crops(len(rows), SEED + 1)
    for j, (d, i) in enumerate(rows):
        w, sr = sf.read(str(d / "wav" / f"{i}_{kind}.wav"), dtype="float64")
        assert sr == SR
        s, L = C[j]
        L = min(L, len(w))
        s = min(s, len(w) - L)
        yield j, w[s:s + L]


def dev_rows(d: str):
    return [json.loads(l) for l in open(VT / f"tokens_v1_{d}.jsonl", encoding="utf-8") if l.strip()]


def dev_prompt(d: str, key: str) -> Path:
    return VT / "prompts" / d / f"{key}.wav"


def twin_prompt(d: str, key: str) -> Path:
    return ORT / "twin_prompts" / d / f"{key}.wav"
