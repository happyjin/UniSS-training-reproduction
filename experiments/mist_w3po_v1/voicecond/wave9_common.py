"""Robust wave 9 (reports/mist_w3po_v1/robust_wave9_anchor_prereg.zh-CN.md): voice-anchored Thinker, shared pieces.

Anchor tokens: u = C(first 1.0 s of the audio) with C = the V1 denoising encoder (voicecond_idea1/m2/ctrl/encoder.pt,
loaded into BiCodecPhi as wave5_m_v1tokens.py does), the pre-FSQ (32, 6) code, flattened to 192 dims; PCA (fitted on
clean wave-5 training audio by wave9_anchor_fit.py, fixed in PCA_FILE) -> the first 8 components -> each into 32 bins
at the training quantiles -> token id ANCHOR_LO + 32 k + bin (k = 0..7). The 256 ids ANCHOR_LO .. ANCHOR_LO + 255 are
the last 256 rows of the Thinker's 152064-row embedding; the tokenizer's largest id (incl. added tokens) is 151675, so
none of them can be produced by the tokenizer (check_unused asserts it). In the prompt the 8 ids follow <|audio_end|>.

Crosstalk (both arms, wave9_train.py): with probability 0.5 (random.Random(sha1("20261120|w9|xt|<s>|<j>"))) another
training sentence of the same source language and a different speaker is added from an onset U[1.0 s, 0.8 dur] to
the end of the target (truncated there, not looped), at SIR ~ U(-5, +10) dB between the active-RMS levels of the two
whole clips; peak limiter 0.99; the wave-5 background noise is added afterwards to the mixture by wave 5's rule.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

W9 = REPO / "eval_outputs/mist_w3po_v1/wave9"
ANCHOR_DIR = W9 / "anchor"
PCA_FILE = ANCHOR_DIR / "pca_bins.npz"
N_DIM, N_BIN = 8, 32
VOCAB = 152064
ANCHOR_N = N_DIM * N_BIN
ANCHOR_LO = VOCAB - ANCHOR_N          # 151808
AUDIO_END = 151670
TOK_N = 151676                         # ids 0..151675: everything the tokenizer (incl. added tokens) can produce
HEAD = 16000                           # first 1.0 s at 16 kHz
SR = 16000


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def check_unused(tok) -> dict:
    """the 256 anchor ids are above every id the tokenizer can produce"""
    v = tok.get_vocab()
    top = max(max(v.values()), max(tok.added_tokens_decoder))
    assert top < ANCHOR_LO and len(tok) <= ANCHOR_LO, (top, len(tok), ANCHOR_LO)
    probe = tok.encode("Hello 你好 <|audio_end|><|im_end|> 12345 ¿¡ 😀")
    assert max(probe) < ANCHOR_LO
    return {"tokenizer_max_id": int(top), "len_tokenizer": len(tok), "anchor_lo": ANCHOR_LO, "anchor_hi": VOCAB - 1}


def head_key(audio: np.ndarray) -> str:
    """the anchor depends only on the first 1.0 s: si_thinker_stream's prompt audio (float32) keyed by those bytes"""
    return hashlib.sha1(np.ascontiguousarray(audio[:HEAD], dtype=np.float32).tobytes()).hexdigest()


def insert_after_audio(ids: list[int], anchor: list[int]) -> list[int]:
    k = [i for i, t in enumerate(ids) if t == AUDIO_END]
    assert len(k) == 1, f"{len(k)} <|audio_end|> tokens"
    return ids[: k[0] + 1] + list(anchor) + ids[k[0] + 1:]


class Anchor:
    """u = C(first 1 s) -> 8 anchor token ids (needs PCA_FILE unless pca=False)"""

    def __init__(self, device: str = "cuda:0", pca: bool = True):
        from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
        from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
        from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import CKPT_C
        self.Pc = load_encoder(BiCodecPhi(device), str(CKPT_C))
        self.Pc.model.eval()
        if pca:
            z = np.load(PCA_FILE)
            self.mean, self.comps, self.edges = z["mean"], z["comps"], z["edges"]
            assert self.comps.shape == (N_DIM, 192) and self.edges.shape == (N_DIM, N_BIN - 1)

    def u(self, wavs) -> np.ndarray:
        """(B, 192) flattened pre-FSQ codes of the first 1.0 s of each waveform"""
        heads = [np.asarray(w[:HEAD], dtype=np.float64) for w in wavs]
        return self.Pc.phi_wave(heads).reshape(len(heads), -1).astype(np.float64)

    def ids_from_u(self, U: np.ndarray) -> np.ndarray:
        z = (U - self.mean) @ self.comps.T                                    # (B, 8)
        b = np.stack([np.searchsorted(self.edges[k], z[:, k], side="right") for k in range(N_DIM)], 1)
        return ANCHOR_LO + N_BIN * np.arange(N_DIM)[None, :] + b              # (B, 8)

    def ids(self, wavs) -> np.ndarray:
        return self.ids_from_u(self.u(wavs))


def crosstalk(x: np.ndarray, row: dict, rows: list[dict], s: int, j: int, held: set):
    """the prereg's crosstalk augmentation of target x (float32) -> (mixture, info or None)"""
    from experiments.mist_w3po_v1.voicecond import bench_build as BB
    rng = random.Random(h(f"20261120|w9|xt|{s}|{j}"))
    if rng.random() >= 0.5:
        return x, None
    for _ in range(1000):
        r2 = rows[int(rng.random() * len(rows)) % len(rows)]
        if r2["spk"] != row["spk"]:
            break
    else:
        raise RuntimeError("no interferer with a different speaker")
    assert r2["spk"] not in held and row["spk"] not in held
    n = len(x)
    on = int(round(rng.uniform(1.0, 0.8 * n / SR) * SR))
    sir = rng.uniform(-5.0, 10.0)
    z = sf.read(str(REPO / r2["audio"]), dtype="float32")[0]
    g = 10 ** ((BB.active_rms_db(x) - BB.active_rms_db(z) - sir) / 20)
    seg = z[: n - on]
    y = x.astype(np.float64).copy()
    y[on: on + len(seg)] += g * seg
    y = (y * min(1.0, 0.99 / float(np.abs(y).max()))).astype(np.float32)
    return y, {"int_id": r2["id"], "onset_s": round(on / SR, 3), "sir_db": round(sir, 2), "int_len_s": round(len(seg) / SR, 3)}
