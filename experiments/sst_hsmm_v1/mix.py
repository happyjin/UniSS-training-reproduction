"""sst_hsmm_v1 noise mixer (reports/sst_hsmm_v1/prereg.zh-CN.md section 1): wave5_m_train.Sampler's mixing with the BVR drawn
from the OPEN dev bench (data/open_voicecond_v1/bench_dev/*/*/meetings.json bvr_db, 60 values) instead of the esports-game
statistic. Clean with probability 0.5, else env uniform over {music, crowd, sfx} (train pools), segment covering the
sentence (looped if short), unit RMS, 0.02 s fades, BVR at a U(0.05, 0.95) quantile, peak limit 0.99.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
POOL = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"}
ENVS = ("music", "crowd", "sfx")


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def dev_bvr_values() -> np.ndarray:
    v = []
    for f in sorted(glob.glob(str(REPO / "data/open_voicecond_v1/bench_dev/*/*/meetings.json"))):
        m = json.load(open(f))
        items = m.values() if isinstance(m, dict) else m
        v += [x["bvr_db"] for x in items if isinstance(x, dict) and "bvr_db" in x]
    assert len(v) == 60, len(v)
    return np.array(sorted(v))


class Mixer:
    def __init__(self):
        from experiments.mist_w3po_v1.voicecond import wave1_data as W1D
        self.pools = W1D.bg_pools("train")
        self.bv = dev_bvr_values()

    def mix(self, x: np.ndarray, key: str, p_clean: float = 0.5):
        """-> (wav, env or None, bvr or None); deterministic in key"""
        from experiments.mist_w3po_v1.voicecond import bench_build as BB
        rng = random.Random(h(key))
        if rng.random() < p_clean:
            return x, None, None
        env = ENVS[int(rng.random() * 3) % 3]
        bvr = float(np.quantile(self.bv, rng.uniform(0.05, 0.95)))
        P = self.pools[POOL[env]]
        f = P[int(rng.random() * len(P)) % len(P)]
        b = BB.load(f["file"])
        n = len(x)
        if len(b) < n:
            b = np.tile(b, 1 + n // max(1, len(b)))
        off = int(rng.random() * max(0, len(b) - n))
        seg = BB.fade(BB.unit_rms(b[off: off + n]), 0.02, 0.02)
        es, eb = float((x.astype(np.float64) ** 2).sum()), float((seg.astype(np.float64) ** 2).sum())
        alpha = math.sqrt(10 ** (bvr / 10) * es / max(eb, 1e-12))
        y = x + alpha * seg
        return (y * min(1.0, 0.99 / float(np.abs(y).max()))).astype(np.float32), env, bvr
