"""Decoder run-to-run noise (review of the v6 draft, coordinator note of 2026-11-01). The MoE Thinker's greedy bf16 decoding
depends on the batch composition (T13: the same 96 Zh->En windows at batch 16 and at batch 4). Per window we take both runs'
cpBLEU statistics (session matching of each run) and resample N windows with replacement: the spread of
BLEU(run A) - BLEU(run B) over the resamples estimates the difference two re-decodes of the same system would show on a test
of N windows. Writes generated/decoder_noise.json: {N: sd} for the test sizes of the paper.

    python scripts/decoder_noise.py      (from the paper folder)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_v6 import score, smap_of, stats, window_data  # noqa: E402
from systems_v6 import V6  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
win = V6 / "t13_zh2en_b16/windows96.jsonl"
A = window_data("zh2en", V6 / "t13_zh2en_b16", win)
B = window_data("zh2en", V6 / "t13_zh2en_b4", win)
ids = sorted(set(A) & set(B))
Xa, Xb = stats("zh2en", A, ids, smap_of(A, ids)), stats("zh2en", B, ids, smap_of(B, ids))
rng = np.random.default_rng(20261101)
out = {"windows_t13": len(ids), "diff_t13": round(score("zh2en", Xa.sum(0)) - score("zh2en", Xb.sum(0)), 3)}
for n in (950, 1378, 1410, 1439, 2375, 3317, 4243):
    d = []
    for _ in range(2000):
        j = rng.integers(0, len(ids), n)
        d.append(score("zh2en", Xa[j].sum(0)) - score("zh2en", Xb[j].sum(0)))
    out[str(n)] = round(float(np.std(d)), 3)
(GEN / "decoder_noise.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
