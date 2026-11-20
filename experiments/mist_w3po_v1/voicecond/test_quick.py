"""Quick point estimates on the robust test (no bootstrap; user 2026-11-20 "先不做bootstrap，先快速验证"): per system x
direction x {realsi/clean, realsi/noisy, obsi/clean, obsi/id2, obsi/ood1} (OOD-3 dropped, prereg amendment 1; OBSI en2zh
without pair e031_0004): ASR-BLEU, Thinker text BLEU, mean D1 / D2 (s), median PESQ, median speaker similarity.

    qwen3omni PYTHONPATH=. python -m experiments.mist_w3po_v1.voicecond.test_quick B:b EW:v1 C:v1 [UE:v1 ...]
"""
from __future__ import annotations

import sys

import numpy as np

from experiments.mist_w3po_v1.voicecond.combo_score import LANG, jl, laal_defs, load
from experiments.mist_w3po_v1.voicecond.test_score import EXCL, T, sysd
from experiments.mist_w3po_v1.voicecond.wave2_s1_score import corpus_bleu

CELLS = ("realsi/clean", "realsi/noisy", "obsi/clean", "obsi/id2", "obsi/ood1")


def main() -> None:
    systems = [x.split(":") for x in sys.argv[1:]]
    LD = laal_defs()
    for d in ("en2zh", "zh2en"):
        meta = {m["id"]: m for m in jl(T / "data" / f"meta_{d}.jsonl")}
        ref = {r["id"]: r["translation"] for r in jl(T / "data" / f"items_{d}.jsonl")}
        P = {f"{a}_{r}": load(sysd(a, r), d, LD) for a, r in systems}
        for cell in CELLS:
            ids = sorted(i for i, m in meta.items() if f"{m['set']}/{m['cond']}" == cell
                         and not (m["set"] == "obsi" and m["pair"] in EXCL[d]))
            row = []
            for s, p in P.items():
                asr = corpus_bleu([p[i]["asr"] for i in ids], [ref[i] for i in ids], LANG[d])
                txt = corpus_bleu([p[i]["text"] or "" for i in ids], [ref[i] for i in ids], LANG[d])
                f = lambda k: np.nanmedian([p[i][k] if p[i][k] is not None else np.nan for i in ids])
                d1 = np.nanmean([p[i]["d1"] if p[i]["d1"] is not None else np.nan for i in ids])
                row.append(f"{s} asr {asr:5.2f} txt {txt:5.2f} D1 {d1:4.2f} pesq {f('pesq'):4.2f}")
            print(f"{d} {cell:13s} n={len(ids):3d} | " + " | ".join(row))


if __name__ == "__main__":
    main()
