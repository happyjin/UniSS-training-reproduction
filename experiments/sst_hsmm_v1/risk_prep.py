"""sst_hsmm_v1 P4 data prep (reports/sst_hsmm_v1/prereg.zh-CN.md change 7). Sample table for steps 0..STEPS-1, slots
0..15: direction ("en2zh", "zh2en")[j % 2], row = sha1("20261123|risk|row|s|j") mod n_rows(dir), audio mixed with key
"20261123|risk|mix|s|j" (mix.Mixer, p_clean 0.5) and written to <out>/audio/<dir>/s<s>_j<j>.wav (16 kHz, PCM16).
Writes <out>/samples.jsonl (s, j, dir, row id, wav) and the rollout items <out>/items_<dir>_<k>.jsonl (k < shards,
contiguous) in e2e_si_thinker's format. Refuses existing outputs.
    qwen3omni OMP_NUM_THREADS=1 PYTHONPATH=. python -m experiments.sst_hsmm_v1.risk_prep --out eval_outputs/sst_hsmm_v1/p4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
SEED, SLOTS = 20261123, 16
_M = {}


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _init():
    from experiments.sst_hsmm_v1.mix import Mixer
    _M["m"] = Mixer()


def _one(job):
    s, j, d, r, wav_p = job
    x = sf.read(r["audio"], dtype="float32")[0]
    if x.ndim > 1:
        x = x.mean(axis=1)
    y, env, _ = _M["m"].mix(x, f"{SEED}|risk|mix|{s}|{j}", p_clean=0.5)
    y = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    sf.write(wav_p, y, 16000, subtype="PCM_16")
    return {"s": s, "j": j, "dir": d, "row": r["id"], "wav": wav_p, "dur_ms": len(y) // 16, "env": env}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--shards", type=int, default=4, help="rollout shards per direction")
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "samples.jsonl").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}/samples.jsonl")
    rows = {d: [json.loads(l) for l in open(DATA / f"train_{d}.jsonl", encoding="utf-8")] for d in ("en2zh", "zh2en")}
    jobs = []
    for d in ("en2zh", "zh2en"):
        (out / "audio" / d).mkdir(parents=True, exist_ok=True)
    for s in range(a.steps):
        for j in range(SLOTS):
            d = ("en2zh", "zh2en")[j % 2]
            r = rows[d][h(f"{SEED}|risk|row|{s}|{j}") % len(rows[d])]
            jobs.append((s, j, d, r, str(out / "audio" / d / f"s{s:04d}_j{j:02d}.wav")))
    with mp.get_context("fork").Pool(a.procs, _init) as pool:
        res = pool.map(_one, jobs, chunksize=16)
    refs = {d: {r["id"]: r["ref"] for r in rows[d]} for d in rows}
    with open(out / "samples.jsonl", "x", encoding="utf-8") as fh:
        for x in res:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    for d in ("en2zh", "zh2en"):
        it = [x for x in res if x["dir"] == d]
        b = [round(k * len(it) / a.shards) for k in range(a.shards + 1)]
        for k in range(a.shards):
            with open(out / f"items_{d}_{k}.jsonl", "x", encoding="utf-8") as fh:
                for x in it[b[k]: b[k + 1]]:
                    fh.write(json.dumps({"id": f"s{x['s']:04d}_j{x['j']:02d}", "audio": x["wav"], "dur_ms": x["dur_ms"],
                                         "translation": refs[d][x["row"]]}, ensure_ascii=False) + "\n")
    print(json.dumps({d: sum(1 for x in res if x["dir"] == d) for d in ("en2zh", "zh2en")}), flush=True)


if __name__ == "__main__":
    main()
