"""Robust wave 5, main line M (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md): the training sentences.

CoVoST 2 TRAIN (data/open_voicecond_v1/idea2/covost_train/.../train-*.parquet), both directions, speakers listed in
bench/HELDOUT_SPEECH.txt excluded, empty source/translation dropped. zh->en: every surviving row; en->zh: the first
N_EN rows ranked by sha1("20261119|w5m|en2zh|<id>"). Each clip goes through bench_build.trim_norm (edge trim,
-26 dBFS active RMS, 1.5-12 s) and is written as PCM16 16 kHz; rows that fail trim_norm are dropped.
Output (refuses to overwrite): eval_outputs/mist_w3po_v1/wave5/M/data/{rows_<dir>.jsonl, clips/<dir>/<id>.wav}.
    <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/wave5_m_data.py --procs 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as BB  # noqa: E402
from experiments.mist_w3po_v1.voicecond import wave1_data as W1D  # noqa: E402

OUT = REPO / "eval_outputs/mist_w3po_v1/wave5/M/data"
N_EN = 120000
SR = 16000


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _job(a):
    r, d = a
    try:
        x = W1D.decode_bytes(r["audio"]["bytes"])
    except Exception:          # a few CoVoST clips are not decodable by libsndfile: dropped like trim_norm failures
        return None
    t = BB.trim_norm(x)
    if t is None:
        return None
    y, _ = t
    p = OUT / "clips" / d / f"{r['id']}.wav"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    sf.write(str(p), y, SR, subtype="PCM_16")
    return {"id": r["id"], "dir": d, "spk": r["client_id"], "src": r["sentence"].strip(),
            "ref": r["translation"].strip(), "dur": round(len(y) / SR, 3), "audio": str(p.relative_to(REPO))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--dirs", default="zh2en,en2zh")
    a = ap.parse_args()
    held = W1D.held_speakers()
    for d in a.dirs.split(","):
        man = OUT / f"rows_{d}.jsonl"
        if man.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {man}")
        (OUT / "clips" / d).mkdir(parents=True, exist_ok=True)
        rows = [r for r in W1D.rows_of(d, "train") if r["client_id"] not in held
                and r["sentence"].strip() and r["translation"].strip()]
        n_all = len(rows)
        if d == "en2zh":
            rows.sort(key=lambda r: h(f"20261119|w5m|en2zh|{r['id']}"))
            rows = rows[:N_EN]
        with mp.get_context("fork").Pool(a.procs) as P:
            res = P.map(_job, [(r, d) for r in rows], chunksize=16)
        keep = [x for x in res if x]
        assert not {x["spk"] for x in keep} & held
        with open(man, "x", encoding="utf-8") as fh:
            for x in keep:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{d}: {len(keep)} kept of {len(rows)} ({n_all} train rows after the held-out filter), "
              f"{sum(x['dur'] for x in keep) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
