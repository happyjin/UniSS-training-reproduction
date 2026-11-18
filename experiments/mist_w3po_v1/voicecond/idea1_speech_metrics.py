"""Idea 1 (ZNE) generated-speech metrics, paired by stream (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §6).

measure  per stream of a talker run (results.jsonl speech_path): SQUIM objective STOI / PESQ / SI-SDR (first 20 s, as
         game_muffle_analysis.squim), spectral centroid / hf share (game_muffle_analysis.spectral) and the voice-texture
         measures of game_voice_texture.feats (flatness 300-4000 Hz, spectral contrast, harmonic share). CPU pool.
         -> <out jsonl> (refuses an existing file)
    python -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure --talker <run>/talker --out <jsonl> [--procs 48]
compare  paired differences B - A over the common streams (several A:B pairs; one Holm family over the PESQ comparisons
         given with --primary): mean / median diff with bootstrap 95 % CI (10000 resamples, seed 0), Wilcoxon signed-rank
         p, share B > A.
    python -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics compare --pair <A.jsonl>:<B.jsonl>[:label] ...
        [--primary <label> ...] [--out <json>]
Run with conda_envs/uniss-eval/bin/python (torchaudio SQUIM weights in cache/torch/hub/checkpoints).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

KEYS = ("pesq", "sisdr", "stoi", "contrast", "flatness", "harm_share", "centroid", "hf_db")


def one(a):
    sid, path = a
    try:
        from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import read16, spectral, squim
        from experiments.mist_w3po_v1.evaluation.game_voice_texture import feats
        x = read16(path)
        out = {"id": sid, "dur_s": round(len(x) / 16000, 2)}
        q = squim(x) or {}
        sp = spectral(x) or {}
        tx = feats(path) or {}
        out.update(q)
        out.update({k: sp[k] for k in ("centroid", "hf_db") if k in sp})
        out.update(tx)
        return out
    except Exception as e:  # noqa: BLE001
        return {"id": sid, "error": repr(e)[:200]}


def measure(a) -> None:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    T = Path(a.talker)
    files = [T / "results.jsonl"] if (T / "results.jsonl").exists() else sorted(T.glob("part_*.jsonl"))
    rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
    jobs = [(r["id"], r["speech_path"]) for r in rows if r.get("speech_path") and r.get("error") in (None, "None")]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    ok = [r for r in res if "pesq" in r]
    print(json.dumps({"out": str(out), "n": len(res), "ok": len(ok),
                      "pesq_median": float(np.median([r["pesq"] for r in ok])) if ok else None}))


def boot(d: np.ndarray, fn, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    v = fn(d[idx], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired(A: dict, B: dict, key: str) -> dict | None:
    from scipy.stats import wilcoxon
    ids = sorted(i for i in set(A) & set(B) if key in A[i] and key in B[i])
    if len(ids) < 5:
        return None
    x = np.array([A[i][key] for i in ids], dtype=float)
    y = np.array([B[i][key] for i in ids], dtype=float)
    d = y - x
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = 1.0
    return {"n": len(ids), "A_median": float(np.median(x)), "B_median": float(np.median(y)),
            "mean_diff": float(d.mean()), "mean_ci": boot(d, np.mean), "median_diff": float(np.median(d)),
            "median_ci": boot(d, np.median), "wilcoxon_p": p, "B_gt_A": float(np.mean(d > 0))}


def load(p: str) -> dict:
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8")) if "error" not in r}


def compare(a) -> None:
    res = {}
    for spec in a.pair:
        parts = spec.split(":")
        A, B = parts[0], parts[1]
        label = parts[2] if len(parts) > 2 else f"{Path(B).stem}-vs-{Path(A).stem}"
        dA, dB = load(A), load(B)
        res[label] = {"A": A, "B": B, **{k: paired(dA, dB, k) for k in KEYS}}
    if a.primary:   # Holm over the primary PESQ comparisons
        ps = sorted((res[l]["pesq"]["wilcoxon_p"], l) for l in a.primary)
        m, run = len(ps), 0.0
        for k, (p, l) in enumerate(ps):
            run = max(run, min(1.0, (m - k) * p))
            res[l]["pesq"]["holm_p"] = run
    for l, r in res.items():
        q = r["pesq"]
        print(f"{l}: n={q['n']} PESQ {q['A_median']:.3f} -> {q['B_median']:.3f} mean diff {q['mean_diff']:+.3f} "
              f"[{q['mean_ci'][0]:+.3f},{q['mean_ci'][1]:+.3f}] median diff {q['median_diff']:+.3f} "
              f"[{q['median_ci'][0]:+.3f},{q['median_ci'][1]:+.3f}] p={q['wilcoxon_p']:.2g}"
              + (f" holm={q['holm_p']:.2g}" if "holm_p" in q else "")
              + "".join(f" | {k} {r[k]['mean_diff']:+.3f} [{r[k]['mean_ci'][0]:+.3f},{r[k]['mean_ci'][1]:+.3f}]"
                        for k in ("sisdr", "contrast", "flatness") if r.get(k)))
    if a.out:
        p = Path(a.out)
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--talker", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--procs", type=int, default=48)
    c = sub.add_parser("compare")
    c.add_argument("--pair", nargs="+", required=True)
    c.add_argument("--primary", nargs="*", default=[])
    c.add_argument("--out", default="")
    a = ap.parse_args()
    measure(a) if a.cmd == "measure" else compare(a)


if __name__ == "__main__":
    main()
