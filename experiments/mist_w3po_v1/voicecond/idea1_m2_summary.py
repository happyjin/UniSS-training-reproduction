"""Idea 1 M2 demo summary and gate G2a (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v3.4): for each arm
<tag> (runs game_b_<dir>__t1c_<tag>{,_tts2}) vs baseline B (game_b_<dir>__t1c{,_tts2}), on all streams of the 10 demo
excerpts, paired by (seed, stream):
  pooled (both directions, both seeds) median PESQ gain and its bootstrap 95 % CI (10000, seed 0), contrast and flatness
  median gains; per direction the same, blind ASR-BLEU per seed (game_blind_bleu.py, written to <out dir> if missing)
  and the two-seed means, speech LAAL (crf/t_eval/<name>/lat_*.json unit.speech_laal_online.all.mean) two-seed means.
G2a (per arm): pooled PESQ median gain >= 0.5 AND contrast median gain >= 1.5 dB AND, in each direction, the two-seed
blind ASR-BLEU mean >= baseline B's. Metrics files: voicecond_idea1/metrics/<run>.jsonl (idea1_speech_metrics measure).
    conda_envs/qwen3omni/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_m2_summary --arms e2e ctrl vocals zne_Rt
        --out <json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EO = REPO / "eval_outputs/mist_w3po_v1"
MET = EO / "voicecond_idea1/metrics"
BB = EO / "voicecond_idea1/e12b"


def load(run):
    p = MET / f"{run}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8")) if "pesq" in r}


def blind(d, run):
    out = BB / f"blind_bleu_{run}.json"
    if not out.exists():
        hyp = EO / "decode" / run / "hyp_asr.jsonl"
        if run in (f"game_b_{d}__t1c", f"game_b_{d}__t1c_tts2"):
            return json.load(open(EO / "decode" / run / "blind_bleu_asr.json"))["corpus"]
        subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.game_blind_bleu", "--dir", d, "--hyp",
                        str(hyp), "--out", str(out)], check=True, cwd=REPO, capture_output=True)
    return json.load(open(out))["corpus"]


def laal(run, base):
    p = EO / "crf/t_eval" / base / f"lat_{run}.json"
    return json.load(open(p))["unit"]["speech_laal_online"]["all"]["mean"]


def med_ci(d):
    rng = np.random.default_rng(0)
    bs = np.median(d[rng.integers(0, len(d), (10000, len(d)))], 1)
    return float(np.median(d)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    for arm in a.arms:
        R = {"pooled": {}, "dirs": {}}
        pool = {k: [] for k in ("pesq", "contrast", "flatness")}
        bleu_ok = True
        for d in ("zh2en", "en2zh"):
            D = {k: [] for k in pool}
            bl_b, bl_z, la_b, la_z = [], [], [], []
            for s in ("", "_tts2"):
                base, run = f"game_b_{d}__t1c{s}", f"game_b_{d}__t1c_{arm}{s}"
                A, B = load(base), load(run)
                for k in D:
                    D[k] += [B[i][k] - A[i][k] for i in A if i in B and k in A[i] and k in B[i]]
                bl_b.append(blind(d, base))
                bl_z.append(blind(d, run))
                la_b.append(laal(base, f"game_b_{d}__t1c"))
                la_z.append(laal(run, f"game_b_{d}__t1c_{arm}"))
            dr = {}
            for k, v in D.items():
                v = np.array(v)
                pool[k] += list(v)
                m, ci = med_ci(v)
                dr[k] = {"median_gain": m, "ci": ci, "n": int(len(v))}
            dr["blind_bleu"] = {"base": bl_b, "arm": bl_z, "mean_diff": float(np.mean(bl_z) - np.mean(bl_b))}
            dr["laal"] = {"base": la_b, "arm": la_z, "mean_diff": float(np.mean(la_z) - np.mean(la_b))}
            bleu_ok &= np.mean(bl_z) >= np.mean(bl_b)
            R["dirs"][d] = dr
        for k, v in pool.items():
            m, ci = med_ci(np.array(v))
            R["pooled"][k] = {"median_gain": m, "ci": ci, "n": len(v)}
        R["G2a"] = {"pesq_ok": R["pooled"]["pesq"]["median_gain"] >= 0.5,
                    "contrast_ok": R["pooled"]["contrast"]["median_gain"] >= 1.5, "bleu_ok": bool(bleu_ok)}
        R["G2a"]["pass"] = all(R["G2a"].values())
        res[arm] = R
        print(f"{arm}: PESQ {R['pooled']['pesq']['median_gain']:+.3f} {R['pooled']['pesq']['ci']} contrast "
              f"{R['pooled']['contrast']['median_gain']:+.2f} flat {R['pooled']['flatness']['median_gain']:+.4f} | "
              + " | ".join(f"{d} BLEU {R['dirs'][d]['blind_bleu']['arm']} vs {R['dirs'][d]['blind_bleu']['base']} "
                           f"LAAL {R['dirs'][d]['laal']['mean_diff']:+.3f} PESQ {R['dirs'][d]['pesq']['median_gain']:+.3f}"
                           for d in ("zh2en", "en2zh")) + f" | G2a {R['G2a']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
