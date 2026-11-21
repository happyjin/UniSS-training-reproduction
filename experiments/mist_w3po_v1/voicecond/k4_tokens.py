"""Robust K4 voice-token tables (reports/mist_w3po_v1/robust_k4_tokencls_prereg.zh-CN.md section 4; copy of k3_tokens.py): for every row of the
wave3 V1 table (wave3_combos/V/tokens/tokens_v1_<dir>.jsonl) the same prompt file (wave3_combos/V/tokens/prompts/<dir>/
<key>.wav, sha1 checked) is encoded by the K4 encoder exactly as V1 did (ids = Q(phi(read_mono64(prompt)))), in the
table format of idea1_si_infer_vo.py. As a check the V1 encoder is re-run on the same prompts and must reproduce the
V1 ids (mismatch count reported, and the run fails if > 5 % of rows differ: single FSQ boundary flips across devices are expected).
Record only (prereg section 3): position / FSQ-level match of the V1, K3 and K4 ids with the oracle table
(oracle_voice/tokens/tokens_oracle_<dir>.jsonl: noisy rows = clean-twin tokens, clean rows = B tokens), split by cond.
    vllm_omni_030 python -I experiments/mist_w3po_v1/voicecond/k4_tokens.py --k4 <encoder.pt> --out <dir> [--device cuda:0]
Writes <out>/tokens_k4_<dir>.jsonl and <out>/tokens_k4_<dir>.stats.json for dir in zh2en, en2zh; refuses existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
VT = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/tokens"
V1 = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
K3T = REPO / "eval_outputs/mist_w3po_v1/k3/tokens"
ORT = REPO / "eval_outputs/mist_w3po_v1/oracle_voice/tokens"


def digits(ids):
    ids = np.asarray(ids)
    return np.stack([(ids // 4 ** k) % 4 for k in range(6)], -1)


def match_stats(rows, ref):
    """rows/ref: {id: ids}; position and FSQ-level match per cond (id prefix a_/b_ + cond)"""
    out = {}
    for cond in ("noisy", "clean"):
        ks = [k for k in rows if f"_{cond}__" in k and k in ref]
        A, B = np.array([rows[k] for k in ks]), np.array([ref[k] for k in ks])
        out[cond] = {"n": len(ks), "pos": round(float((A == B).mean()), 4),
                     "level": round(float((digits(A) == digits(B)).mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows per direction")
    a = ap.parse_args()
    out = Path(a.out)
    files = {d: out / f"tokens_k4_{d}.jsonl" for d in ("zh2en", "en2zh")}
    for f in [*files.values(), *[f.with_suffix(".stats.json") for f in files.values()]]:
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    out.mkdir(parents=True, exist_ok=True)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    Pv = load_encoder(BiCodecPhi(a.device), str(V1))
    Pk = load_encoder(BiCodecPhi(a.device), a.k4)
    Pv.model.eval()
    Pk.model.eval()
    for d, f in files.items():
        rows, mism, ham = [], 0, []
        src = [json.loads(l) for l in open(VT / f"tokens_v1_{d}.jsonl", encoding="utf-8")]
        src = src[: a.limit] if a.limit else src
        for n, r in enumerate(src, 1):
            p = VT / "prompts" / d / f"{r['key']}.wav"
            assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
            y = read_mono64(p)
            iv = [int(v) for v in Pv.tokens(Pv.phi_wave(y))]
            ik = [int(v) for v in Pk.tokens(Pk.phi_wave(y))]
            mism += iv != r["ids"]
            ham.append(sum(x != z for x, z in zip(ik, r["ids"])))
            rows.append({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": ik})
            if n % 500 == 0:
                print(f"[k4-tokens] {d} {n}/{len(src)}", flush=True)
        if mism > 0.05 * len(src):
            sys.exit(f"V1 ids not reproduced on {mism}/{len(src)} rows ({d})")
        with open(f, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        orc = {r["id"]: r["ids"] for r in map(json.loads, open(ORT / f"tokens_oracle_{d}.jsonl", encoding="utf-8"))}
        arms = {"v1": {r["id"]: r["ids"] for r in src},
                "k3": {r["id"]: r["ids"] for r in map(json.loads, open(K3T / f"tokens_k3_{d}.jsonl", encoding="utf-8"))},
                "k4": {r["id"]: r["ids"] for r in rows}}
        st = {"dir": d, "n": len(rows), "v1_mismatch": mism, "ham_k4_v1_mean": float(np.mean(ham)),
              "vs_oracle": {k: match_stats(v, orc) for k, v in arms.items()}}
        with open(f.with_suffix(".stats.json"), "x", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1)
        print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
