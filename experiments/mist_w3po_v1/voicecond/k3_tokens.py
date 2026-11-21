"""Robust K3 voice-token tables (reports/mist_w3po_v1/robust_k3_discrim_prereg.zh-CN.md section 5): for every row of the
wave3 V1 table (wave3_combos/V/tokens/tokens_v1_<dir>.jsonl) the same prompt file (wave3_combos/V/tokens/prompts/<dir>/
<key>.wav, sha1 checked) is encoded by the K3 encoder exactly as V1 did (ids = Q(phi(read_mono64(prompt)))), in the
table format of idea1_si_infer_vo.py. As a check the V1 encoder is re-run on the same prompts and must reproduce the
V1 ids (mismatch count reported, and the run fails if > 5 % of rows differ: single FSQ boundary flips across devices are expected).
    qwen3omni python -I experiments/mist_w3po_v1/voicecond/k3_tokens.py --k3 <encoder.pt> --out <dir> [--device cuda:0]
Writes <out>/tokens_k3_<dir>.jsonl for dir in zh2en, en2zh; refuses existing files.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows per direction")
    a = ap.parse_args()
    out = Path(a.out)
    files = {d: out / f"tokens_k3_{d}.jsonl" for d in ("zh2en", "en2zh")}
    for f in files.values():
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    out.mkdir(parents=True, exist_ok=True)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    Pv = load_encoder(BiCodecPhi(a.device), str(V1))
    Pk = load_encoder(BiCodecPhi(a.device), a.k3)
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
                print(f"[k3-tokens] {d} {n}/{len(src)}", flush=True)
        if mism > 0.05 * len(src):
            sys.exit(f"V1 ids not reproduced on {mism}/{len(src)} rows ({d})")
        with open(f, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(json.dumps({"dir": d, "n": len(rows), "v1_mismatch": mism, "ham_k3_v1_mean": float(np.mean(ham))}), flush=True)


if __name__ == "__main__":
    main()
