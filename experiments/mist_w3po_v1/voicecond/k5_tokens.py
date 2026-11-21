"""Robust K5 soft-voice tables (reports/mist_w3po_v1/robust_k5_softvoice_prereg.zh-CN.md sections 3-4): for every row of the
wave3 V1 table (wave3_combos/V/tokens/tokens_v1_<dir>.jsonl) the same prompt file (sha1 checked) is encoded by the V1
encoder exactly as V1 did, but the BOUNDED code b = FSQ.bound(phi_V1(y)) (32x6) is kept instead of round(b).
Row: {key, id, sha1, ids (= the V1 table's hard ids, unchanged), b}. Gate: the continuous decoder path on the hard
tokens' level centres must equal speaker_encoder.detokenize(ids) (max abs diff <= 1e-5) on every row, else exit 1.
Record only: rows where Q(b) != V1 ids; share of dims with |b - round(b)| > 0.25; median cosine of d(b) and of d(V1 ids)
to d(oracle ids) (oracle_voice table: noisy rows = clean-twin tokens, clean rows = B tokens), split by cond.
    vllm_omni_030 python -I experiments/mist_w3po_v1/voicecond/k5_tokens.py --out <dir> [--device cuda:0] [--limit N]
Writes <out>/tokens_k5_<dir>.jsonl and .stats.json for dir in en2zh, zh2en; refuses existing files.
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
ORT = REPO / "eval_outputs/mist_w3po_v1/oracle_voice/tokens"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows per direction")
    a = ap.parse_args()
    out = Path(a.out)
    files = {d: out / f"tokens_k5_{d}.jsonl" for d in ("en2zh", "zh2en")}
    for f in [*files.values(), *[f.with_suffix(".stats.json") for f in files.values()]]:
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_grad_enabled(False)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    from experiments.mist_w3po_v1.voicecond.k5_soft import centres, corners, d_cont
    P = load_encoder(BiCodecPhi(a.device), str(V1))
    P.model.eval()
    se = P.se

    def dv(ids):
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=P.device)[None, None]
        return se.detokenize(t)[0]

    def cos(x, y):
        return float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))

    for d, f in files.items():
        src = [json.loads(l) for l in open(VT / f"tokens_v1_{d}.jsonl", encoding="utf-8")]
        src = src[: a.limit] if a.limit else src
        orc = {r["id"]: r["ids"] for r in map(json.loads, open(ORT / f"tokens_oracle_{d}.jsonl", encoding="utf-8"))}
        rows, mism, maxdiff, near, cs = [], 0, 0.0, [], {"noisy": [], "clean": []}
        for n, r in enumerate(src, 1):
            p = VT / "prompts" / d / f"{r['key']}.wav"
            assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
            u = P.phi_wave(read_mono64(p))
            b = P.bounded(u).astype(np.float64)
            mism += [int(v) for v in P.tokens(u)] != r["ids"]
            near.append(float((np.abs(b - np.round(b)) > 0.25).mean()))
            dh = dv(r["ids"])
            maxdiff = max(maxdiff, float((d_cont(se, centres(r["ids"]))[0] - dh).abs().max()))
            idx, w = corners(centres(r["ids"]))
            assert (w.max(1) == 1.0).all() and (idx[np.arange(32), w.argmax(1)] == np.asarray(r["ids"])).all(), r["key"]
            cond = "noisy" if "_noisy__" in r["key"] else "clean"
            if r["id"] in orc:
                do = dv(orc[r["id"]])
                cs[cond].append((cos(d_cont(se, b)[0], do), cos(dh, do)))
            rows.append({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": r["ids"],
                         "b": np.round(b, 6).tolist()})
            if n % 500 == 0:
                print(f"[k5-tokens] {d} {n}/{len(src)}", flush=True)
        st = {"dir": d, "n": len(rows), "q_b_ne_v1_rows": mism, "dec_centre_max_abs_diff": maxdiff,
              "near_boundary_dim_share_mean": float(np.mean(near)),
              "dvec_cos_to_oracle_median": {c: {"n": len(v), "soft_b": float(np.median([x[0] for x in v])) if v else None,
                                                "hard_v1": float(np.median([x[1] for x in v])) if v else None}
                                            for c, v in cs.items()}}
        print(json.dumps(st), flush=True)
        if maxdiff > 1e-5:
            sys.exit(f"continuous decoder path does not reproduce detokenize on level centres: {maxdiff} ({d})")
        if mism > 0.05 * len(src):
            sys.exit(f"V1 ids not reproduced on {mism}/{len(src)} rows ({d})")
        with open(f, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        with open(f.with_suffix(".stats.json"), "x", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1)


if __name__ == "__main__":
    main()
