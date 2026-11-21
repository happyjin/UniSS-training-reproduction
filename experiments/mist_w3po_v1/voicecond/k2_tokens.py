"""K2 W2 arm voice-token table (reports/mist_w3po_v1/robust_k2_w2map_prereg.zh-CN.md, step 2): for every row of the V1
table wave3_combos/V/tokens/tokens_v1_<d>.jsonl, the same prompt (wave3_combos/V/tokens/prompts/<d>/<key>.wav, sha1
checked), u = V1 encoder code (as wave3_v_tokens.py V1), w = T_lang(u) (k2_fit_w2.py map of the SOURCE language),
ids = FSQ(w) exactly as V1 quantizes u. Output rows in the idea1_si_infer_vo.py table format. Refuses an existing --out.
    CUDA_VISIBLE_DEVICES= vllm_omni_030/bin/python -I experiments/mist_w3po_v1/voicecond/k2_tokens.py --dir en2zh \
        --map <k2>/fit/w2_map.npz --out <k2>/tokens/tokens_w2_en2zh.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import CKPT_C  # noqa: E402

V = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=("en2zh", "zh2en"))
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    lg = a.dir[:2]
    M = np.load(a.map)
    mv, mc, A = M[f"mu_v_{lg}"], M[f"mu_c_{lg}"], M[f"A_{lg}"]
    Pc = load_encoder(BiCodecPhi(a.device), str(CKPT_C))
    Pc.model.eval()
    rows, same_v1, ham = [], 0, []
    for r in map(json.loads, open(V / "tokens" / f"tokens_v1_{a.dir}.jsonl", encoding="utf-8")):
        p = V / "tokens" / "prompts" / a.dir / f"{r['key']}.wav"
        sha = hashlib.sha1(p.read_bytes()).hexdigest()
        assert sha == r["sha1"], (p, sha, r["sha1"])
        u = Pc.phi_wave(read_mono64(p)).reshape(192).astype(np.float64)
        same_v1 += int([int(v) for v in Pc.tokens(u.reshape(32, 6).astype(np.float32))] == r["ids"])
        w = (mc + A @ (u - mv)).reshape(32, 6).astype(np.float32)
        ids = [int(v) for v in Pc.tokens(w)]
        ham.append(sum(x != z for x, z in zip(ids, r["ids"])))
        rows.append({"key": r["key"], "id": r["id"], "sha1": sha, "ids": ids, "ham_vs_v1": ham[-1]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)
    print(json.dumps({"dir": a.dir, "n": len(rows), "v1_ids_reproduced": same_v1,
                      "ham_vs_v1_median": float(np.median(ham)), "ham_vs_v1_mean": float(np.mean(ham))}), flush=True)


if __name__ == "__main__":
    main()
