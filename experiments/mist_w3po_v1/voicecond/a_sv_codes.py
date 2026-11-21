"""Experiment A step 2 (vllm_omni_030): BiCodec pre-bound codes (32x6) for the SV-anchor MLP.
  train / mon : v1 = phi_V1(y crop) (V1 encoder, phi_wave as V1's training), u0 = phi_0(s crop) (frozen tokenizer)
  dev_d       : v1 = phi_V1(read_mono64(prompt)) exactly as k4_tokens / V1's table (Q(v1) must reproduce the V1 ids on
                >= 95 % of rows), u0 = phi_0 of the clean-twin prompt (noisy rows; oracle_voice twin_prompts) or of the
                row's own prompt (clean rows) via the tokenizer's file path (phi_path; Q(u0) must reproduce the oracle
                table ids on >= 95 % of rows)
    venvs/vllm_omni_030/bin/python -m experiments.mist_w3po_v1.voicecond.a_sv_codes --out <dir> [--device cuda:0]
Writes codes_{train,mon,dev_<d>}.npz (arrays v1, u0); refuses existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import a_sv_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    names = ["codes_train.npz", "codes_mon.npz"] + [f"codes_dev_{d}.npz" for d in C.DIRS]
    for n in names:
        if (out / n).exists():
            sys.exit(f"拒绝覆盖已有输出: {out / n}")
    out.mkdir(parents=True, exist_ok=True)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    P0 = BiCodecPhi(a.device)
    Pv = load_encoder(BiCodecPhi(a.device), str(C.V1))
    P0.model.eval()
    Pv.model.eval()
    stats = {}

    def batched(itv, ity, name, bs=128):
        t0, V, U, by, bs_ = time.time(), [], [], [], []
        for (j, y), (j2, s) in zip(itv, ity):
            assert j == j2
            if a.limit and j >= a.limit:
                break
            by.append(y)
            bs_.append(s)
            if len(by) == bs:
                V.append(Pv.phi_wave(by)); U.append(P0.phi_wave(bs_)); by, bs_ = [], []
            if (j + 1) % 5000 == 0:
                print(f"[codes] {name} {j + 1} {time.time() - t0:.0f}s", flush=True)
        if by:
            V.append(Pv.phi_wave(by)); U.append(P0.phi_wave(bs_))
        V, U = np.concatenate(V).astype(np.float32), np.concatenate(U).astype(np.float32)
        with open(out / name, "xb") as fh:
            np.savez(fh, v1=V, u0=U)
        print(f"[codes] {name} {V.shape} {time.time() - t0:.0f}s", flush=True)

    batched(C.mon_iter("y"), C.mon_iter("s"), "codes_mon.npz")
    for d in C.DIRS:
        rows = C.dev_rows(d)
        orc = {r["id"]: r["ids"] for r in map(json.loads, open(C.ORT / f"tokens_oracle_{d}.jsonl", encoding="utf-8"))}
        V, U, mv, mo = [], [], 0, 0
        for n, r in enumerate(rows):
            if a.limit and n >= a.limit:
                break
            p = C.dev_prompt(d, r["key"])
            assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
            v = Pv.phi_wave(read_mono64(p))
            src = C.twin_prompt(d, r["key"]) if "_noisy__" in r["id"] else p
            u = P0.phi_path(str(src))
            mv += [int(x) for x in Pv.tokens(v)] != r["ids"]
            mo += [int(x) for x in P0.tokens(u)] != orc[r["id"]]
            V.append(v); U.append(u)
        n = len(V)
        stats[d] = {"n": n, "v1_id_mismatch_rows": mv, "oracle_id_mismatch_rows": mo}
        print(json.dumps(stats[d]), flush=True)
        if mv > 0.05 * n or mo > 0.05 * n:
            sys.exit(f"V1 / oracle ids not reproduced ({d}): {stats[d]}")
        with open(out / f"codes_dev_{d}.npz", "xb") as fh:
            np.savez(fh, v1=np.stack(V).astype(np.float32), u0=np.stack(U).astype(np.float32),
                     ids_v1=np.array([r["ids"] for r in rows[:n]]), ids_oracle=np.array([orc[r["id"]] for r in rows[:n]]))
    batched(C.train_iter("y"), C.train_iter("s"), "codes_train.npz")
    with open(out / "codes_stats.json", "x", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
