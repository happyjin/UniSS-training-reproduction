"""Robust wave 6, gate 1 (duration variability across voice conditions): token tables for the NOISY CoVoST items (set a)
in idea1_si_infer_vo.py format, all keyed to the same noisy prompt file (sha1 of wave3's V tokens/prompts, the bytes si_infer
writes), so the four runs differ only in the 32 voice ids:
  raw    frozen BiCodec on the noisy prompt (what si_infer itself computes; B's condition)
  c      Idea 1 C on the noisy prompt (= wave3 V1 table rows)
  v2     wave3 V2 (Idea 2 frozen posterior) table rows
  cclean Idea 1 C on the clean twin cut to the same samples (the twins are sample-aligned)
Also writes items_a_noisy_<dir>.jsonl (the 300 noisy CoVoST items of S1's items file, in order).

    qwen3omni python experiments/mist_w3po_v1/voicecond/wave6_durtables.py --out <dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
V = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/tokens"
CENC = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    P0 = BiCodecPhi(a.device)
    PC = load_encoder(BiCodecPhi(a.device), str(CENC))
    for d in ("en2zh", "zh2en"):
        items = [r for r in jl(S1 / f"data/items_{d}.jsonl") if r["id"].startswith("a_noisy__")]
        byid = {r["id"]: r for r in jl(S1 / f"data/items_{d}.jsonl")}
        v1 = {r["id"]: r for r in jl(V / f"tokens_v1_{d}.jsonl")}
        v2 = {r["id"]: r for r in jl(V / f"tokens_v2_{d}.jsonl")}
        with open(out / f"items_a_noisy_{d}.jsonl", "x", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        T = {k: [] for k in ("raw", "c", "v2", "cclean")}
        for r in items:
            i = r["id"]
            pr = V / "prompts" / d / f"{i}_voice.wav"
            sha = hashlib.sha1(pr.read_bytes()).hexdigest()
            assert sha == v1[i]["sha1"] == v2[i]["sha1"], i
            y, sr = sf.read(str(pr), dtype="float32")
            assert sr == 16000
            clean_id = "a_clean__" + i[len("a_noisy__"):]
            c, sr2 = sf.read(byid[clean_id]["audio"], dtype="float32")
            assert sr2 == 16000 and len(c) >= len(y), clean_id
            base = {"key": v1[i]["key"], "id": i, "sha1": sha}
            T["raw"].append({**base, "ids": [int(v) for v in P0.tokens(P0.phi_path(str(pr)))]})
            T["c"].append({**base, "ids": [int(v) for v in v1[i]["ids"]]})
            T["v2"].append({**base, "ids": [int(v) for v in v2[i]["ids"]]})
            T["cclean"].append({**base, "ids": [int(v) for v in PC.tokens(PC.phi_wave(c[: len(y)].astype(np.float64)))]})
        for k, rows in T.items():
            with open(out / f"tokens_{k}_{d}.jsonl", "x", encoding="utf-8") as fh:
                for x in rows:
                    fh.write(json.dumps(x) + "\n")
        print(d, len(items), {k: len(v) for k, v in T.items()}, flush=True)


if __name__ == "__main__":
    main()
