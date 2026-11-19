"""Idea 1 voice-token tables for the Talker override (idea1_si_infer_vo.py), one row per stream prompt:
{key = prompt stem, id, sha1 of the prompt file, ids (32), raw_ids, ham_vs_raw}. Modes (prereg v3, M2):
  V      frozen encoder on the HDemucs vocals stem, exactly the tokenizer's file path (phi_path of <vocals>/<id>.wav)
         -> the listening page's reference arm (game_muffle analysis arm V)
  e2e    ZNE-in-the-loop encoder phi_theta (--ckpt): u = 2 phi_theta(y) - phi_theta(y + (sqrt2 - 1) bh), bh = y - vocals
  ctrl   denoising-encoder control phi_theta (--ckpt): u = phi_theta(y)
    python -m experiments.mist_w3po_v1.voicecond.idea1_tokens --mode V|e2e|ctrl [--ckpt <encoder.pt>] --items
        <talker_in/items.jsonl> --vocals <dir> --out <tokens.jsonl> [--device cpu|cuda:0]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def load_encoder(P, ckpt: str):
    """load a fine-tuned speaker encoder (idea1_e2e_train.py state dict) into a BiCodecPhi"""
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    for name, mod in (("speaker_encoder", P.se.speaker_encoder), ("perceiver_sampler", P.se.perceiver_sampler),
                      ("project_in", P.rq.project_in)):
        missing, unexpected = mod.load_state_dict(sd[name], strict=True), None
    return P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["V", "e2e", "ctrl"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--items", required=True)
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_zne import SQ2, aligned_bh, read_mono64
    P0 = BiCodecPhi(a.device)
    P = P0
    if a.mode in ("e2e", "ctrl"):
        assert a.ckpt, "--ckpt required"
        P = load_encoder(BiCodecPhi(a.device), a.ckpt)
    rows, seen = [], set()
    for it in map(json.loads, open(a.items, encoding="utf-8")):
        p = Path(it["voice_audio"])
        p = p if p.is_absolute() else REPO / p
        if p.stem in seen:
            continue
        seen.add(p.stem)
        vp = Path(a.vocals) / f"{it['id']}.wav"
        raw = P0.tokens(P0.phi_path(str(p))).tolist()
        if a.mode == "V":
            ids = P0.tokens(P0.phi_path(str(vp))).tolist()
        else:
            y = read_mono64(p)
            bh = aligned_bh(y, read_mono64(vp))
            if a.mode == "e2e":
                u1, u2 = P.phi_wave([y, y + (SQ2 - 1) * bh])
                u = 2 * u1 - u2
            else:
                u = P.phi_wave(y)
            ids = P.tokens(u).tolist()
        rows.append({"key": p.stem, "id": it["id"], "mode": a.mode, "ckpt": a.ckpt, "ids": [int(v) for v in ids],
                     "raw_ids": [int(v) for v in raw], "sha1": hashlib.sha1(p.read_bytes()).hexdigest(),
                     "ham_vs_raw": int(sum(x != z for x, z in zip(ids, raw)))})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    h = np.array([r["ham_vs_raw"] for r in rows])
    print(json.dumps({"out": str(out), "mode": a.mode, "n": len(rows), "ham_vs_raw_median": float(np.median(h))}))


if __name__ == "__main__":
    main()
