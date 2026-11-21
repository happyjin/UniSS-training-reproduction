"""Robust test copy of a_sv_ecapa2_spksim.py (data root = $S1_DATA). Experiment A independent evaluator: speaker similarity of each generated sentence to the CLEAN source of the same
sentence (clean twin, as wave3_v_spksim.py), with ECAPA2 (Jenthe/ECAPA2 TorchScript, HF download, VoxCeleb-trained;
independent of TitaNet, which is both a metric and the method's input, and of BiCodec's ECAPA). fp32, whole utterance,
L2-normalised 192-d embedding, cosine. One output row per item: {id, sim}.
    venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.a_sv_ecapa2_spksim --dir zh2en --run <si_infer run dir>
        --out <jsonl> [--device cuda:0]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
ECAPA2 = ("/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub/models--Jenthe--ECAPA2/snapshots/"
          "207cb6d137c671a12ba820ebec3b719549b06c0f/ecapa2.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    m = torch.jit.load(ECAPA2, map_location=a.device).eval()

    @torch.no_grad()
    def emb(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        assert sr == 16000, p
        e = m(torch.from_numpy(x.mean(1))[None].to(a.device))[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    data = Path(__import__("os").environ["S1_DATA"])   # test copy: data root from S1_DATA (required)
    meta = {r["id"]: r for r in map(json.loads, open(data / f"meta_{a.dir}.jsonl", encoding="utf-8"))}
    audio = {r["id"]: r["audio"] for r in map(json.loads, open(data / f"items_{a.dir}.jsonl", encoding="utf-8"))}
    clean = {(r["set"], r["pair"]): i for i, r in meta.items() if r["cond"] == "clean"}
    refs, sims = {}, []
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for r in map(json.loads, open(Path(a.run) / "results.jsonl", encoding="utf-8")):
            mm = meta[r["id"]]
            ci = clean.get((mm["set"], mm["pair"]), r["id"])   # no clean twin (OOD-3): own source, as wave3_v_spksim
            p = r.get("speech_path")
            s = None
            if p and Path(p).exists() and sf.info(p).frames > 1600:
                if ci not in refs:
                    refs[ci] = emb(REPO / audio[ci])
                s = float(emb(p) @ refs[ci])
                sims.append(s)
            fh.write(json.dumps({"id": r["id"], "sim": s}) + "\n")
    print(json.dumps({"run": a.run, "n": len(sims), "median": float(np.median(sims)) if sims else None}))


if __name__ == "__main__":
    main()
