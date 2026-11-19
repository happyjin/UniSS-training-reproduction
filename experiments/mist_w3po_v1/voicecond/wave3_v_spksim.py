"""Robust wave 3 V group: speaker similarity of each generated sentence to the CLEAN source of the same sentence (the
clean twin of its pair in S1's data: set a = the CoVoST clean wav, set b = the clean-dev cut of the same turn), TitaNet-
large cosine (the idea1_spksim model, independent of BiCodec). One output row per item: {id, sim}.

    venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir zh2en --run <si_infer run dir>
        --out <jsonl> [--device cuda:0]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO

REPO = Path(__file__).resolve().parents[3]
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"


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
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()

    @torch.no_grad()
    def emb(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        assert sr == 16000, p
        x = torch.from_numpy(x.mean(1))[None].to(a.device)
        _, e = m.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]], device=a.device))
        e = e[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    meta = {r["id"]: r for r in map(json.loads, open(S1 / "data" / f"meta_{a.dir}.jsonl", encoding="utf-8"))}
    audio = {r["id"]: r["audio"] for r in map(json.loads, open(S1 / "data" / f"items_{a.dir}.jsonl", encoding="utf-8"))}
    clean = {(r["set"], r["pair"]): i for i, r in meta.items() if r["cond"] == "clean"}
    refs = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    sims = []
    with open(out, "x", encoding="utf-8") as fh:
        for r in map(json.loads, open(Path(a.run) / "results.jsonl", encoding="utf-8")):
            mm = meta[r["id"]]
            ci = clean[(mm["set"], mm["pair"])]
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
