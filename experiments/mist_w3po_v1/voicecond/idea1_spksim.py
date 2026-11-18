"""Independent speaker similarity for Idea 1 E1.1 (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §5): TitaNet-large
(nvidia/speakerverification_en_titanet_large, local .nemo in cache/hf_v6; not related to BiCodec's ECAPA) embeddings of
each decoded wav and of its clean reference s; cosine similarity per pair and arm -> <out jsonl> (refuses existing).

    venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_spksim --e11 <E1.1 out dir> --diag <dir> [...]
        --out <jsonl> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NEMO = ("/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--nvidia--speakerverification_en_titanet_large/snapshots/"
        "0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--diag", nargs="+", required=True)
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
    wav = {}
    for d in a.diag:
        for r in map(json.loads, open(Path(d) / "manifest.jsonl", encoding="utf-8")):
            wav[r["id"]] = Path(d) / "wav" / f"{r['id']}_s.wav"

    @torch.no_grad()
    def emb(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = torch.from_numpy(x.mean(1))[None].to(a.device)
        _, e = m.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]], device=a.device))
        e = e[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    E = Path(a.e11)
    arms = sorted(p.name for p in E.iterdir() if (p / "results.jsonl").exists())
    rows = {}
    for arm in arms:
        for r in map(json.loads, open(E / arm / "results.jsonl", encoding="utf-8")):
            rows.setdefault(r["id"], {})[arm] = r["speech_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    sims = {arm: [] for arm in arms}
    with open(out, "x", encoding="utf-8") as fh:
        for pid, d in rows.items():
            ref = emb(wav[pid])
            rec = {"id": pid}
            for arm, p in d.items():
                rec[arm] = float(emb(p) @ ref)
                sims[arm].append(rec[arm])
            fh.write(json.dumps(rec) + "\n")
    print(json.dumps({arm: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} for arm, v in sims.items()}))


if __name__ == "__main__":
    main()
