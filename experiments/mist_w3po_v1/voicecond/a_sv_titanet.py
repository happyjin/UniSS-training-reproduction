"""Experiment A step 1 (nemo_diar): TitaNet-large embeddings (L2-normalised, 192-d; the idea1_spksim model, forward exactly
as wave3_v_spksim.emb) of
  train : the noisy y prompt-span crop of every training pair (a_sv_common.train_iter("y"))    -> titanet_train.npy
  mon   : the noisy y crop of the 200 monitor pairs                                              -> titanet_mon.npy
  dev_d : the S1 dev prompt file of every wave3 V1 table row (sha1 checked; noisy AND clean rows) -> titanet_dev_<d>.npy
    venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.a_sv_titanet --out <dir> [--device cuda:0]
Refuses existing output files.
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
from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N per part")
    a = ap.parse_args()
    out = Path(a.out)
    names = ["titanet_train.npy", "titanet_mon.npy"] + [f"titanet_dev_{d}.npy" for d in C.DIRS]
    for n in names:
        if (out / n).exists():
            sys.exit(f"拒绝覆盖已有输出: {out / n}")
    out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()

    @torch.no_grad()
    def emb(x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))[None].to(a.device)
        _, e = m.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]], device=a.device))
        e = e[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def run(it, name):
        t0, E = time.time(), []
        for j, w in it:
            if a.limit and j >= a.limit:
                break
            E.append(emb(w))
            if (j + 1) % 5000 == 0:
                print(f"[titanet] {name} {j + 1} {time.time() - t0:.0f}s", flush=True)
        E = np.stack(E).astype(np.float32)
        with open(out / name, "xb") as fh:
            np.save(fh, E)
        print(f"[titanet] {name} {E.shape} {time.time() - t0:.0f}s", flush=True)

    run(C.mon_iter("y"), "titanet_mon.npy")
    for d in C.DIRS:
        def dev_it(d=d):
            for j, r in enumerate(C.dev_rows(d)):
                p = C.dev_prompt(d, r["key"])
                assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                assert sr == C.SR, p
                yield j, x.mean(1)
        run(dev_it(), f"titanet_dev_{d}.npy")
    run(C.train_iter("y"), "titanet_train.npy")
    print(json.dumps({"done": names}), flush=True)


if __name__ == "__main__":
    main()
