"""Idea 2 prereg v3 §D identity metric for OOD-3 (same definition as v2.1 for the esports demo): TitaNet-large cosine
between each generated stream and the reference of its (meeting, diarizer slot) = HDemucs vocals (idea2_demucs.separate)
over ALL solo runs of that slot (p_k > .5, others < .5, runs >= 0.4 s) in the whole session.

refs (venvs/vllm_omni_030):  python -I .../idea2_spksim_generic.py refs --data <dir> --session <s> --diar <d> --tag <t>
                                 --meetings m ... [--device cuda:0]   -> data/open_voicecond_v1/idea2/spkref/<t>/<m>_S<k>.wav
sim  (venvs/nemo_diar):      python .../idea2_spksim_generic.py sim --tag <t> --runs <run> ... [--device cuda:0]
                                 -> eval_outputs/mist_w3po_v1/voicecond_idea2/spksim/<run>.jsonl (skips existing files)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
I2 = REPO / "data/open_voicecond_v1/idea2"
SR = 16000


def cmd_refs(a):
    import soundfile as sf
    import torch
    from torchaudio.models import hdemucs_high
    from experiments.mist_w3po_v1.voicecond.idea2_demucs import separate
    from experiments.mist_w3po_v1.voicecond.idea2_game_spksim import solo_runs
    data = REPO / a.data
    od = I2 / "spkref" / a.tag
    od.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints/hdemucs_high_trained.pt",
                                     map_location="cpu"))
    model = model.eval().to(dev)
    index = {}
    for m in a.meetings:
        p = data / "session_audio" / a.session / (f"{m}.Mix-Headset.wav" if a.session == "ihm_mix" else f"{m}.wav")
        y, sr = sf.read(str(p), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        assert sr == SR, sr
        vh, _ = separate(model, y, dev)
        post = torch.load(data / "diar" / a.diar / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        for s in range(post.shape[1]):
            runs = solo_runs(post, s)
            if not runs:
                continue
            key = f"{m}_S{s + 1}"
            x = np.concatenate([vh[int(b0 * SR): int(b1 * SR)] for b0, b1 in runs])
            rp = od / f"{key}.wav"
            if not rp.exists():
                sf.write(str(rp), x, SR, subtype="PCM_16")
            index[key] = {"dur_s": round(len(x) / SR, 2), "n_runs": len(runs)}
        print(f"[refs] {a.tag} {m}", flush=True)
    ip = od / "index.json"
    if not ip.exists():
        ip.write_text(json.dumps(index, indent=1))
    print(json.dumps({"tag": a.tag, "refs": len(index)}))


def cmd_sim(a):
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from experiments.mist_w3po_v1.voicecond.idea2_game_spksim import NEMO
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()

    @torch.no_grad()
    def emb(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            import torchaudio.functional as AF
            x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
        t = torch.from_numpy(x)[None].to(a.device)
        _, e = m.forward(input_signal=t, input_signal_length=torch.tensor([t.shape[1]], device=a.device))
        e = e[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    od = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/spksim"
    od.mkdir(parents=True, exist_ok=True)
    refs = {}
    for run in a.runs:
        out = od / f"{run}.jsonl"
        if out.exists():
            print(f"skip existing {out}")
            continue
        res = []
        for r in map(json.loads, open(REPO / "eval_outputs/mist_w3po_v1/speech" / run / "talker/results.jsonl")):
            if r.get("error") is not None or not r.get("speech_path"):
                continue
            wid, k = r["id"].rsplit("__S", 1)
            key = f"{wid.rsplit('_on_', 1)[0]}_S{k}"
            rp = I2 / "spkref" / a.tag / f"{key}.wav"
            if not rp.exists():
                continue
            if key not in refs:
                refs[key] = emb(rp)
            res.append({"id": r["id"], "ref": key, "sim": float(emb(r["speech_path"]) @ refs[key])})
        with open(out, "x") as fh:
            for x in res:
                fh.write(json.dumps(x) + "\n")
        print(json.dumps({"run": run, "n": len(res), "median": float(np.median([x["sim"] for x in res])) if res else None}),
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refs", "sim"])
    ap.add_argument("--data", default="")
    ap.add_argument("--session", default="")
    ap.add_argument("--diar", default="nemotron_lowlat")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--meetings", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cmd_refs(a) if a.cmd == "refs" else cmd_sim(a)
