"""Idea 2 prereg v2.1: speaker-identity check on the esports demo (OOD-2, evaluation only).

refs   (venvs/vllm_omni_030 python): per excerpt, HDemucs vocals of the whole 300 s (idea2_demucs.separate); per
       (excerpt, diarizer slot) the reference = concatenation of the vocals over ALL solo runs of that slot (p_k > .5,
       other slots < .5, runs >= 0.4 s, meeting-level Nemotron lowlat posteriors) -> data/open_voicecond_v1/idea2/game/
       <dir>/spkref/<meeting>_S<k>.wav (+ index.json). Existing files are kept.
    python -I experiments/mist_w3po_v1/voicecond/idea2_game_spksim.py refs --dir zh2en|en2zh [--device cuda:0]
sim    (venvs/nemo_diar python): TitaNet-large (idea1_spksim.NEMO) cosine between each generated stream
       (speech/<run>/talker/results.jsonl speech_path) and its slot reference -> eval_outputs/mist_w3po_v1/
       voicecond_idea2/spksim/<run>.jsonl (one file per run; an existing file is skipped, never rewritten).
    python experiments/mist_w3po_v1/voicecond/idea2_game_spksim.py sim --runs <run> ... [--device cuda:0]
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
NEMO = ("/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--nvidia--speakerverification_en_titanet_large/snapshots/"
        "0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo")


def solo_runs(post, s):
    alone = (post[:, s] > 0.5) & (np.delete(post, s, axis=1).max(1) < 0.5)
    runs, i = [], 0
    while i < len(alone):
        if alone[i]:
            j = i
            while j < len(alone) and alone[j]:
                j += 1
            if (j - i) * 0.08 >= 0.4:
                runs.append((i * 0.08, j * 0.08))
            i = j
        else:
            i += 1
    return runs


def cmd_refs(a):
    import soundfile as sf
    import torch
    from torchaudio.models import hdemucs_high
    from experiments.mist_w3po_v1.voicecond.idea2_demucs import separate
    from experiments.mist_w3po_v1.voicecond.idea2_game_tokens import GD, meeting_audio
    od = I2 / "game" / a.dir / "spkref"
    od.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    model.load_state_dict(torch.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints/hdemucs_high_trained.pt",
                                     map_location="cpu"))
    model = model.eval().to(dev)
    index = {}
    meetings = sorted(p.stem.split(".")[0] for p in (GD / a.dir / "diar/nemotron_lowlat").glob("*.post.pt"))
    for m in meetings:
        vp = I2 / "game" / a.dir / f"{m}.vhat.wav"
        if not vp.exists():
            y, sr = sf.read(str(meeting_audio(a.dir, m)), dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            vh, _ = separate(model, y, dev)
            tmp = vp.parent / f".{m}.vhat.partial.wav"
            sf.write(str(tmp), np.clip(vh, -1, 1), SR, subtype="PCM_16")
            os.rename(tmp, vp)
        vh = sf.read(str(vp), dtype="float32")[0]
        post = torch.load(GD / a.dir / "diar/nemotron_lowlat" / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        for s in range(post.shape[1]):
            runs = solo_runs(post, s)
            if not runs:
                continue
            key = f"{m}_S{s + 1}"
            rp = od / f"{key}.wav"
            x = np.concatenate([vh[int(b0 * SR): int(b1 * SR)] for b0, b1 in runs])
            if not rp.exists():
                sf.write(str(rp), x, SR, subtype="PCM_16")
            index[key] = {"dur_s": round(len(x) / SR, 2), "n_runs": len(runs)}
    ip = od / "index.json"
    if not ip.exists():
        ip.write_text(json.dumps(index, indent=1))
    print(json.dumps({"dir": a.dir, "refs": len(index), "dur_s_median": float(np.median([v["dur_s"] for v in index.values()]))}))


def cmd_sim(a):
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
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
        d = "zh2en" if "_zh2en_" in run else "en2zh"
        rows = [json.loads(l) for l in open(REPO / "eval_outputs/mist_w3po_v1/speech" / run / "talker/results.jsonl")]
        res = []
        for r in rows:
            if r.get("error") is not None or not r.get("speech_path"):
                continue
            wid, k = r["id"].rsplit("__S", 1)
            key = f"{wid.split('_')[0]}_S{k}"
            rp = I2 / "game" / d / "spkref" / f"{key}.wav"
            if not rp.exists():
                continue
            if key not in refs:
                refs[key] = emb(rp)
            res.append({"id": r["id"], "ref": key, "sim": float(emb(r["speech_path"]) @ refs[key])})
        with open(out, "x") as fh:
            for x in res:
                fh.write(json.dumps(x) + "\n")
        print(json.dumps({"run": run, "n": len(res), "median": float(np.median([x["sim"] for x in res]))}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refs", "sim"])
    ap.add_argument("--dir", choices=["zh2en", "en2zh"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cmd_refs(a) if a.cmd == "refs" else cmd_sim(a)
