"""Idea 2: Nemotron-3-Diarization (lowlat preset, 1.04 s) posteriors for the synthetic streams (prereg §2).
Same model / preset / pooling as experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py; runs in venvs/nemo_diar.

    python experiments/mist_w3po_v1/voicecond/idea2_diarize.py --which pilot|all --shard 0 --num-shards 4

Per stream: data/open_voicecond_v1/idea2/pre/diar/<sid>.post.pt ({"post80": fp16 [T80, n_spk], "frame_sec": 0.08, ...})
and <sid>.rttm. A stream already written is skipped, never rewritten.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "pretrained_models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo"
I2 = Path(__import__("os").environ.get("IDEA2_ROOT", str(REPO / "data/open_voicecond_v1/idea2")))   # IDEA2_ROOT: e.g. the cvt root
LOWLAT = (264, 264, 9, 4, 222)   # spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period
N_PILOT = {"train": 32, "dev": 4, "test": 4}


def is_pilot(sid: str) -> bool:
    split, _, i = sid.split("_")
    return int(i) < N_PILOT[split]


def stream_list(which: str):
    s = sorted(p.name for p in (I2 / "streams").iterdir() if p.is_dir() and not p.name.startswith(".")
               and (p / "meta.json").exists())
    return [x for x in s if which == "all" or is_pilot(x)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    from nemo.collections.asr.models import SortformerEncLabelModel
    out = I2 / "pre/diar"
    out.mkdir(parents=True, exist_ok=True)
    sids = stream_list(a.which)[a.shard:: a.num_shards]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SortformerEncLabelModel.restore_from(str(MODEL), map_location=dev).eval()
    sm = model.sortformer_modules
    (sm.spkcache_len, sm.fifo_len, sm.chunk_len, sm.chunk_right_context, sm.spkcache_update_period) = LOWLAT
    print(f"[diar] {a.which} shard {a.shard}/{a.num_shards}: {len(sids)} streams on {dev}", flush=True)
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        if (out / f"{sid}.post.pt").exists():
            continue
        wav = I2 / "streams" / sid / "mix.wav"
        dur = sf.info(str(wav)).duration
        with torch.no_grad():
            segs, probs = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True, verbose=False)
        pr = probs[0][0].float().cpu()
        pool = int(round(pr.shape[0] / max(dur, 1e-6) * 0.08))
        t80 = pr.shape[0] // max(pool, 1)
        post80 = pr[: t80 * pool].reshape(t80, pool, -1).mean(1).half() if pool > 1 else pr.half()
        tmp = out / f".{sid}.post.pt.partial"
        torch.save({"post80": post80, "frame_sec": 0.08, "n_spk": int(post80.shape[1]), "duration": dur,
                    "preset": "lowlat", "model": "Nemotron-3-Diarization"}, tmp)
        tmp.rename(out / f"{sid}.post.pt")
        with open(out / f"{sid}.rttm", "w" if not (out / f"{sid}.rttm").exists() else "x") as f:
            for s in segs[0]:
                st, ed, spk = s.split()
                st, ed = float(st), float(ed)
                if ed - st > 1e-3:
                    f.write(f"SPEAKER {sid} 1 {st:.3f} {ed - st:.3f} <NA> <NA> spk{int(spk.split('_')[-1]) + 1} <NA> <NA>\n")
        print(f"[diar] {i}/{len(sids)} {sid} dur={dur:.0f}s T80={t80} elapsed={time.time() - t0:.0f}s", flush=True)
    print("[diar] DONE", flush=True)


if __name__ == "__main__":
    main()
