"""Nemotron-3-Diarization on whole AMI IHM-mix sessions (docs/multi_speaker_si_plan_v7.zh-CN.md, step D2).

Runs in venvs/nemo_diar. After the gstar-nemotron branch (nemotron/dump_posteriors.py): the Streaming Sortformer is
set to one of the model card's presets and diarizes each session as one stream; per session it writes
  {meeting}.rttm     who spoke when: the model's own segments, slots spk1..spk8 consistent over the session
  {meeting}.post.pt  {"post80": fp16 [T80, 8] (10 ms posteriors mean-pooled to 80 ms), "frame_sec": 0.08, ...}
A session already written is skipped (resume), never rewritten. --shard / --num-shards split the sessions.

    python data_clean/dump_nemotron_posteriors.py --preset offline --shard 0 --num-shards 8
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
PRESETS = {  # spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period (80 ms frames)
    "offline": (264, 40, 340, 40, 300),
    "lowlat": (264, 264, 9, 4, 222),   # 1.04 s: the streaming configuration
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    ap.add_argument("--data", default=str(REPO / "data/ami_multispk_si_v1"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel

    data = Path(args.data)
    out = data / "diar" / f"nemotron_{args.preset}"
    out.mkdir(parents=True, exist_ok=True)
    meetings = sorted(json.load(open(data / "meetings.json")))[args.shard:: args.num_shards]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SortformerEncLabelModel.restore_from(str(MODEL), map_location=device).eval()
    sm = model.sortformer_modules
    (sm.spkcache_len, sm.fifo_len, sm.chunk_len, sm.chunk_right_context, sm.spkcache_update_period) = PRESETS[args.preset]
    print(f"[dump] {args.preset} shard {args.shard}/{args.num_shards}: {len(meetings)} sessions on {device} -> {out}", flush=True)
    t0 = time.time()
    for i, m in enumerate(meetings, 1):
        if (out / f"{m}.post.pt").exists():
            continue
        wav = data / "session_audio" / "ihm_mix" / f"{m}.Mix-Headset.wav"
        dur = sf.info(str(wav)).duration
        with torch.no_grad():
            segs, probs = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True, verbose=False)
        pr = probs[0][0].float().cpu()                     # [T, n_spk], 10 ms
        pool = int(round(pr.shape[0] / max(dur, 1e-6) * 0.08))
        t80 = pr.shape[0] // max(pool, 1)
        post80 = pr[: t80 * pool].reshape(t80, pool, -1).mean(1).half() if pool > 1 else pr.half()
        with open(out / f"{m}.rttm", "x") as f:
            for s in segs[0]:
                st, ed, spk = s.split()
                st, ed = float(st), float(ed)
                if ed - st > 1e-3:
                    f.write(f"SPEAKER {m} 1 {st:.3f} {ed - st:.3f} <NA> <NA> spk{int(spk.split('_')[-1]) + 1} <NA> <NA>\n")
        torch.save({"post80": post80, "frame_sec": 0.08, "n_spk": int(post80.shape[1]), "duration": dur,
                    "preset": args.preset, "model": "Nemotron-3-Diarization"}, out / f"{m}.post.pt")
        print(f"[dump] {i}/{len(meetings)} {m} dur={dur:.0f}s T80={t80} segs={len(segs[0])} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    print("[dump] DONE", flush=True)


if __name__ == "__main__":
    main()
