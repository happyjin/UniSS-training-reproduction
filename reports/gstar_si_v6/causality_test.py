"""Paper v6, review M02 (prereg_v6 §2): is the speaker stream the streaming system reads exactly what a prefix-only
diarizer would have produced?

The streaming runs read posteriors from ONE run of the low-latency Streaming Sortformer over the whole session
(data_clean/dump_nemotron_posteriors.py, preset lowlat), with every frame later than t - 1.04 s zeroed at a read at t.
That equals a causal system only if the frames <= t - 1.04 s do not change when the audio after t is never given. Here,
per test meeting and cut point t (read times of randomly drawn reference windows, fixed seed), the session audio is
truncated at t and diarized from scratch with the same model and preset; the frames before t - 1.04 s are compared
with the whole-session dump: max / mean absolute difference, frames whose >0.5 decision differs, and whether the
per-frame most active slot agrees (a slot permutation would show here).

Runs in venvs/nemo_diar. Writes <out>/cuts.jsonl (one line per cut, appended; a cut already there is skipped) and,
per cut, the prefix posteriors <out>/post/<set>_<meeting>_<t_ms>.pt. Nothing existing is overwritten.

    CUDA_VISIBLE_DEVICES=3 venvs/nemo_diar/bin/python reports/gstar_si_v6/causality_test.py --out <dir> [--per-meeting 3]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "pretrained_models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo"
LOWLAT = (264, 264, 9, 4, 222)   # spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period
LOOKAHEAD_S = 1.04
D = REPO / "data/ami_multispk_si_v1"
A = REPO / "data/alimeeting_si_v1"
SETS = {   # name: (windows of the streaming test, session audio of a meeting, whole-session lowlat dump)
    "en": (D / "e2e_gstar_stream_v1/windows_test_ref.jsonl", lambda m: D / f"session_audio/ihm_mix/{m}.Mix-Headset.wav",
           lambda m: D / f"diar/nemotron_lowlat/{m}.post.pt"),
    "zh": (D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl", lambda m: D / f"session_audio/zh_mix_v8/{m}.wav",
           lambda m: D / f"diar/nemotron_lowlat_zh8/{m}.post.pt"),
    "ali": (A / "windows_test_ref.jsonl", lambda m: A / f"session_audio/near_mix/{m}.wav",
            lambda m: A / f"diar/nemotron_lowlat/{m}.post.pt"),
}


def cuts(per_meeting: int, seed: int) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    out = []
    for name, (wf, _, _) in SETS.items():
        by_m = {}
        for w in map(json.loads, open(wf, encoding="utf-8")):
            by_m.setdefault(w["meeting"], []).append(w)
        for m in sorted(by_m):
            for w in rng.sample(by_m[m], min(per_meeting, len(by_m[m]))):
                reads = list(range(1000, w["dur_ms"], 1000)) + [w["dur_ms"]]
                out.append((name, m, round(w["start_s"] + rng.choice(reads) / 1000, 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-meeting", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()
    from nemo.collections.asr.models import SortformerEncLabelModel

    out = Path(args.out)
    (out / "post").mkdir(parents=True, exist_ok=True)
    (out / "wav").mkdir(exist_ok=True)
    done = set()
    if (out / "cuts.jsonl").exists():
        done = {(r["set"], r["meeting"], r["t_s"]) for r in map(json.loads, open(out / "cuts.jsonl"))}
    model = SortformerEncLabelModel.restore_from(str(MODEL), map_location=torch.device("cuda")).eval()
    sm = model.sortformer_modules
    (sm.spkcache_len, sm.fifo_len, sm.chunk_len, sm.chunk_right_context, sm.spkcache_update_period) = LOWLAT
    todo = [c for c in cuts(args.per_meeting, args.seed) if c not in done]
    print(f"[causality] {len(todo)} cuts to run", flush=True)
    t0 = time.time()
    for k, (name, m, t) in enumerate(todo, 1):
        _, audio_of, dump_of = SETS[name]
        x, sr = sf.read(str(audio_of(m)), dtype="float32", always_2d=True)
        x = x.mean(1)[: int(round(t * sr))]
        wav = out / "wav" / f"{name}_{m}_{int(t * 1000)}.wav"
        if not wav.exists():
            sf.write(str(wav), x, sr)
        with torch.no_grad():
            _, probs = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True, verbose=False)
        pre = probs[0][0].float().cpu().numpy()
        torch.save({"post80": torch.from_numpy(pre).half(), "t_s": t}, out / "post" / f"{wav.stem}.pt")
        full = torch.load(dump_of(m), weights_only=False)["post80"].float().numpy()
        n = int((t - LOOKAHEAD_S) / 0.08)      # frames the reader at t may see (later ones are zeroed)
        n = max(0, min(n, len(pre), len(full)))
        a, b = pre[:n], full[:n]
        # the dump is stored in fp16: compare with the prefix run rounded the same way
        a16 = a.astype(np.float16).astype(np.float32)
        diff = np.abs(a16 - b)
        act = (a16 > 0.5) != (b > 0.5)
        live = (b.max(1) > 0.5) | (a16.max(1) > 0.5)
        arg = (a16.argmax(1) == b.argmax(1))[live] if live.any() else np.array([True])
        rec = {"set": name, "meeting": m, "t_s": t, "frames": n,
               "max_abs": float(diff.max()) if n else 0.0, "mean_abs": float(diff.mean()) if n else 0.0,
               "frac_decision_diff": float(act.any(1).mean()) if n else 0.0,
               "argmax_agree_active": float(arg.mean()),
               "bit_equal_fp16": bool(n == 0 or (a16 == b).all()),
               "tail_max_abs_after_cut": float(np.abs(pre[n: n + 13].astype(np.float16).astype(np.float32)
                                                      - full[n: n + 13]).max()) if len(pre) > n and len(full) > n else 0.0}
        with open(out / "cuts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[causality] {k}/{len(todo)} {name} {m} t={t:.1f}s frames={n} max={rec['max_abs']:.2e} "
              f"dec_diff={rec['frac_decision_diff']:.4f} elapsed={time.time() - t0:.0f}s", flush=True)
    print("[causality] DONE", flush=True)


if __name__ == "__main__":
    main()
