"""Listening package: HSC2 vs B_b / N / SVC-JM on the test cells (2 sentences per cell and direction).
Selection (no result peeking): per (direction, set, cond) cell, items present in all four systems' play outputs with
3 s <= dur <= 12 s, random.Random(20261125).sample(., 2) over the sorted ids. Audio = the dip player's listening wav
(voice_onset_v1.s0_apply --wav-ids, the same player as the scores) per system, plus the source; MP3. Text: source, reference,
each system's Thinker text and speech ASR, per-item D2 (speech LAAL, s), NP_in. Output demo_packages/hsc2_listen_v1.
    <qwen3omni python> -m experiments.sst_hsmm_v1.listen_hsc2
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "demo_packages/hsc2_listen_v1"
T = REPO / "eval_outputs/mist_w3po_v1/test_robust"
S = REPO / "eval_outputs/sst_hsmm_v1"
SYS = {"B_b": (T / "B/b", S / "p3z2/play/B_b_{d}.jsonl"), "N": (T / "UE_mbr/mbr", S / "p3z2/play/N_{d}.jsonl"),
       "SVC-JM": (S / "p3w/new", S / "p3w/play/NEW_{d}.jsonl"), "HSC2": (S / "p3z2/new", S / "p3z2/play/NEW_{d}.jsonl")}
CELLS = [("realsi", "clean"), ("realsi", "noisy"), ("obsi", "clean"), ("obsi", "id2"), ("obsi", "ood1")]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    if OUT.exists():
        sys.exit(f"拒绝覆盖: {OUT}")
    (OUT / "audio").mkdir(parents=True)
    tmp = OUT / "render"
    rng = random.Random(20261125)
    samples = []
    for d in ("en2zh", "zh2en"):
        items = {r["id"]: r for r in jl(T / "data" / f"items_{d}.jsonl")}
        meta = {r["id"]: r for r in jl(T / "data" / f"meta_{d}.jsonl")}
        play = {k: {r["id"]: r for r in jl(str(p).format(d=d))} for k, (_, p) in SYS.items()}
        chosen = []
        for st, cd in CELLS:
            pool = sorted(i for i, m in meta.items() if m["set"] == st and m["cond"] == cd and 3000 <= m["dur_ms"] <= 12000
                          and all(i in play[k] and play[k][i].get("D2") is not None for k in SYS))
            for i in rng.sample(pool, 2):
                chosen.append((st, cd, i))
        ids = ",".join(i for _, _, i in chosen)
        for k, (run, _) in SYS.items():
            wd = tmp / f"{k}_{d}"
            env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", S1_DATA=str(T / "data"))
            subprocess.run([sys.executable, "-m", "experiments.voice_onset_v1.s0_apply", "--run", str(run / f"talker_{d}"),
                            "--dir", d, "--policy", "dip", "--out", str(tmp / f"{k}_{d}.jsonl"), "--wav-ids", ids,
                            "--wav-dir", str(wd), "--procs", "32"], check=True, env=env, cwd=REPO,
                           stdout=subprocess.DEVNULL)
        res = {k: {r["id"]: r for r in jl(run / f"talker_{d}" / "results.jsonl")} for k, (run, _) in SYS.items()}
        asr = {k: {r["id"]: r.get("asr_text", "") for r in jl(run / f"talker_{d}" / "asr_speech.jsonl")} for k, (run, _) in SYS.items()}
        for n_, (st, cd, i) in enumerate(chosen):
            it = items[i]
            x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
            src = x.mean(axis=1)[: int(it["dur_ms"]) * 16]
            key = f"{d}_{st}_{cd}_{n_ % 2}"
            sf.write(str(OUT / "audio" / f"{key}__source.mp3"), src, 16000, format="MP3")
            entry = {"key": key, "id": i, "dir": d, "set": st, "cond": cd, "dur_s": round(len(src) / 16000, 2),
                     "src_text": it.get("src_text", ""), "ref": it.get("translation", ""), "systems": {}}
            for k in SYS:
                tr, _ = sf.read(str(tmp / f"{k}_{d}" / f"{i}.wav"), dtype="float32")
                L = max(len(tr), len(src))
                mono = np.zeros(L, np.float32); mono[: len(tr)] = tr
                st2 = np.zeros((L, 2), np.float32); st2[: len(src), 0] = 0.6 * src; st2[:, 1] = mono
                peak = max(1e-6, float(np.abs(st2).max())); st2 = st2 * min(1.0, 0.95 / peak)
                sf.write(str(OUT / "audio" / f"{key}__{k}__tr.mp3"), mono * min(1.0, 0.95 / max(1e-6, float(np.abs(mono).max()))), 16000, format="MP3")
                sf.write(str(OUT / "audio" / f"{key}__{k}__mix.mp3"), st2, 16000, format="MP3")
                p = play[k][i]
                entry["systems"][k] = {"text": res[k].get(i, {}).get("translated_text", ""), "asr": asr[k].get(i, ""),
                                       "laal": round(p["D2"] / 1000, 2), "np_in": p["NP_in"], "end_offset": p["EndOffset"]}
            samples.append(entry)
    (OUT / "data.json").write_text(json.dumps({"systems": list(SYS), "samples": samples}, ensure_ascii=False, indent=1))
    print("samples", len(samples), "files", len(list((OUT / "audio").iterdir())))


if __name__ == "__main__":
    main()
