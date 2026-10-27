"""Manifest + OGG audio for the M2-vs-new fluency listening page (index.html beside this script).

    python build.py <compare.json> en2zh:<m2 realsi dir>:<new realsi dir>:<new label> zh2en:...
Per direction: the 6 sentences whose excess pause burden fell most (source >= 3 s) and 4 drawn at random (seed 0).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/opt/dlami/nvme/jasonleeeli/projects/UniSS")
from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import TOL, seam_types, vad_pauses  # noqa: E402

PAGE = Path(__file__).resolve().parent


def ogg(src: str, dst: Path) -> float:
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    for c in range(x.shape[1]):   # each channel to the same loud-speech level (RealSI's source is quieter than the TTS)
        y = x[:, c]
        n = max(1, len(y) // (sr // 10))
        loud = float(np.percentile(np.sqrt(np.mean(y[: n * (sr // 10)].reshape(n, -1) ** 2, axis=1)), 99)) or 1e-4
        g = min(0.12 / max(loud, 1e-4), 3.0 / (float(np.abs(y).max()) or 1.0))
        x[:, c] = 0.95 * np.tanh(y * g / 0.95)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(dst), "x", sr, x.shape[1], format="OGG", subtype="VORBIS") as fh:
        for i in range(0, len(x), 32768):
            fh.write(x[i: i + 32768])
    return round(len(x) / sr, 2)


def side(r: dict) -> dict:
    pauses, first, last, _ = vad_pauses(r["audio_path"])
    b = r["blocks"]
    text = r.get("translated_text") or "".join(x["text"] for x in b)
    types = seam_types(b, r["tgt_lang"], text) if len(b) > 1 else []
    seam_t = [((x["start_ms"] + x["dur_ms"]) / 1000, y["start_ms"] / 1000) for x, y in zip(b, b[1:])]
    excess = 0.0
    for p0, p1 in pauses:
        k = next((k for k, (a, c) in enumerate(seam_t) if p0 <= c + 0.05 and p1 >= a - 0.05), None)
        excess += max(0.0, (p1 - p0) - (TOL[types[k]] if k is not None else TOL["P"]))
    return {"blocks": [{"t": x["write_ms"] / 1000, "s": x["start_ms"] / 1000, "d": x["dur_ms"] / 1000, "text": x["text"]} for x in b],
            "pauses": [[round(a, 2), round(c, 2)] for a, c in pauses], "excess": round(excess, 2),
            "n_long": sum(1 for a, c in pauses if c - a >= 1.0), "end_after": round(last - r["source_duration_ms"] / 1000, 2)}


def main():
    cmp = json.load(open(sys.argv[1]))
    out = {"summary": cmp, "dirs": []}
    for spec in sys.argv[2:]:
        d, m2_dir, new_dir, label = spec.split(":")
        A = {r["id"]: r for r in map(json.loads, open(Path(m2_dir) / "results.jsonl"))}
        B = {r["id"]: r for r in map(json.loads, open(Path(new_dir) / "results.jsonl"))}
        ids = [i for i in A if i in B and A[i]["source_duration_ms"] >= 3000]
        sides = {i: (side(A[i]), side(B[i])) for i in ids}
        best = sorted(ids, key=lambda i: sides[i][1]["excess"] - sides[i][0]["excess"])[:6]
        rest = [i for i in ids if i not in best]
        random.Random(0).shuffle(rest)
        items = []
        for i, why in [(i, "改善最多") for i in best] + [(i, "随机抽样") for i in sorted(rest[:4])]:
            a, b = sides[i]
            ra, rb = A[i], B[i]
            ogg(ra["audio_path"].replace("_simul.wav", "_stereo.wav"), PAGE / f"audio/{d}/{i}_m2.ogg")
            ogg(rb["audio_path"].replace("_simul.wav", "_stereo.wav"), PAGE / f"audio/{d}/{i}_new.ogg")
            items.append({"id": i, "why": why, "src": ra["transcription_ref"], "ref": ra["translation_ref"],
                          "src_s": ra["source_duration_ms"] / 1000, "m2": a, "new": b,
                          "audio": {"m2": f"audio/{d}/{i}_m2.ogg", "new": f"audio/{d}/{i}_new.ogg"}})
        out["dirs"].append({"dir": d, "label": label, "items": items})
    (PAGE / "manifest.json").write_text(json.dumps(out, ensure_ascii=False))
    print("items", sum(len(x["items"]) for x in out["dirs"]))


if __name__ == "__main__":
    main()
