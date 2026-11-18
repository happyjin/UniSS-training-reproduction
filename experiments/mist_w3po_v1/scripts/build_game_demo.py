"""Game (HOK esports commentary) listening page data (reports/mist_w3po_v1/game_multispk_demo_plan.zh-CN.md §5).

Read-only on all inputs. Per excerpt (data/game_esports_si_v1/<dir>/meetings.json) one STEREO MP3: left = the original
5-min excerpt, right = baseline B's simultaneous translation (every speaker stream's talker wav
<window>__S<k>.wav placed at its window's start_s; t=0 of the stream = window start), both brought to the same active
RMS. The page remixes the two channels in the browser. Subtitles: source = the data pipeline's turn transcripts (silver),
speaker = the Nemotron lowlat slot with the largest overlap in the excerpt's rttm; translation = the talker blocks'
text, speaker = stream slot k. Scores (silver refs, indicative): score_text.json / score_asr.json of the runs.
Writes <out>/audio/<m>.mp3 and <out>/data.json. Refuses an existing <out>.

    python experiments/mist_w3po_v1/scripts/build_game_demo.py --out demo_packages/game_multispk_b_demo_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
GD = ROOT / "data/game_esports_si_v1"
SR = 16000
TAIL_S = 8.0
TARGET_RMS = 0.07


def active_rms(x):
    a = np.abs(x)
    thr = max(1e-4, 0.05 * float(np.percentile(a, 99.5)) if a.size else 1e-4)
    v = x[a > thr]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def norm(x):
    r = active_rms(x)
    return x * (TARGET_RMS / r) if r > 0 else x


def read16(p):
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def rttm(p):
    out = []
    for l in open(p):
        f = l.split()
        out.append((float(f[3]), float(f[3]) + float(f[4]), int(f[7].removeprefix("spk")) - 1))
    return out


def find(pattern):
    g = sorted(glob.glob(str(ROOT / pattern)))
    return json.load(open(g[0])) if g else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-suffix", default="", help="'' (TTS seed 1) or _tts2")
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    samples, runs = [], {}
    for d in ("zh2en", "en2zh"):
        name = f"game_b_{d}__t1c{a.seed_suffix}"
        wins = [json.loads(l) for l in open(ROOT / f"eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1/windows_test_online_{d}.jsonl")]
        streams = defaultdict(list)
        for l in open(ROOT / f"eval_outputs/mist_w3po_v1/speech/{name}/talker/results.jsonl"):
            r = json.loads(l)
            if r.get("error") is None and r.get("audio_path"):
                streams[r["id"].rsplit("__S", 1)[0]].append(r)
        runs[d] = {"name": name,
                   "text": find(f"eval_outputs/mist_w3po_v1/decode/game_b_{d}/score_text.json"),
                   "asr": find(f"eval_outputs/mist_w3po_v1/decode/{name}/score_asr.json"),
                   "lat": find(f"eval_outputs/mist_w3po_v1/crf/t_eval/{name}/lat_{name}.json")}
        refs = defaultdict(list)
        for l in open(GD / d / "refs_windows.jsonl"):
            r = json.loads(l)
            refs[r["meeting"]].append(r)
        meetings = json.load(open(GD / d / "meetings.json"))
        sdir = "mix" if d == "zh2en" else "ihm_mix"
        for m, info in sorted(meetings.items()):
            src = read16(GD / d / "session_audio" / sdir / (f"{m}.wav" if d == "zh2en" else f"{m}.Mix-Headset.wav"))
            dur = len(src) / SR
            n = int(round((dur + TAIL_S) * SR))
            tr = np.zeros(n, np.float32)
            tl, ns = [], 0
            ws = sorted((w for w in wins if w["meeting"] == m), key=lambda w: w["start_s"])
            for w in ws:
                off = w["start_s"]
                for r in streams.get(w["id"], []):
                    y = read16(r["audio_path"])
                    i = int(round(off * SR))
                    j = min(n, i + len(y))
                    if j > i:
                        tr[i:j] += y[: j - i]
                    ns += 1
                    k = int(r["id"].rsplit("__S", 1)[1])
                    for b in r.get("blocks") or []:
                        if (b.get("text") or "").strip():
                            s = off + b["start_ms"] / 1000.0
                            tl.append({"t": round(s, 2), "e": round(s + b["dur_ms"] / 1000.0, 2), "spk": k,
                                       "text": b["text"].strip()})
            last = max([x["e"] for x in tl] + [dur])
            n = min(n, int(round(min(last + 0.5, dur + TAIL_S) * SR)))
            tr = tr[:n]
            src = np.pad(src, (0, max(0, n - len(src))))[:n]
            rt = rttm(GD / d / "diar/nemotron_lowlat" / f"{m}.rttm")
            sl = []
            for r in sorted(refs[m], key=lambda r: r["start_s"]):
                u = r["utts"][0]
                b, e = r["start_s"], r["start_s"] + u["end_s"]
                ov = defaultdict(float)
                for s0, s1, k in rt:
                    ov[k] += max(0.0, min(e, s1) - max(b, s0))
                k = max(ov, key=ov.get) if ov and max(ov.values()) > 0 else -1
                sl.append({"t": round(b, 2), "e": round(e, 2), "spk": k,
                           "text": u["zh"] if d == "zh2en" else u["en"], "ref": u["ref"]})
            x = np.stack([norm(src), norm(tr)], 1)
            pk = float(np.max(np.abs(x))) if x.size else 0.0
            if pk > 0.97:
                x = x * (0.97 / pk)
            sf.write(str(out / "audio" / f"{m}.mp3"), x.astype(np.float32), SR, format="MP3")
            spk = sorted({k for _, _, k in rt})
            samples.append({"id": m, "dir": d, "title": info.get("title", ""), "url": info["url"], "video": info["video"],
                            "part": info["part"], "t0": info["t0"], "dur": round(n / SR, 2), "windows": len(ws),
                            "streams": ns, "speakers": len(spk), "audio": f"audio/{m}.mp3",
                            "src": sl, "tgt": sorted(tl, key=lambda x: x["t"])})
            print(m, d, "windows", len(ws), "streams", ns, "spk", len(spk), "lines", len(sl), len(tl), flush=True)
    (out / "data.json").write_text(json.dumps({"samples": samples, "runs": runs}, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
