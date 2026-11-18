#!/usr/bin/env python
"""OpenBroadcast-SI v1 listening subset (plan §7.2f; reports/mist_w3po_v1/voicecond_bench.zh-CN.md): for a few streams
per direction, chosen WITHOUT looking at any system output (K >= 3, stream >= 120 s for zh2en / >= 45 s for en2zh, then
the 3 whose BVR target is closest to the esports prompt-BVR median, ties by stream key), the same stream in id2, ood1
and clean: one STEREO MP3 each in build_game_demo.py's layout -- left = the condition's source mix, right = the run's
simultaneous translation (every talker stream wav placed at its window start), both at the same active RMS -- plus
data.json with source subtitles (real speaker, CoVoST source text, reference) and translation subtitles (talker blocks).
Default run: baseline B (obsi_b_<cond>_<dir>__t1c, TTS seed 1). Read-only on all inputs. Refuses an existing <out>.

    python scripts/voicecond_bench_listen.py --out demo_packages/voicecond_bench_listen_v1 [--run-fmt obsi_b_{cond}_{dir}__t1c]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/mist_w3po_v1/scripts"))
from build_game_demo import TAIL_S, norm, read16  # noqa: E402

BD = ROOT / "data/open_voicecond_v1/bench"
SR = 16000
CONDS = ("id2", "ood1", "clean")


def pick(d: str) -> list[str]:
    P = json.load(open(BD / "src" / d / "plan.json"))
    med = float(np.median(json.load(open(BD / "meta/stats.json"))["bvr_db"]["values"]))
    lo = 120.0 if d == "zh2en" else 45.0
    c = [s for s in P["streams"] if s["K"] >= 3 and s["dur"] >= lo]
    return [s["key"] for s in sorted(c, key=lambda s: (abs(s["bvr_db"] - med), s["key"]))[:3]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-fmt", default="obsi_b_{cond}_{dir}__t1c")
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    samples = []
    for d in ("zh2en", "en2zh"):
        keys = pick(d)
        for cond in CONDS:
            run = a.run_fmt.format(cond=cond, dir=d)
            M = json.load(open(BD / cond / d / "meetings.json"))
            by_key = {v["stream"]: m for m, v in M.items()}
            wins = [json.loads(l) for l in open(ROOT / f"eval_outputs/mist_w3po_v1/data/windows_obsi_{cond}_{d}_c2g06_v1/"
                                                       f"windows_test_online_{d}.jsonl")]
            streams = defaultdict(list)
            for l in open(ROOT / f"eval_outputs/mist_w3po_v1/speech/{run}/talker/results.jsonl"):
                r = json.loads(l)
                if r.get("error") is None and r.get("audio_path"):
                    streams[r["id"].rsplit("__S", 1)[0]].append(r)
            refs = defaultdict(list)
            for l in open(BD / cond / d / "refs_windows.jsonl", encoding="utf-8"):
                r = json.loads(l)
                refs[r["meeting"]].append(r)
            sdir = "mix" if d == "zh2en" else "ihm_mix"
            for key in keys:
                m = by_key[key]
                src = read16(BD / cond / d / "session_audio" / sdir / (f"{m}.wav" if d == "zh2en" else f"{m}.Mix-Headset.wav"))
                dur = len(src) / SR
                n = int(round((dur + TAIL_S) * SR))
                tr = np.zeros(n, np.float32)
                tl, ns = [], 0
                for w in sorted((w for w in wins if w["meeting"] == m), key=lambda w: w["start_s"]):
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
                tr, src = tr[:n], np.pad(src, (0, max(0, n - len(src))))[:n]
                spk_ix = {s: i for i, s in enumerate(M[m]["speakers"])}
                sl = [{"t": round(r["start_s"], 2), "e": round(r["start_s"] + r["utts"][0]["end_s"], 2),
                       "spk": spk_ix[r["utts"][0]["speaker_id"]],
                       "text": r["utts"][0]["zh" if d == "zh2en" else "en"], "ref": r["utts"][0]["ref"]}
                      for r in sorted(refs[m], key=lambda r: r["start_s"])]
                x = np.stack([norm(src), norm(tr)], 1)
                pk = float(np.max(np.abs(x))) if x.size else 0.0
                if pk > 0.97:
                    x = x * (0.97 / pk)
                sf.write(str(out / "audio" / f"{m}.mp3"), x.astype(np.float32), SR, format="MP3")
                samples.append({"id": m, "stream": key, "dir": d, "cond": cond, "run": run, "dur": round(n / SR, 2),
                                "speakers": M[m]["K"], "bvr_db": M[m].get("bvr_db"), "twins": M[m]["twins"],
                                "streams": ns, "audio": f"audio/{m}.mp3", "src": sl, "tgt": sorted(tl, key=lambda x: x["t"])})
                print(m, d, cond, key, "streams", ns, flush=True)
    (out / "data.json").write_text(json.dumps({"samples": samples, "layout": "left = source mix, right = translation"},
                                              ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
