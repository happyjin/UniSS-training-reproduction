"""Listening page v2 data (A/B of several systems) for the HOK esports demo (plan §7.2f: the deployed listening page is
the non-open OOD acceptance test). Read-only on all inputs. Per excerpt and per ARM (a talker run of the same Thinker text:
baseline B, Idea 2 shrink, Idea 1 ZNE, HDemucs-vocals reference, ...) one STEREO MP3 (left = original excerpt, right =
that arm's simultaneous translation, every stream placed at its window start, same normalisation for every arm) plus
the arm's translation lines; the source lines (pipeline transcripts, silver) are shared. Arms whose run does not exist
are skipped (logged). Writes <out>/audio/<m>__<arm>.mp3 and <out>/data.json. Refuses an existing <out>.

    python experiments/mist_w3po_v1/scripts/build_game_demo_v2.py --out demo_packages/game_multispk_ab_v2 \
        --arm B:"现有（baseline B）":game_b_{d}__t1c --arm idea2:"Idea 2":game_b_{d}__t1c_shrink ...
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_game_demo import GD, ROOT, SR, TAIL_S, norm, read16, rttm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="key:label:run pattern with {d}")
    ap.add_argument("--stats", default="", help="optional json {arm: {sample_id: {metric: value}}} shown on the page")
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    arms = [tuple(x.split(":", 2)) for x in a.arm]
    samples = []
    for d in ("zh2en", "en2zh"):
        wins = [json.loads(l) for l in open(ROOT / f"eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1/windows_test_online_{d}.jsonl")]
        refs = defaultdict(list)
        for l in open(GD / d / "refs_windows.jsonl"):
            r = json.loads(l)
            refs[r["meeting"]].append(r)
        meetings = json.load(open(GD / d / "meetings.json"))
        sdir = "mix" if d == "zh2en" else "ihm_mix"
        streams = {}
        for key, label, pat in arms:
            run = pat.format(d=d)
            rp = ROOT / f"eval_outputs/mist_w3po_v1/speech/{run}/talker/results.jsonl"
            if not rp.exists():
                print("skip arm (no run):", key, run, flush=True)
                continue
            st = defaultdict(list)
            for l in open(rp):
                r = json.loads(l)
                if r.get("error") is None and r.get("audio_path"):
                    st[r["id"].rsplit("__S", 1)[0]].append(r)
            streams[key] = (label, run, st)
        for m, info in sorted(meetings.items()):
            src = read16(GD / d / "session_audio" / sdir / (f"{m}.wav" if d == "zh2en" else f"{m}.Mix-Headset.wav"))
            dur = len(src) / SR
            ws = sorted((w for w in wins if w["meeting"] == m), key=lambda w: w["start_s"])
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
            arm_out, n_max = {}, int(round((dur + TAIL_S) * SR))
            tracks = {}
            for key, (label, run, st) in streams.items():
                tr = np.zeros(n_max, np.float32)
                tl = []
                for w in ws:
                    for r in st.get(w["id"], []):
                        y = read16(r["audio_path"])
                        i = int(round(w["start_s"] * SR))
                        j = min(n_max, i + len(y))
                        if j > i:
                            tr[i:j] += y[: j - i]
                        k = int(r["id"].rsplit("__S", 1)[1])
                        for b in r.get("blocks") or []:
                            if (b.get("text") or "").strip():
                                s = w["start_s"] + b["start_ms"] / 1000.0
                                tl.append({"t": round(s, 2), "e": round(s + b["dur_ms"] / 1000.0, 2), "spk": k,
                                           "text": b["text"].strip()})
                tracks[key] = (tr, sorted(tl, key=lambda x: x["t"]), label)
            last = max([x["e"] for _, (tr, tl, _) in tracks.items() for x in tl] + [dur])
            n = min(n_max, int(round(min(last + 0.5, dur + TAIL_S) * SR)))
            s16 = np.pad(src, (0, max(0, n - len(src))))[:n]
            left = norm(s16)
            for key, (tr, tl, label) in tracks.items():
                x = np.stack([left, norm(tr[:n])], 1)
                pk = float(np.max(np.abs(x))) if x.size else 0.0
                if pk > 0.97:
                    x = x * (0.97 / pk)
                p = out / "audio" / f"{m}__{key}.mp3"
                sf.write(str(p), x.astype(np.float32), SR, format="MP3")
                arm_out[key] = {"label": label, "run": streams[key][1], "audio": f"audio/{p.name}", "tgt": tl}
            spk = sorted({k for _, _, k in rt})
            samples.append({"id": m, "dir": d, "title": info.get("title", ""), "url": info["url"], "video": info["video"],
                            "part": info["part"], "t0": info["t0"], "dur": round(n / SR, 2), "windows": len(ws),
                            "speakers": len(spk), "src": sl, "arms": arm_out})
            print(m, d, "arms", list(arm_out), flush=True)
    stats = json.load(open(a.stats)) if a.stats else {}
    (out / "data.json").write_text(json.dumps({"samples": samples, "arm_order": [k for k, _, _ in arms],
                                               "arm_labels": {k: l for k, l, _ in arms}, "stats": stats},
                                              ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
