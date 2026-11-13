"""Offline listening page for AMI multi-speaker streaming SI (zh->en, test meetings), for colleagues to listen to.

Read-only on all inputs. Per meeting it cuts an excerpt of the Chinese session audio (zh_mix_v8) and places every
speaker stream's streaming translation (talker wav/<window>__S<k>_simul.wav, whose t=0 is the window start) at the
window's start_s on the meeting timeline. Writes four MP3 versions (simultaneous mix, stereo L zh / R en, translation
only, original only), a self-contained index.html (subtitles + timeline, no external resources) and README.txt.
No metrics. Refuses to write into an existing output directory or over an existing zip.

    python experiments/mist_w3po_v1/scripts/build_ami_stream_demo.py \\
        --system ft_ami_aprime_s3000__rlplem --meetings EN2002a ES2004a IS1009a TS3003a --dur 240 \\
        --out demo_packages/ami_multispk_stream_demo_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
WIN = ROOT / "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl"
SESSION = ROOT / "data/ami_multispk_si_v1/session_audio/zh_mix_v8"
UTTS = ROOT / "data/ami_multispk_si_v1/zh_mix_v8/utts"
SPEECH = ROOT / "eval_outputs/mist_w3po_v1/speech"
TEMPLATE = Path(__file__).with_name("ami_stream_demo_template.html")
SR = 16000
TAIL_S = 8.0          # translation allowed to run past the excerpt end
TARGET_RMS = 0.07     # active-speech RMS both tracks are brought to
BED_GAIN = 0.28       # original under the translation in the simultaneous mix


def active_rms(x: np.ndarray) -> float:
    a = np.abs(x)
    thr = max(1e-4, 0.05 * float(np.percentile(a, 99.5)) if a.size else 1e-4)
    v = x[a > thr]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


def norm(x: np.ndarray) -> np.ndarray:
    r = active_rms(x)
    return x * (TARGET_RMS / r) if r > 0 else x


def limit(x: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > peak else x


def write_mp3(path: Path, x: np.ndarray):
    sf.write(str(path), x.astype(np.float32), SR, format="MP3")


def build_meeting(meeting: str, system: str, dur: float, wins: list[dict], streams: dict, out: Path) -> dict:
    ws = sorted((w for w in wins if w["meeting"] == meeting), key=lambda w: w["start_s"])
    t0 = max(0.0, round(ws[0]["start_s"] - 1.0, 2))
    t1 = t0 + dur
    ws = [w for w in ws if w["start_s"] < t1]
    sess, sr = sf.read(str(SESSION / f"{meeting}.wav"), dtype="float32")
    assert sr == SR, sr
    n = int(round((dur + TAIL_S) * SR))
    en = np.zeros(n, np.float32)
    en_lines, n_streams = [], 0
    for w in ws:
        off = w["start_s"] - t0
        for r in streams.get(w["id"], []):
            y, sr = sf.read(r["audio_path"], dtype="float32")
            assert sr == SR, sr
            i = int(round(off * SR))
            j = min(n, i + len(y))
            if j > i:
                en[i:j] += y[: j - i]
            n_streams += 1
            k = r["id"].rsplit("__S", 1)[1]
            for b in r.get("blocks") or []:
                s = off + b["start_ms"] / 1000.0
                if s < dur + TAIL_S and (b.get("text") or "").strip():
                    en_lines.append({"t": round(s, 2), "e": round(s + b["dur_ms"] / 1000.0, 2), "spk": int(k),
                                     "text": b["text"].strip()})
    last = max([l["e"] for l in en_lines] + [dur])
    n = min(n, int(round((min(last + 0.5, dur + TAIL_S)) * SR)))
    en = en[:n]
    i0 = int(round(t0 * SR))
    zh = sess[i0:i0 + n].copy()
    if len(zh) < n:
        zh = np.pad(zh, (0, n - len(zh)))
    # Chinese lines: the redub's placed utterances, speakers lettered by first appearance
    zh_lines, letters = [], {}
    for l in open(UTTS / f"{meeting}.jsonl"):
        u = json.loads(l)
        if not u.get("placed") or u.get("zh_begin") is None:
            continue
        s, e = u["zh_begin"] - t0, u["zh_end"] - t0
        if e <= 0 or s >= n / SR:
            continue
        zh_lines.append({"t": round(max(0.0, s), 2), "e": round(e, 2), "who": u["speaker_id"], "text": u["zh"]})
    zh_lines.sort(key=lambda x: x["t"])
    for z in zh_lines:
        z["who"] = letters.setdefault(z["who"], chr(ord("A") + len(letters)))
    en_lines.sort(key=lambda x: x["t"])
    zh_n, en_n = norm(zh), norm(en)
    files = {}
    d = out / "audio"
    tracks = {
        "mix": limit(BED_GAIN * zh_n + en_n),
        "stereo": limit(np.stack([zh_n, en_n], 1)),
        "en": limit(en_n),
        "zh": limit(zh_n),
    }
    for key, x in tracks.items():
        p = d / f"{meeting}_{key}.mp3"
        write_mp3(p, x)
        files[key] = f"audio/{p.name}"
    return {"meeting": meeting, "t0": t0, "dur": round(n / SR, 2), "windows": len(ws), "streams": n_streams,
            "speakers": len(letters), "files": files, "zh": zh_lines, "en": en_lines}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="ft_ami_aprime_s3000__rlplem")
    ap.add_argument("--meetings", nargs="+", default=["EN2002a", "ES2004a", "IS1009a", "TS3003a"])
    ap.add_argument("--dur", type=float, default=240.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    zp = out.with_suffix(".zip")
    if out.exists() or zp.exists():
        sys.exit(f"refusing: {out} or {zp} already exists")
    wins = [json.loads(l) for l in open(WIN)]
    streams = defaultdict(list)
    for l in open(SPEECH / a.system / "talker" / "results.jsonl"):
        r = json.loads(l)
        if r.get("error") is None and r.get("audio_path"):
            streams[r["id"].rsplit("__S", 1)[0]].append(r)
    (out / "audio").mkdir(parents=True)
    data = [build_meeting(m, a.system, a.dur, wins, streams, out) for m in a.meetings]
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "README.txt").write_text(
        "AMI 多人会议流式同传试听（离线版）\n\n"
        "解压后双击 index.html，用 Chrome / Edge / Firefox / Safari 打开即可试听，不需要联网。\n"
        "audio/ 文件夹要和 index.html 放在同一目录。\n"
        "每场会议有四个版本：同传混音（原声压低、译音在上）、左右声道（左原声、右译音，建议戴耳机）、只听译音、只听原声。\n"
        "点击字幕或时间条可以跳到对应位置。\n", encoding="utf-8")
    meta = {"system": a.system, "meetings": a.meetings, "dur": a.dur,
            "per_meeting": [{k: d[k] for k in ("meeting", "t0", "dur", "windows", "streams", "speakers")} for d in data]}
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.make_archive(str(out), "zip", root_dir=str(out.parent), base_dir=out.name)
    print(json.dumps(meta, ensure_ascii=False))
    print(zp)


if __name__ == "__main__":
    main()
