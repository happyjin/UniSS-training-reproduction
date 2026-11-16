"""Listening material for the pause problem (2026-11-16, user request; dev meetings only).

For one direction: B (deadline Thinker + base Talker, __t1c) and MIST (B's text + rl + PA Talker, __rlplem), TTS seed 1,
and MIST re-timed with a 0.5 s playout buffer at the first block of every sentence (pause_buffer_explore "sent").
Each stream's speech-only wav is cut into its blocks (dur_ms) and laid on the meeting timeline at
window start + start_ms; streams of all speakers are summed (per-speaker layout). Clips:
  typical  a 60 s excerpt whose MIST mid-sentence pause rate is closest to the dev-wide rate (>= 30 s of speech)
  severe   the 60 s excerpt with the highest MIST mid-sentence pause rate (>= 30 s of speech)
  closeup  the single window stream with the most MIST mid-sentence pauses while B has at most one
Files per clip: source.wav, {B,MIST,MIST_buffer0.5}_interp.wav (translation only) and _mix.wav (source at -14 dB +
translation). Writes into a new directory (refuses an existing one) with clips.json and a block transcript.

    python -m experiments.mist_w3po_v1.evaluation.pause_listening_demo --dir bz|be --out <new dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.mist_w3po_v1.evaluation import v11_metrics as V
from experiments.mist_w3po_v1.evaluation.pause_buffer_explore import schedule
from experiments.mist_w3po_v1.talker.tspa_precheck_p1 import sent_end

SR = 16000
REPO = Path(__file__).resolve().parents[3]
TXT = {"bz": ("jdev_bz_joint_v8h6k_20261113T073658Z_s06000", "devz"),
       "be": ("jdev_be_joint_v8h6k_20261113T073658Z_s06000", "deve")}
SPAN = 60.0
TAIL = 6.0
SRC_DB = -14.0


def load_run(run: str) -> dict:
    out = {}
    for line in open(REPO / f"eval_outputs/mist_w3po_v1/speech/{run}/talker/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") or not r.get("blocks"):
            continue
        out[r["id"]] = r
    return out


def blocks_audio(r: dict) -> list[np.ndarray]:
    x, sr = sf.read(r["speech_path"], dtype="float32")
    assert sr == SR
    segs, o = [], 0
    for b in r["blocks"]:
        n = int(round(b["dur_ms"] * SR / 1000))
        segs.append(x[o:o + n])
        o += n
    return segs


def gaps(r: dict, starts=None) -> tuple[int, int, float]:
    bl = r["blocks"]
    st = starts or [float(b["start_ms"]) for b in bl]
    p = pm = 0
    for n in range(1, len(bl)):
        g = st[n] - (st[n - 1] + float(bl[n - 1]["dur_ms"]))
        if g >= 500:
            p += 1
            pm += not sent_end(bl[n - 1].get("text", ""))
    return p, pm, sum(float(b["dur_ms"]) for b in bl) / 1000.0


def render(streams: list[tuple[float, dict, list | None]], t0: float, t1: float) -> np.ndarray:
    y = np.zeros(int((t1 - t0) * SR) + 1, dtype=np.float32)
    for ws, r, starts in streams:
        st = starts or [float(b["start_ms"]) for b in r["blocks"]]
        for s, seg in zip(st, blocks_audio(r)):
            i = int((ws + s / 1000.0 - t0) * SR)
            if 0 <= i < len(y):
                j = min(len(y), i + len(seg))
                y[i:j] += seg[:j - i]
    return y


def render_src(wins: list[dict], t0: float, t1: float) -> np.ndarray:
    y = np.zeros(int((t1 - t0) * SR) + 1, dtype=np.float32)
    for w in wins:
        x, sr = sf.read(w["audio"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        i = int((w["start_s"] - t0) * SR)
        if 0 <= i < len(y):
            j = min(len(y), i + len(x))
            y[i:j] += x[:j - i]
    return y


def norm(y: np.ndarray) -> np.ndarray:
    m = float(np.abs(y).max()) or 1.0
    return (y / m * 10 ** (-1 / 20)).astype(np.float32)


def transcript(r: dict, starts=None) -> str:
    bl = r["blocks"]
    st = starts or [float(b["start_ms"]) for b in bl]
    out = []
    for n, b in enumerate(bl):
        if n:
            g = st[n] - (st[n - 1] + float(bl[n - 1]["dur_ms"]))
            if g >= 500:
                out.append(f"【停 {g / 1000:.1f}s】")
        out.append(b.get("text", ""))
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", choices=("bz", "be"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    text, s = TXT[a.dir]
    W = {w["id"]: w for w in map(json.loads, open(REPO / V.WIN[s], encoding="utf-8"))}
    B, M = load_run(text + "__t1c"), load_run(text + "__rlplem")
    ids = sorted(set(B) & set(M), key=lambda i: (W[i.split("__S")[0]]["meeting"], W[i.split("__S")[0]]["start_s"]))
    bufd = {i: schedule(M[i]["blocks"], "sent", 500.0) for i in ids}
    # dev-wide MIST mid-sentence rate
    tot = np.array([gaps(M[i])[1:] for i in ids]).sum(0)
    rate_all = tot[0] / (tot[1] / 60)
    # 60 s excerpts starting at each window
    by_m: dict = {}
    for i in ids:
        by_m.setdefault(W[i.split("__S")[0]]["meeting"], []).append(i)
    cands = []
    for m, its in by_m.items():
        starts = sorted({W[i.split("__S")[0]]["start_s"] for i in its})
        for t0 in starts:
            sel = [i for i in its if t0 <= W[i.split("__S")[0]]["start_s"] < t0 + SPAN]
            pm = sum(gaps(M[i])[1] for i in sel)
            sp = sum(gaps(M[i])[2] for i in sel)
            if sp >= 30:
                cands.append((m, t0, pm / (sp / 60), sel))
    typical = min(cands, key=lambda c: abs(c[2] - rate_all))
    severe = max(cands, key=lambda c: c[2])
    close = max((i for i in ids if gaps(B[i])[1] <= 1 and 8000 <= W[i.split("__S")[0]]["dur_ms"] <= 20000),
                key=lambda i: (gaps(M[i])[1], -gaps(B[i])[1]))
    clips = {"typical": typical[3], "severe": severe[3], "closeup": [close]}
    info = {"dir": a.dir, "text_run": text, "dev_mist_mid_pauses_per_min": round(rate_all, 2), "clips": {}}
    lines = []
    for name, sel in clips.items():
        wins = sorted({i.split("__S")[0] for i in sel}, key=lambda w: W[w]["start_s"])
        t0 = W[wins[0]]["start_s"]
        t1 = max(W[w]["start_s"] + W[w]["dur_ms"] / 1000 for w in wins) + TAIL
        d = out / name
        d.mkdir()
        src = render_src([W[w] for w in wins], t0, t1)
        sf.write(d / "source.wav", norm(src), SR)
        stat = {}
        for tag, runs, starts in (("B", B, None), ("MIST", M, None), ("MIST_buffer0.5", M, bufd)):
            ss = [(W[i.split("__S")[0]]["start_s"], runs[i], starts[i] if starts else None) for i in sel]
            y = render(ss, t0, t1)
            sf.write(d / f"{tag}_interp.wav", norm(y), SR)
            sf.write(d / f"{tag}_mix.wav", norm(norm(y) + 10 ** (SRC_DB / 20) * norm(src)), SR)
            g = np.array([gaps(runs[i], starts[i] if starts else None) for i in sel]).sum(0)
            stat[tag] = {"pauses": int(g[0]), "mid_pauses": int(g[1]), "speech_s": round(float(g[2]), 1),
                         "pauses_per_min_speech": round(g[0] / (g[2] / 60), 1),
                         "mid_per_min_speech": round(g[1] / (g[2] / 60), 1)}
        info["clips"][name] = {"meeting": W[wins[0]]["meeting"], "t0_s": round(t0, 2), "dur_s": round(t1 - t0, 1),
                               "windows": len(wins), "streams": len(sel), "stats": stat}
        lines.append(f"## {name}（{W[wins[0]]['meeting']}，会议第 {t0:.0f} 秒起，{t1 - t0:.0f} 秒）\n")
        for i in sel[:12]:
            lines.append(f"- `{i}`\n  - B：{transcript(B[i])}\n  - MIST：{transcript(M[i])}\n"
                         f"  - MIST+缓冲：{transcript(M[i], bufd[i])}\n")
    (out / "clips.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    (out / "transcript.zh-CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
