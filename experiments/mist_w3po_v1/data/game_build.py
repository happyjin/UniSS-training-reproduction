"""Game (HOK esports commentary) listening demo (reports/mist_w3po_v1/game_multispk_demo_plan.zh-CN.md §3-§4).

  data     the selected 5-min excerpts (select_<dir>.json) cut from the ORIGINAL part audio, 16 kHz mono
           -> data/game_esports_si_v1/<dir>/session_audio/<sess>/<m>(.Mix-Headset).wav, meetings.json, and the SILVER
           reference windows refs_windows.jsonl: one window per pipeline turn fully inside the excerpt (turn times
           mapped to the part timeline by the clip's source_mapping), utts = [that turn], times relative to the window,
           "ref" = the pipeline's GPT translation, speaker = clip-local "<segment>:<S>" (no cross-clip identity).
           zh2en: sess "mix" (<m>.wav); en2zh: sess "ihm_mix" (<m>.Mix-Headset.wav), as gstar_talker_v6 expects.
  windows  build_test_windows_c2's C2 online windows (build_windows_zh2en.online_windows(post, 0.6): open at the first
           frame with any slot > 0.5, close after 0.6 s of no activity or at 20 s; audio [open - 0.2, last + 0.3];
           detection times with the 1.04 s look-ahead) on the lowlat Nemotron posteriors diar/nemotron_lowlat
           -> eval_outputs/mist_w3po_v1/data/windows_game_<dir>_c2g06_v1/windows_test_online_<dir>.jsonl
Meeting names gz01..gz05 (zh2en) / ge01..ge05 (en2zh): no "_" or "-" in ids. Refuses existing outputs. CPU only.

    python -m experiments.mist_w3po_v1.data.game_build data|windows --dir zh2en|en2zh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
GD = REPO / "data/game_esports_si_v1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from game_select import META, load_video  # noqa: E402

SR = 16000


def x_open(p: Path):
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, "x", encoding="utf-8")


def sess_path(d: str, m: str) -> Path:
    return GD / d / "session_audio" / ("mix" if d == "zh2en" else "ihm_mix") / (f"{m}.wav" if d == "zh2en" else f"{m}.Mix-Headset.wav")


def load_16k(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def build_data(d: str) -> None:
    sel = json.load(open(GD / f"select_{d}.json"))
    dur = float(sel["dur"])
    pre = "gz" if d == "zh2en" else "ge"
    meetings, refs = {}, []
    for k, s in enumerate(sel["selected"], 1):
        m = f"{pre}{k:02d}"
        part = GD / "src_parts" / d / f"{s['video']}_part{s['part']:02d}.wav"
        x = load_16k(part)
        t0 = float(s["t0"])
        seg = x[round(t0 * SR): round((t0 + dur) * SR)]
        out = sess_path(d, m)
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), seg, SR)
        P = load_video(META / d / s["video"])[int(s["part"])]
        turns = sorted((t for t in P["turns"] if t["b"] >= t0 - 0.05 and t["e"] <= t0 + dur + 0.05 and t["e"] > t["b"]),
                       key=lambda t: t["b"])
        for j, t in enumerate(turns):
            b, e = max(0.0, t["b"] - t0), min(dur, t["e"] - t0)
            u = {"record_id": f"{m}_{j:04d}", "speaker_id": t["spk"], "begin_s": 0.0, "end_s": round(e - b, 3),
                 "hesitation": False, "human": False}
            if d == "zh2en":
                u.update(zh=t["src"], en=t["tgt"], ref=t["tgt"], ref_mt=t["tgt"])
            else:
                u.update(en=t["src"], zh=t["tgt"], ref=t["tgt"])
            refs.append({"id": f"{m}_ref_{j:04d}", "meeting": m, "split": "test", "wtype": "ref", "dir": d,
                         "start_s": round(b, 3), "dur_ms": round((e - b) * 1000), "audio": str(out),
                         "strict_review": False, "utts": [u]})
        meetings[m] = {"video": s["video"], "part": s["part"], "t0": t0, "dur": dur, "title": s.get("title", ""),
                       "url": f"https://www.youtube.com/watch?v={s['video']}", "turns": len(turns),
                       "spk_changes": s["spk_changes"], "speech_cov": s["speech_cov"]}
        print(m, s["video"], s["part"], t0, "turns", len(turns), flush=True)
    with x_open(GD / d / "meetings.json") as fh:
        json.dump(meetings, fh, ensure_ascii=False, indent=1)
    with x_open(GD / d / "refs_windows.jsonl") as fh:
        for r in refs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_windows(d: str) -> None:
    import torch
    from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, LOOKAHEAD, online_windows
    out = REPO / f"eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1"
    wf = out / f"windows_test_online_{d}.jsonl"
    for sub in ("audio", "post"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meetings = json.load(open(GD / d / "meetings.json"))
    n = 0
    with x_open(wf) as fh:
        for m in sorted(meetings):
            post = torch.load(GD / d / "diar/nemotron_lowlat" / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
            x, sr = sf.read(str(sess_path(d, m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            sess_s = len(x) / sr
            for k, (st, en, j) in enumerate(online_windows(post, 0.6)):
                ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
                close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
                wid = f"{m}_on_{k:04d}"
                wav, pp = out / "audio" / f"{wid}.wav", out / "post" / f"{wid}.npy"
                sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
                s = post[f0: f0 + nf]
                if len(s) < nf:
                    s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                np.save(pp, s.astype(np.float16))
                fh.write(json.dumps({"id": wid, "meeting": m, "split": "test", "wtype": "online", "dir": d,
                                     "start_s": round(ws, 3), "dur_ms": round((we - ws) * 1000),
                                     "audio": str(wav.resolve()), "post": {"lowlat": str(pp.resolve())},
                                     "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                                     "close_detect_ms": round((close - ws) * 1000), "utts": [],
                                     "strict_review": False}) + "\n")
                n += 1
            print(m, "windows so far", n, flush=True)
    print("windows", n, "->", wf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["data", "windows"])
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    a = ap.parse_args()
    (build_data if a.step == "data" else build_windows)(a.dir)


if __name__ == "__main__":
    main()
