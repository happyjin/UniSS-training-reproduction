"""S1 (reports/mist_w3po_v1/talker_silence_s1_prereg.zh-CN.md): silence inside the talker's blocks, and the re-timed upper
bound with every silence run compressed to <= 150 ms. CPU only.

Block audio: results.jsonl's speech_path (<id>_speech.wav, 16 kHz) is the stream's blocks concatenated in order, block
k at [sum_{j<k} dur_j, sum_{j<=k} dur_j) ms. Frames of 10 ms (160 samples, no overlap), energy 20 log10(RMS + 1e-10);
a frame is silent below (block peak frame - 40 dB); a silence run = a maximal run of silent frames >= 150 ms, kind
"lead" (starts at the block's first frame), "trail" (ends at its last frame), "internal", or "whole" (the block is
silent throughout).

measure  -> json: per kind total seconds, share of the total speech time (sum of dur_ms), run counts / median / p90,
            share of blocks with such a run; per block runs (compact) for retime
retime   -> <out-dir>/results.jsonl: every block's dur_ms := dur_ms - sum over runs of max(0, run - 150 ms) (dur_ms_orig
            kept); everything else verbatim -- speech_latency_lit then re-times the queue unchanged
Both refuse to overwrite.

    python -m experiments.mist_w3po_v1.talker.silence_share measure --talker <talker dir> --out <json>
    python -m experiments.mist_w3po_v1.talker.silence_share retime --talker <talker dir> --measure <json> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16000
FRAME_MS = 10
REL_DB = 40.0
MIN_RUN_MS = 150
CAP_MS = 150
KINDS = ("lead", "trail", "internal", "whole")


def frame_db(x: np.ndarray, sr: int = SR, frame_ms: int = FRAME_MS) -> np.ndarray:
    n = sr * frame_ms // 1000
    m = len(x) // n
    if m == 0:
        return np.zeros(0)
    f = np.asarray(x[: m * n], dtype=np.float64).reshape(m, n)
    return 20.0 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-10)


def silence_runs(x: np.ndarray, sr: int = SR, frame_ms: int = FRAME_MS, rel_db: float = REL_DB,
                 min_run_ms: int = MIN_RUN_MS) -> list[tuple[int, int, str]]:
    """-> [(start_ms, end_ms, kind)] of the block's silence runs (block-relative ms)."""
    e = frame_db(x, sr, frame_ms)
    if not len(e):
        return []
    sil = e < e.max() - rel_db   # never all True (the peak frame itself); all-silent blocks: is_whole_silent
    out, i, n = [], 0, len(e)
    while i < n:
        if not sil[i]:
            i += 1
            continue
        j = i
        while j < n and sil[j]:
            j += 1
        if (j - i) * frame_ms >= min_run_ms:
            kind = "lead" if i == 0 else ("trail" if j == n else "internal")
            out.append((i * frame_ms, j * frame_ms, kind))
        i = j
    return out


def is_whole_silent(x: np.ndarray, floor_db: float = -80.0) -> bool:
    """an all-silent block (every frame below -80 dBFS): its peak-relative test cannot fire, so it is caught here."""
    e = frame_db(x)
    return bool(len(e)) and bool((e < floor_db).all())


def block_runs(x: np.ndarray) -> list[tuple[int, int, str]]:
    if is_whole_silent(x):
        return [(0, len(frame_db(x)) * FRAME_MS, "whole")]
    return silence_runs(x)


def new_dur(dur_ms: int, runs: list, cap_ms: int = CAP_MS) -> int:
    return int(dur_ms - sum(max(0, (b - a) - cap_ms) for a, b, _ in runs))


def streams(talker: str):
    for l in open(Path(talker) / "results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("error") or not r.get("blocks"):
            continue
        yield r


def measure(talker: str) -> dict:
    import soundfile as sf

    tot_ms, per = 0, {k: [] for k in KINDS}
    blocks_with = {k: 0 for k in KINDS}
    nblk, runs_all, bad = 0, {}, 0
    for r in streams(talker):
        x, sr = sf.read(r["speech_path"], dtype="float32")
        assert sr == SR, (r["id"], sr)
        if x.ndim > 1:
            x = x.mean(1)
        c, rr = 0, []
        for b in r["blocks"]:
            s0, s1 = c * SR // 1000, (c + b["dur_ms"]) * SR // 1000
            c += b["dur_ms"]
            seg = x[s0:s1]
            if len(seg) < s1 - s0:
                bad += 1
            runs = block_runs(seg)
            rr.append([[a, e, k] for a, e, k in runs])
            for k in {k for _, _, k in runs}:
                blocks_with[k] += 1
            for a, e, k in runs:
                per[k].append(e - a)
            tot_ms += b["dur_ms"]
            nblk += 1
        runs_all[r["id"]] = rr
    q = lambda v: {"n": len(v), "total_s": round(sum(v) / 1000, 2), "share": round(sum(v) / max(tot_ms, 1), 4),
                   "median_ms": float(np.median(v)) if v else None, "p90_ms": float(np.percentile(v, 90)) if v else None,
                   "block_share": None}
    res = {"talker": talker, "blocks": nblk, "speech_s": round(tot_ms / 1000, 2), "short_audio_blocks": bad,
           "def": {"frame_ms": FRAME_MS, "rel_db": REL_DB, "min_run_ms": MIN_RUN_MS, "cap_ms": CAP_MS}, "kinds": {}}
    for k in KINDS:
        res["kinds"][k] = q(per[k])
        res["kinds"][k]["block_share"] = round(blocks_with[k] / max(nblk, 1), 4)
    allv = [v for k in KINDS for v in per[k]]
    res["all"] = {"total_s": round(sum(allv) / 1000, 2), "share": round(sum(allv) / max(tot_ms, 1), 4), "n": len(allv)}
    excess = sum(max(0, v - CAP_MS) for v in allv)
    res["compressible_s"] = round(excess / 1000, 2)
    res["compressible_share"] = round(excess / max(tot_ms, 1), 4)
    res["runs"] = runs_all
    return res


def retime(talker: str, meas: dict, out_dir: str) -> dict:
    out = Path(out_dir) / "results.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    n, cut = 0, 0
    with open(out, "x", encoding="utf-8") as fh:
        for l in open(Path(talker) / "results.jsonl", encoding="utf-8"):
            r = json.loads(l)
            rr = meas["runs"].get(r["id"]) if not r.get("error") and r.get("blocks") else None
            if rr is not None:
                assert len(rr) == len(r["blocks"]), r["id"]
                for b, runs in zip(r["blocks"], rr):
                    d = new_dur(b["dur_ms"], runs)
                    b["dur_ms_orig"] = b["dur_ms"]
                    cut += b["dur_ms"] - d
                    b["dur_ms"] = d
                    n += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"blocks": n, "cut_s": round(cut / 1000, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--talker", required=True)
    m.add_argument("--out", required=True)
    r = sub.add_parser("retime")
    r.add_argument("--talker", required=True)
    r.add_argument("--measure", required=True)
    r.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "measure":
        if Path(a.out).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
        res = measure(a.talker)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "x", encoding="utf-8") as fh:
            json.dump(res, fh)
        print(json.dumps({k: v for k, v in res.items() if k != "runs"}, ensure_ascii=False))
    else:
        print(json.dumps(retime(a.talker, json.load(open(a.measure)), a.out_dir)))


if __name__ == "__main__":
    main()
