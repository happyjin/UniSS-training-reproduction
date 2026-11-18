"""Idea 1 (ZNE) open diagnostic set (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §2): per language 1000
pairs = 4 background categories (music / crowd / sfx / mixed) x 5 BVR bins (quintiles of the esports prompt BVR,
a statistic only) x 50; per (category, bin) cell 25 dev + 25 test. Each pair: 6 s clean speech s (one speaker, trimmed
utterances concatenated), true background b scaled to the target BVR = 10 log10(sum b^2 / sum s^2), mixture y = s + b
(s, b scaled together if peak > 0.99), HDemucs vocals vhat (game_muffle_analysis.demucs, CPU) and the separator fit
bh = y - vhat = c b + kappa s + e (least squares on the waveform).

Speech  en: LibriSpeech dev-clean -> dev, test-clean -> test; zh: AISHELL-3 test speakers, sorted by sha1(speaker id),
        first half -> dev, second half -> test.
Background files split dev / test by sha1(file name) parity; files listed in data/open_voicecond_v1/bench/
HELDOUT_BACKGROUNDS.txt / HELDOUT_BACKGROUNDS_MUSAN.txt (OpenBroadcast-SI) and every file matching the benchmark's
own hold-out rule are excluded (both lists must exist). FMA / DEMAND / ESC-50 (OOD-1) and CoVoST 2 are never used.
  music  MUSAN music hd-classical / jamendo / rfm (the Free-Music-Archive subsets fma, fma-western-art are skipped:
         FMA is an OOD-1 corpus), ANNOTATIONS vocals = N, random 6 s with RMS >= 0.1 x track RMS
  crowd  FSD50K eval, labels contain Crowd / Cheering / Applause and none of the speech / singing / music labels
  sfx    FSD50K eval, labels contain Explosion / Gunshot_and_gunfire / Whoosh_and_swoosh_and_swish / Boom / Fireworks /
         Crack and none of those labels; crowd / sfx: random clips of the category concatenated to 6 s
  mixed  b = m + 0.7 c + 0.7 x (each component first normalised to unit RMS)
Writes float32 16 kHz wavs <out>/wav/<id>_{s,b,y,vhat}.wav and <out>/manifest.jsonl. Refuses an existing --out.

    conda_envs/uniss-eval/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_build_diag --lang en|zh --out <dir>
        --speech-dev <dir> --speech-test <dir> | --speech-zh <AISHELL-3 test wav dir>
        --musan <musan/music dir> --fsd-audio <FSD50K.eval_audio dir> --fsd-gt <eval.csv> [--procs 40]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
SR, N6 = 16000, 96000
SEED = 20261119
CATS = ("music", "crowd", "sfx", "mixed")
NBIN, PER_CELL = 5, 50
HELDOUT = REPO / "data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS.txt"
CROWD = {"Crowd", "Cheering", "Applause"}
SFX = {"Explosion", "Gunshot_and_gunfire", "Whoosh_and_swoosh_and_swish", "Boom", "Fireworks", "Crack"}
BAN = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
       "Conversation", "Singing", "Music"}


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def read16(p) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def trim(x: np.ndarray, db: float = 25.0) -> np.ndarray:
    fr, hop = 400, 160
    if len(x) < fr:
        return x
    n = 1 + (len(x) - fr) // hop
    e = np.array([10 * np.log10((x[i * hop:i * hop + fr] ** 2).mean() + 1e-12) for i in range(n)])
    keep = np.where(e > e.max() - db)[0]
    a = max(0, keep[0] * hop - 800)
    b = min(len(x), keep[-1] * hop + fr + 800)
    return x[a:b]


def esports_bvr() -> np.ndarray:
    v = []
    for d in ("zh2en", "en2zh"):
        for r in map(json.loads, open(REPO / f"eval_outputs/mist_w3po_v1/game_muffle_v1/game_{d}.jsonl")):
            b = (r.get("prompt") or {}).get("bvr_db")
            if b is not None:
                v.append(float(b))
    return np.array(v)


def heldout() -> set:
    """stems listed in HELDOUT_BACKGROUNDS.txt (FSD50K) and HELDOUT_BACKGROUNDS_MUSAN.txt; both must exist"""
    out = set()
    for f in (HELDOUT, HELDOUT.with_name("HELDOUT_BACKGROUNDS_MUSAN.txt")):
        if not f.exists():
            raise SystemExit(f"{f} missing: backgrounds are drawn only after the held-out lists exist")
        for line in open(f, encoding="utf-8"):
            line = line.strip().split("\t")[0].strip()
            if line and not line.startswith("#"):
                out.add(Path(line).stem)
    return out


def bench_rule(member: str) -> bool:
    """the benchmark's own rule (HELDOUT_BACKGROUNDS.txt header), applied as a second check"""
    return int(hashlib.sha1(("OpenBroadcast-SI-v1|" + member).encode()).hexdigest(), 16) % 1000 < 150


def member_path(p: Path) -> str:
    parts = p.parts
    if "musan" in parts:
        return "/".join(parts[parts.index("musan"):])          # musan/music/<subset>/<file>.wav (tar member)
    return f"FSD50K.eval_audio/{p.name}"


def bg_pools(a, held: set) -> dict:
    """{split: {cat: [paths]}} (music, crowd, sfx)"""
    pools = {"dev": {c: [] for c in CATS[:3]}, "test": {c: [] for c in CATS[:3]}}
    n_held = 0
    music = []
    for ann in glob.glob(os.path.join(a.musan, "*", "ANNOTATIONS")):
        if Path(ann).parent.name in ("fma", "fma-western-art"):   # Free Music Archive: FMA is an OOD-1 corpus
            continue
        for line in open(ann, encoding="utf-8", errors="replace"):
            f = line.split()
            if len(f) >= 3 and f[2] == "N":
                p = Path(ann).parent / f"{f[0]}.wav"
                if p.exists():
                    music.append(p)
    rows = list(csv.reader(open(a.fsd_gt, encoding="utf-8")))[1:]
    crowd, sfx = [], []
    for fname, labels, _ in rows:
        L = set(labels.split(","))
        if L & BAN:
            continue
        p = Path(a.fsd_audio) / f"{fname}.wav"
        if L & CROWD:
            crowd.append(p)
        elif L & SFX:
            sfx.append(p)
    for cat, lst in (("music", music), ("crowd", crowd), ("sfx", sfx)):
        for p in sorted(lst):
            if p.stem in held or bench_rule(member_path(p)):
                n_held += 1
                continue
            pools["dev" if h(p.name) % 2 == 0 else "test"][cat].append(p)
    print(json.dumps({"held_out_excluded": n_held, **{f"{s}_{c}": len(v) for s in pools for c, v in pools[s].items()}}))
    return pools


def speakers(a, lang: str) -> dict:
    """{split: {spk: [utt paths]}}"""
    out = {"dev": {}, "test": {}}
    if lang == "en":
        for split, root in (("dev", a.speech_dev), ("test", a.speech_test)):
            for spk in sorted(os.listdir(root)):
                fs = sorted(glob.glob(os.path.join(root, spk, "*", "*.flac")))
                if fs:
                    out[split][spk] = fs
    else:
        spks = sorted(os.listdir(a.speech_zh), key=lambda s: h(s))
        half = len(spks) // 2
        for i, spk in enumerate(spks):
            fs = sorted(glob.glob(os.path.join(a.speech_zh, spk, "*.wav")))
            if fs:
                out["dev" if i < half else "test"][spk] = fs
    return out


def take6(rng, paths, pick_offset=True) -> tuple[np.ndarray, list]:
    """concatenate random clips until >= 6 s (random start in the first); returns (x, [(file, start)])"""
    parts, used, n = [], [], 0
    first = True
    while n < N6:
        p = paths[rng.integers(len(paths))]
        x = read16(p)
        if len(x) < 400:
            continue
        st = int(rng.integers(0, max(1, len(x) - SR))) if (first and pick_offset and len(x) > N6) else 0
        x = x[st:]
        first = False
        parts.append(x)
        used.append((Path(p).name, st))
        n += len(x)
    return np.concatenate(parts)[:N6], used


def music6(rng, paths) -> tuple[np.ndarray, list]:
    for _ in range(20):
        p = paths[rng.integers(len(paths))]
        info = sf.info(str(p))
        dur = info.frames / info.samplerate
        if dur < 7:
            continue
        st = float(rng.uniform(0, dur - 6.5))
        x, sr = sf.read(str(p), start=int(st * info.samplerate), frames=int(6.2 * info.samplerate), dtype="float32",
                        always_2d=True)
        x = x.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            g = np.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x = x[:N6]
        if len(x) < N6:
            continue
        full = read16(p)
        if np.sqrt((x ** 2).mean()) >= 0.1 * np.sqrt((full ** 2).mean()):
            return x, [(p.name, round(st, 2))]
    raise RuntimeError("no usable music segment")


def unit(x):
    return x / (np.sqrt((x ** 2).mean()) + 1e-9)


def speech6(rng, utts: list, used: set) -> tuple[np.ndarray, list]:
    order = [u for u in rng.permutation(len(utts)) if utts[u] not in used] + list(rng.permutation(len(utts)))
    parts, names, n = [], [], 0
    for k in order:
        x = trim(read16(utts[k]))
        if len(x) < SR // 2:
            continue
        parts.append(x)
        names.append(Path(utts[k]).name)
        used.add(utts[k])
        n += len(x)
        if n >= N6:
            break
    s = np.concatenate(parts)
    if len(s) < N6:
        s = np.tile(s, 1 + N6 // len(s))
    return s[:N6], names


def separate(job):
    wd, pid = job
    from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import demucs
    s = sf.read(f"{wd}/{pid}_s.wav", dtype="float32")[0]
    b = sf.read(f"{wd}/{pid}_b.wav", dtype="float32")[0]
    y = sf.read(f"{wd}/{pid}_y.wav", dtype="float32")[0]
    bvr_h, v = demucs(y)
    m = min(len(v), len(y))
    vh = np.zeros_like(y)
    vh[:m] = v[:m]
    sf.write(f"{wd}/{pid}_vhat.wav", vh, SR, subtype="FLOAT")
    bh = y - vh
    X = np.stack([b, s], 1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(X, bh.astype(np.float64), rcond=None)
    e = bh - X @ coef
    return pid, {"bvr_hdemucs": round(bvr_h, 3), "c": float(coef[0]), "kappa": float(coef[1]),
                 "e_rel": float(np.linalg.norm(e) / (np.linalg.norm(bh) + 1e-12))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["en", "zh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--speech-dev", default="")
    ap.add_argument("--speech-test", default="")
    ap.add_argument("--speech-zh", default="")
    ap.add_argument("--musan", required=True)
    ap.add_argument("--fsd-audio", required=True)
    ap.add_argument("--fsd-gt", required=True)
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = np.random.default_rng(SEED + (0 if a.lang == "en" else 1))
    held = heldout()
    pools = bg_pools(a, held)
    spk = speakers(a, a.lang)
    bv = esports_bvr()
    edges = np.percentile(bv, np.linspace(0, 100, NBIN + 1))
    wd = out / "wav"
    wd.mkdir(parents=True)
    rows = []
    for split in ("dev", "test"):
        names = sorted(spk[split])
        used = {s: set() for s in names}
        k = 0
        for cat in CATS:
            for bi in range(NBIN):
                lo, hi = edges[bi], edges[bi + 1]
                vals = bv[(bv >= lo) & (bv <= hi)]
                for j in range(PER_CELL // 2):
                    sp = names[k % len(names)]
                    k += 1
                    s, utts = speech6(rng, spk[split][sp], used[sp])
                    P = pools[split]
                    if cat == "music":
                        b, src = music6(rng, P["music"])
                    elif cat in ("crowd", "sfx"):
                        b, src = take6(rng, P[cat])
                    else:
                        m, s1 = music6(rng, P["music"])
                        c, s2 = take6(rng, P["crowd"])
                        x, s3 = take6(rng, P["sfx"])
                        b, src = unit(m) + 0.7 * unit(c) + 0.7 * unit(x), s1 + s2 + s3
                    target = float(vals[rng.integers(len(vals))])
                    g = np.sqrt((s ** 2).sum() / ((b ** 2).sum() + 1e-12) * 10 ** (target / 10))
                    b = (g * b).astype(np.float32)
                    y = s + b
                    pk = float(np.abs(y).max())
                    if pk > 0.99:
                        s, b = s * (0.99 / pk), b * (0.99 / pk)
                        y = s + b
                    pid = f"{a.lang}_{split}_{cat}_b{bi}_{j:02d}"
                    for nm, w in (("s", s), ("b", b), ("y", y)):
                        sf.write(str(wd / f"{pid}_{nm}.wav"), w.astype(np.float32), SR, subtype="FLOAT")
                    rows.append({"id": pid, "lang": a.lang, "split": split, "cat": cat, "bin": bi,
                                 "bin_edges": [round(float(lo), 2), round(float(hi), 2)], "bvr_target": round(target, 3),
                                 "bvr_true": round(float(10 * np.log10((b ** 2).sum() / (s ** 2).sum())), 3),
                                 "speaker": sp, "utts": utts, "bg": src, "peak_scaled": pk > 0.99})
    with mp.get_context("fork").Pool(a.procs) as pool:
        sep = dict(pool.map(separate, [(str(wd), r["id"]) for r in rows], chunksize=2))
    with open(out / "manifest.jsonl", "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, **sep[r["id"]]}) + "\n")
    cs = np.array([sep[r["id"]]["c"] for r in rows])
    print(json.dumps({"out": str(out), "pairs": len(rows), "bvr_edges": [round(float(e), 2) for e in edges],
                      "c_median": float(np.median(cs)), "kappa_median": float(np.median([sep[r["id"]]["kappa"] for r in rows]))}))


if __name__ == "__main__":
    main()
