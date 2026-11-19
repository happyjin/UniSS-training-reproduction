"""Robust wave 1, Thinker prechecks K2 / K3 / K7 / K8 / K9 (prereg reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md):
the shared CoVoST 2 VALIDATION sentence set and every CPU-side statistic. Open, non-test data only.

Sentences: CoVoST 2 validation (data/open_voicecond_v1/idea2/covost_train/.../validation-*.parquet; bench/covost_hf only
holds the TEST split, never read), speakers in bench/HELDOUT_SPEECH.txt excluded. Rows ranked by
sha1("20261119|sel|<dir>|<id>"); per-speaker cap (en2zh 1, zh2en 6); bench_build.trim_norm (edge trim, -26 dBFS active
RMS, 1.5-12 s, no clipping); the first N_SENT = 300 survivors per direction in rank order.
Per sentence (RNG = random.Random(sha1("20261119|mix|<id>")), so nothing depends on the worker count):
  BVR ~ quantile of the OpenBroadcast-SI esports prompt-BVR values (bench/meta/stats.json) at q ~ U(0.05, 0.95)
  (bench_build.plan's stratified range); one background timeline from bench_build.bg_schedule / render_bg (bed = music,
  amb = crowd, sfx = sfx + MUSAN noise) over the Idea 2 background index split "dev" (held-out benchmark files already
  excluded there; asserted again against HELDOUT_BACKGROUNDS*.txt):
      [2.5 s lead-in, dropped (bed fade-in)] [bg-only clip, T s] [gap, 3 s] [under-speech background, T s]
  scaled once so that energy(under) / energy(speech) = BVR (bench_build.mix's alpha), no ducking (sentence-level mix:
  bg-only clip and gap keep the level of the background under the speech), the bench limiter (whole mixture <= 0.99).
  Files <id>.{clean,noisy,bgonly,gap,bgunder}.wav (PCM16 16 kHz) and <id>.post.npy: synthetic diarizer posteriors
  [F80, 8] (slot 0 = 1 on the 80 ms frames that overlap the clean speech span (frame dB > max(peak - 40, -55)), shared by
  every condition and branch, so the comparisons only move the Thinker's audio path).
Steps (each refuses to overwrite; CPU only; --procs workers):
  select    data/sel_<dir>.jsonl + data/clips/<dir>/<id>.clean.wav
  mix       data/audio/<dir>/<id>.*.wav + data/manifest.jsonl
  sep       HDemucs (idea2_demucs.separate, CPU) of every clean and noisy sentence -> data/sep/<dir>/<id>.<cond>.vhat.wav
            (float) and the Idea 2 lambda head (idea2_shrink.ShrinkParams of shrink_fit_all/full.npz) on the sentence
            window feature v~ (idea2_feats2.window_vt over the whole sentence: ends = [T], L = T) -> data/lambda.jsonl
  k8cal     sigma2_a = sum ||HDemucs(noisy) - clean||^2 / sum ||clean||^2 over 200 CoVoST 2 TRAIN pairs (100 en, 100 zh;
            mixed like the sentences but with the Idea 2 background split "train") -> data/k8cal.json
  k3sil     training-silence encoder-input statistics: Whisper log-mel (the Thinker's feature extractor) of 1000 CoVoST 2
            TRAIN clips (500 per language, gain to -26 dBFS active RMS, not trimmed), frames > 0.1 s outside the speech
            span (the clip's non-speech edges) -> data/k3_silence.npz (mean [128], count)
  k7src     K7 calibration audio: 300 CoVoST 2 TRAIN clean sentences (150 per language, trim_norm) and 300 training-noise
            clips (render_bg over the "train" background split, 3-8 s, active RMS = -26 dBFS + a BVR draw) ->
            data/k7src/{clean,noise}/*.wav + data/k7src.jsonl

    python -I experiments/mist_w3po_v1/voicecond/wave1_data.py select|mix|sep|k8cal|k3sil|k7src [--procs 64]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as BB  # noqa: E402

OUT = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5"
DATA = OUT / "data"
CV = REPO / "data/open_voicecond_v1/idea2/covost_train/s2tt/covost/hf"
BG_INDEX = REPO / "data/open_voicecond_v1/idea2/index/backgrounds.jsonl"
FIT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all/full.npz"
SEED = "20261119"
SR = 16000
N_SENT = 300
LEAD, GAP = 2.5, 3.0
DIRS = {"en2zh": ("en_zh-CN", 1), "zh2en": ("zh-CN_en", 6)}   # parquet dir, per-speaker cap


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def rng_of(tag: str) -> random.Random:
    return random.Random(h(f"{SEED}|{tag}"))


def refuse(p: Path) -> None:
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")


def wjsonl(p: Path, rows) -> None:
    refuse(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def w16(p: Path, x: np.ndarray, sub: str = "PCM_16") -> None:
    refuse(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), x, SR, subtype=sub)


def held_speakers() -> set:
    ids = set()
    for l in open(REPO / "data/open_voicecond_v1/bench/HELDOUT_SPEECH.txt", encoding="utf-8"):
        t = l.rstrip("\n").split("\t")
        if len(t) == 3 and t[0] == "client_id":
            ids.add(t[2])
    return ids


def heldout_bg_paths() -> set:
    """bench-relative paths of every held-out background (MUSAN member / FSD50K eval clip)"""
    H = BB.heldout_set()
    return set().union(*H.values())


def bg_pools(split: str) -> dict:
    held = heldout_bg_paths()
    P = {"musan_music": [], "fsd_crowd": [], "fsd_sfx_musan_noise": []}
    key = {"music": "musan_music", "crowd": "fsd_crowd", "sfx": "fsd_sfx_musan_noise", "noise": "fsd_sfx_musan_noise"}
    for r in map(json.loads, open(BG_INDEX, encoding="utf-8")):
        if r["split"] != split:
            continue
        bp = r["rel"].split("/", 1)[1]
        assert bp not in held and not BB.held(bp), f"held-out background in the index: {r['rel']}"
        P[key[r["kind"]]].append({"file": r["path"], "src": r["rel"], "dur": r["dur"]})
    for k in P:
        P[k].sort(key=lambda r: r["src"])
        assert P[k], f"empty pool {k} ({split})"
    return P


def bvr_values() -> np.ndarray:
    return np.array(json.load(open(REPO / "data/open_voicecond_v1/bench/meta/stats.json"))["bvr_db"]["values"])


def decode_bytes(b: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return BB.to16(x, sr)


def speech_span(x: np.ndarray) -> tuple[float, float]:
    db = BB.frame_db(x)
    keep = np.where(db > max(db.max() - 40.0, -55.0))[0]
    return float(keep[0] * 0.01), float((keep[-1] + 1) * 0.01)


def synth_post(x: np.ndarray) -> np.ndarray:
    a, b = speech_span(x)
    F = int(math.ceil(len(x) / SR / 0.08))
    post = np.zeros((F, 8), np.float32)
    for f in range(F):
        if f * 0.08 < b and (f + 1) * 0.08 > a:
            post[f, 0] = 1.0
    return post


# ------------------------------------------------------------------------------------------------------------ select
def rows_of(d: str, split: str, cols=("client_id", "id", "sentence", "translation", "audio")) -> list[dict]:
    import pyarrow.parquet as pq
    out = []
    for f in sorted(glob.glob(str(CV / DIRS[d][0] / f"{split}-*.parquet"))):
        out += pq.read_table(f, columns=list(cols)).to_pylist()
    return out


def _sel_job(a):
    r, d = a
    t = BB.trim_norm(decode_bytes(r["audio"]["bytes"]))
    if t is None:
        return None
    y, meta = t
    p = DATA / "clips" / d / f"{r['id']}.clean.wav"
    w16(p, y)
    return {"id": r["id"], "dir": d, "spk": r["client_id"], "src": r["sentence"].strip(), "ref": r["translation"].strip(),
            "dur": round(len(y) / SR, 3), "clean_clip": str(p.relative_to(REPO)), **meta}


def ranked_capped(rows: list[dict], tag: str, cap: int, n: int, held: set) -> list[dict]:
    rows = [r for r in rows if r["client_id"] not in held and r["sentence"].strip() and r["translation"].strip()]
    rows.sort(key=lambda r: h(f"{SEED}|{tag}|{r['id']}"))
    cnt, out = {}, []
    for r in rows:
        if cnt.get(r["client_id"], 0) >= cap:
            continue
        cnt[r["client_id"]] = cnt.get(r["client_id"], 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return out


def cmd_select(a):
    import multiprocessing as mp
    held = held_speakers()
    for d, (_, cap) in DIRS.items():
        man = DATA / f"sel_{d}.jsonl"
        refuse(man)
        cand = ranked_capped(rows_of(d, "validation"), f"sel|{d}", cap, int(N_SENT * 1.4), held)
        with mp.get_context("fork").Pool(a.procs) as P:
            res = P.map(_sel_job, [(r, d) for r in cand], chunksize=1)
        keep = [x for x in res if x][:N_SENT]
        assert len(keep) == N_SENT, (d, len(keep))
        assert not {x["spk"] for x in keep} & held
        wjsonl(man, keep)
        print(f"{d}: {len(keep)} sentences of {len({x['spk'] for x in keep})} speakers, "
              f"{sum(x['dur'] for x in keep) / 60:.1f} min (candidates {len(cand)}, failed {sum(x is None for x in res)})",
              flush=True)


# --------------------------------------------------------------------------------------------------------------- mix
_POOLS: dict = {}


def mix_one(s: np.ndarray, rng: random.Random, pools: dict, bv: np.ndarray, with_context: bool = True) -> dict:
    """bench_build recipe on one sentence: background timeline, BVR scaling, limiter."""
    ns = len(s)
    q = rng.uniform(0.05, 0.95)
    bvr = float(np.quantile(bv, q))
    T = ns / SR
    L = LEAD + (T + GAP if with_context else 0.0) + T
    n = int(round(L * SR)) + SR // 10
    sched = BB.bg_schedule(rng, L)
    b, used = BB.render_bg(sched, pools, "id2", n)
    i0 = int(LEAD * SR)
    if with_context:
        bgonly, gap = b[i0: i0 + ns], b[i0 + ns: i0 + ns + int(GAP * SR)]
        under = b[i0 + ns + int(GAP * SR): i0 + ns + int(GAP * SR) + ns]
    else:
        bgonly = gap = None
        under = b[i0: i0 + ns]
    es, eb = float((s.astype(np.float64) ** 2).sum()), float((under.astype(np.float64) ** 2).sum())
    alpha = math.sqrt(10 ** (bvr / 10) * es / max(eb, 1e-12))
    y = s + alpha * under
    lim = min(1.0, 0.99 / float(np.abs(y).max()))
    r = {"bvr_q": round(q, 4), "bvr_target_db": round(bvr, 2), "alpha": round(alpha, 6), "limiter": round(lim, 5),
         "bvr_db": round(10 * math.log10(max(alpha * alpha * eb, 1e-20) / es), 2),
         "noisy": (y * lim).astype(np.float32), "under": (alpha * lim * under).astype(np.float32),
         "events": [e for e in used if e["t"] < L]}
    if with_context:
        r["bgonly"] = (alpha * lim * bgonly).astype(np.float32)
        r["gap"] = (alpha * lim * gap).astype(np.float32)
        eo = float((r["bgonly"].astype(np.float64) ** 2).mean()) + 1e-20
        eu = float((r["under"].astype(np.float64) ** 2).mean()) + 1e-20
        r["bgonly_vs_under_db"] = round(10 * math.log10(eo / eu), 2)
    return r


def _mix_job(x):
    s = sf.read(str(REPO / x["clean_clip"]), dtype="float32")[0]
    m = mix_one(s, rng_of(f"mix|{x['id']}"), _POOLS["dev"], _POOLS["bv"])
    D = DATA / "audio" / x["dir"]
    paths = {}
    for k, v in (("clean", s), ("noisy", m["noisy"]), ("bgonly", m["bgonly"]), ("gap", m["gap"]), ("bgunder", m["under"])):
        p = D / f"{x['id']}.{k}.wav"
        w16(p, v)
        paths[k] = str(p.relative_to(REPO))
    pp = D / f"{x['id']}.post.npy"
    refuse(pp)
    np.save(pp, synth_post(s).astype(np.float16))
    paths["post"] = str(pp.relative_to(REPO))
    a, b = speech_span(s)
    return {**{k: x[k] for k in ("id", "dir", "spk", "src", "ref", "dur")}, **paths,
            **{k: m[k] for k in ("bvr_q", "bvr_target_db", "bvr_db", "alpha", "limiter", "bgonly_vs_under_db")},
            "speech_span_s": [round(a, 2), round(b, 2)], "events": m["events"]}


def cmd_mix(a):
    import multiprocessing as mp
    man = DATA / "manifest.jsonl"
    refuse(man)
    _POOLS["dev"], _POOLS["bv"] = bg_pools("dev"), bvr_values()
    sel = [json.loads(l) for d in DIRS for l in open(DATA / f"sel_{d}.jsonl", encoding="utf-8")]
    with mp.get_context("fork").Pool(a.procs) as P:
        res = P.map(_mix_job, sel, chunksize=1)
    wjsonl(man, res)
    for d in DIRS:
        b = [r["bvr_db"] for r in res if r["dir"] == d]
        print(f"{d}: {len(b)} mixed, BVR median {np.median(b):.1f} dB (p10 {np.percentile(b, 10):.1f}, p90 "
              f"{np.percentile(b, 90):.1f}), limited {sum(r['limiter'] < 1 for r in res if r['dir'] == d)}", flush=True)


# --------------------------------------------------------------------------------------------------------------- sep
_SEP: dict = {}


def _sep_init():
    import torch
    from torchaudio.models import hdemucs_high
    torch.set_num_threads(2)
    m = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    m.load_state_dict(torch.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints/hdemucs_high_trained.pt",
                                 map_location="cpu"))
    _SEP["model"] = m.eval()


def separate(y: np.ndarray):
    import torch
    from experiments.mist_w3po_v1.voicecond.idea2_demucs import separate as sep2
    return sep2(_SEP["model"], y.astype(np.float32), torch.device("cpu"))


def _sep_job(a):
    from experiments.mist_w3po_v1.voicecond.idea2_feats2 import window_vt
    x, cond = a
    y = sf.read(str(REPO / x[cond]), dtype="float32")[0]
    vh, bh = separate(y)
    p = DATA / "sep" / x["dir"] / f"{x['id']}.{cond}.vhat.wav"
    w16(p, vh.astype(np.float32), "FLOAT")
    T = len(y) / SR
    vt = window_vt(vh.astype(np.float32), bh.astype(np.float32), np.array([T]), T)[0]
    return {"id": x["id"], "dir": x["dir"], "cond": cond, "vhat": str(p.relative_to(REPO)),
            "vt": [round(float(v), 4) for v in vt]}


def cmd_sep(a):
    import multiprocessing as mp
    from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams
    out = DATA / "lambda.jsonl"
    refuse(out)
    man = [json.loads(l) for l in open(DATA / "manifest.jsonl", encoding="utf-8")]
    jobs = [(x, c) for x in man for c in ("clean", "noisy")]
    with mp.get_context("fork").Pool(a.procs, initializer=_sep_init) as P:
        res = P.map(_sep_job, jobs, chunksize=1)
    p = ShrinkParams.load(FIT)
    for r in res:
        r["lam"] = round(float(p.lam(np.array(r["vt"]))), 6)
    wjsonl(out, res)
    for d in DIRS:
        for c in ("clean", "noisy"):
            l = [r["lam"] for r in res if r["dir"] == d and r["cond"] == c]
            print(f"{d} {c}: lambda median {np.median(l):.3f} (p10 {np.percentile(l, 10):.3f}, p90 {np.percentile(l, 90):.3f})")


# ------------------------------------------------------------------------------------------------------------- k8cal
def with_audio(d: str, split: str, rows: list[dict]) -> list[dict]:
    """fetch the audio column only for the given rows (the en train parquets are 13 GB), order kept"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array([r["id"] for r in rows])
    got = {}
    for f in sorted(glob.glob(str(CV / DIRS[d][0] / f"{split}-*.parquet"))):
        t = pq.read_table(f, columns=["id", "audio"])
        t = t.filter(pc.is_in(t.column("id"), value_set=want))
        for x in t.to_pylist():
            got[x["id"]] = x["audio"]
    return [dict(r, audio=got[r["id"]]) for r in rows]


def train_clean(n_per_lang: int, tag: str) -> list[dict]:
    """n_per_lang CoVoST 2 TRAIN clips per source language (held-out speakers excluded), ranked; per-speaker cap 2 for
    en and 40 for zh (zh-CN train has only 22 speakers; v0 capped zh at 2 too and got 44 zh clips)"""
    held = held_speakers()
    out = []
    for d in DIRS:
        rows = rows_of(d, "train", ("client_id", "id", "sentence", "translation"))
        sel = ranked_capped(rows, f"{tag}|{d}", 2 if d == "en2zh" else 40, int(n_per_lang * 1.5), held)
        out += [dict(r, dir=d) for r in with_audio(d, "train", sel)]
    return out


def _k8_job(r):
    t = BB.trim_norm(decode_bytes(r["audio"]["bytes"]))
    if t is None:
        return None
    s = t[0]
    m = mix_one(s, rng_of(f"k8cal|{r['id']}"), _POOLS["train"], _POOLS["bv"], with_context=False)
    vh, _ = separate(m["noisy"])
    e = vh[: len(s)].astype(np.float64) - s
    return {"id": r["id"], "dir": r["dir"], "es": float((s.astype(np.float64) ** 2).sum()), "ee": float((e ** 2).sum()),
            "bvr_db": m["bvr_db"]}


def cmd_k8cal(a):
    import multiprocessing as mp
    out = DATA / "k8cal.json"
    refuse(out)
    _POOLS["train"], _POOLS["bv"] = bg_pools("train"), bvr_values()
    rows = train_clean(100, "k8cal")
    with mp.get_context("fork").Pool(a.procs, initializer=_sep_init) as P:
        res = P.map(_k8_job, rows, chunksize=1)
    keep = []
    for d in DIRS:
        keep += [x for x in res if x and x["dir"] == d][:100]
    assert len(keep) == 200, len(keep)
    s2a = sum(x["ee"] for x in keep) / sum(x["es"] for x in keep)
    per = [x["ee"] / x["es"] for x in keep]
    refuse(out)
    with open(out, "x") as fh:
        json.dump({"sigma2_a": s2a, "n": len(keep), "per_pair_median": float(np.median(per)),
                   "per_lang": {d: sum(x["ee"] for x in keep if x["dir"] == d) / sum(x["es"] for x in keep if x["dir"] == d)
                                for d in DIRS}, "pairs": keep}, fh, indent=0)
    print(f"sigma2_a = {s2a:.4f} (per-pair median {np.median(per):.4f}) on {len(keep)} pairs")


# ------------------------------------------------------------------------------------------------------------- k3sil
_FE: dict = {}


def feat_extractor():
    if "fe" not in _FE:
        from transformers import WhisperFeatureExtractor
        _FE["fe"] = WhisperFeatureExtractor.from_pretrained(str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"))
    return _FE["fe"]


def logmel(x: np.ndarray) -> np.ndarray:
    """[128, F] log-mel exactly as engine.audio_embeds computes it for one unpadded clip"""
    f = feat_extractor()([x], sampling_rate=SR, padding=True, truncation=False, return_attention_mask=True,
                         return_tensors="np")
    n = int(f["attention_mask"][0].sum())
    return f["input_features"][0][:, :n]


def _k3_job(r):
    x = decode_bytes(r["audio"]["bytes"])
    db = BB.frame_db(x)
    if db.max() < -50:
        return None
    x = x * 10 ** ((BB.SPEECH_DBFS - BB.active_rms_db(x)) / 20)
    keep = np.where(db > max(db.max() - 40.0, -55.0))[0]
    F = logmel(x.astype(np.float32))
    lo, hi = keep[0] - 10, keep[-1] + 1 + 10           # feature frames are 10 ms apart, as frame_db's
    idx = [i for i in range(F.shape[1]) if i < lo or i > hi]
    if not idx:
        return {"id": r["id"], "n": 0, "sum": None}
    return {"id": r["id"], "n": len(idx), "sum": F[:, idx].astype(np.float64).sum(1)}


def cmd_k3sil(a):
    import multiprocessing as mp
    out = DATA / "k3_silence.npz"
    refuse(out)
    rows = train_clean(500, "k3sil")
    rows = [r for d in DIRS for r in [r for r in rows if r["dir"] == d][:500]]
    feat_extractor()
    with mp.get_context("fork").Pool(a.procs) as P:
        res = [x for x in P.map(_k3_job, rows, chunksize=4) if x]
    S = sum(x["sum"] for x in res if x["n"])
    N = sum(x["n"] for x in res)
    np.savez(out, mean=(S / N).astype(np.float32), count=N, clips=len(res), clips_with_edges=sum(x["n"] > 0 for x in res))
    print(f"silence frames {N} from {sum(x['n'] > 0 for x in res)}/{len(res)} clips; mean of the mean {(S / N).mean():.3f}")


# ------------------------------------------------------------------------------------------------------------- k7src
def _k7c_job(r):
    t = BB.trim_norm(decode_bytes(r["audio"]["bytes"]))
    if t is None:
        return None
    p = DATA / "k7src/clean" / f"{r['id']}.wav"
    w16(p, t[0])
    return {"kind": "clean", "id": r["id"], "dir": r["dir"], "wav": str(p.relative_to(REPO)), "dur": round(len(t[0]) / SR, 3)}


def _k7n_job(i):
    rng = rng_of(f"k7noise|{i}")
    L = rng.uniform(3.0, 8.0)
    bvr = float(np.quantile(_POOLS["bv"], rng.uniform(0.05, 0.95)))
    sched = BB.bg_schedule(rng, LEAD + L)
    b, used = BB.render_bg(sched, _POOLS["train"], "id2", int((LEAD + L) * SR))
    x = b[int(LEAD * SR):]
    x = x * 10 ** ((BB.SPEECH_DBFS + bvr - BB.active_rms_db(x)) / 20)
    if np.abs(x).max() > 0.99:
        x = x * 0.99 / np.abs(x).max()
    p = DATA / "k7src/noise" / f"noise_{i:04d}.wav"
    w16(p, x.astype(np.float32))
    return {"kind": "noise", "id": f"noise_{i:04d}", "wav": str(p.relative_to(REPO)), "dur": round(len(x) / SR, 3),
            "level_dbfs": round(BB.SPEECH_DBFS + bvr, 2), "srcs": sorted({e["src"] for e in used})}


def cmd_k7src(a):
    import multiprocessing as mp
    out = DATA / "k7src.jsonl"
    refuse(out)
    _POOLS["train"], _POOLS["bv"] = bg_pools("train"), bvr_values()
    rows = train_clean(150, "k7clean")
    with mp.get_context("fork").Pool(a.procs) as P:
        cl = P.map(_k7c_job, rows, chunksize=1)
        nz = P.map(_k7n_job, list(range(300)), chunksize=1)
    keep = []
    for d in DIRS:
        keep += [x for x in cl if x and x["dir"] == d][:150]
    wjsonl(out, keep + nz)
    print(f"k7src: {len(keep)} clean train sentences, {len(nz)} training-noise clips")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["select", "mix", "sep", "k8cal", "k3sil", "k7src"])
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    {"select": cmd_select, "mix": cmd_mix, "sep": cmd_sep, "k8cal": cmd_k8cal, "k3sil": cmd_k3sil,
     "k7src": cmd_k7src}[a.step](a)


if __name__ == "__main__":
    main()
