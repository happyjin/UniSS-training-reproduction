"""Idea 2: synthetic broadcast-style multi-speaker streams (prereg reports/mist_w3po_v1/voicecond_idea2_prereg.zh-CN.md §1).

    python -I experiments/mist_w3po_v1/voicecond/idea2_build_streams.py streams --set pilot|rest [--workers 48]
    python -I experiments/mist_w3po_v1/voicecond/idea2_build_streams.py refs                     # clean 6 s prompts

Stream ids <split>_<lang>_<i:04d>: train i < 320, dev/test i < 40 per language; pilot = train i < 32, dev/test i < 4.
Each stream -> data/open_voicecond_v1/idea2/streams/<sid>/{mix,speech,bg}.wav (16 kHz PCM16) + meta.json, written in
streams/.tmp_<sid> and renamed when complete; an existing <sid> is skipped, never rewritten. All randomness of a stream
comes from sha1("20261119:<sid>"). refs: for every speaker used in a built stream, refs/<spk>/p<k>.wav (up to 5 clean
6 s prompts from the speaker's reserved utterances) + refs/<spk>/info.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[3]
I2 = REPO / "data/open_voicecond_v1/idea2"
IDX = I2 / "index"
OUT = I2 / "streams"
REFS = I2 / "refs"
SR = 16000
SEED = 20261119
N_STREAMS = {"train": 320, "dev": 40, "test": 40}
N_PILOT = {"train": 32, "dev": 4, "test": 4}


def rng_for(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(f"{SEED}:{key}".encode()).hexdigest(), 16) % (2 ** 63))


def read16(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def trim(x: np.ndarray, db: float = -35.0) -> np.ndarray:
    n = SR // 100
    m = len(x) // n
    if m < 3:
        return x
    e = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12)
    act = np.nonzero(20 * np.log10(e / (e.max() + 1e-12) + 1e-12) > db)[0]
    if len(act) == 0:
        return x[:0]
    return x[act[0] * n: (act[-1] + 1) * n]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-12))


def active_rms(x: np.ndarray, db: float = -30.0) -> float:
    n = SR // 100
    m = len(x) // n
    if m < 1:
        return rms(x)
    e = (x[: m * n].reshape(m, n).astype(np.float64) ** 2).mean(1)
    a = e > e.max() * 10 ** (db / 10)
    return float(np.sqrt(e[a].mean() + 1e-12))


# ---------------------------------------------------------------- pools
def heldout_speech():
    """OpenBroadcast-SI v1 held-out Common Voice speakers and clips (bench/HELDOUT_SPEECH.txt): never used here"""
    ids, clips = set(), set()
    for l in open(REPO / "data/open_voicecond_v1/bench/HELDOUT_SPEECH.txt"):
        t = l.rstrip("\n").split("\t")
        if len(t) == 3 and t[0] == "client_id":
            ids.add(t[2])
        elif len(t) == 3 and t[0] == "clip":
            clips.add(t[2].split(".")[0])
    return ids, clips


def load_speakers(pilot: bool):
    spk = {}
    hid, hclip = heldout_speech()
    for f in sorted(IDX.glob("speakers_*.jsonl")):
        tag = f.stem[len("speakers_"):]
        if pilot and tag == "ls_train-clean-360":
            continue
        for l in open(f):
            r = json.loads(l)
            if r.get("client_id") in hid:
                continue
            r["utts"] = [u for u in r["utts"] if Path(u[0]).stem not in hclip]
            if not r["utts"]:
                continue
            r["tag"] = tag
            spk[r["spk"]] = r
    return spk


def reserved_split(r):
    """speaker's utterances in a fixed sha1 order; the first ones (>= min(30 s, total/3)) are reserved for refs"""
    u = sorted(r["utts"], key=lambda x: hashlib.sha1((r["spk"] + x[0]).encode()).hexdigest())
    tot = sum(x[1] for x in u)
    need, acc, k = min(30.0, tot / 3), 0.0, 0
    while k < len(u) and acc < need:
        acc += u[k][1]
        k += 1
    return u[:k], (u[k:] or u)


def bvr_values():
    vals = []
    for f in sorted((REPO / "eval_outputs/mist_w3po_v1/game_muffle_v1").glob("game_*.jsonl")):
        for l in open(f):
            b = json.loads(l).get("prompt", {}).get("bvr_db")
            if b is not None and -25 <= b <= -3:
                vals.append(float(b))
    return np.array(sorted(vals))


# ---------------------------------------------------------------- background
def bed(rng, pool, T, xfade=0.5):
    """random pieces end to end with crossfades, each RMS-normalised to 1, until T samples"""
    out = np.zeros(0, np.float32)
    used = []
    xf = int(xfade * SR)
    first = True
    while len(out) < T:
        r = pool[rng.integers(len(pool))]
        x = read16(r["path"])
        if len(x) < 2 * xf:
            continue
        if first and len(x) > T // 4:
            off = int(rng.integers(0, len(x) - T // 4))
            x = x[off:]
        x = x / (active_rms(x) + 1e-9)
        used.append(r["rel"])
        if len(out) == 0:
            out = x.astype(np.float32)
        else:
            w = np.linspace(0, 1, xf, dtype=np.float32)
            out = np.concatenate([out[:-xf], out[-xf:] * (1 - w) + x[:xf] * w, x[xf:]])
        first = False
    return out[:T], used


def gain_walk(rng, T, step_s=5.0, sd_db=1.0, lim_db=4.0):
    n = int(T / (step_s * SR)) + 2
    g = np.clip(np.cumsum(rng.normal(0, sd_db, n)), -lim_db, lim_db)
    t = np.arange(n) * step_s * SR
    return (10 ** (np.interp(np.arange(T), t, g) / 20)).astype(np.float32)


def events(rng, pool, T, rate_s, dur_rng, lvl_db, fade_in, fade_out, concat):
    out = np.zeros(T, np.float32)
    ev = []
    t = rng.exponential(rate_s)
    while t * SR < T - SR:
        d = rng.uniform(*dur_rng)
        n = int(d * SR)
        x = np.zeros(0, np.float32)
        srcs = []
        while len(x) < n:
            r = pool[rng.integers(len(pool))]
            y = trim(read16(r["path"]), -40)
            if len(y) < SR // 10:
                continue
            x = np.concatenate([x, y]) if concat else y
            srcs.append(r["rel"])
            if not concat:
                break
        x = x[:n]
        if len(x) < SR // 10:
            t += rng.exponential(rate_s)
            continue
        fi, fo = min(int(fade_in * SR), len(x) // 4), min(int(fade_out * SR), len(x) // 4)
        x = x.copy()
        if fi:
            x[:fi] *= np.linspace(0, 1, fi)
        if fo:
            x[-fo:] *= np.linspace(1, 0, fo)
        lv = rng.uniform(*lvl_db)
        x = x / (active_rms(x) + 1e-9) * 10 ** (lv / 20)
        s = int(t * SR)
        e = min(T, s + len(x))
        out[s:e] += x[: e - s]
        ev.append({"t": round(t, 3), "dur": round((e - s) / SR, 3), "level_db": round(lv, 2), "src": srcs})
        t += rng.exponential(rate_s)
    return out, ev


def duck(speech, rng):
    n = SR // 100
    m = len(speech) // n + 1
    pad = np.zeros(m * n, np.float32)
    pad[: len(speech)] = speech
    env = np.sqrt((pad.reshape(m, n) ** 2).mean(1) + 1e-12)
    act = env > env.max() * 10 ** (-40 / 20)
    d = rng.uniform(3, 8)
    tgt = np.where(act, -d, 0.0)
    g = np.zeros(m)
    a_att, a_rel = np.exp(-0.01 / 0.15), np.exp(-0.01 / 0.6)
    cur = 0.0
    for i in range(m):
        a = a_att if tgt[i] < cur else a_rel
        cur = a * cur + (1 - a) * tgt[i]
        g[i] = cur
    gs = 10 ** (np.interp(np.arange(len(speech)), np.arange(m) * n + n / 2, g) / 20)
    return gs.astype(np.float32), act, d


# ---------------------------------------------------------------- stream
def build(job):
    sid, split, lang, spk_pool, bg, ami, bvrs, pool_tag = job
    final = OUT / sid
    if final.exists():
        return sid, "skip"
    tmp = OUT / f".tmp_{sid}"
    if tmp.exists():
        return sid, "REFUSE: stale tmp dir exists"
    rng = rng_for(sid)
    T = int(rng.uniform(300, 600) * SR)
    nspk = int(rng.integers(2, 5))
    spks = list(rng.choice(sorted(spk_pool), nspk, replace=False))
    w = rng.dirichlet([2.0] * nspk)
    lvl = rng.uniform(-3, 3, nspk) - 23.0
    td = ami["turn_dur"][(ami["turn_dur"] >= 0.5) & (ami["turn_dur"] <= 40)]
    gp = ami["gap"]
    # utterance iterators (stream-specific order over the non-reserved pool)
    its = {}
    for s in spks:
        _, pool = reserved_split(spk_pool[s])
        its[s] = {"pool": pool, "order": list(rng.permutation(len(pool))), "k": 0}

    def next_utt(s):
        it = its[s]
        if it["k"] >= len(it["order"]):
            it["order"], it["k"] = list(rng.permutation(len(it["pool"]))), 0
        u = it["pool"][it["order"][it["k"]]]
        it["k"] += 1
        return u

    speech = np.zeros(T + SR, np.float32)
    segs, turns = [], []
    t = rng.uniform(0.5, 3.0)
    cur = int(rng.choice(nspk, p=w))
    dur = float(rng.choice(td))
    while t < T / SR - 0.5:
        end = min(t + dur, T / SR)
        turns.append({"spk": spks[cur], "start": round(t, 3), "end": round(end, 3)})
        # fill the turn with utterances
        p = t
        while p < end - 0.2:
            path, _ = next_utt(spks[cur])
            x = trim(read16(path))
            if len(x) < SR // 5:
                continue
            x = x / (rms(x) + 1e-9) * 10 ** (lvl[cur] / 20)
            n = min(len(x), int((end - p) * SR))
            y = x[:n].copy()
            if n < len(x):
                fo = min(len(y), SR // 50)
                y[-fo:] *= np.linspace(1, 0, fo)
            s0 = int(p * SR)
            speech[s0: s0 + n] += y
            segs.append({"spk": spks[cur], "start": round(s0 / SR, 4), "end": round((s0 + n) / SR, 4), "utt": path,
                         "cut": n < len(x)})
            p = (s0 + n) / SR + rng.uniform(0.1, 0.4)
        # next turn
        others = [i for i in range(nspk) if i != cur]
        wo = w[others] / w[others].sum()
        nxt = int(others[rng.choice(len(others), p=wo)])
        ndur = float(rng.choice(td))
        g = float(rng.choice(gp))
        if g < 0:
            g = -min(-g, 0.8 * min(end - t, ndur))
        t, cur, dur = end + g, nxt, ndur
    speech = speech[:T]
    # background
    music = [r for r in bg if r["kind"] == "music" and r["split"] == split]
    noise = [r for r in bg if r["kind"] == "noise" and r["split"] == split]
    crowd = [r for r in bg if r["kind"] == "crowd" and r["split"] == split]
    sfx = [r for r in bg if r["kind"] == "sfx" and r["split"] == split]
    mb, m_used = bed(rng, music, T)
    mb = mb * gain_walk(rng, T)
    layers = {"music": m_used}
    b = mb.copy()
    if rng.random() < 0.5:
        nb, n_used = bed(rng, noise, T)
        nl = rng.uniform(-15, -6)
        b += nb * 10 ** (nl / 20)
        layers["noise"] = {"src": n_used, "level_db": round(nl, 2)}
    cb, cev = events(rng, crowd, T, 25.0, (2.0, 8.0), (0.0, 8.0), 0.05, 0.05, True)
    sb, sev = events(rng, sfx, T, 15.0, (0.3, 3.0), (-3.0, 6.0), 0.01, 0.05, False)
    b += cb + sb
    layers["crowd"], layers["sfx"] = cev, sev
    gd, act, ddb = duck(speech, rng)
    b = b * gd
    n = SR // 100
    am = np.repeat(act, n)[:T]
    am = np.pad(am, (0, max(0, T - len(am))))
    bvr = float(rng.choice(bvrs))
    es, eb = float((speech[am] ** 2).mean()), float((b[am] ** 2).mean())
    b = b * np.sqrt(10 ** (bvr / 10) * es / (eb + 1e-12))
    mix = speech + b
    pk = float(np.abs(mix).max())
    sc = 0.99 / pk if pk > 0.99 else 1.0
    tmp.mkdir(parents=True)
    for name, x in (("mix", mix), ("speech", speech), ("bg", b)):
        sf.write(str(tmp / f"{name}.wav"), np.clip(x * sc, -1, 1), SR, subtype="PCM_16")
    meta = {"sid": sid, "split": split, "lang": lang, "pool": pool_tag, "dur": T / SR, "speakers": spks,
            "weights": [round(float(x), 4) for x in w], "level_db": [round(float(x), 2) for x in lvl],
            "bvr_db": round(bvr, 3), "duck_db": round(ddb, 2), "scale": sc, "turns": turns, "segments": segs,
            "background": layers}
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    os.rename(tmp, final)
    return sid, "ok"


def stream_ids(which: str):
    out = []
    for split, n in N_STREAMS.items():
        for lang in ("en", "zh"):
            for i in range(n):
                pil = i < N_PILOT[split]
                if (which == "pilot" and pil) or (which == "rest" and not pil):
                    out.append((f"{split}_{lang}_{i:04d}", split, lang))
    return out


def cmd_streams(a):
    spk = load_speakers(pilot=(a.set == "pilot"))
    if a.set == "rest" and not (IDX / "speakers_ls_train-clean-360.jsonl").exists():
        sys.exit("rest needs speakers_ls_train-clean-360.jsonl")
    bg = [json.loads(l) for l in open(IDX / "backgrounds.jsonl")]
    ami = json.load(open(REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/ami_turn_stats.json"))
    ami = {"turn_dur": np.array(ami["turn_dur"]), "gap": np.array(ami["gap"])}
    bvrs = bvr_values()
    print(f"BVR values in [-25,-3]: {len(bvrs)}, median {np.median(bvrs):.2f}", flush=True)
    jobs = []
    for sid, split, lang in stream_ids(a.set):
        if a.only and not any(sid.startswith(o) for o in a.only):
            continue
        pool = {k: r for k, r in spk.items() if r["split"] == split and r["lang"] == lang}
        jobs.append((sid, split, lang, pool, bg, ami, bvrs, a.set))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} streams ({a.set})", flush=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, st in ex.map(build, jobs, chunksize=1):
            done += 1
            if st != "ok" or done % 20 == 0:
                print(f"[{done}/{len(jobs)}] {sid} {st}", flush=True)
    if not a.only and not (OUT / f"BUILD_{a.set.upper()}_DONE").exists():
        (OUT / f"BUILD_{a.set.upper()}_DONE").write_text("ok\n")


def ref_one(job):
    spk, r = job
    d = REFS / spk.replace(":", "_")
    if d.exists():
        return spk, "skip"
    res, _ = reserved_split(r)
    x = np.concatenate([trim(read16(p)) for p, _ in res] or [np.zeros(0, np.float32)])
    L = 6 * SR
    k = min(5, len(x) // L) or 1
    tmp = REFS / f".tmp_{spk.replace(':', '_')}"
    tmp.mkdir(parents=True)
    for i in range(k):
        sf.write(str(tmp / f"p{i}.wav"), x[i * L:(i + 1) * L], SR, subtype="PCM_16")
    (tmp / "info.json").write_text(json.dumps({"spk": spk, "n_prompts": k, "reserved_s": round(len(x) / SR, 2),
                                                "utts": [p for p, _ in res]}))
    os.rename(tmp, d)
    return spk, "ok"


def cmd_refs(a):
    spk = load_speakers(pilot=False)
    used = set()
    for m in OUT.glob("*/meta.json"):
        used |= set(json.load(open(m))["speakers"])
    jobs = [(s, spk[s]) for s in sorted(used)]
    REFS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(ref_one, jobs, chunksize=4))
    print(f"refs: {sum(st == 'ok' for _, st in res)} new, {sum(st == 'skip' for _, st in res)} existing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["streams", "refs"])
    ap.add_argument("--set", choices=["pilot", "rest"], default="pilot")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--only", nargs="*", default=[], help="build only stream ids with these prefixes (e.g. dev_en test_en)")
    a = ap.parse_args()
    cmd_streams(a) if a.cmd == "streams" else cmd_refs(a)
