"""OpenBroadcast-SI DEV twin (D0 of reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md): bench_build.py's recipe on
CoVoST 2 VALIDATION speech and NON-held-out backgrounds, for method selection without touching the test benchmark.
Writes ONLY under data/open_voicecond_v1/bench_dev/ and eval_outputs/mist_w3po_v1/data/windows_obsidev_*; the test
benchmark (data/open_voicecond_v1/bench/, windows_obsi_*) is only read (meta/stats.json, HELDOUT_*.txt).

Conditions (same CoVoST 2 validation speech streams in both; only the background differs):
  id2    NON-held-out MUSAN music / noise + NON-held-out FSD50K eval clips (the complement of HELDOUT_BACKGROUNDS*.txt;
         in-distribution background pools, same layer recipe and label filters as the test id2)   meeting ids db[ze]NNN
  clean  no background (the clean twin)                                                         meeting ids dc[ze]NNN
Directions: zh2en (CoVoST 2 zh-CN->en validation; session "mix", <m>.wav), en2zh (en->zh-CN validation; session
"ihm_mix", <m>.Mix-Headset.wav), the layouts of the test benchmark.

Speakers (Common Voice client_id): never in HELDOUT_SPEECH.txt (test speakers / clips), and never in a TRAINING split of
the voice-condition ideas (Idea 2: idea2/index/speakers_cv_zh.jsonl and idea2/cvt/index/speakers_cvt.jsonl split
"train"; Idea 1 trains on LibriSpeech / AISHELL-1 only), so trained and training-free methods meet unseen voices.
  zh2en  the 15 such zh validation speakers (all others of the 82 non-held-out ones are Idea 2 training speakers), each
         with >= 30 clips: every speaker's clips are split round-robin into ZH_UNITS = 4 disjoint units, and a stream
         takes K units of K DIFFERENT speakers (the K with the most units left; ties in a seeded order). So a speaker
         recurs in about 4 streams with different sentences (the test has one stream per speaker; deviation forced
         by the 15 clean speakers). N_STREAMS zh2en = 20 (K 2/3/4 cycling, as the test).
  en2zh  validation speakers with exactly 3 clips (the test's rule), streams of K = 2/3/4 speakers in a seeded order,
         N_STREAMS en2zh = 40 (short streams, about 47 s each: 40 short streams ~ 0.5 h, 20 long zh2en streams ~ 1 h).
Turn statistics and the BVR quantiles are the test benchmark's recipe parameters (bench/meta/stats.json, read only);
all random draws use the salt "OpenBroadcast-SI-dev-v1|" (the held-out rule keeps its own salt).

Steps (each refuses to overwrite its outputs; CPU only):
  src --dir D     validation clips -> bench_dev/src/<D>/clips/<id>.wav + pool.jsonl (bench_build.trim_norm unchanged)
  plan --dir D    bench_dev/src/<D>/plan.json
  bgprep          bench_dev/bgsrc/<pool>/*.wav + pools.json: musan_music (no vocals, >= 30 s), fsd_crowd,
                  fsd_sfx_musan_noise -- non-held-out files only (also re-checked with the sha1 rule)
  mix --dir D --phase clean|bg   bench_dev/<clean|id2>/<D>/{session_audio,meetings.json,refs_windows.jsonl,utts,bg_track}
  windows --cond C --dir D       C2 windows (gap 0.6) on diar/nemotron_lowlat -> eval_outputs/mist_w3po_v1/data/
                  windows_obsidev_<C>_<D>_c2g06_v1/windows_test_online_<D>.jsonl + ..._c2g06_chunk_v1/ (specclose, judge off)
  check           bench_dev/meta/leak_check.json: no test speaker / clip / held-out background, no Idea 2 training
                  speaker, meeting ids disjoint from the test benchmark
  summary         bench_dev/summary.json

    python -m experiments.mist_w3po_v1.voicecond.bench_dev_build <step> [--dir zh2en|en2zh] [--cond id2|clean]
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as B  # noqa: E402

TEST = B.BENCH                                     # read only: meta/stats.json, HELDOUT_*.txt, FSD50K meta
DEV = REPO / "data/open_voicecond_v1/bench_dev"
HF = REPO / "data/open_voicecond_v1/idea2/covost_train/s2tt/covost/hf"
I2 = REPO / "data/open_voicecond_v1/idea2"
DD = B.DD
SR = B.SR
DSALT = "OpenBroadcast-SI-dev-v1|"
CONDS = {"id2": "b", "clean": "c"}
DIRS = B.DIRS
N_STREAMS = {"zh2en": 20, "en2zh": 40}
ZH_UNITS = 4


def mid(cond: str, d: str, k: int) -> str:
    return f"d{CONDS[cond]}{DIRS[d][0]}{k:03d}"


def sess_path(cond: str, d: str, m: str) -> Path:
    s = DIRS[d][2]
    return DEV / cond / d / "session_audio" / s / (f"{m}.wav" if s == "mix" else f"{m}.Mix-Headset.wav")


def win_dirs(cond: str, d: str) -> tuple[Path, Path]:
    return DD / f"windows_obsidev_{cond}_{d}_c2g06_v1", DD / f"windows_obsidev_{cond}_{d}_c2g06_chunk_v1"


def held_speech() -> tuple[set, set]:
    spk, clip = set(), set()
    for l in open(TEST / "HELDOUT_SPEECH.txt", encoding="utf-8"):
        t = l.rstrip("\n").split("\t")
        if len(t) == 3:
            (spk if t[0] == "client_id" else clip).add(t[2])
    return spk, clip


def idea2_train_speakers() -> set:
    out = set()
    for f in (I2 / "index/speakers_cv_zh.jsonl", I2 / "cvt/index/speakers_cvt.jsonl"):
        for r in map(json.loads, open(f, encoding="utf-8")):
            if r["split"] == "train":
                out.add(r["client_id"])
    return out


# -------------------------------------------------------------------------------------------------------------- src
def src(d: str) -> None:
    import multiprocessing as mp
    import pyarrow.parquet as pq
    out = DEV / "src" / d / "clips"
    pool = DEV / "src" / d / "pool.jsonl"
    if pool.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {pool}")
    files = sorted((HF / DIRS[d][1]).glob("validation-*.parquet"))
    assert files, HF / DIRS[d][1]
    hs, hc = held_speech()
    tr = idea2_train_speakers()
    cnt = Counter()
    for f in files:
        cnt.update(pq.read_table(f, columns=["client_id"]).column("client_id").to_pylist())
    ok = {s for s in cnt if s not in hs and s not in tr}
    want = {s for s in ok if (cnt[s] >= 8 if d == "zh2en" else cnt[s] == 3)}
    info = {"validation_speakers": len(cnt), "heldout_speakers": len(set(cnt) & hs),
            "idea2_train_speakers": len((set(cnt) - hs) & tr), "eligible": len(ok), "wanted": len(want)}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            if r["client_id"] in want and r["id"] not in hc and r["sentence"].strip() and r["translation"].strip():
                rows.append(r)
    print(f"{d}: {json.dumps(info)}; {len(rows)} clips to prepare", flush=True)
    with mp.get_context("fork").Pool(32) as P:
        res = [x for x in P.map(B._src_job, [(r, out) for r in rows], chunksize=8) if x]
    with B.x_open(pool) as fh:
        for x in sorted(res, key=lambda x: (x["spk"], x["id"])):
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with B.x_open(DEV / "src" / d / "speaker_filter.json") as fh:
        json.dump(dict(info, kept_clips=len(res), prepared=len(rows)), fh, indent=1)
    print(f"{d}: {len(res)} clips kept of {len(rows)} ({sum(x['dur'] for x in res) / 3600:.2f} h), "
          f"{len({x['spk'] for x in res})} speakers")


# ------------------------------------------------------------------------------------------------------------- plan
def plan(d: str) -> None:
    out = DEV / "src" / d / "plan.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    st = json.load(open(TEST / "meta/stats.json"))
    A = st["ami"]
    pool = [json.loads(l) for l in open(DEV / "src" / d / "pool.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for c in pool:
        by[c["spk"]].append(c)
    N = N_STREAMS[d]
    streams = []
    if d == "zh2en":
        spks = sorted(s for s, v in by.items() if len(v) >= 8)
        units = {}
        for s in spks:
            c = sorted(by[s], key=lambda x: x["id"])
            random.Random(f"{DSALT}units:{s}").shuffle(c)
            units[s] = [c[i::ZH_UNITS] for i in range(ZH_UNITS)]
        left = {s: list(range(ZH_UNITS)) for s in spks}
        for k in range(N):
            K = [2, 3, 4][k % 3]
            r = random.Random(f"{DSALT}stream:{d}:{k}")
            order = spks[:]
            r.shuffle(order)
            cand = sorted((s for s in order if left[s]), key=lambda s: -len(left[s]))   # stable: shuffled ties
            if len(cand) < K:
                break
            pick = cand[:K]
            uo = {s: left[s].pop(0) for s in pick}
            utts = B.sequence(r, {s: units[s][uo[s]] for s in pick}, A)
            dur = round(max(u["start"] + u["dur"] for u in utts) + 0.8, 3)
            gains = {s: round(r.uniform(-2.0, 2.0), 2) for s in pick}
            streams.append({"key": f"s{k + 1:03d}", "K": K, "speakers": pick, "unit": uo, "spk_gain_db": gains, "dur": dur,
                            "utts": utts})
    else:
        spks = sorted(s for s, v in by.items() if len(v) >= 3)
        rng = random.Random(f"{DSALT}plan:{d}")
        rng.shuffle(spks)
        for k in range(N):
            K = [2, 3, 4][k % 3]
            if len(spks) < K:
                break
            pick = spks[:K]
            spks = spks[K:]
            r = random.Random(f"{DSALT}stream:{d}:{k}")
            utts = B.sequence(r, {s: by[s] for s in pick}, A)
            dur = round(max(u["start"] + u["dur"] for u in utts) + 0.8, 3)
            gains = {s: round(r.uniform(-2.0, 2.0), 2) for s in pick}
            streams.append({"key": f"s{k + 1:03d}", "K": K, "speakers": pick, "spk_gain_db": gains, "dur": dur, "utts": utts})
    bv = np.array(st["bvr_db"]["values"])
    n = len(streams)
    perm = list(range(n))
    random.Random(f"{DSALT}bvr:{d}").shuffle(perm)
    for i, s in enumerate(streams):
        q = 0.05 + 0.90 * (perm[i] + 0.5) / n          # stratified quantiles of the pooled prompt BVR (test recipe)
        s["bvr_q"] = round(q, 4)
        s["bvr_db"] = round(float(np.quantile(bv, q)), 2)
        s["duck_db"] = round(random.Random(f"{DSALT}duck:{d}:{i}").uniform(6.0, 12.0), 2)
    total = sum(s["dur"] for s in streams)
    used = Counter(u["clip"] for s in streams for u in s["utts"])
    assert max(used.values()) == 1, "a clip used twice"
    with B.x_open(out) as fh:
        json.dump({"dir": d, "split": "dev", "n": n, "total_s": round(total, 1), "streams": streams}, fh, ensure_ascii=False)
    print(f"{d}: {n} streams, {total / 3600:.2f} h, speaker slots {sum(s['K'] for s in streams)} "
          f"({len({x for s in streams for x in s['speakers']})} speakers), utts {sum(len(s['utts']) for s in streams)}, "
          f"K {Counter(s['K'] for s in streams)}, dur min/med/max {min(s['dur'] for s in streams):.0f}/"
          f"{np.median([s['dur'] for s in streams]):.0f}/{max(s['dur'] for s in streams):.0f} s")


# ----------------------------------------------------------------------------------------------------------- bgprep
def bgprep() -> None:
    import tarfile
    import zipfile
    out = DEV / "bgsrc"
    pj = out / "pools.json"
    if pj.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {pj}")
    H = B.heldout_set()
    pools = defaultdict(list)
    # MUSAN music vocals annotations (ANNOTATIONS col 3: Y/N), from the test benchmark's extracted copy (read only); the
    # tar's own ANNOTATIONS are read below as well and must agree
    vocals = {}
    for f in sorted((TEST / "meta/musan_ann/musan/music").glob("*/ANNOTATIONS")):
        for line in open(f, encoding="utf-8", errors="replace").read().splitlines():
            t = line.split()
            if len(t) >= 3:
                vocals[t[0]] = t[2]
    assert vocals, "no MUSAN music annotations"
    ann = {}
    skipped = Counter()
    with tarfile.open(B.RAW / "musan/musan.tar.gz", "r|gz") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            if ti.name.endswith("ANNOTATIONS") or ti.name.endswith("LICENSE"):
                ann[ti.name] = tf.extractfile(ti).read().decode("utf-8", "replace")
                continue
            parts = ti.name.split("/")
            if not ti.name.endswith(".wav") or len(parts) < 3 or parts[1] not in ("music", "noise"):
                continue
            if ti.name in H["musan"] or B.held(ti.name):
                skipped["heldout"] += 1
                continue
            kind, name = parts[1], parts[-1]
            if kind == "music" and vocals.get(Path(name).stem, "Y") != "N":
                skipped["musan_music_vocals"] += 1
                continue
            x, sr = sf.read(io.BytesIO(tf.extractfile(ti).read()), dtype="float32", always_2d=True)
            x = B._norm_peak(B.to16(x, sr))
            if kind == "music":
                if len(x) < SR * 30.0:
                    skipped["musan_music_short"] += 1
                    continue
                p = out / "musan_music" / name
                pools["musan_music"].append({"file": str(p.relative_to(REPO)), "src": ti.name, "dur": B._w16(p, x),
                                             "licence": "MUSAN (see musan LICENSE; CC BY 4.0 per OpenSLR 17)"})
            else:
                p = out / "musan_noise" / name
                pools["noise"].append({"file": str(p.relative_to(REPO)), "src": ti.name, "dur": B._w16(p, x),
                                       "licence": "MUSAN noise (CC BY 4.0 / public domain per ANNOTATIONS)"})
    for k, v in ann.items():   # the tar's annotations agree with the cached copy used above
        if "/music/" in k and k.endswith("ANNOTATIONS"):
            for line in v.splitlines():
                t = line.split()
                if len(t) >= 3:
                    assert vocals.get(t[0]) == t[2], (k, t[0])
    lic = {k: v[:400] for k, v in ann.items() if k.endswith("LICENSE")}
    info = json.load(open(TEST / "meta/FSD50K.metadata/eval_clips_info_FSD50K.json"))
    lab = {r["fname"]: set(r["labels"].split(",")) for r in csv.DictReader(open(TEST / "meta/FSD50K.ground_truth/eval.csv"))}
    z = zipfile.ZipFile(B.RAW / "fsd50k/FSD50K.eval_audio.merged.zip")
    for f in sorted(lab, key=int):
        key = f"FSD50K.eval_audio/{f}.wav"
        if key in H["fsd50k_eval"] or B.held(key) or lab[f] & B.FSD_BAD:
            continue
        layer = "fsd_crowd" if lab[f] & B.FSD_CROWD else "fsd_sfx" if lab[f] & B.FSD_SFX else None
        if layer is None:
            continue
        x, sr = sf.read(io.BytesIO(z.read(key)), dtype="float32", always_2d=True)
        x = B.to16(x, sr)
        if len(x) < SR * 0.3:
            continue
        p = out / layer / f"{f}.wav"
        dur = B._w16(p, B._norm_peak(x))
        pools[layer].append({"file": str(p.relative_to(REPO)), "src": key, "dur": dur, "labels": sorted(lab[f]),
                             "licence": info.get(f, {}).get("license", "")})
    pools["fsd_sfx_musan_noise"] = pools.pop("fsd_sfx") + pools.pop("noise")
    summ = {k: {"n": len(v), "hours": round(sum(x["dur"] for x in v) / 3600, 3)} for k, v in pools.items()}
    with B.x_open(pj) as fh:
        json.dump({"pools": pools, "summary": summ, "skipped": dict(skipped), "musan_licence_files": lic,
                   "rule": "complement of bench/HELDOUT_BACKGROUNDS*.txt (and the sha1 rule) within MUSAN music/noise and "
                           "FSD50K eval; test id2 label filters"}, fh, ensure_ascii=False, indent=0)
    print(json.dumps({"summary": summ, "skipped": dict(skipped)}))


# -------------------------------------------------------------------------------------------------------------- mix
def _mix_job(a):
    d, k, s, pools, clips, phase = a
    n = int(s["dur"] * SR)
    sp = np.zeros(n, np.float32)
    act = np.zeros(n, bool)
    for u in s["utts"]:
        x = B.load(clips[u["clip"]]["path"]) * (10 ** (s["spk_gain_db"][u["spk"]] / 20))
        t0 = int(u["start"] * SR)
        sp[t0: t0 + len(x)] += x[: n - t0]
        act[t0: t0 + len(x)] = True
    rng = random.Random(f"{DSALT}bg:{d}:{s['key']}")
    sched = B.bg_schedule(rng, s["dur"])
    g = B.duck_env(act, s["duck_db"])
    res = {"key": s["key"], "conds": {}}
    es = float((sp[act] ** 2).sum())
    if phase == "bg":
        cond = "id2"
        b, used = B.render_bg(sched, pools, cond, n)
        bd = b * g
        eb = float((bd[act] ** 2).sum())
        alpha = math.sqrt(10 ** (s["bvr_db"] / 10) * es / max(eb, 1e-12))
        bd = bd * alpha
        y = sp + bd
        peak = float(np.abs(y).max())
        lim = 1.0
        if peak > 0.99:
            lim = 0.99 / peak
        m = mid(cond, d, k)
        p = sess_path(cond, d, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        sf.write(str(p), y * lim, SR, subtype="PCM_16")
        q = DEV / cond / d / "bg_track" / f"{m}.wav"
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {q}")
        sf.write(str(q), bd * lim, SR, subtype="PCM_16")
        res["conds"][cond] = {"meeting": m, "alpha": round(alpha, 5), "limiter": round(lim, 4),
                              "bvr_db": round(10 * math.log10(float((bd[act] ** 2).sum()) / es), 2),
                              "bvr_all_db": round(10 * math.log10(float((bd ** 2).sum()) / float((sp ** 2).sum())), 2),
                              "events": used}
    else:
        m = mid("clean", d, k)
        p = sess_path("clean", d, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        sf.write(str(p), sp, SR, subtype="PCM_16")
        res["conds"]["clean"] = {"meeting": m}
    ov = np.zeros(n, np.int8)
    for u in s["utts"]:
        t0 = int(u["start"] * SR)
        ov[t0: t0 + int(u["dur"] * SR)] += 1
    res["overlap_share"] = round(float((ov >= 2).sum() / max(1, (ov >= 1).sum())), 4)
    res["schedule"] = sched
    return res


def mix(d: str, phase: str) -> None:
    import multiprocessing as mp
    P = json.load(open(DEV / "src" / d / "plan.json"))
    pools = json.load(open(DEV / "bgsrc/pools.json"))["pools"] if phase == "bg" else {}
    clips = {c["id"]: c for c in map(json.loads, open(DEV / "src" / d / "pool.jsonl", encoding="utf-8"))}
    cond = "clean" if phase == "clean" else "id2"
    if (DEV / cond / d / "meetings.json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {DEV / cond / d / 'meetings.json'}")
    jobs = [(d, k, s, pools, clips, phase) for k, s in enumerate(P["streams"], 1)]
    with mp.get_context("fork").Pool(min(32, len(jobs))) as pool:
        R = {r["key"]: r for r in pool.map(_mix_job, jobs, chunksize=1)}
    srcf, tgtf = ("zh", "en") if d == "zh2en" else ("en", "zh")
    spkref, meetings, refs = {}, {}, []
    C = DEV / cond / d
    (C / "utts").mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(P["streams"], 1):
        r = R[s["key"]]
        m = r["conds"][cond]["meeting"]
        au = sess_path(cond, d, m)
        us = sorted(s["utts"], key=lambda u: u["start"])
        with B.x_open(C / "utts" / f"{m}.jsonl") as fh:
            for j, u in enumerate(us):
                fh.write(json.dumps({"record_id": f"{m}_{j:04d}", "speaker_id": u["spk"], "begin_s": u["start"],
                                     "end_s": round(u["start"] + u["dur"], 3), "hesitation": False, "clip": u["clip"]}) + "\n")
        for j, u in enumerate(us):
            c = clips[u["clip"]]
            uu = {"record_id": f"{m}_{j:04d}", "speaker_id": u["spk"], "begin_s": 0.0, "end_s": u["dur"],
                  "hesitation": False, "human": True, srcf: c["src"], tgtf: c["ref"], "ref": c["ref"], "clip": u["clip"]}
            if d == "zh2en":
                uu["ref_mt"] = c["ref"]
            refs.append({"id": f"{m}_ref_{j:04d}", "meeting": m, "split": "dev", "wtype": "ref", "dir": d,
                         "start_s": u["start"], "dur_ms": round(u["dur"] * 1000), "audio": str(au), "strict_review": False,
                         "utts": [uu]})
        info = {k2: v for k2, v in r["conds"][cond].items() if k2 != "events"}
        meetings[m] = {"stream": s["key"], "cond": cond, "dir": d, "split": "dev", "dur": s["dur"], "K": s["K"],
                       "speakers": s["speakers"], "n_utts": len(us), "overlap_share": r["overlap_share"],
                       "bvr_target_db": s["bvr_db"], "duck_db": s["duck_db"], **info,
                       "twins": {c2: mid(c2, d, k) for c2 in CONDS}}
        if cond == "clean":
            spkref[s["key"]] = {sp: [u["clip"] for u in us if u["spk"] == sp] for sp in s["speakers"]}
    with B.x_open(C / "meetings.json") as fh:
        json.dump(meetings, fh, ensure_ascii=False, indent=1)
    with B.x_open(C / "refs_windows.jsonl") as fh:
        for x in refs:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    if phase == "bg":
        with B.x_open(DEV / "src" / d / "bg_plan.json") as fh:
            json.dump({k: {"schedule": v["schedule"], "events": {"id2": v["conds"]["id2"].get("events", [])}}
                       for k, v in R.items()}, fh)
        b = [R[s["key"]]["conds"]["id2"]["bvr_db"] for s in P["streams"]]
        print(f"{d}: mixed {len(R)} streams (id2); realised BVR median {np.median(b):.1f} dB; limited "
              f"{sum(R[s['key']]['conds']['id2']['limiter'] < 1 for s in P['streams'])}")
    else:
        with B.x_open(DEV / "src" / d / "spkref.json") as fh:
            json.dump(spkref, fh)
        print(f"{d}: clean twin of {len(R)} streams written")


# ---------------------------------------------------------------------------------------------------------- windows
def windows(cond: str, d: str) -> None:
    import torch
    from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, LOOKAHEAD, online_windows
    C = DEV / cond / d
    out, chunk = win_dirs(cond, d)
    wf = out / f"windows_test_online_{d}.jsonl"
    if wf.exists() or chunk.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {wf} / {chunk}")
    for sub in ("audio", "post"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meetings = json.load(open(C / "meetings.json"))
    n = 0
    with B.x_open(wf) as fh:
        for m in sorted(meetings):
            post = torch.load(C / "diar/nemotron_lowlat" / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
            x, sr = sf.read(str(sess_path(cond, d, m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            sess_s = len(x) / sr
            for k, (st, en, j) in enumerate(online_windows(post, 0.6)):
                ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
                close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
                wid = f"{m}_on_{k:04d}"
                wav, pp = out / "audio" / f"{wid}.wav", out / "post" / f"{wid}.npy"
                if wav.exists() or pp.exists():
                    raise SystemExit(f"拒绝覆盖已有输出: {wav}")
                sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
                s = post[f0: f0 + nf]
                if len(s) < nf:
                    s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                np.save(pp, s.astype(np.float16))
                fh.write(json.dumps({"id": wid, "meeting": m, "split": "dev", "wtype": "online", "dir": d,
                                     "start_s": round(ws, 3), "dur_ms": round((we - ws) * 1000),
                                     "audio": str(wav.resolve()), "post": {"lowlat": str(pp.resolve())},
                                     "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                                     "close_detect_ms": round((close - ws) * 1000), "utts": [],
                                     "strict_review": False}) + "\n")
                n += 1
    print(f"{cond} {d}: {n} C2 windows -> {wf}", flush=True)
    from experiments.mist_w3po_v1.data import specclose as S

    def load_meeting(m: str):
        post = torch.load(C / "diar/nemotron_lowlat" / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        x, sr = sf.read(str(sess_path(cond, d, m)), dtype="float32", always_2d=True)
        return post, x.mean(1), sr

    S.DATA, S.DIAR, S.MIX = C, "nemotron_lowlat", DIRS[d][2]
    S.load_meeting = load_meeting
    S.cmd_build(argparse.Namespace(judge="off", base_win=str(wf), out=str(chunk)))


# ------------------------------------------------------------------------------------------------------------ check
def check() -> None:
    out = DEV / "meta/leak_check.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    hs, hc = held_speech()
    tr = idea2_train_speakers()
    H = B.heldout_set()
    res, bad = {}, []
    test_m = set()
    for c in ("id2", "ood1", "clean"):
        for d in DIRS:
            f = TEST / c / d / "meetings.json"
            if f.exists():
                test_m |= set(json.load(open(f)))
    test_clips = {json.loads(l)["id"] for d in DIRS for l in open(TEST / "src" / d / "pool.jsonl", encoding="utf-8")}
    for d in DIRS:
        P = json.load(open(DEV / "src" / d / "plan.json"))
        spk = {s for st in P["streams"] for s in st["speakers"]}
        clips = {u["clip"] for st in P["streams"] for u in st["utts"]}
        r = {"speakers": len(spk), "clips": len(clips), "spk_in_heldout": len(spk & hs), "clip_in_heldout": len(clips & hc),
             "clip_in_test_pool": len(clips & test_clips), "spk_in_idea2_train": len(spk & tr)}
        bg = json.load(open(DEV / "src" / d / "bg_plan.json"))
        srcs = {e["src"] for v in bg.values() for e in v["events"]["id2"]}
        r["bg_files_used"] = len(srcs)
        r["bg_in_heldout"] = len({s for s in srcs if s in H["musan"] or s in H["fsd50k_eval"] or B.held(s)})
        ms = set()
        for c in CONDS:
            ms |= set(json.load(open(DEV / c / d / "meetings.json")))
        r["meetings"] = len(ms)
        r["meeting_ids_in_test"] = len(ms & test_m)
        r["meeting_id_pattern_ok"] = all(re.fullmatch(r"d[bc][ze]\d\d\d", m) for m in ms)
        res[d] = r
        for k in ("spk_in_heldout", "clip_in_heldout", "clip_in_test_pool", "spk_in_idea2_train", "bg_in_heldout",
                  "meeting_ids_in_test"):
            if r[k]:
                bad.append(f"{d}:{k}={r[k]}")
        if not r["meeting_id_pattern_ok"]:
            bad.append(f"{d}:meeting_id_pattern")
    pools = json.load(open(DEV / "bgsrc/pools.json"))["pools"]
    pb = [x["src"] for v in pools.values() for x in v]
    res["pools_in_heldout"] = sum(1 for s in pb if s in H["musan"] or s in H["fsd50k_eval"] or B.held(s))
    if res["pools_in_heldout"]:
        bad.append(f"pools_in_heldout={res['pools_in_heldout']}")
    res["ok"] = not bad
    res["problems"] = bad
    with B.x_open(out) as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    if bad:
        raise SystemExit(f"leak check failed: {bad}")


# ---------------------------------------------------------------------------------------------------------- summary
def summary() -> None:
    out = DEV / "summary.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    for d in DIRS:
        P = json.load(open(DEV / "src" / d / "plan.json"))
        sp = [s for st in P["streams"] for s in st["speakers"]]
        utt = [u for st in P["streams"] for u in st["utts"]]
        r = {"streams": P["n"], "hours": round(P["total_s"] / 3600, 3), "speech_hours": round(sum(u["dur"] for u in utt) / 3600, 3),
             "speakers": len(set(sp)), "speaker_slots": len(sp), "utterances": len(utt),
             "K": dict(Counter(st["K"] for st in P["streams"])),
             "stream_s": {q: round(float(np.percentile([st["dur"] for st in P["streams"]], q)), 1) for q in (0, 50, 100)},
             "utts_per_speaker_stream": {q: float(np.percentile([sum(1 for u in st["utts"] if u["spk"] == s)
                                                                 for st in P["streams"] for s in st["speakers"]], q))
                                         for q in (0, 50, 100)}}
        for cond in CONDS:
            M = json.load(open(DEV / cond / d / "meetings.json"))
            r[cond] = {"meetings": len(M), "first": sorted(M)[0], "last": sorted(M)[-1]}
            if cond != "clean":
                b = [v["bvr_db"] for v in M.values()]
                r[cond]["bvr_db"] = {q: round(float(np.percentile(b, q)), 2) for q in (0, 10, 50, 90, 100)}
                r[cond]["limited_streams"] = sum(v["limiter"] < 1 for v in M.values())
            w, c = win_dirs(cond, d)
            wf = w / f"windows_test_online_{d}.jsonl"
            if wf.exists():
                r[cond]["c2_windows"] = sum(1 for _ in open(wf))
        r["overlap_share_of_speech"] = round(float(np.mean([v["overlap_share"] for v in M.values()])), 4)
        res[d] = r
    res["pools"] = json.load(open(DEV / "bgsrc/pools.json"))["summary"]
    with B.x_open(out) as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["src", "plan", "bgprep", "mix", "windows", "check", "summary"])
    ap.add_argument("--dir", choices=list(DIRS))
    ap.add_argument("--cond", choices=list(CONDS))
    ap.add_argument("--phase", choices=["clean", "bg"])
    a = ap.parse_args()
    if a.step in ("src", "plan", "mix", "windows") and not a.dir:
        raise SystemExit("--dir required")
    if a.step == "src":
        src(a.dir)
    elif a.step == "plan":
        plan(a.dir)
    elif a.step == "bgprep":
        bgprep()
    elif a.step == "mix":
        if not a.phase:
            raise SystemExit("--phase clean|bg required")
        mix(a.dir, a.phase)
    elif a.step == "windows":
        if not a.cond:
            raise SystemExit("--cond required")
        windows(a.cond, a.dir)
    elif a.step == "check":
        check()
    else:
        summary()


if __name__ == "__main__":
    main()
