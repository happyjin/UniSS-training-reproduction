"""OpenBroadcast-SI v1: the shared open full-chain TEST benchmark of the voice-condition ideas
(reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md §7.2d-§7.2f; doc reports/mist_w3po_v1/voicecond_bench.zh-CN.md).
TEST ONLY: nothing is selected or tuned on it. Open data only (esports data enter only as the prompt-BVR quantiles).

Conditions (same CoVoST 2 TEST speech streams in all three; only the background differs):
  id2    held-out MUSAN music / noise + held-out FSD50K eval clips (HELDOUT_BACKGROUNDS.txt)   meeting ids ob[ze]NNN
  ood1   FMA-small (Instrumental) + DEMAND (5 crowded environments) + ESC-50 (8 SFX classes,
         Freesound ids in FSD50K excluded) -- corpora neither idea uses                      meeting ids oo[ze]NNN
  clean  no background (the clean twin)                                                     meeting ids oc[ze]NNN
Directions: zh2en (CoVoST 2 zh-CN->en test; session "mix", <m>.wav) and en2zh (en->zh-CN test; session "ihm_mix",
<m>.Mix-Headset.wav), the layouts game_build.py / gstar_talker_v6 expect.

Steps (each refuses to overwrite its outputs; CPU only):
  heldout --part fsd50k|musan   HELDOUT_BACKGROUNDS.txt / HELDOUT_BACKGROUNDS_MUSAN.txt: a file is held out iff sha1("OpenBroadcast-SI-v1|" + <path>) % 1000 < 150,
                   for every MUSAN music / noise file and every FSD50K eval clip (15 %, fixed rule)
  stats            meta/stats.json: AMI train turn statistics (non-hesitation segments >= 1 s; same-speaker segments
                   < 1.5 s apart form a turn): turn durations, within-turn pauses, gaps at speaker changes (negative =
                   overlap); the pooled esports prompt-BVR values (game_muffle_v1 game_*.jsonl prompt.bvr_db)
  src --dir D      CoVoST 2 TEST clips -> src/<D>/clips/<id>.wav (16 kHz, edge silence trimmed, -26 dBFS active RMS)
                   + src/<D>/pool.jsonl. zh2en: speakers with >= 8 test clips; en2zh: speakers with 3 (the maximum)
  plan --dir D     src/<D>/plan.json: streams of 2-4 speakers (K balanced), turns from the AMI statistics
  bgprep           bgsrc/<pool>/*.wav (16 kHz mono) + bgsrc/pools.json for the five pools of each background condition
  mix --dir D --phase clean|bg   (clean twin first; then id2 + ood1)  the conditions' data dirs <cond>/<D>/: session_audio, meetings.json, refs_windows.jsonl (one
                   window per utterance, real CV client_id speakers, human CoVoST references), utts/<m>.jsonl,
                   bg_track/<m>.wav (the scaled, ducked background as mixed); src/<D>/spkref.json (speaker -> clean clips)
  windows --cond C --dir D   C2 online windows (gap 0.6, game_build.build_windows' recipe) on diar/nemotron_lowlat
                   -> eval_outputs/mist_w3po_v1/data/windows_obsi_<C>_<D>_c2g06_v1/windows_test_online_<D>.jsonl and
                   the chunk-protocol windows (specclose.cmd_build, judge off) -> ..._c2g06_chunk_v1/
  summary          bench/summary.json (hours, speakers, overlap, BVR per condition / direction)

    python -m experiments.mist_w3po_v1.voicecond.bench_build <step> [--dir zh2en|en2zh] [--cond id2|ood1|clean]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "data/open_voicecond_v1/bench"
RAW = REPO / "data/open_voicecond_v1/raw"
RAWO = REPO / "data/open_voicecond_v1/raw_ood"
HF = BENCH / "covost_hf/s2tt/covost/hf"
DD = REPO / "eval_outputs/mist_w3po_v1/data"
SR = 16000
SALT = "OpenBroadcast-SI-v1|"
CONDS = {"id2": "b", "ood1": "o", "clean": "c"}
DIRS = {"zh2en": ("z", "zh-CN_en", "mix"), "en2zh": ("e", "en_zh-CN", "ihm_mix")}
TARGET_S = 7200.0            # stream time per direction
STREAM_MAX_S = 300.0
SPEECH_DBFS = -26.0
FSD_CROWD = {"Crowd", "Cheering", "Applause", "Clapping", "Chatter"}
FSD_SFX = {"Explosion", "Gunshot_and_gunfire", "Whoosh_and_swoosh_and_swish", "Boom", "Fireworks"}
# clips with intelligible speech or singing are excluded; crowd shouting / chatter (Human_voice, Shout, Chatter) is kept
# (v0 also excluded Human_voice and Chatter: only 14 held-out crowd clips, ~100 s; kept as bgsrc.v0_strict_speech_filter)
FSD_BAD = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
           "Conversation", "Singing", "Speech_synthesizer", "Male_singing", "Female_singing", "Narration_and_monologue"}
DEMAND_ENVS = ["PCAFETER", "PRESTO", "PSTATION", "SPSQUARE", "TMETRO"]
ESC_SFX = {"fireworks", "glass_breaking", "door_wood_knock", "can_opening", "clock_alarm", "thunderstorm", "mouse_click",
           "keyboard_typing"}
# background layers: bed (persistent music), amb (crowd / ambience bursts), sfx (sparse effects)
LAYER_POOLS = {"id2": {"bed": "musan_music", "amb": "fsd_crowd", "sfx": "fsd_sfx_musan_noise"},
               "ood1": {"bed": "fma_instrumental", "amb": "demand_crowded", "sfx": "esc50_sfx"}}


def held(path: str) -> bool:
    return int(hashlib.sha1((SALT + path).encode()).hexdigest(), 16) % 1000 < 150


def x_open(p: Path, mode: str = "x"):
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode, encoding="utf-8")


def mid(cond: str, d: str, k: int) -> str:
    return f"o{CONDS[cond]}{DIRS[d][0]}{k:03d}"


def sess_path(cond: str, d: str, m: str) -> Path:
    s = DIRS[d][2]
    return BENCH / cond / d / "session_audio" / s / (f"{m}.wav" if s == "mix" else f"{m}.Mix-Headset.wav")


def to16(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def frame_db(x: np.ndarray, hop: int = 160) -> np.ndarray:
    n = max(1, len(x) // hop)
    e = (x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)
    return 10 * np.log10(e + 1e-12)


def active_rms_db(x: np.ndarray, rel: float = 30.0) -> float:
    db = frame_db(x)
    a = db > db.max() - rel
    return float(10 * np.log10((10 ** (db[a] / 10)).mean() + 1e-12))


# ---------------------------------------------------------------------------------------------------------- heldout
def musan_members() -> list[str]:
    import tarfile
    lst = BENCH / "meta/musan_members.txt"
    if not lst.exists():
        with tarfile.open(RAW / "musan/musan.tar.gz", "r|gz") as tf, x_open(lst) as fh:
            for ti in tf:
                if ti.isfile():
                    fh.write(ti.name + "\n")
    return [l.strip() for l in open(lst, encoding="utf-8") if l.strip()]


def heldout(part: str) -> None:
    """part fsd50k -> HELDOUT_BACKGROUNDS.txt (FSD50K eval list + the rule, written first: FSD50K was ready);
    part musan -> HELDOUT_BACKGROUNDS_MUSAN.txt (once musan.tar.gz is downloaded)"""
    hdr = ("# OpenBroadcast-SI v1 held-out background files (TEST ONLY; never use them for training, fitting or\n"
           "# selection). Rule: held out iff int(sha1('OpenBroadcast-SI-v1|' + <path>).hexdigest(), 16) % 1000 < 150,\n"
           "# <path> = MUSAN tar member (musan/music/..., musan/noise/...) or 'FSD50K.eval_audio/<fname>.wav'.\n"
           "# The MUSAN list is HELDOUT_BACKGROUNDS_MUSAN.txt (same rule).\n"
           "# OOD-1 corpora (FMA, DEMAND, ESC-50) are reserved WHOLE for the benchmark: neither idea may use them.\n")
    if part == "fsd50k":
        out = BENCH / "HELDOUT_BACKGROUNDS.txt"
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        fsd = [r["fname"] for r in csv.DictReader(open(BENCH / "meta/FSD50K.ground_truth/eval.csv", encoding="utf-8"))]
        hf = sorted((f for f in fsd if held(f"FSD50K.eval_audio/{f}.wav")), key=int)
        with x_open(out) as fh:
            fh.write(hdr + f"# FSD50K eval: {len(hf)} of {len(fsd)} clips.\n")
            for f in hf:
                fh.write(f"fsd50k_eval\tFSD50K.eval_audio/{f}.wav\n")
        print(f"held out: FSD50K eval {len(hf)}/{len(fsd)} -> {out}")
        return
    out = BENCH / "HELDOUT_BACKGROUNDS_MUSAN.txt"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if not (RAW / "musan/musan.tar.gz.DONE").exists():
        raise SystemExit("MUSAN download not done (musan.tar.gz.DONE missing)")
    mus = [m for m in musan_members() if m.endswith(".wav") and m.split("/")[1] in ("music", "noise")]
    hm = sorted(m for m in mus if held(m))
    with x_open(out) as fh:
        fh.write(hdr + f"# MUSAN music+noise: {len(hm)} of {len(mus)} files.\n")
        for m in hm:
            fh.write(f"musan\t{m}\n")
    print(f"held out: MUSAN {len(hm)}/{len(mus)} -> {out}")


def heldout_set() -> dict[str, set]:
    H = defaultdict(set)
    for f in ("HELDOUT_BACKGROUNDS.txt", "HELDOUT_BACKGROUNDS_MUSAN.txt"):
        for l in open(BENCH / f, encoding="utf-8"):
            if l.startswith("#") or not l.strip():
                continue
            k, p = l.rstrip("\n").split("\t")
            H[k].add(p)
    return H


# ------------------------------------------------------------------------------------------------------------ stats
def stats() -> None:
    out = BENCH / "meta/stats.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sys.path.insert(0, str(REPO))
    import pyarrow.dataset as pads
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation
    D = REPO / "data/ami_multispk_si_v1"
    split = json.load(open(D / "meetings.json"))
    src = pads.dataset(D / "index/source_ihm", format="parquet", partitioning="hive").to_table(
        columns=["meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
    M = defaultdict(list)
    for u in src:
        if split.get(u["meeting_id"]) == "train" and not is_hesitation(u["transcript"]) and u["end_time"] - u["begin_time"] >= 1.0:
            M[u["meeting_id"]].append((u["begin_time"], u["end_time"], u["speaker_id"]))
    turn, intra, change, ov = [], [], [], []
    for m, us in M.items():
        us.sort()
        T = []
        for b, e, s in us:
            if T and T[-1][2] == s and b - T[-1][1] < 1.5:
                intra.append(max(0.0, b - T[-1][1]))
                T[-1][1] = max(T[-1][1], e)
            else:
                T.append([b, e, s])
        turn += [t[1] - t[0] for t in T]
        change += [c[0] - a[1] for a, c in zip(T, T[1:]) if a[2] != c[2]]
        n = int(max(e for _, e, _ in us) * 100) + 1
        cnt = np.zeros(n, np.int16)
        for b, e, s in us:
            cnt[int(b * 100): int(e * 100)] += 1
        ov.append(float((cnt >= 2).sum() / max(1, (cnt >= 1).sum())))
    bvr = []
    for f in sorted((REPO / "eval_outputs/mist_w3po_v1/game_muffle_v1").glob("game_*.jsonl")):
        bvr += [r["prompt"]["bvr_db"] for r in map(json.loads, open(f)) if r.get("prompt", {}).get("bvr_db") is not None]
    P = lambda x: {str(q): round(float(np.percentile(x, q)), 3) for q in (5, 10, 25, 50, 75, 90, 95)}  # noqa: E731
    res = {"ami": {"meetings": len(M), "turn_s": [round(x, 3) for x in turn], "intra_gap_s": [round(x, 3) for x in intra],
                   "change_gap_s": [round(x, 3) for x in change], "overlap_share_of_speech": round(float(np.mean(ov)), 4),
                   "summary": {"turn": P(turn), "intra": P(intra), "change": P(change),
                               "change_overlap_frac": round(float(np.mean(np.array(change) < 0)), 4)}},
           "bvr_db": {"values": [round(x, 3) for x in bvr], "n": len(bvr), "summary": P(bvr),
                      "source": "eval_outputs/mist_w3po_v1/game_muffle_v1/game_{zh2en,en2zh}.jsonl prompt.bvr_db (statistic only)"}}
    with x_open(out) as fh:
        json.dump(res, fh)
    print(json.dumps({"ami": res["ami"]["summary"], "ami_overlap": res["ami"]["overlap_share_of_speech"],
                      "bvr": res["bvr_db"]["summary"], "n_bvr": len(bvr)}))


# -------------------------------------------------------------------------------------------------------------- src
def trim_norm(x: np.ndarray) -> tuple[np.ndarray, dict] | None:
    db = frame_db(x)
    if db.max() < -50:
        return None
    keep = np.where(db > max(db.max() - 40.0, -55.0))[0]
    a, b = max(0, keep[0] - 15), min(len(db), keep[-1] + 16)
    y = x[a * 160: b * 160].astype(np.float64)
    if len(y) < SR * 1.5 or len(y) > SR * 12.0:
        return None
    clip = float(np.mean(np.abs(x) > 0.999))
    if clip > 1e-3:
        return None
    g = 10 ** ((SPEECH_DBFS - active_rms_db(y)) / 20)
    y = y * g
    if np.abs(y).max() > 0.98:
        y = y * 0.98 / np.abs(y).max()
    return y.astype(np.float32), {"trim_s": [round(a * 0.01, 2), round(b * 0.01, 2)], "gain_db": round(20 * math.log10(g), 2)}


def _src_job(a):
    r, out = a
    try:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        t = trim_norm(to16(x, sr))
        if t is None:
            return None
        y, meta = t
        p = out / f"{r['id']}.wav"
        sf.write(str(p), y, SR, subtype="PCM_16")
        return {"id": r["id"], "spk": r["client_id"], "src": r["sentence"].strip(), "ref": r["translation"].strip(),
                "dur": round(len(y) / SR, 3), "path": str(p.relative_to(REPO)), **meta}
    except Exception as e:  # noqa: BLE001
        print("decode failed", r["id"], repr(e)[:200], flush=True)
        return None


def src(d: str) -> None:
    import multiprocessing as mp
    import pyarrow.parquet as pq
    out = BENCH / "src" / d / "clips"
    pool = BENCH / "src" / d / "pool.jsonl"
    if pool.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {pool}")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((HF / DIRS[d][1]).glob("test-*.parquet"))
    cnt = Counter()
    for f in files:
        cnt.update(pq.read_table(f, columns=["client_id"]).column("client_id").to_pylist())
    want = {s for s, n in cnt.items() if (n >= 8 if d == "zh2en" else n == 3)}
    rows = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            if r["client_id"] in want and r["sentence"].strip() and r["translation"].strip():
                rows.append(r)
    print(f"{d}: {len(rows)} clips of {len(want)} speakers to prepare", flush=True)
    with mp.get_context("fork").Pool(48) as P:
        res = [x for x in P.map(_src_job, [(r, out) for r in rows], chunksize=8) if x]
    with x_open(pool) as fh:
        for x in sorted(res, key=lambda x: (x["spk"], x["id"])):
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{d}: {len(res)} clips kept of {len(rows)} ({sum(x['dur'] for x in res) / 3600:.2f} h)")


# ------------------------------------------------------------------------------------------------------------- plan
def sequence(rng: random.Random, spk: dict[str, list[dict]], A: dict) -> list[dict]:
    """turns from the AMI statistics: target turn duration ~ AMI turns; within-turn pauses ~ AMI (0.1-1.5 s); the gap
    at a speaker change ~ AMI change gaps clipped to [-3, 4] s (negative = overlap), but the new turn starts no earlier
    than the middle of the previous turn's last utterance and after every earlier turn (at most two talkers at once).
    Next speaker: any other speaker with clips left, weighted by clips left. Stops at STREAM_MAX_S or exhaustion."""
    q = {s: list(v) for s, v in spk.items()}
    for v in q.values():
        rng.shuffle(v)
    utts, prev, last, end_all, end_before_prev = [], None, None, 0.0, 0.0
    while True:
        cands = [s for s in sorted(q) if q[s] and s != prev]
        if not cands:
            break
        s = rng.choices(cands, weights=[len(q[c]) for c in cands])[0]
        target = rng.choice(A["turn_s"])
        if prev is None:
            t = 0.6
        else:
            g = min(4.0, max(-3.0, rng.choice(A["change_gap_s"])))
            t = max(end_all + g, last["start"] + 0.5 * last["dur"], end_before_prev)
        t0, turn_end = t, t
        end_before_prev = end_all
        while q[s]:
            c = q[s].pop()
            u = {"clip": c["id"], "spk": s, "start": round(t, 3), "dur": c["dur"]}
            utts.append(u)
            last = u
            turn_end = t + c["dur"]
            if turn_end - t0 >= target or not q[s]:
                break
            t = turn_end + min(1.5, max(0.1, rng.choice(A["intra_gap_s"])))
        end_all = max(end_all, turn_end)
        prev = s
        if end_all >= STREAM_MAX_S:
            break
    return utts


def plan(d: str) -> None:
    out = BENCH / "src" / d / "plan.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    st = json.load(open(BENCH / "meta/stats.json"))
    A = st["ami"]
    pool = [json.loads(l) for l in open(BENCH / "src" / d / "pool.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for c in pool:
        by[c["spk"]].append(c)
    need = 8 if d == "zh2en" else 3
    spks = sorted(s for s, v in by.items() if len(v) >= need)
    rng = random.Random(f"{SALT}plan:{d}")
    rng.shuffle(spks)
    streams, total, k = [], 0.0, 0
    while total < TARGET_S and spks:
        K = [2, 3, 4][k % 3]
        if len(spks) < K:
            break
        if d == "zh2en":   # fill towards STREAM_MAX_S: the K speakers with the most clips among the next 3K in line
            cand = spks[: 3 * K]
            pick = sorted(cand, key=lambda s: -len(by[s]))[:K]
        else:
            pick = spks[:K]
        for s in pick:
            spks.remove(s)
        r = random.Random(f"{SALT}stream:{d}:{k}")
        utts = sequence(r, {s: by[s] for s in pick}, A)
        dur = round(max(u["start"] + u["dur"] for u in utts) + 0.8, 3)
        gains = {s: round(r.uniform(-2.0, 2.0), 2) for s in pick}
        streams.append({"key": f"s{k + 1:03d}", "K": K, "speakers": pick, "spk_gain_db": gains, "dur": dur, "utts": utts})
        total += dur
        k += 1
    bv = np.array(st["bvr_db"]["values"])
    n = len(streams)
    perm = list(range(n))
    random.Random(f"{SALT}bvr:{d}").shuffle(perm)
    for i, s in enumerate(streams):
        q = 0.05 + 0.90 * (perm[i] + 0.5) / n          # stratified quantiles of the pooled esports prompt BVR
        s["bvr_q"] = round(q, 4)
        s["bvr_db"] = round(float(np.quantile(bv, q)), 2)
        s["duck_db"] = round(random.Random(f"{SALT}duck:{d}:{i}").uniform(6.0, 12.0), 2)
    with x_open(out) as fh:
        json.dump({"dir": d, "n": n, "total_s": round(total, 1), "streams": streams}, fh, ensure_ascii=False)
    print(f"{d}: {n} streams, {total / 3600:.2f} h, speakers {sum(s['K'] for s in streams)}, "
          f"utts {sum(len(s['utts']) for s in streams)}, K {Counter(s['K'] for s in streams)}")


# ----------------------------------------------------------------------------------------------------------- bgprep
def _w16(p: Path, x: np.ndarray) -> float:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    sf.write(str(p), x, SR, subtype="PCM_16")
    return round(len(x) / SR, 3)


def _norm_peak(x: np.ndarray) -> np.ndarray:
    m = float(np.abs(x).max()) if len(x) else 0.0
    return (x * (0.9 / m)).astype(np.float32) if m > 0 else x


def bgprep() -> None:
    import tarfile
    out = BENCH / "bgsrc"
    pj = out / "pools.json"
    if pj.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {pj}")
    H = heldout_set()
    pools = defaultdict(list)
    # MUSAN held-out: music without vocals (ANNOTATIONS), every noise file
    ann = {}
    want = {m for m in H["musan"]}
    with tarfile.open(RAW / "musan/musan.tar.gz", "r|gz") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            if ti.name.endswith("ANNOTATIONS") or ti.name.endswith("LICENSE"):
                ann[ti.name] = tf.extractfile(ti).read().decode("utf-8", "replace")
                continue
            if ti.name in want:
                x, sr = sf.read(io.BytesIO(tf.extractfile(ti).read()), dtype="float32", always_2d=True)
                kind = ti.name.split("/")[1]
                name = ti.name.split("/")[-1]
                p = out / f"musan_{kind}" / name
                dur = _w16(p, _norm_peak(to16(x, sr)))
                pools[f"musan_{kind}_all"].append({"file": str(p.relative_to(REPO)), "src": ti.name, "dur": dur})
    vocals = {}
    for k, v in ann.items():
        if "/music/" in k and k.endswith("ANNOTATIONS"):
            for line in v.splitlines():
                f = line.split()
                if len(f) >= 3:
                    vocals[f[0]] = f[2]
    lic = {k: v[:400] for k, v in ann.items() if k.endswith("LICENSE")}
    for r in pools.pop("musan_music_all"):
        stem = Path(r["src"]).stem
        if vocals.get(stem, "Y") == "N" and r["dur"] >= 30.0:
            pools["musan_music"].append(dict(r, licence="MUSAN (see musan LICENSE; CC BY 4.0 per OpenSLR 17)"))
    noise = [dict(r, licence="MUSAN noise (CC BY 4.0 / public domain per ANNOTATIONS)") for r in pools.pop("musan_noise_all")]
    # FSD50K held-out eval clips of the chosen classes (no speech / singing labels)
    info = json.load(open(BENCH / "meta/FSD50K.metadata/eval_clips_info_FSD50K.json"))
    lab = {r["fname"]: set(r["labels"].split(",")) for r in csv.DictReader(open(BENCH / "meta/FSD50K.ground_truth/eval.csv"))}
    z = zipfile.ZipFile(RAW / "fsd50k/FSD50K.eval_audio.merged.zip")
    for f in sorted(lab, key=int):
        key = f"FSD50K.eval_audio/{f}.wav"
        if key not in H["fsd50k_eval"] or lab[f] & FSD_BAD:
            continue
        layer = "fsd_crowd" if lab[f] & FSD_CROWD else "fsd_sfx" if lab[f] & FSD_SFX else None
        if layer is None:
            continue
        x, sr = sf.read(io.BytesIO(z.read(key)), dtype="float32", always_2d=True)
        x = to16(x, sr)
        if len(x) < SR * 0.3:
            continue
        p = out / layer / f"{f}.wav"
        dur = _w16(p, _norm_peak(x))
        pools[layer].append({"file": str(p.relative_to(REPO)), "src": key, "dur": dur, "labels": sorted(lab[f]),
                             "licence": info.get(f, {}).get("license", "")})
    pools["fsd_sfx_musan_noise"] = pools.pop("fsd_sfx") + noise
    # OOD-1: FMA-small Instrumental
    zm = zipfile.ZipFile(RAWO / "fma/fma_metadata.zip")
    raw = zm.read("fma_metadata/tracks.csv").decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    h0, h1 = rows[0], rows[1]
    col = {(a, b): i for i, (a, b) in enumerate(zip(h0, h1))}
    ci = lambda a, b: col[(a, b)]  # noqa: E731
    musan_artists = set()
    for k, v in ann.items():
        if "/music/" in k and k.endswith("ANNOTATIONS"):
            for line in v.splitlines():
                f = line.split()
                if len(f) >= 4:
                    musan_artists.add(f[3].replace("_", " ").strip().lower())
    fma = []
    for r in rows[3:]:
        if len(r) < len(h0):
            continue
        if r[ci("set", "subset")] == "small" and r[ci("track", "genre_top")] == "Instrumental":
            fma.append((int(r[0]), r[ci("artist", "name")], r[ci("track", "license")], r[ci("track", "title")]))
    zf = zipfile.ZipFile(RAWO / "fma/fma_small.zip")
    names = set(zf.namelist())
    skipped = Counter()
    for tid, artist, licence, title in sorted(fma):
        key = f"fma_small/{tid:06d}"[:13] + f"/{tid:06d}.mp3"
        if key not in names:
            skipped["missing"] += 1
            continue
        if artist.replace("_", " ").strip().lower() in musan_artists:
            skipped["musan_artist"] += 1
            continue
        try:
            x, sr = sf.read(io.BytesIO(zf.read(key)), dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001
            skipped["decode"] += 1
            continue
        x = to16(x, sr)
        if len(x) < SR * 25:
            skipped["short"] += 1
            continue
        p = out / "fma_instrumental" / f"{tid:06d}.wav"
        dur = _w16(p, _norm_peak(x))
        pools["fma_instrumental"].append({"file": str(p.relative_to(REPO)), "src": key, "dur": dur, "artist": artist,
                                          "title": title, "licence": licence})
    # DEMAND: channel 1 of the crowded public environments
    for e in DEMAND_ENVS:
        zd = zipfile.ZipFile(RAWO / f"demand/{e}_16k.zip")
        key = next(n for n in zd.namelist() if n.endswith("ch01.wav"))
        x, sr = sf.read(io.BytesIO(zd.read(key)), dtype="float32", always_2d=True)
        p = out / "demand_crowded" / f"{e}_ch01.wav"
        dur = _w16(p, _norm_peak(to16(x, sr)))
        pools["demand_crowded"].append({"file": str(p.relative_to(REPO)), "src": f"{e}_16k.zip:{key}", "dur": dur,
                                        "licence": "CC BY 4.0 (zenodo 1227121)"})
    # ESC-50 SFX classes, Freesound ids that are in FSD50K (dev or eval) excluded
    fsd_ids = set(lab) | {r["fname"] for r in csv.DictReader(open(BENCH / "meta/FSD50K.ground_truth/dev.csv"))}
    ze = zipfile.ZipFile(RAWO / "esc50/ESC-50-master.zip")
    meta = list(csv.DictReader(io.StringIO(ze.read("ESC-50-master/meta/esc50.csv").decode("utf-8"))))
    for r in sorted(meta, key=lambda r: r["filename"]):
        if r["category"] not in ESC_SFX:
            continue
        if r["src_file"] in fsd_ids:
            skipped["esc_in_fsd50k"] += 1
            continue
        x, sr = sf.read(io.BytesIO(ze.read(f"ESC-50-master/audio/{r['filename']}")), dtype="float32", always_2d=True)
        x = to16(x, sr)
        db = frame_db(x)
        keep = np.where(db > db.max() - 40)[0]
        if len(keep) == 0:
            continue
        x = x[max(0, keep[0] - 5) * 160: (keep[-1] + 6) * 160]
        if len(x) < SR * 0.3:
            continue
        p = out / "esc50_sfx" / r["filename"]
        dur = _w16(p, _norm_peak(x))
        pools["esc50_sfx"].append({"file": str(p.relative_to(REPO)), "src": r["filename"], "dur": dur,
                                   "category": r["category"], "freesound_id": r["src_file"],
                                   "licence": "ESC-50 (CC BY-NC 3.0; per-clip Freesound licences)"})
    summ = {k: {"n": len(v), "hours": round(sum(x["dur"] for x in v) / 3600, 3)} for k, v in pools.items()}
    with x_open(pj) as fh:
        json.dump({"pools": pools, "summary": summ, "skipped": dict(skipped), "musan_licence_files": lic}, fh,
                  ensure_ascii=False, indent=0)
    print(json.dumps({"summary": summ, "skipped": dict(skipped)}))


# -------------------------------------------------------------------------------------------------------------- mix
_CACHE: dict = {}


def load(p: str) -> np.ndarray:
    if p not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        x, sr = sf.read(str(REPO / p), dtype="float32", always_2d=True)
        _CACHE[p] = to16(x, sr)
    return _CACHE[p]


def fade(x: np.ndarray, fi: float, fo: float) -> np.ndarray:
    x = x.copy()
    a, b = min(len(x), int(fi * SR)), min(len(x), int(fo * SR))
    if a:
        x[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b:
        x[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return x


def unit_rms(x: np.ndarray) -> np.ndarray:
    return (x / (10 ** (active_rms_db(x) / 20) + 1e-9)).astype(np.float32)


def bg_schedule(rng: random.Random, T: float) -> dict:
    """corpus-independent background schedule (identical for id2 and ood1): bed = 30 s excerpts with 2 s crossfades;
    amb = Poisson bursts (1 / 20 s, 2-8 s, +-0.3 s fades, -3..+6 dB re bed); sfx = Poisson events (1 / 15 s, <= 3 s,
    -6..+4 dB re bed). u_file / u_off pick the file and the offset in whichever corpus."""
    bed, t = [], 0.0
    while t < T:
        bed.append({"t": round(t, 3), "len": 30.0, "u_file": rng.random(), "u_off": rng.random(),
                    "gain_db": round(rng.uniform(-2, 2), 2)})
        t += 28.0
    ev = {}
    for layer, rate, lo, hi, g0, g1 in (("amb", 20.0, 2.0, 8.0, -3.0, 6.0), ("sfx", 15.0, 0.3, 3.0, -6.0, 4.0)):
        L, t = [], rng.expovariate(1 / rate)
        while t < T:
            L.append({"t": round(t, 3), "len": round(rng.uniform(lo, hi), 3), "u_file": rng.random(), "u_off": rng.random(),
                      "gain_db": round(rng.uniform(g0, g1), 2)})
            t += rng.expovariate(1 / rate)
        ev[layer] = L
    return {"bed": bed, **ev}


def render_bg(sched: dict, pools: dict, cond: str, n: int) -> tuple[np.ndarray, list]:
    b = np.zeros(n, np.float32)
    used = []
    for layer in ("bed", "amb", "sfx"):
        P = pools[LAYER_POOLS[cond][layer]]
        for e in sched[layer]:
            r = P[int(e["u_file"] * len(P)) % len(P)]
            x = load(r["file"])
            L = int(e["len"] * SR)
            if layer == "sfx":
                L = min(L, len(x))
                off = int(e["u_off"] * max(0, len(x) - L)) if len(x) > L else 0
            else:
                off = int(e["u_off"] * max(0, len(x) - L))
            seg = x[off: off + L]
            if layer == "bed" and len(seg) < L:      # short track: loop it
                seg = np.tile(x, 1 + L // max(1, len(x)))[:L]
            if len(seg) < SR * 0.1:
                continue
            seg = unit_rms(seg) * (10 ** (e["gain_db"] / 20))
            seg = fade(seg, 2.0 if layer == "bed" else 0.3 if layer == "amb" else 0.02,
                       2.0 if layer == "bed" else 0.3 if layer == "amb" else 0.15)
            t0 = int(e["t"] * SR)
            if t0 >= n:
                continue
            seg = seg[: n - t0]
            b[t0: t0 + len(seg)] += seg
            used.append({"layer": layer, "t": e["t"], "src": r["src"], "off_s": round(off / SR, 3), "len_s": round(len(seg) / SR, 3),
                         "gain_db": e["gain_db"]})
    return b, used


def duck_env(act: np.ndarray, depth_db: float) -> np.ndarray:
    """sidechain ducking on 10 ms frames: attack 50 ms, release 400 ms; gain -depth_db * envelope"""
    hop = 160
    n = int(math.ceil(len(act) / hop))
    a = np.array([act[i * hop: (i + 1) * hop].any() for i in range(n)], np.float32)
    env, e = np.zeros(n, np.float32), 0.0
    ka, kr = 1 - math.exp(-1 / 5), 1 - math.exp(-1 / 40)
    for i in range(n):
        e += (ka if a[i] > e else kr) * (a[i] - e)
        env[i] = e
    g = 10 ** (-depth_db * env / 20)
    return np.repeat(g, hop)[: len(act)].astype(np.float32)


def _mix_job(a):
    d, k, s, pools, clips, phase = a
    n = int(s["dur"] * SR)
    sp = np.zeros(n, np.float32)
    act = np.zeros(n, bool)
    for u in s["utts"]:
        x = load(clips[u["clip"]]["path"]) * (10 ** (s["spk_gain_db"][u["spk"]] / 20))
        t0 = int(u["start"] * SR)
        sp[t0: t0 + len(x)] += x[: n - t0]
        act[t0: t0 + len(x)] = True
    rng = random.Random(f"{SALT}bg:{d}:{s['key']}")
    sched = bg_schedule(rng, s["dur"])
    g = duck_env(act, s["duck_db"])
    res = {"key": s["key"], "conds": {}}
    es = float((sp[act] ** 2).sum())
    for cond in (("id2", "ood1") if phase == "bg" else ()):
        b, used = render_bg(sched, pools, cond, n)
        bd = b * g
        eb = float((bd[act] ** 2).sum())
        alpha = math.sqrt(10 ** (s["bvr_db"] / 10) * es / max(eb, 1e-12))
        bd = bd * alpha
        y = sp + bd
        peak = float(np.abs(y).max())
        lim = 1.0
        if peak > 0.99:   # rare: scale the whole mixture (speech and background alike) under 0.99
            lim = 0.99 / peak
        m = mid(cond, d, k)
        p = sess_path(cond, d, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        sf.write(str(p), y * lim, SR, subtype="PCM_16")
        q = BENCH / cond / d / "bg_track" / f"{m}.wav"
        q.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(q), bd * lim, SR, subtype="PCM_16")
        bvr_real = 10 * math.log10(float((bd[act] ** 2).sum()) / es)
        res["conds"][cond] = {"meeting": m, "alpha": round(alpha, 5), "limiter": round(lim, 4), "bvr_db": round(bvr_real, 2),
                              "bvr_all_db": round(10 * math.log10(float((bd ** 2).sum()) / float((sp ** 2).sum())), 2),
                              "events": used}
    if phase == "clean":
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
    """phase clean: the clean twin (no background corpus needed); phase bg: id2 and ood1 (needs bgsrc/pools.json)"""
    import multiprocessing as mp
    P = json.load(open(BENCH / "src" / d / "plan.json"))
    pools = json.load(open(BENCH / "bgsrc/pools.json"))["pools"] if phase == "bg" else {}
    clips = {c["id"]: c for c in map(json.loads, open(BENCH / "src" / d / "pool.jsonl", encoding="utf-8"))}
    conds = ["clean"] if phase == "clean" else ["id2", "ood1"]
    for cond in conds:
        if (BENCH / cond / d / "meetings.json").exists():
            raise SystemExit(f"拒绝覆盖已有输出: {BENCH / cond / d / 'meetings.json'}")
    jobs = [(d, k, s, pools, clips, phase) for k, s in enumerate(P["streams"], 1)]
    with mp.get_context("fork").Pool(32) as pool:
        R = {r["key"]: r for r in pool.map(_mix_job, jobs, chunksize=1)}
    srcf, tgtf = ("zh", "en") if d == "zh2en" else ("en", "zh")
    spkref = {}
    for cond in conds:
        meetings, refs = {}, []
        C = BENCH / cond / d
        (C / "utts").mkdir(parents=True, exist_ok=True)
        for k, s in enumerate(P["streams"], 1):
            r = R[s["key"]]
            m = r["conds"][cond]["meeting"]
            au = sess_path(cond, d, m)
            us = sorted(s["utts"], key=lambda u: u["start"])
            with x_open(C / "utts" / f"{m}.jsonl") as fh:
                for j, u in enumerate(us):
                    fh.write(json.dumps({"record_id": f"{m}_{j:04d}", "speaker_id": u["spk"], "begin_s": u["start"],
                                         "end_s": round(u["start"] + u["dur"], 3), "hesitation": False,
                                         "clip": u["clip"]}) + "\n")
            for j, u in enumerate(us):
                c = clips[u["clip"]]
                uu = {"record_id": f"{m}_{j:04d}", "speaker_id": u["spk"], "begin_s": 0.0, "end_s": u["dur"],
                      "hesitation": False, "human": True, srcf: c["src"], tgtf: c["ref"], "ref": c["ref"], "clip": u["clip"]}
                if d == "zh2en":
                    uu["ref_mt"] = c["ref"]
                refs.append({"id": f"{m}_ref_{j:04d}", "meeting": m, "split": "test", "wtype": "ref", "dir": d,
                             "start_s": u["start"], "dur_ms": round(u["dur"] * 1000), "audio": str(au), "strict_review": False,
                             "utts": [uu]})
            info = {k2: v for k2, v in r["conds"][cond].items() if k2 != "events"}
            meetings[m] = {"stream": s["key"], "cond": cond, "dir": d, "dur": s["dur"], "K": s["K"], "speakers": s["speakers"],
                           "n_utts": len(us), "overlap_share": r["overlap_share"], "bvr_target_db": s["bvr_db"],
                           "duck_db": s["duck_db"], **info,
                           "twins": {c2: mid(c2, d, k) for c2 in CONDS}}
            if cond == "clean":
                spkref[s["key"]] = {sp: [u["clip"] for u in us if u["spk"] == sp] for sp in s["speakers"]}
        with x_open(C / "meetings.json") as fh:
            json.dump(meetings, fh, ensure_ascii=False, indent=1)
        with x_open(C / "refs_windows.jsonl") as fh:
            for x in refs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    if phase == "bg":
        with x_open(BENCH / "src" / d / "bg_plan.json") as fh:
            json.dump({k: {"schedule": v["schedule"], "events": {c: v["conds"][c].get("events", []) for c in ("id2", "ood1")}}
                       for k, v in R.items()}, fh)
        b = [R[s["key"]]["conds"]["id2"]["bvr_db"] for s in P["streams"]]
        print(f"{d}: mixed {len(R)} streams (id2, ood1); realised BVR id2 median {np.median(b):.1f} dB")
    else:
        with x_open(BENCH / "src" / d / "spkref.json") as fh:
            json.dump(spkref, fh)
        print(f"{d}: clean twin of {len(R)} streams written")


# ---------------------------------------------------------------------------------------------------------- windows
def windows(cond: str, d: str) -> None:
    import torch
    from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, LOOKAHEAD, online_windows
    C = BENCH / cond / d
    out = DD / f"windows_obsi_{cond}_{d}_c2g06_v1"
    wf = out / f"windows_test_online_{d}.jsonl"
    if wf.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {wf}")
    for sub in ("audio", "post"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meetings = json.load(open(C / "meetings.json"))
    n = 0
    with x_open(wf) as fh:
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
    print(f"{cond} {d}: {n} C2 windows -> {wf}", flush=True)
    from experiments.mist_w3po_v1.data import specclose as S

    def load_meeting(m: str):
        post = torch.load(C / "diar/nemotron_lowlat" / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        x, sr = sf.read(str(sess_path(cond, d, m)), dtype="float32", always_2d=True)
        return post, x.mean(1), sr

    S.DATA, S.DIAR, S.MIX = C, "nemotron_lowlat", DIRS[d][2]
    S.load_meeting = load_meeting
    S.cmd_build(argparse.Namespace(judge="off", base_win=str(wf), out=str(DD / f"windows_obsi_{cond}_{d}_c2g06_chunk_v1")))


# ---------------------------------------------------------------------------------------------------------- summary
def summary() -> None:
    out = BENCH / "summary.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    for d in DIRS:
        P = json.load(open(BENCH / "src" / d / "plan.json"))
        sp = [s for st in P["streams"] for s in st["speakers"]]
        utt = [u for st in P["streams"] for u in st["utts"]]
        r = {"streams": P["n"], "hours": round(P["total_s"] / 3600, 3), "speech_hours": round(sum(u["dur"] for u in utt) / 3600, 3),
             "speakers": len(set(sp)), "utterances": len(utt), "K": dict(Counter(st["K"] for st in P["streams"])),
             "stream_s": {q: round(float(np.percentile([st["dur"] for st in P["streams"]], q)), 1) for q in (0, 50, 100)},
             "utts_per_speaker": {q: float(np.percentile(list(Counter(u["spk"] for u in utt).values()), q)) for q in (0, 50, 100)}}
        for cond in CONDS:
            M = json.load(open(BENCH / cond / d / "meetings.json"))
            r[cond] = {"meetings": len(M), "first": sorted(M)[0], "last": sorted(M)[-1]}
            if cond != "clean":
                b = [v["bvr_db"] for v in M.values()]
                r[cond]["bvr_db"] = {q: round(float(np.percentile(b, q)), 2) for q in (0, 10, 50, 90, 100)}
                r[cond]["limited_streams"] = sum(v["limiter"] < 1 for v in M.values())
        r["overlap_share_of_speech"] = round(float(np.mean([v["overlap_share"] for v in M.values()])), 4)
        res[d] = r
    with x_open(out) as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["heldout", "stats", "src", "plan", "bgprep", "mix", "windows", "summary"])
    ap.add_argument("--dir", choices=list(DIRS))
    ap.add_argument("--cond", choices=list(CONDS))
    ap.add_argument("--phase", choices=["clean", "bg"])
    ap.add_argument("--part", choices=["fsd50k", "musan"])
    a = ap.parse_args()
    if a.step in ("src", "plan", "mix", "windows") and not a.dir:
        raise SystemExit("--dir required")
    if a.step == "windows" and not a.cond:
        raise SystemExit("--cond required")
    if a.step == "heldout":
        if a.part not in ("fsd50k", "musan"):
            raise SystemExit("--part fsd50k|musan required")
        heldout(a.part)
    {"stats": stats, "bgprep": bgprep, "summary": summary}.get(a.step, lambda: None)()
    if a.step == "src":
        src(a.dir)
    elif a.step == "plan":
        plan(a.dir)
    elif a.step == "mix":
        if not a.phase:
            raise SystemExit("--phase clean|bg required")
        mix(a.dir, a.phase)
    elif a.step == "windows":
        windows(a.cond, a.dir)


if __name__ == "__main__":
    main()
