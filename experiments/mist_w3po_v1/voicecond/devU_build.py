"""Robust line, UniST in-domain dev D-U (reports/mist_w3po_v1/robust_devU_20261120.zh-CN.md). CPU only; open, non-test
data only; refuses an existing output directory (eval_outputs/mist_w3po_v1/devU/data).

Source: the UniST pilot manifest (data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source), rows of the
UniST held-out validation subset crc32(id utf-8) % 50 == 0 -- the subset wave14_data.py excluded from training.
Per direction (zh2en = src_lang cmn, en2zh = eng):
  1. candidates: held-out rows with non-empty transcription and translation whose normalised source text (wave14_data.norm)
     is not in any test / evaluation string set (wave14_data.harvest, the same sources as the training exclusion) and not
     equal to the source text of any wave-14 training row; ranked by sha1("20261120|devU|<dir>|<id>");
  2. the first POOL = 3000 candidates are trim_norm'ed (bench_build.trim_norm: edge trim, -26 dBFS, 1.5-12 s), as for training;
  3. duration matching to S1 dev (a): for every clean (a) sentence of that direction (300, ascending duration), the unused
     trimmed candidate with the nearest duration (ties: rank order) -> 300 sentences;
  4. noisy twin: wave1_data.mix_one (the recipe of S1 dev (a) noisy: bench_build background timeline bed / amb / sfx over the
     Idea 2 background index split "dev", BVR = quantile of the OpenBroadcast-SI esports prompt-BVR values at q ~ U(0.05,
     0.95), sentence-level scaling, no ducking, limiter), without the bg-only / gap context, RNG
     random.Random(sha1("20261120|devU|mix|<id>")). The training noise is split "train" (wave14_train.py) and the test
     backgrounds are HELDOUT_BACKGROUNDS* (asserted absent by wave1_data.bg_pools).
Output (wave2_S1/data format): items_<dir>.jsonl {id, audio, dur_ms, translation}, meta_<dir>.jsonl {id, set="u", cond,
pair, stream, ovl=false, dur_ms}, audio/<dir>/<uid>.{clean,noisy}.wav (PCM16 16 kHz), manifest_<dir>.jsonl (source row,
BVR, alpha, limiter, background events), build.json, verify.json.

    <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/devU_build.py --procs 96
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as BB  # noqa: E402
from experiments.mist_w3po_v1.voicecond import wave1_data as W1D  # noqa: E402
from experiments.mist_w3po_v1.voicecond import wave14_data as W14  # noqa: E402

OUT = REPO / "eval_outputs/mist_w3po_v1/devU/data"
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
TRAIN = REPO / "eval_outputs/mist_w3po_v1/wave14/data"
SEED = "20261120|devU"
SR = 16000
POOL = 3000
N = 300
_P: dict = {}


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _trim_job(r):
    x, sr = sf.read(r["src_audio"], dtype="float32", always_2d=True)
    t = BB.trim_norm(BB.to16(x, sr))
    if t is None:
        return None
    return r, t[0]


def _mix_job(a):
    uid, d, clean_path = a
    s = sf.read(str(clean_path), dtype="float32")[0]
    m = W1D.mix_one(s, random.Random(h(f"{SEED}|mix|{uid}")), _P["dev"], _P["bv"], with_context=False)
    p = OUT / "audio" / d / f"{uid}.noisy.wav"
    W1D.w16(p, m["noisy"])
    return uid, str(p), {k: m[k] for k in ("bvr_q", "bvr_target_db", "bvr_db", "alpha", "limiter", "events")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=96)
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {OUT}")
    t0 = time.time()
    cats, info = W14.harvest(a.procs)
    allt = set().union(*cats.values())
    print(f"test strings {len(allt)} {time.time() - t0:.0f} s", flush=True)
    parts = sorted(os.listdir(W14.SRC / "parts"))
    with mp.get_context("fork").Pool(len(parts)) as P:
        rows = [r for part in P.map(W14._read_part, parts) for r in part]
    print(f"manifest rows {len(rows)} {time.time() - t0:.0f} s", flush=True)
    (OUT / "audio").mkdir(parents=True)
    _P["dev"], _P["bv"] = W1D.bg_pools("dev"), W1D.bvr_values()
    build = {"seed": SEED, "pool": POOL, "n": N, "test_sources": info, "per_dir": {},
             "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
             "noise": "Idea 2 background index split 'dev' via wave1_data.bg_pools (training uses 'train'; HELDOUT_BACKGROUNDS* asserted absent)"}
    verify = {"per_dir": {}}
    for d in ("zh2en", "en2zh"):
        train = [json.loads(l) for l in open(TRAIN / f"rows_{d}.jsonl", encoding="utf-8")]
        train_ids = {r["id"] for r in train}
        train_src = {W14.norm(r["src"]) for r in train}
        train_ref = {W14.norm(r["ref"]) for r in train}
        c = collections.Counter()
        cand = []
        for r in rows:
            if W14.DIR_OF.get(r["src_lang"]) != d or not W14.held_out(r["id"]):
                continue
            c["heldout_rows"] += 1
            if not r["src"] or not r["ref"]:
                c["empty_text"] += 1
                continue
            n = W14.norm(r["src"])
            hit = [k for k, v in cats.items() if n in v]
            for k in hit:
                c[f"overlap_{k}"] += 1
            if hit:
                c["overlap_test_any"] += 1
                continue
            if n in train_src:
                c["overlap_train_src"] += 1
                continue
            cand.append(r)
        c["eligible"] = len(cand)
        cand.sort(key=lambda r: h(f"{SEED}|{d}|{r['id']}"))
        rank = {r["id"]: k for k, r in enumerate(cand)}
        with mp.get_context("fork").Pool(a.procs) as P:
            res = [x for x in P.map(_trim_job, cand[:POOL], chunksize=8) if x]
        c["pool_trimmed_ok"] = len(res)
        target = sorted(m["dur_ms"] for m in map(json.loads, open(S1 / f"meta_{d}.jsonl", encoding="utf-8"))
                        if m["set"] == "a" and m["cond"] == "clean")
        assert len(target) == N, (d, len(target))
        avail = sorted(((len(y) * 1000 // SR, rank[r["id"]], r["id"]) for r, y in res))
        byid = {r["id"]: (r, y) for r, y in res}
        used, sel = set(), []
        for t in target:
            best = min((x for x in avail if x[2] not in used), key=lambda x: (abs(x[0] - t), x[1]))
            used.add(best[2])
            sel.append(best[2])
        (OUT / "audio" / d).mkdir()
        items, meta, man, jobs = [], [], {}, []
        for k, i in enumerate(sel):
            r, y = byid[i]
            uid = f"u{k:03d}__{i}"
            pc = OUT / "audio" / d / f"{uid}.clean.wav"
            W1D.w16(pc, y)
            dur = int(round(len(y) / SR * 1000))
            man[uid] = {"uid": uid, "unist_id": i, "src": r["src"], "ref": r["ref"], "unist_audio": r["src_audio"],
                        "dur_ms": dur, "clean": str(pc)}
            jobs.append((uid, d, pc))
        with mp.get_context("fork").Pool(a.procs) as P:
            mixed = P.map(_mix_job, jobs, chunksize=4)
        for uid, pn, m in mixed:
            man[uid].update(noisy=pn, **m)
        for uid, m in man.items():
            for cond in ("clean", "noisy"):
                sid = f"u_{cond}__{uid}"
                items.append({"id": sid, "audio": m[cond], "dur_ms": m["dur_ms"], "translation": m["ref"]})
                meta.append({"id": sid, "set": "u", "cond": cond, "pair": uid, "stream": uid, "ovl": False,
                             "dur_ms": m["dur_ms"]})
        W1D.wjsonl(OUT / f"items_{d}.jsonl", items)
        W1D.wjsonl(OUT / f"meta_{d}.jsonl", meta)
        W1D.wjsonl(OUT / f"manifest_{d}.jsonl", man.values())
        durs = np.array([m["dur_ms"] for m in man.values()]) / 1000
        bvr = np.array([m["bvr_db"] for m in man.values()])
        build["per_dir"][d] = {**dict(c), "selected": len(man), "minutes": round(float(durs.sum()) / 60, 2),
                               "corpus_mix": dict(collections.Counter(m["unist_id"].rsplit("_", 1)[0]
                                                                      for m in man.values()).most_common())}
        # verification
        sel_ids = [m["unist_id"] for m in man.values()]
        v = {"n_items": len(items), "n_pairs": len(man),
             "all_heldout_crc32_mod50": all(W14.held_out(i) for i in sel_ids),
             "id_in_training_rows": sum(i in train_ids for i in sel_ids),
             "src_in_training_src": sum(W14.norm(m["src"]) in train_src for m in man.values()),
             "ref_in_training_ref": sum(W14.norm(m["ref"]) in train_ref for m in man.values()),
             "src_in_test_strings": {k: sum(W14.norm(m["src"]) in cats[k] for m in man.values()) for k in cats},
             "ref_in_test_strings": {k: sum(W14.norm(m["ref"]) in cats[k] for m in man.values()) for k in cats},
             "dur_s": {q: round(float(np.percentile(durs, q)), 2) for q in (0, 10, 50, 90, 100)},
             "s1a_dur_s": {q: round(float(np.percentile(np.array(target) / 1000, q)), 2) for q in (0, 10, 50, 90, 100)},
             "bvr_db": {q: round(float(np.percentile(bvr, q)), 2) for q in (5, 25, 50, 75, 95)},
             "limiter_active": int(sum(m["limiter"] < 1 for m in man.values())),
             "background_files": len({e["src"] for m in man.values() for e in m["events"]})}
        peaks, dur_ok = [], 0
        for m in man.values():
            xc, _ = sf.read(m["clean"], dtype="float32")
            xn, _ = sf.read(m["noisy"], dtype="float32")
            dur_ok += int(len(xc) == len(xn) and abs(len(xc) / SR * 1000 - m["dur_ms"]) <= 1)
            peaks.append(float(np.abs(xn).max()))
        v["clean_noisy_same_length_and_dur_ms"] = dur_ok
        v["noisy_peak_max"] = round(max(peaks), 4)
        held = W1D.heldout_bg_paths()
        v["events_in_heldout_backgrounds"] = sum(e["src"].split("/", 1)[1] in held or BB.held(e["src"].split("/", 1)[1])
                                                 for m in man.values() for e in m["events"] if "/" in e["src"])
        verify["per_dir"][d] = v
        print(f"{d}: {json.dumps(build['per_dir'][d], ensure_ascii=False)}\n  verify {json.dumps(v, ensure_ascii=False)}"
              f" {time.time() - t0:.0f} s", flush=True)
    with open(OUT / "build.json", "x", encoding="utf-8") as fh:
        json.dump(build, fh, ensure_ascii=False, indent=1)
    with open(OUT / "verify.json", "x", encoding="utf-8") as fh:
        json.dump(verify, fh, ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
