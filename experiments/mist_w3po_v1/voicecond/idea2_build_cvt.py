"""Idea 2 end-to-end data (e2e prereg §5): `cvt` multi-speaker streams from CoVoST 2 NON-test splits (source audio with
human translations), the same broadcast recipe as idea2_build_streams.py (train / dev background splits, AMI turn
statistics, esports BVR quantiles; held-out benchmark backgrounds already excluded in index/backgrounds.jsonl), and the
teacher-target inputs for the frozen T1c (si_infer --text-policy file items / blocks).

Speakers (Common Voice client_id; CoVoST 2 test split never read; bench/HELDOUT_SPEECH.txt speakers excluded):
  zh source (zh-CN->en): train + validation speakers, int(sha1(client_id), 16) % 10: 0-7 train, 8-9 dev
  en source (en->zh-CN): train split speakers with >= 8 clips, first 700 by sha1 -> train (<= 20 clips each);
                         validation speakers with >= 5 clips, first 80 by sha1 -> dev (<= 5 clips each)
Streams (seed 20261119, ids cvtr_<lang>_<i> train / cvtd_<lang>_<i> dev): zh 120 + 12, en 200 + 12.
Teacher items: uncut segments >= 1.5 s, sha1-ranked, 4000 per language (train) + 400 per language (dev).

    python -I experiments/mist_w3po_v1/voicecond/idea2_build_cvt.py export|streams|refs|teacher [--workers 48]
Root data/open_voicecond_v1/idea2/cvt/ {clips/, index/, streams/, refs/, teacher_in/}; never overwrites.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import experiments.mist_w3po_v1.voicecond.idea2_build_streams as BS  # noqa: E402

I2 = REPO / "data/open_voicecond_v1/idea2"
CV = I2 / "cvt"
HF = I2 / "covost_train/s2tt/covost/hf"
SR = 16000
N_STREAMS = {("train", "zh"): 120, ("dev", "zh"): 12, ("train", "en"): 200, ("dev", "en"): 12}
N_TEACHER = {"train": 4000, "dev": 400}


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def held_ids():
    ids = set()
    for l in open(REPO / "data/open_voicecond_v1/bench/HELDOUT_SPEECH.txt"):
        t = l.rstrip("\n").split("\t")
        if len(t) == 3 and t[0] == "client_id":
            ids.add(t[2])
    return ids


def decode_one(job):
    cid, b, out = job
    if out.exists():
        return cid, float(sf.info(str(out)).duration)
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    tmp = out.parent / f".{out.name}.partial.wav"
    sf.write(str(tmp), x, SR, subtype="PCM_16")
    os.rename(tmp, out)
    return cid, len(x) / SR


def cmd_export(a):
    import pyarrow.parquet as pq
    man = CV / "index/clips.jsonl"
    if man.exists():
        sys.exit(f"refuse to overwrite {man}")
    held = held_ids()
    sel = []   # (lang, split, client_id, id, sentence, translation, parquet file, row)
    # zh source: train + validation, by speaker hash
    for sp in ("train", "validation"):
        for f in sorted(glob.glob(str(HF / f"zh-CN_en/{sp}-*.parquet"))):
            t = pq.read_table(f, columns=["client_id", "id", "sentence", "translation"]).to_pylist()
            for k, r in enumerate(t):
                if r["client_id"] in held:
                    continue
                split = "train" if h(r["client_id"]) % 10 < 8 else "dev"
                sel.append(("zh", split, r["client_id"], r["id"], r["sentence"], r["translation"], f, k))
    # en source
    for sp, split, nspk, cap, minc in (("train", "train", 700, 20, 8), ("validation", "dev", 80, 5, 5)):
        rows, cnt = [], {}
        for f in sorted(glob.glob(str(HF / f"en_zh-CN/{sp}-*.parquet"))):
            t = pq.read_table(f, columns=["client_id", "id", "sentence", "translation"]).to_pylist()
            for k, r in enumerate(t):
                rows.append((r, f, k))
                cnt[r["client_id"]] = cnt.get(r["client_id"], 0) + 1
        spk = sorted([c for c, n in cnt.items() if n >= minc and c not in held], key=h)[:nspk]
        keep, used = set(spk), {}
        for r, f, k in sorted(rows, key=lambda x: h(x[0]["id"])):
            c = r["client_id"]
            if c in keep and used.get(c, 0) < cap:
                used[c] = used.get(c, 0) + 1
                sel.append(("en", split, c, r["id"], r["sentence"], r["translation"], f, k))
    print(f"[export] selected {len(sel)} clips", flush=True)
    (CV / "clips").mkdir(parents=True, exist_ok=True)
    by_file = {}
    for s in sel:
        by_file.setdefault(s[6], []).append(s)
    durs = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for f, ss in by_file.items():
            tab = pq.read_table(f, columns=["audio"])
            col = tab.column("audio")
            jobs = [(s[3], col[s[7]].as_py()["bytes"], CV / "clips" / f"{s[3]}.wav") for s in ss]
            for cid, dur in ex.map(decode_one, jobs, chunksize=16):
                durs[cid] = dur
            print(f"[export] {Path(f).name}: {len(ss)}", flush=True)
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "x", encoding="utf-8") as fh:
        for s in sel:
            fh.write(json.dumps({"id": s[3], "lang": s[0], "split": s[1], "client_id": s[2], "sentence": s[4],
                                 "translation": s[5], "wav": str(CV / "clips" / f"{s[3]}.wav"), "dur": round(durs[s[3]], 3)},
                                ensure_ascii=False) + "\n")
    spk = {}
    for r in map(json.loads, open(man, encoding="utf-8")):
        k = f"cvt:{r['client_id'][:16]}"
        spk.setdefault(k, {"spk": k, "client_id": r["client_id"], "lang": r["lang"], "corpus": "covost2_nontest",
                           "split": r["split"], "utts": []})["utts"].append([r["wav"], r["dur"]])
    with open(CV / "index/speakers_cvt.jsonl", "x", encoding="utf-8") as fh:
        for r in spk.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("speakers", Counter((r["lang"], r["split"]) for r in spk.values()))
    print("hours", {k: round(sum(r['dur'] for r in map(json.loads, open(man)) if (r['lang'], r['split']) == k) / 3600, 2)
                    for k in [("zh", "train"), ("zh", "dev"), ("en", "train"), ("en", "dev")]})


def patch_bs():
    BS.OUT, BS.REFS = CV / "streams", CV / "refs"


def cmd_streams(a):
    patch_bs()
    spk = {json.loads(l)["spk"]: json.loads(l) for l in open(CV / "index/speakers_cvt.jsonl", encoding="utf-8")}
    bg = [json.loads(l) for l in open(I2 / "index/backgrounds.jsonl")]
    ami = json.load(open(REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/ami_turn_stats.json"))
    ami = {"turn_dur": np.array(ami["turn_dur"]), "gap": np.array(ami["gap"])}
    bvrs = BS.bvr_values()
    jobs = []
    for (split, lang), n in N_STREAMS.items():
        pool = {k: r for k, r in spk.items() if r["split"] == split and r["lang"] == lang}
        for i in range(n):
            sid = f"cvt{'r' if split == 'train' else 'd'}_{lang}_{i:04d}"
            jobs.append((sid, split, lang, pool, bg, ami, bvrs, "cvt"))
    BS.OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, st in ex.map(BS.build, jobs, chunksize=1):
            done += 1
            if st != "ok" or done % 50 == 0:
                print(f"[{done}/{len(jobs)}] {sid} {st}", flush=True)
    (BS.OUT / "BUILD_CVT_DONE").write_text("ok\n") if not (BS.OUT / "BUILD_CVT_DONE").exists() else None


def cmd_refs(a):
    patch_bs()
    spk = {json.loads(l)["spk"]: json.loads(l) for l in open(CV / "index/speakers_cvt.jsonl", encoding="utf-8")}
    used = set()
    for m in (CV / "streams").glob("*/meta.json"):
        used |= set(json.load(open(m))["speakers"])
    BS.REFS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(BS.ref_one, [(s, spk[s]) for s in sorted(used)], chunksize=4))
    print(f"refs: {sum(st == 'ok' for _, st in res)} new")


def cmd_teacher(a):
    out = CV / "teacher_in"
    if (out / "DONE").exists():
        sys.exit("teacher_in exists")
    clips = {r["wav"]: r for r in map(json.loads, open(CV / "index/clips.jsonl", encoding="utf-8"))}
    segs = []
    for m in sorted((CV / "streams").glob("cvt*/meta.json")):
        meta = json.load(open(m))
        split = "train" if meta["sid"].startswith("cvtr") else "dev"
        for k, g in enumerate(meta["segments"]):
            if g["cut"] or g["end"] - g["start"] < 1.5 or g["utt"] not in clips:
                continue
            segs.append((split, meta["lang"], meta["sid"], k, g))
    rows = {}
    for split in ("train", "dev"):
        for lang in ("zh", "en"):
            cand = sorted([s for s in segs if s[0] == split and s[1] == lang], key=lambda s: h(f"{s[2]}|{s[3]}"))
            rows[(split, lang)] = cand[:N_TEACHER[split]]
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "voice").mkdir(parents=True, exist_ok=True)
    mixes = {}
    for (split, lang), ss in rows.items():
        d = "zh2en" if lang == "zh" else "en2zh"
        items, blocks = [], []
        for _, _, sid, k, g in sorted(ss, key=lambda x: (x[2], x[3])):
            if sid not in mixes:
                mixes = {sid: sf.read(str(CV / "streams" / sid / "mix.wav"), dtype="float32")[0]}
            iid = f"{sid}__{k:04d}"
            ap = out / "audio" / f"{iid}.wav"
            if not ap.exists():
                x = mixes[sid][int(g["start"] * SR): int(g["end"] * SR)]
                sf.write(str(ap), x, SR, subtype="PCM_16")
            vs = g["spk"].replace(":", "_")
            vp = out / "voice" / f"{vs}.wav"
            if not vp.exists():
                shutil.copyfile(CV / "refs" / vs / "p0.wav", vp)
            dur_ms = int(round((g["end"] - g["start"]) * 1000))
            tr = clips[g["utt"]]["translation"]
            items.append({"id": iid, "audio": str(ap), "dur_ms": dur_ms, "translation": tr, "voice_audio": str(vp),
                          "spk": g["spk"], "lang": lang, "split": split})
            blocks.append({"id": iid, "blocks": [{"write_ms": dur_ms, "text": tr, "final": True, "start_s": 0.0,
                                                  "end_s": round(dur_ms / 1000, 3)}]})
        for name, rr in (("items", items), ("blocks", blocks)):
            with open(out / f"{name}_{d}_{split}.jsonl", "x", encoding="utf-8") as fh:
                for r in rr:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[teacher] {d} {split}: {len(items)} items", flush=True)
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "streams", "refs", "teacher"])
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    {"export": cmd_export, "streams": cmd_streams, "refs": cmd_refs, "teacher": cmd_teacher}[a.cmd](a)
