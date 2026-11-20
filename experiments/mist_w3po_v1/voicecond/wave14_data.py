"""Robust wave 14 (reports/mist_w3po_v1/robust_wave14_unist_prereg.zh-CN.md): the training sentences from the UniST pilot
(data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source), in wave 5's row/clip format.

Per manifest row: dir = zh2en if src_lang == cmn, en2zh if eng; src = transcription, ref = translation (both non-empty);
audio = the row's own `source_audio` path (16 kHz FLAC). Exclusions (asserted, counted in data/stats.json):
  * UniST held-out validation: crc32(id utf-8) % 50 == 0;
  * overlap with any evaluation / test item: normalised source text (NFKC, lower, only str.isalnum() characters)
    equal to any string harvested from the test sources below (CoVoST 2 dev + test both directions, S1, OB-SI bench /
    bench_dev, Idea 2 cvt, OOD-3, RealSI, ACL 60/60, AliMeeting, AMI, esports). UniST has no original filename,
    Common Voice clip id or speaker field, so text is the only usable key (recorded in stats.json).
Then per direction rank by sha1("20261120|w14|<dir>|<id>"), take the first 120000, bench_build.trim_norm each clip
(edge trim, -26 dBFS, 1.5-12 s) and write PCM16 16 kHz; failures are dropped (as in wave 5).
Output (refuses to overwrite): eval_outputs/mist_w3po_v1/wave14/data/{rows_<dir>.jsonl, clips/<dir>/<id>.wav, stats.json}.
    <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/wave14_data.py --procs 96
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
import zlib
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as BB  # noqa: E402

SRC = REPO / "data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave14/data"
N_PER_DIR = 120000
SR = 16000
DIR_OF = {"cmn": "zh2en", "eng": "en2zh"}
CV_TRAIN = REPO / "data/open_voicecond_v1/idea2/covost_train/s2tt/covost/hf"
CV_TEST = REPO / "data/open_voicecond_v1/bench/covost_hf/s2tt/covost/hf"
D = "data/open_voicecond_v1"
# test / evaluation sources: category -> glob patterns (json / jsonl), relative to the repo
TEST_GLOBS = {
    "s1_dev": ["eval_outputs/mist_w3po_v1/wave2_S1/data/*.jsonl", "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/*.jsonl"],
    "obsi_bench": [f"{D}/bench/**/*.json", f"{D}/bench/**/*.jsonl", f"{D}/bench_dev/**/*.json", f"{D}/bench_dev/**/*.jsonl"],
    "idea2_cvt": [f"{D}/idea2/cvt/index/*.jsonl", f"{D}/idea2/cvt/refs/*/info.json"],
    "ood3": [f"{D}/idea2/ood3/*.jsonl"],
    "realsi": ["data/raw/realsi/RealSI_data/data/*/json/*.json"],
    "acl6060": ["data/acl6060_dev_v1/*.jsonl", "data/acl6060_dev_v1/*.json"],
    "alimeeting": ["data/alimeeting_si_v1/*.jsonl", "data/alimeeting_si_v1/*.json", "data/alimeeting_eval_si_v1/*.jsonl",
                   "data/alimeeting_eval_si_v1/*.json", "data/diarist_alimeeting_refs_v1/*.jsonl",
                   "data/alimeeting_si_v1/utts/**/*.jsonl", "data/alimeeting_eval_si_v1/utts/**/*.jsonl"],
    "ami": ["data/ami_multispk_si_v1/*.jsonl", "data/ami_multispk_si_v1/*.json", "data/ami_multispk_si_v1/index/**/*.jsonl"],
    "esports": ["data/game_esports_si_v1/**/*.json", "data/game_esports_si_v1/**/*.jsonl"],
}
MAX_FILE = 400 * 2**20
ID_LIKE = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", s).lower() if c.isalnum())


def held_out(i: str) -> bool:
    return zlib.crc32(i.encode("utf-8")) % 50 == 0


def keep_str(s: str) -> bool:
    if "/" in s or "\\" in s:
        return False
    t = s.strip()
    if ID_LIKE.match(t) and (any(c.isdigit() for c in t) or "_" in t):
        return False
    return True


def walk(o, acc: set) -> None:
    if isinstance(o, str):
        if keep_str(o):
            n = norm(o)
            if n:
                acc.add(n)
    elif isinstance(o, dict):
        for v in o.values():
            walk(v, acc)
    elif isinstance(o, list):
        for v in o:
            walk(v, acc)


def _harvest_file(p: str) -> tuple[str, set, str | None]:
    acc: set = set()
    try:
        if os.path.getsize(p) > MAX_FILE:
            return p, acc, "too large, skipped"
        if p.endswith(".jsonl"):
            with open(p, encoding="utf-8") as fh:
                for l in fh:
                    l = l.strip()
                    if l:
                        try:
                            walk(json.loads(l), acc)
                        except json.JSONDecodeError:
                            pass
        else:
            walk(json.load(open(p, encoding="utf-8")), acc)
        return p, acc, None
    except Exception as e:  # unreadable file: recorded
        return p, acc, f"{type(e).__name__}: {e}"


def harvest(procs: int) -> tuple[dict, dict]:
    import pyarrow.parquet as pq
    cats: dict[str, set] = {}
    info: dict[str, dict] = {}
    # CoVoST 2 dev (HF validation) + test, both directions: sentence + translation; ids for the HELDOUT clip check
    cv = set()
    cv_ids = set()
    nfile = 0
    for root, split in ((CV_TRAIN, "validation"), (CV_TEST, "test")):
        for f in sorted(glob.glob(str(root / "*" / f"{split}-*.parquet"))):
            t = pq.read_table(f, columns=["id", "sentence", "translation"]).to_pylist()
            nfile += 1
            for r in t:
                cv_ids.add(r["id"])
                for k in ("sentence", "translation"):
                    if r[k] and norm(r[k]):
                        cv.add(norm(r[k]))
    cats["covost_dev_test"] = cv
    info["covost_dev_test"] = {"files": nfile, "strings": len(cv), "clip_ids": len(cv_ids)}
    for cat, pats in TEST_GLOBS.items():
        files = sorted({p for pat in pats for p in glob.glob(str(REPO / pat), recursive=True) if os.path.isfile(p)})
        acc: set = set()
        errs = {}
        with mp.get_context("fork").Pool(procs) as P:
            for p, a, e in P.imap_unordered(_harvest_file, files, chunksize=8):
                acc |= a
                if e:
                    errs[str(Path(p).relative_to(REPO))] = e
        cats[cat] = acc
        info[cat] = {"files": len(files), "strings": len(acc), "errors": errs}
        assert files, f"no files for test source {cat}"
    held = [l.rstrip("\n").split("\t") for l in open(REPO / D / "bench/HELDOUT_SPEECH.txt", encoding="utf-8")
            if not l.startswith("#")]
    hclips = {t[2] for t in held if len(t) == 3 and t[0] == "clip"}
    hc_found = {c for c in hclips if c in cv_ids or c + ".mp3" in cv_ids}
    info["heldout_speech"] = {
        "speakers": sum(1 for t in held if len(t) == 3 and t[0] == "client_id"), "clips": len(hclips),
        "clips_found_in_covost_dev_test": len(hc_found), "clips_not_found": len(hclips - hc_found),
        "speaker_match": "not possible: the UniST manifest has no speaker / client_id field",
        "clip_match": "UniST ids are internal sequence numbers (e.g. CommonVoice_EN_0000499569) with no original filename; "
                      "held-out clips are covered through the CoVoST test texts"}
    return cats, info


def _read_part(part: str) -> list[dict]:
    out = []
    with open(SRC / "parts" / part / "manifest.jsonl", "rb") as fh:
        for l in fh:
            r = json.loads(l)
            out.append({"id": r["id"], "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                        "src": (r.get("transcription") or "").strip(), "ref": (r.get("translation") or "").strip(),
                        "src_audio": r["source_audio"], "src_dur_ms": r.get("source_duration_ms")})
    return out


def _job(a):
    r, d = a
    x, sr = sf.read(r["src_audio"], dtype="float32", always_2d=True)
    t = BB.trim_norm(BB.to16(x, sr))
    if t is None:
        return None
    y, _ = t
    p = OUT / "clips" / d / f"{r['id']}.wav"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    sf.write(str(p), y, SR, subtype="PCM_16")
    return {"id": r["id"], "dir": d, "spk": None, "src": r["src"], "ref": r["ref"], "dur": round(len(y) / SR, 3),
            "audio": str(p.relative_to(REPO)), "unist_audio": r["src_audio"], "corpus": re.sub(r"_\d+$", "", r["id"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=96)
    a = ap.parse_args()
    for f in ("rows_zh2en.jsonl", "rows_en2zh.jsonl", "stats.json"):
        if (OUT / f).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {OUT / f}")
    t0 = time.time()
    cats, info = harvest(a.procs)
    allt = set().union(*cats.values())
    print(f"test strings: {len(allt)} ({ {k: len(v) for k, v in cats.items()} }) {time.time() - t0:.0f} s", flush=True)
    parts = sorted(os.listdir(SRC / "parts"))
    with mp.get_context("fork").Pool(len(parts)) as P:
        rows = [r for part in P.map(_read_part, parts) for r in part]
    print(f"manifest rows: {len(rows)} {time.time() - t0:.0f} s", flush=True)
    st = {"manifest_rows": len(rows), "test_sources": info, "per_dir": {}}
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate UniST ids"
    for d in ("zh2en", "en2zh"):
        man = OUT / f"rows_{d}.jsonl"
        rs = [r for r in rows if DIR_OF.get(r["src_lang"]) == d]
        c = collections.Counter()
        c["rows"] = len(rs)
        cand = []
        for r in rs:
            bad = False
            if not r["src"] or not r["ref"]:
                c["empty_text"] += 1
                bad = True
            if held_out(r["id"]):
                c["unist_heldout_crc32_mod50"] += 1
                bad = True
            n = norm(r["src"])
            hit = [k for k, v in cats.items() if n in v]
            for k in hit:
                c[f"overlap_{k}"] += 1
                if r["id"].startswith("CommonVoice_EN"):
                    c[f"overlap_{k}__CommonVoice_EN"] += 1
            if hit:
                c["overlap_any"] += 1
                bad = True
            if bad:
                c["excluded_total"] += 1
            else:
                cand.append(r)
        c["eligible"] = len(cand)
        cand.sort(key=lambda r: h(f"20261120|w14|{d}|{r['id']}"))
        sel = cand[:N_PER_DIR]
        # assertions: nothing selected is held out or overlaps any test text
        assert all(not held_out(r["id"]) for r in sel)
        assert all(norm(r["src"]) not in allt for r in sel)
        assert all(r["src"] and r["ref"] for r in sel)
        assert len(sel) == min(N_PER_DIR, len(cand))
        (OUT / "clips" / d).mkdir(parents=True, exist_ok=True)
        with mp.get_context("fork").Pool(a.procs) as P:
            res = P.map(_job, [(r, d) for r in sel], chunksize=32)
        keep = [x for x in res if x]
        c["selected"] = len(sel)
        c["trim_norm_failed"] = len(sel) - len(keep)
        c["kept"] = len(keep)
        with open(man, "x", encoding="utf-8") as fh:
            for x in keep:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        hours = sum(x["dur"] for x in keep) / 3600
        st["per_dir"][d] = {**dict(c), "hours": round(hours, 2),
                            "corpus_mix_kept": dict(collections.Counter(x["corpus"] for x in keep).most_common())}
        print(f"{d}: {json.dumps(st['per_dir'][d], ensure_ascii=False)} {time.time() - t0:.0f} s", flush=True)
    st["key"] = 'sha1("20261120|w14|<dir>|<id>") ascending, first 120000, then trim_norm'
    with open(OUT / "stats.json", "x", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
