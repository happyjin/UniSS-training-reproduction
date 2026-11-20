"""OOD single-speaker test sets v1 (reports/mist_w3po_v1/robust_test_prereg_20261120.zh-CN.md sections 3-4).
TEST ONLY: nothing is trained, fitted, selected or tuned on them. CPU only; refuses an existing output directory.

Two sources, both cut as wave2_s1_prep.py (b) cuts OpenBroadcast-SI dev: one reference utterance = one sentence of a
single-speaker stream (stream = meeting x speaker), cut at its reference times, kept only when no OTHER speaker's
utterance (hesitations included) overlaps it anywhere in the meeting's full timeline (no crosstalk in the cut).

  OOD-3 (data/open_voicecond_v1/idea2/ood3/<set>_ref.jsonl, AMI test EN2002d + IS1009c, AliMeeting test R8009_M8021 +
  R8009_M8024; every utterance of every reference window, cut from the window audio at [begin_s, end_s)):
    en   en2zh  AMI original audio (e2e_v0 windows)        ref = zh    full timeline = AMI source_ihm index (all utts)
    ali  zh2en  AliMeeting far-field (alimeeting_si_v1)    ref = ref (human, DiariST)   full = alimeeting_si_v1/utts
    ami  zh2en  Chinese-dubbed AMI (synthetic speech)      ref = ref (AMI English)      full = e2e_gstar_zh2en_v1
                                                                                         windows_test_fullref_v9
    kept: not hesitation, duration >= 1 s, reference non-empty and in the target language (CJK for en2zh, Latin
    letters for zh2en: drops untranslated fragments), no crosstalk; cond = "real".
  OpenBroadcast-SI TEST (data/open_voicecond_v1/bench/{clean,id2,ood1}/<dir>/refs_windows.jsonl, one utterance per
  window, cut from the session mix at the window): kept when no other speaker's turn overlaps it (meeting utts/*.jsonl);
  the three conditions are the same plan (asserted: same turns, same times, same references) -> one-to-one pairs.

Writes (FLOAT wav = the decoded samples, channels averaged):
  <out>/audio/<set>/<cond>/<dir>/<id>.wav
  <out>/items_<dir>.jsonl  {id, audio, dur_ms, translation, src_text}  (si_thinker_stream / si_infer --items-file)
  <out>/meta_<dir>.jsonl   {id, set, cond, pair, stream, ovl, dur_ms}   (wave2_S1/data/meta_*.jsonl conventions)
  <out>/items/<set>_<cond>_<dir>.jsonl  the same rows per subset
  <out>/manifest_<dir>.jsonl  per item: source window, absolute times, speaker, source / reference text, audio source
  <out>/build_info.json, <out>/verify.json

    python experiments/mist_w3po_v1/voicecond/ood_single_build.py build
    python experiments/mist_w3po_v1/voicecond/ood_single_build.py verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/open_voicecond_v1/ood_single_v1"
SR = 16000
OOD3 = REPO / "data/open_voicecond_v1/idea2/ood3"
BENCH = REPO / "data/open_voicecond_v1/bench"
O3 = {"en": {"dir": "en2zh", "ref": "zh", "src": "en", "full": "ami_ihm"},
      "ali": {"dir": "zh2en", "ref": "ref", "src": "zh", "full": "ali"},
      "ami": {"dir": "zh2en", "ref": "ref", "src": "zh", "full": "ami_dub"}}
OBSI_CONDS = ("clean", "id2", "ood1")
TRAIN_ROWS = {"wave5_covost_erm": "eval_outputs/mist_w3po_v1/wave5/M/data/rows_{d}.jsonl",
              "wave14_unist": "eval_outputs/mist_w3po_v1/wave14/data/rows_{d}.jsonl"}
TGT_RE = {"en2zh": re.compile(r"[\u4e00-\u9fff]"), "zh2en": re.compile(r"[A-Za-z]")}
MIN_NORM = 10          # text-overlap check also reported for normalised texts of >= 10 characters (short phrases match by chance)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(s: str) -> str:   # = wave14_data.norm
    return "".join(c for c in unicodedata.normalize("NFKC", s or "").lower() if c.isalnum())


def read_mono(p) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, p
    return x.mean(axis=1)


def full_timeline(kind: str, meetings: set) -> dict:
    """meeting -> [(begin_s, end_s, speaker, record_id)] over the WHOLE meeting"""
    T = defaultdict(list)
    if kind == "ami_ihm":
        import pyarrow.dataset as pads
        ds = pads.dataset(REPO / "data/ami_multispk_si_v1/index/source_ihm", format="parquet", partitioning="hive")
        for u in ds.to_table(columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time"]).to_pylist():
            if u["meeting_id"] in meetings:
                T[u["meeting_id"]].append((u["begin_time"], u["end_time"], u["speaker_id"], u["record_id"]))
    elif kind == "ali":
        for m in meetings:
            for u in jl(REPO / f"data/alimeeting_si_v1/utts/{m}.jsonl"):
                T[m].append((u["begin_s"], u["end_s"], u["speaker_id"], u["record_id"]))
    elif kind == "ami_dub":
        for r in jl(REPO / "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_fullref_v9.jsonl"):
            if r["meeting"] in meetings:
                for u in r["utts"]:
                    T[r["meeting"]].append((r["start_s"] + u["begin_s"], r["start_s"] + u["end_s"], u["speaker_id"], u["record_id"]))
    return T


def crosstalk(T: list, b: float, e: float, spk: str) -> bool:
    return any(sp != spk and tb < e and te > b for tb, te, sp, _ in T)


def x_write(p: Path, rows: list) -> None:
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def wav_out(set_: str, cond: str, d: str, sid: str, x: np.ndarray) -> Path:
    p = OUT / "audio" / set_ / cond / d / f"{sid}.wav"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), x, SR, subtype="FLOAT")
    return p


def build_ood3(items, meta, man, stats) -> None:
    for s, C in O3.items():
        d = C["dir"]
        W = jl(OOD3 / f"{s}_ref.jsonl")
        T = full_timeline(C["full"], {w["meeting"] for w in W})
        tl = {rid: (b, e) for m in T for b, e, _, rid in T[m]}
        c = Counter()
        for w in W:
            x = read_mono(w["audio"])
            assert abs(len(x) - w["dur_ms"] * 16) <= 16, (w["id"], len(x), w["dur_ms"])
            for u in w["utts"]:
                c["utts"] += 1
                b, e = w["start_s"] + u["begin_s"], w["start_s"] + u["end_s"]
                assert u["record_id"] in tl and abs(tl[u["record_id"]][0] - b) < 0.01 and abs(tl[u["record_id"]][1] - e) < 0.01, \
                    (s, u["record_id"], "reference times differ from the full timeline")
                ref = (u.get(C["ref"]) or "").strip()
                if u.get("hesitation"):
                    c["drop_hesitation"] += 1
                    continue
                if u["end_s"] - u["begin_s"] < 1.0:
                    c["drop_short"] += 1
                    continue
                if not ref:
                    c["drop_noref"] += 1
                    continue
                if not TGT_RE[d].search(ref):   # untranslated fragments (e.g. AMI 'WHAT RENA' copied into zh)
                    c["drop_ref_not_target_language"] += 1
                    continue
                if crosstalk(T[w["meeting"]], b, e, u["speaker_id"]):
                    c["drop_crosstalk"] += 1
                    continue
                seg = x[int(round(u["begin_s"] * SR)): int(round(u["end_s"] * SR))]
                sid = f"o3{s}__{u['record_id']}"
                p = wav_out(f"ood3_{s}", "real", d, sid, seg)
                dur = len(seg) // 16
                items[d].append({"id": sid, "audio": str(p), "dur_ms": dur, "translation": ref, "src_text": u.get(C["src"], "")})
                meta[d].append({"id": sid, "set": f"ood3_{s}", "cond": "real", "pair": u["record_id"],
                                "stream": f"{w['meeting']}:{u['speaker_id'][:16]}", "ovl": False, "dur_ms": dur})
                man[d].append({"id": sid, "set": f"ood3_{s}", "cond": "real", "window": w["id"], "window_audio": w["audio"],
                               "meeting": w["meeting"], "speaker": u["speaker_id"], "record_id": u["record_id"],
                               "abs_begin_s": round(b, 3), "abs_end_s": round(e, 3), "cut_in_window_s": [u["begin_s"], u["end_s"]],
                               "src_text": u.get(C["src"], ""), "translation": ref, "ref_field": C["ref"],
                               "human_ref": u.get("human"), "synthetic_speech": s == "ami"})
                c["kept"] += 1
                c["kept_s"] += dur / 1000
        c["kept_s"] = round(c["kept_s"], 1)
        stats[f"ood3_{s}"] = {"dir": d, **c, "full_timeline": C["full"], "full_timeline_utts": sum(len(v) for v in T.values()),
                              "meetings": sorted(T)}


def build_obsi(items, meta, man, stats) -> None:
    for d in ("zh2en", "en2zh"):
        W = {cond: jl(BENCH / cond / d / "refs_windows.jsonl") for cond in OBSI_CONDS}
        n = len(W["clean"])
        assert all(len(W[c]) == n for c in OBSI_CONDS)
        turns = {}
        for cond in OBSI_CONDS:
            tt = defaultdict(list)
            for f in sorted((BENCH / cond / d / "utts").glob("*.jsonl")):
                for u in jl(f):
                    tt[f.stem[2:]].append((u["begin_s"], u["end_s"], u["speaker_id"], u["record_id"][2:]))
            turns[cond] = tt
        assert turns["clean"] == turns["id2"] == turns["ood1"], "OBSI conditions differ in turns"
        T = turns["clean"]
        c = Counter()
        cache, cur = {}, None
        for k in range(n):
            ws = {cond: W[cond][k] for cond in OBSI_CONDS}
            w0 = ws["clean"]
            for cond, w in ws.items():
                assert w["split"] == "test" and len(w["utts"]) == 1 and w["meeting"][2:] == w0["meeting"][2:] \
                    and w["start_s"] == w0["start_s"] and w["dur_ms"] == w0["dur_ms"] \
                    and w["utts"][0]["ref"] == w0["utts"][0]["ref"] and w["id"].split("_", 1)[1] == w0["id"].split("_", 1)[1], (cond, w["id"])
            u = w0["utts"][0]
            c["windows"] += 1
            b, e = w0["start_s"], w0["start_s"] + w0["dur_ms"] / 1000
            if crosstalk(T[w0["meeting"][2:]], b, e, u["speaker_id"]):
                c["drop_crosstalk"] += 1
                continue
            pair = w0["meeting"][2:] + "_" + w0["id"].rsplit("_", 1)[1]
            for cond, w in ws.items():
                if cur != w0["meeting"][2:]:
                    cache, cur = {}, w0["meeting"][2:]
                if w["audio"] not in cache:
                    cache[w["audio"]] = read_mono(w["audio"])
                x = cache[w["audio"]]
                s0 = int(round(w["start_s"] * SR))
                seg = x[s0: s0 + w["dur_ms"] * 16]
                sid = f"ob{cond}__{w['id']}"
                p = wav_out("obsi", cond, d, sid, seg)
                dur = len(seg) // 16
                items[d].append({"id": sid, "audio": str(p), "dur_ms": dur, "translation": u["ref"],
                                 "src_text": u.get("zh" if d == "zh2en" else "en", "")})
                meta[d].append({"id": sid, "set": "obsi", "cond": cond, "pair": pair,
                                "stream": w0["meeting"][2:] + ":" + u["speaker_id"][:16], "ovl": False, "dur_ms": dur})
                man[d].append({"id": sid, "set": "obsi", "cond": cond, "window": w["id"], "session_audio": w["audio"],
                               "meeting": w["meeting"], "speaker": u["speaker_id"][:16], "clip": u.get("clip"),
                               "abs_begin_s": b, "abs_end_s": round(e, 3), "src_text": items[d][-1]["src_text"],
                               "translation": u["ref"], "human_ref": u.get("human")})
            c["kept"] += 1
            c["kept_s"] += w0["dur_ms"] / 1000
        c["kept_s"] = round(c["kept_s"], 1)
        stats[f"obsi_{d}"] = {"dir": d, **c, "per_condition": c["kept"], "conditions": list(OBSI_CONDS)}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(a) -> None:
    if OUT.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {OUT}")
    OUT.mkdir(parents=True)
    items, meta, man, stats = defaultdict(list), defaultdict(list), defaultdict(list), {}
    build_ood3(items, meta, man, stats)
    build_obsi(items, meta, man, stats)
    for d in ("zh2en", "en2zh"):
        x_write(OUT / f"items_{d}.jsonl", items[d])
        x_write(OUT / f"meta_{d}.jsonl", meta[d])
        x_write(OUT / f"manifest_{d}.jsonl", man[d])
        M = {m["id"]: m for m in meta[d]}
        sub = defaultdict(list)
        for it in items[d]:
            sub[(M[it["id"]]["set"], M[it["id"]]["cond"])].append(it)
        for (s, c), rows in sorted(sub.items()):
            x_write(OUT / "items" / f"{s}_{c}_{d}.jsonl", rows)
    info = {"git_rev": git_rev(), "script": str(Path(__file__).relative_to(REPO)),
            "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(), "stats": stats,
            "rules": {"ood3": "not hesitation, >= 1 s, reference non-empty and in the target language, no other-speaker utterance overlapping in the "
                              "full meeting timeline", "obsi": "no other-speaker turn overlapping in the meeting; clean / "
                              "id2 / ood1 one-to-one"}}
    with open(OUT / "build_info.json", "x", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def verify(a) -> None:
    out = OUT / "verify.json"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    train = {}
    for name, pat in TRAIN_ROWS.items():
        for d in ("zh2en", "en2zh"):
            for r in jl(REPO / pat.format(d=d)):
                for k in ("src", "ref"):
                    train.setdefault((name, k), set()).add(norm(r.get(k, "")))
                train.setdefault((name, "id"), set()).add(r["id"])
                if r.get("spk"):
                    train.setdefault((name, "spk"), set()).add(r["spk"][:16])
    for d in ("zh2en", "en2zh"):
        I, M, Mf = jl(OUT / f"items_{d}.jsonl"), jl(OUT / f"meta_{d}.jsonl"), jl(OUT / f"manifest_{d}.jsonl")
        assert [i["id"] for i in I] == [m["id"] for m in M] == [m["id"] for m in Mf]
        assert len({i["id"] for i in I}) == len(I), "duplicate ids"
        cells = defaultdict(lambda: {"n": 0, "s": 0.0, "dur_lt_1s": 0, "peak_max": 0.0, "clip_0.999": 0, "len_mismatch": 0,
                                     "empty_ref": 0, "silent": 0})
        mm = {m["id"]: m for m in M}
        for it in I:
            m = mm[it["id"]]
            x, sr = sf.read(it["audio"], dtype="float32")
            c = cells[f"{m['set']}/{m['cond']}"]
            c["n"] += 1
            c["s"] += len(x) / SR
            c["dur_lt_1s"] += int(len(x) < SR)
            c["len_mismatch"] += int(len(x) // 16 != it["dur_ms"] or sr != SR)
            pk = float(np.abs(x).max()) if len(x) else 0.0
            c["peak_max"] = max(c["peak_max"], round(pk, 4))
            c["clip_0.999"] += int(pk >= 0.999)
            c["silent"] += int(pk < 1e-3)
            c["empty_ref"] += int(not it["translation"].strip())
        for c in cells.values():
            c["s"] = round(c["s"], 1)
            c["h"] = round(c["s"] / 3600, 3)
        # OBSI pairing: every pair has exactly the three conditions, same duration and reference
        P = defaultdict(dict)
        for it in I:
            m = mm[it["id"]]
            if m["set"] == "obsi":
                P[m["pair"]][m["cond"]] = (it["dur_ms"], it["translation"], m["stream"])
        pair_ok = all(set(v) == set(OBSI_CONDS) and len({x for x in v.values()}) == 1 for v in P.values())
        # overlap with training data (both directions' rows)
        ov = {}
        for name in TRAIN_ROWS:
            src_hit = [it["id"] for it in I if norm(it["src_text"]) and norm(it["src_text"]) in train[(name, "src")]]
            ref_hit = [it["id"] for it in I if norm(it["translation"]) in train[(name, "ref")]]
            long_src = [i for i in src_hit if len(norm(next(x for x in I if x["id"] == i)["src_text"])) >= MIN_NORM]
            long_ref = [i for i in ref_hit if len(norm(next(x for x in I if x["id"] == i)["translation"])) >= MIN_NORM]
            clip_hit = [m["id"] for m in Mf if m.get("clip") and m["clip"] in train[(name, "id")]]
            spk_hit = sorted({m["speaker"] for m in Mf if m["set"] == "obsi" and m["speaker"] in train.get((name, "spk"), set())})
            ov[name] = {"src_text_exact": len(src_hit), "src_text_exact_ge10chars": len(long_src),
                        "ref_text_exact": len(ref_hit), "ref_text_exact_ge10chars": len(long_ref),
                        "obsi_clip_id": len(clip_hit), "obsi_speaker": len(spk_hit),
                        "examples_ge10": sorted(set(long_src + long_ref))[:10]}
        res[d] = {"n": len(I), "cells": dict(cells), "obsi_pairs": len(P), "obsi_pairs_complete_and_equal": pair_ok,
                  "train_overlap": ov}
        assert pair_ok and all(c["len_mismatch"] == 0 and c["empty_ref"] == 0 and c["silent"] == 0 for c in cells.values()) \
            and all(c["dur_lt_1s"] == 0 for k, c in cells.items() if k.startswith("ood3")), res[d]
    res["PASS"] = True
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({d: {k: v for k, v in res[d].items()} for d in ("zh2en", "en2zh")}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "verify"))
    a = ap.parse_args()
    build(a) if a.step == "build" else verify(a)


if __name__ == "__main__":
    main()
