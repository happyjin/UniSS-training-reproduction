"""Robust wave 2, diagnosis S1 (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md): the inputs of the
single-speaker baseline (the paper's LocalAgreement-2 at a 1 s read window: base Qwen3-Omni Thinker + T1c Talker),
on clean vs noisy dev data. CPU only; refuses an existing output directory.

(a) CoVoST 2 validation clean / noisy pairs of wave 1 (eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl,
    300 sentences per direction): the wavs as they are.
(b) OpenBroadcast-SI DEV (data/open_voicecond_v1/bench_dev/{id2,clean}/{zh2en,en2zh}): every reference turn (one
    reference utterance = one window of refs_windows.jsonl) cut out of the session mix at its reference times, as
    one sentence of a single-speaker stream (stream = meeting twin x speaker), noisy (id2) and clean twin alike.
    A turn is flagged `ovl` when another speaker's reference turn overlaps it in time (crosstalk in the cut).

Writes <out>/items_<dir>.jsonl ({id, audio, dur_ms, translation}: the si_thinker_stream / si_infer --items-file
format; both sets and both conditions in one file per direction) and <out>/meta_<dir>.jsonl ({id, set, cond, pair,
stream, ovl, dur_ms}).

    python experiments/mist_w3po_v1/voicecond/wave2_s1_prep.py --out eval_outputs/mist_w3po_v1/wave2_S1/data
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
WAVE1 = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"
BENCH = REPO / "data/open_voicecond_v1/bench_dev"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    items, meta = defaultdict(list), defaultdict(list)

    # (a) CoVoST 2 validation pairs
    for r in map(json.loads, open(WAVE1, encoding="utf-8")):
        d = r["dir"]
        for cond, key in (("clean", "clean"), ("noisy", "noisy")):
            p = REPO / r[key]
            info = sf.info(str(p))
            assert info.samplerate == 16000 and info.channels == 1, p
            sid = f"a_{cond}__{r['id']}"
            dur = int(round(info.duration * 1000))
            items[d].append({"id": sid, "audio": str(p), "dur_ms": dur, "translation": r["ref"]})
            meta[d].append({"id": sid, "set": "a", "cond": cond, "pair": r["id"], "stream": r["id"], "ovl": False,
                            "dur_ms": dur})

    # (b) OpenBroadcast-SI dev, single-speaker cuts at the reference turns
    for d in ("zh2en", "en2zh"):
        (out / "audio" / d).mkdir()
        for cond, label in (("id2", "noisy"), ("clean", "clean")):
            base = BENCH / cond / d
            turns = defaultdict(list)   # meeting -> (begin, end, speaker) of every reference turn
            for f in sorted((base / "utts").glob("*.jsonl")):
                for u in map(json.loads, open(f, encoding="utf-8")):
                    turns[f.stem].append((u["begin_s"], u["end_s"], u["speaker_id"]))
            cache = {}
            for w in map(json.loads, open(base / "refs_windows.jsonl", encoding="utf-8")):
                assert w["split"] == "dev" and w["wtype"] == "ref" and len(w["utts"]) == 1, w["id"]
                u = w["utts"][0]
                if w["audio"] not in cache:
                    x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
                    assert sr == 16000, w["audio"]
                    cache = {w["audio"]: x.mean(axis=1)}
                x = cache[w["audio"]]
                s0 = int(round(w["start_s"] * 16000))
                seg = x[s0: s0 + w["dur_ms"] * 16]
                pair = w["meeting"][2:] + "_" + w["id"].rsplit("_", 1)[1]          # dbz001 / dcz001 -> z001
                sid = f"b_{label}__{w['id']}"
                p = out / "audio" / d / f"{sid}.wav"
                sf.write(str(p), seg, 16000)
                b, e = w["start_s"], w["start_s"] + w["dur_ms"] / 1000
                ovl = any(sp != u["speaker_id"] and tb < e and te > b for tb, te, sp in turns[w["meeting"]])
                items[d].append({"id": sid, "audio": str(p), "dur_ms": len(seg) // 16, "translation": u["ref"]})
                meta[d].append({"id": sid, "set": "b", "cond": label, "pair": pair,
                                "stream": w["meeting"][2:] + ":" + u["speaker_id"][:16], "ovl": ovl,
                                "dur_ms": len(seg) // 16})
    for d in ("zh2en", "en2zh"):
        with open(out / f"items_{d}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in items[d])
        with open(out / f"meta_{d}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in meta[d])
        n = defaultdict(int)
        for m in meta[d]:
            n[(m["set"], m["cond"])] += 1
        print(d, dict(n), "ovl(b,noisy)", sum(m["ovl"] for m in meta[d] if m["set"] == "b" and m["cond"] == "noisy"))


if __name__ == "__main__":
    main()
