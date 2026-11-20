"""Robust wave 13 (reports/mist_w3po_v1/robust_wave13_cleandrop_diag_prereg.zh-CN.md), H4: 600 clean zh2en TRAINING
clips of wave 5 (diagnosis only, never used for training or selection). Rows i (0-based line of rows_zh2en.jsonl) ranked
by int(sha1("20261120|w13|train|i")), first 600 kept; items in the S1 format {id, audio, dur_ms, translation} with id
t_clean__<row id>, plus meta {id, set "t", cond "clean", pair, stream, ovl false, dur_ms}. Refuses to overwrite.
    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave13_items.py
"""
import hashlib
import json
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
ROWS = REPO / "eval_outputs/mist_w3po_v1/wave5/M/data/rows_zh2en.jsonl"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave13/data"
N = 600


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def main():
    fi, fm = OUT / "items_zh2en_train600.jsonl", OUT / "meta_zh2en_train600.jsonl"
    for f in (fi, fm):
        if f.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {f}")
    rows = [json.loads(l) for l in open(ROWS, encoding="utf-8") if l.strip()]
    keep = sorted(range(len(rows)), key=lambda i: h(f"20261120|w13|train|{i}"))[:N]
    OUT.mkdir(parents=True, exist_ok=True)
    spk = set()
    with open(fi, "x", encoding="utf-8") as a, open(fm, "x", encoding="utf-8") as b:
        for i in keep:
            r = rows[i]
            wav = REPO / r["audio"]
            info = sf.info(str(wav))
            assert info.samplerate == 16000, wav
            dur = int(round(info.frames / 16))
            iid = f"t_clean__{r['id']}"
            spk.add(r["spk"])
            a.write(json.dumps({"id": iid, "audio": str(wav), "dur_ms": dur, "translation": r["ref"], "row": i},
                               ensure_ascii=False) + "\n")
            b.write(json.dumps({"id": iid, "set": "t", "cond": "clean", "pair": r["id"], "stream": r["id"], "ovl": False,
                                "dur_ms": dur}) + "\n")
    print(json.dumps({"n": len(keep), "speakers": len(spk), "first_rows": keep[:5]}))


if __name__ == "__main__":
    main()
