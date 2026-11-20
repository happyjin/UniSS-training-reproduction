"""Robust wave 17 (reports/mist_w3po_v1/robust_wave17_enclora_prereg.zh-CN.md, implementation note 5), mechanism check:
600 clean zh2en TRAINING clips of wave 14's UniST data (diagnosis only, never used for training or selection). Wave 13's
600 CoVoST clips are not in wave 14's data, so wave 13's rule (wave13_items.py) is re-applied to wave 14's rows: rows i
(0-based line of wave14/data/rows_zh2en.jsonl) ranked by int(sha1("20261120|w13|train|i")), first 600 kept. Items in
the S1 format {id, audio, dur_ms, translation, row, n_sampled} with id t_clean__<row id>; n_sampled = how often the
wave-14/17 sampler drew the row in the 2300 steps (zh2en slots j = 0..31, row = sha1("20261119|w5m|row|s|j") mod n),
for the auxiliary seen-only subset. Plus meta {id, set "t", cond "clean", pair, stream, ovl false, dur_ms}.
Refuses to overwrite.
    <uniss-eval python> experiments/mist_w3po_v1/voicecond/wave17_items.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
ROWS = REPO / "eval_outputs/mist_w3po_v1/wave14/data/rows_zh2en.jsonl"
OUT = REPO / "eval_outputs/mist_w3po_v1/wave17/data"
N, STEPS, ZH_SLOTS = 600, 2300, 32


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def main():
    fi, fm = OUT / "items_zh2en_train600.jsonl", OUT / "meta_zh2en_train600.jsonl"
    for f in (fi, fm):
        if f.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {f}")
    rows = [json.loads(l) for l in open(ROWS, encoding="utf-8") if l.strip()]
    assert all(r["dir"] == "zh2en" for r in rows)
    drawn = Counter(h(f"20261119|w5m|row|{s}|{j}") % len(rows) for s in range(STEPS) for j in range(ZH_SLOTS))
    keep = sorted(range(len(rows)), key=lambda i: h(f"20261120|w13|train|{i}"))[:N]
    OUT.mkdir(parents=True, exist_ok=True)
    seen = 0
    with open(fi, "x", encoding="utf-8") as a, open(fm, "x", encoding="utf-8") as b:
        for i in keep:
            r = rows[i]
            wav = REPO / r["audio"]
            info = sf.info(str(wav))
            assert info.samplerate == 16000, wav
            dur = int(round(info.frames / 16))
            iid = f"t_clean__{r['id']}"
            seen += drawn[i] > 0
            a.write(json.dumps({"id": iid, "audio": str(wav), "dur_ms": dur, "translation": r["ref"], "row": i,
                                "n_sampled": drawn[i]}, ensure_ascii=False) + "\n")
            b.write(json.dumps({"id": iid, "set": "t", "cond": "clean", "pair": r["id"], "stream": r["id"], "ovl": False,
                                "dur_ms": dur}) + "\n")
    print(json.dumps({"n": len(keep), "n_rows": len(rows), "draws": STEPS * ZH_SLOTS, "sampled_at_least_once": seen,
                      "first_rows": keep[:5]}))


if __name__ == "__main__":
    main()
