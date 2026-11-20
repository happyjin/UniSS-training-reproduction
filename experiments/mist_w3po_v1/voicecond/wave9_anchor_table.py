"""Robust wave 9: the anchor tokens of every evaluation item (wave2_S1/data/items_<dir>.jsonl), for
wave9_si_thinker_anchor.py. The item audio is read exactly as si_thinker_stream reads it (sf.read float32, 2-D, channel
mean, [0 : dur_ms * 16]); anchor = wave9_common.Anchor.ids(audio) (first 1.0 s, fixed PCA_FILE); key = sha1 of the
float32 bytes of that first second (head_key), which is how the wrapper finds an item from a vLLM prompt.
Output (refuses to overwrite): wave9/anchor/table_<dir>.jsonl ({id, key, ids}).
    CUDA_VISIBLE_DEVICES=<c> <qwen3omni python> -I experiments/mist_w3po_v1/voicecond/wave9_anchor_table.py --dir zh2en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave9_common as C  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=("zh2en", "en2zh"))
    a = ap.parse_args()
    out = C.ANCHOR_DIR / f"table_{a.dir}.jsonl"
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    A = C.Anchor("cuda:0")
    items = [json.loads(l) for l in open(S1 / "data" / f"items_{a.dir}.jsonl", encoding="utf-8")]
    rows, keys = [], {}
    for k in range(0, len(items), 64):
        part = items[k: k + 64]
        auds = []
        for it in part:
            x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
            assert sr == 16000
            auds.append(x.mean(axis=1)[0: int(it["dur_ms"]) * 16])
        ids = A.ids(auds)
        for it, au, row in zip(part, auds, ids):
            key = C.head_key(au)
            row = [int(v) for v in row]
            assert keys.get(key, row) == row          # same first second -> same anchor by construction
            keys[key] = row
            rows.append({"id": it["id"], "key": key, "ids": row})
    with open(out, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({"dir": a.dir, "n": len(rows), "distinct_keys": len(keys)}), flush=True)


if __name__ == "__main__":
    main()
