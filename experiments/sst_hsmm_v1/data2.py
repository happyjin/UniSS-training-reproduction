"""sst_hsmm_v1 prereg change 37-4 (SAP2 data): data.py's train selection (same timing rows, v2 index, 3-15 s, UniST held-out excluded
by W14.held_out, evaluation / test text excluded, same sha1("20261122|hsmm|<dir>|<id>") order and the same per-row units()) with
N_TRAIN = 80000 per direction; the first 20000 rows are data/train_<d>.jsonl's rows. Train split only (no held-out wavs).
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.data2 --procs 96
Output eval_outputs/sst_hsmm_v1/data2/ (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import multiprocessing as mp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import data as D0  # noqa: E402

OUT = REPO / "eval_outputs/sst_hsmm_v1/data2"
N_TRAIN = 80000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=96)
    a = ap.parse_args()
    if OUT.exists():
        sys.exit(f"拒绝覆盖已有输出: {OUT}")
    import pyarrow.parquet as pq
    from experiments.mist_w3po_v1.voicecond import wave14_data as W14
    keys = set(l.strip() for l in open(REPO / "data/unist_simul_train_v2/keys.txt", encoding="utf-8"))
    rows = []
    for f in sorted(glob.glob(str(REPO / "data/unist_simul_v1/rows/*.parquet"))):
        rows += pq.read_table(f, columns=D0.COLS).to_pylist()
    cats, _ = W14.harvest(a.procs)
    allt = set().union(*cats.values())
    st = {"timing_rows": len(rows), "per_dir": {}}
    OUT.mkdir(parents=True)
    for d in ("en2zh", "zh2en"):
        c = collections.Counter(); tr = []
        for r in rows:
            if D0.DIR_OF.get(r["src_lang"]) != d:
                continue
            if r["key"] not in keys:
                continue
            if not 3000 <= int(r["source_duration_ms"]) <= 15000:
                continue
            if W14.norm(r["transcription"] or "") in allt:
                continue
            if W14.held_out(r["id"]):
                continue
            tr.append(r)
        tr.sort(key=lambda r: D0.h(f"20261122|hsmm|{d}|{r['id']}"))
        c["eligible_train"] = len(tr)
        cand = tr[: int(N_TRAIN * 1.1)]
        with mp.get_context("fork").Pool(a.procs) as P:
            res = P.map(D0._job, cand, chunksize=32)
        byid = {i: (u, why) for i, u, why in res}
        kept, drop = [], collections.Counter()
        for r in cand:
            u, why = byid[r["id"]]
            if u is None:
                drop[why] += 1; continue
            kept.append({"id": r["id"], "dir": d, "audio": r["source_audio"], "dur_ms": int(r["source_duration_ms"]),
                         "src": r["transcription"], "ref": r["translation"].strip(), **u})
            if len(kept) == N_TRAIN:
                break
        assert len(kept) == N_TRAIN, (d, len(kept))
        c["dropped"] = dict(drop)
        old = [json.loads(l)["id"] for l in open(REPO / f"eval_outputs/sst_hsmm_v1/data/train_{d}.jsonl", encoding="utf-8")]
        assert [x["id"] for x in kept[: len(old)]] == old, f"{d}: first rows differ from data/train"
        with open(OUT / f"train_{d}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
        st["per_dir"][d] = c
        print(d, json.dumps(c), flush=True)
    (OUT / "stats.json").write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
