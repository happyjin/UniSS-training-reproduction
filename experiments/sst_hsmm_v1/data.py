"""sst_hsmm_v1 data (reports/sst_hsmm_v1/prereg.zh-CN.md section 1). UniST read/write timing rows (data/unist_simul_v1, hybrid)
that are in the v2 quality index (data/unist_simul_train_v2/keys.txt), source 3-15 s, not UniST held-out-for-train
(crc32 % 50 != 0 for train, == 0 for heldout) and not overlapping any evaluation / test source text (wave14_data.harvest /
norm). Per direction: train = first 20000 by sha1("20261122|hsmm|<dir>|<id>"), heldout = first 1000 of the held-out rows.
Per row: Thinker tokens of the reference translation with per-token chunk lower bound g (from cummax of the word's
support_hybrid_ms), speech duration d_ms (word duration min(next start, end + 300) - start in the target speech, split
over the word's tokens; 0 for tokens without word characters) and punctuation flag. Held-out rows also get 16 kHz clean
and noisy (mix.Mixer, key "20261122|hsmm|heldout|<id>", p_clean 0) wavs + items/meta files for the Thinker scripts.
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.data --procs 96
Output eval_outputs/sst_hsmm_v1/data/ (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import multiprocessing as mp
import re
import sys
import unicodedata
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "eval_outputs/sst_hsmm_v1/data"
THINKER = REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"
N_TRAIN, N_HELD = 20000, 1000
DIR_OF = {"eng": "en2zh", "cmn": "zh2en"}
PUNCT = re.compile(r"[，。,.!?！？；;：:、…\"“”'‘’()（）《》\-—]")
COLS = ["key", "id", "src_lang", "tgt_lang", "transcription", "translation", "source_audio", "source_duration_ms",
        "target_words", "support_hybrid_ms"]
_TOK = None


def h(s):
    import hashlib
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(THINKER))
    return _TOK


def units(r):
    """-> dict with tokens and per-token g / tt / d_ms / punct, or (None, reason)"""
    ref = r["translation"].strip()
    X = int(r["source_duration_ms"])
    C = math.ceil(X / 1000)
    words = json.loads(r["target_words"])
    sup = json.loads(r["support_hybrid_ms"]) if isinstance(r["support_hybrid_ms"], str) else list(r["support_hybrid_ms"])
    if len(sup) != len(words) or not words:
        return None, "support_len"
    low = ref.lower()
    cw = [-1] * len(ref)            # char -> word index
    cur = 0
    for wi, w in enumerate(words):
        t = w["text"].strip().lower()
        if not t:
            continue
        p = low.find(t, cur)
        if p < 0:
            return None, "word_not_found"
        for k in range(p, p + len(t)):
            cw[k] = wi
        cur = p + len(t)
    enc = tok()(ref, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > 120:
        return None, "too_long"
    tw = []
    for (a, b) in offs:
        ws = [cw[k] for k in range(a, b) if 0 <= k < len(cw) and cw[k] >= 0]
        tw.append(ws[0] if ws else -1)
    ntok = collections.Counter(w for w in tw if w >= 0)
    wd = []
    for wi, w in enumerate(words):
        nxt = words[wi + 1]["start_ms"] if wi + 1 < len(words) else w["end_ms"]
        wd.append(max(0, min(nxt, w["end_ms"] + 300) - w["start_ms"]))
    g, tt, d, pu = [], [], [], []
    run = 0
    for k, w in enumerate(tw):
        if w >= 0:
            run = max(run, int(sup[w]))
            d.append(round(wd[w] / ntok[w], 1))
            pu.append(0)
        else:
            d.append(0.0)
            a, b = offs[k]
            pu.append(int(bool(PUNCT.search(ref[a:b]))))
        tt.append(run)
        g.append(int(min(max(math.ceil(run / 1000), 1), C)))
    return {"tok": ids, "g": g, "tt": tt, "d_ms": d, "punct": pu, "C": C, "X": X}, None


def _job(r):
    u, why = units(r)
    return r["id"], u, why


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
        rows += pq.read_table(f, columns=COLS).to_pylist()
    print("timing rows", len(rows), flush=True)
    cats, _ = W14.harvest(a.procs)
    allt = set().union(*cats.values())
    st = {"timing_rows": len(rows), "per_dir": {}}
    sel = {}
    for d in ("en2zh", "zh2en"):
        c = collections.Counter()
        tr, he = [], []
        for r in rows:
            if DIR_OF.get(r["src_lang"]) != d:
                continue
            c["rows"] += 1
            if r["key"] not in keys:
                c["not_in_v2_index"] += 1; continue
            X = int(r["source_duration_ms"])
            if not 3000 <= X <= 15000:
                c["dur_out_of_3_15s"] += 1; continue
            if W14.norm(r["transcription"] or "") in allt:
                c["overlap_test_text"] += 1; continue
            (he if W14.held_out(r["id"]) else tr).append(r)
        tr.sort(key=lambda r: h(f"20261122|hsmm|{d}|{r['id']}"))
        he.sort(key=lambda r: h(f"20261122|hsmm|{d}|{r['id']}"))
        c["eligible_train"], c["eligible_heldout"] = len(tr), len(he)
        sel[d] = {"train": tr[: int(N_TRAIN * 1.1)], "heldout": he[: int(N_HELD * 1.2)]}
        st["per_dir"][d] = c
    OUT.mkdir(parents=True)
    from experiments.sst_hsmm_v1.mix import Mixer
    import soundfile as sf
    mixer = Mixer()
    for d in ("en2zh", "zh2en"):
        for split, n in (("train", N_TRAIN), ("heldout", N_HELD)):
            cand = sel[d][split]
            with mp.get_context("fork").Pool(a.procs) as P:
                res = P.map(_job, cand, chunksize=32)
            byid = {i: (u, why) for i, u, why in res}
            kept, drop = [], collections.Counter()
            for r in cand:
                u, why = byid[r["id"]]
                if u is None:
                    drop[why] += 1; continue
                kept.append({"id": r["id"], "dir": d, "audio": r["source_audio"], "dur_ms": int(r["source_duration_ms"]),
                             "src": r["transcription"], "ref": r["translation"].strip(), **u})
                if len(kept) == n:
                    break
            assert len(kept) == n, (d, split, len(kept))
            st["per_dir"][d][f"{split}_dropped"] = dict(drop)
            with open(OUT / f"{split}_{d}.jsonl", "x", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
            if split == "heldout":
                wd = OUT / "heldout_wav" / d
                wd.mkdir(parents=True)
                items, meta = [], []
                for x in kept:
                    w = sf.read(x["audio"], dtype="float32")[0]
                    y, env, bvr = mixer.mix(w, f"20261122|hsmm|heldout|{x['id']}", p_clean=0.0)
                    for cond, sig in (("clean", w), ("noisy", y)):
                        p = wd / f"{cond}__{x['id']}.wav"
                        sf.write(str(p), sig, 16000, subtype="PCM_16")
                        iid = f"{cond}__{x['id']}"
                        items.append({"id": iid, "audio": str(p), "dur_ms": x["dur_ms"], "translation": x["ref"]})
                        meta.append({"id": iid, "set": "unist_heldout", "cond": cond, "pair": x["id"], "stream": iid,
                                     "env": env if cond == "noisy" else None, "bvr": bvr if cond == "noisy" else None,
                                     "dur_ms": x["dur_ms"]})
                with open(OUT / f"items_{d}.jsonl", "x", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
                with open(OUT / f"meta_{d}.jsonl", "x", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in meta)
            print(d, split, len(kept), dict(drop), flush=True)
    json.dump(st, open(OUT / "stats.json", "x"), indent=1, default=int)


if __name__ == "__main__":
    main()
