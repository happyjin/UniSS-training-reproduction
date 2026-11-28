"""change 65 (reversed-pair en2zh adapter) data: UniST zh2en rows used in reverse -- input = UniST's own ENGLISH target speech
(decoded from its BiCodec codes), target = the ORIGINAL spoken CHINESE transcription. Stages (each refuses existing outputs):
  select  rows: data/unist_simul_v1 timing rows with src_lang cmn, in the v2 quality index, not UniST held-out (W14.held_out),
          neither the Chinese transcription nor the English translation in the evaluation / test text set (W14.norm),
          English target speech 3-15 s, not one of the 20000 rows of data/train_zh2en.jsonl; sorted by sha1("20261128|rev|<id>"),
          first N_ROWS -> rev/rows.jsonl. Slots: the 16000 p4 (s, j); en2zh slots (j even) take rev row
          h("20261123|risk|row|s|j") mod N_ROWS (p4's draw rule on the rev table) -> rev/slots.jsonl.
  decode  (GPU) every row used by a slot: raw UniST shard data/raw/UniST/<source_parquet> row <source_row_index> (id checked),
          BiCodec.detokenize(target_bicodec, bicodec_global) one row at a time (no padding) -> rev/clean/<id>.wav (16 kHz PCM16).
  build   en2zh slots: mix.Mixer key "20261123|risk|mix|s|j", p_clean 0.5 (= p4) -> rev/audio/en2zh/s<s>_j<j>.wav; zh2en slots are
          p4's records unchanged -> rev/samples.jsonl. Targets: transcription with the filler interjections 呃 / 嗯 removed (and the
          punctuation runs / leading punctuation this leaves cleaned), Thinker tokens (<= 120), C = ceil(X / 1000), X = clean speech
          ms -> rev/train_en2zh.jsonl (id, dir, ref, src, tok, C, X, dur_ms).
    qwen3omni PYTHONPATH=. python -m experiments.sst_hsmm_v1.rev_prep select --procs 96
    qwen3omni PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python -m experiments.sst_hsmm_v1.rev_prep decode
    qwen3omni PYTHONPATH=. OMP_NUM_THREADS=1 python -m experiments.sst_hsmm_v1.rev_prep build --procs 48
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "eval_outputs/sst_hsmm_v1/rev"
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
P4 = REPO / "eval_outputs/sst_hsmm_v1/p4"
RAW = REPO / "data/raw/UniST"
THINKER = REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05"
N_ROWS, SEED, SLOTS, STEPS = 20000, 20261123, 16, 1000
COLS = ["key", "id", "src_lang", "transcription", "translation", "target_duration_ms", "source_parquet", "source_row_index"]
FILLER = re.compile(r"[呃嗯]+")
PUN = "，。、,.!?！？；;：:"
_M = {}


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def clean_ref(t: str) -> str:
    s = FILLER.sub("", t.strip())
    s = re.sub(r"([，、,；;：:])[，、,；;：:\s]+", r"\1", s)          # comma runs left by a removed filler
    s = re.sub(r"[，、,；;：:]+([。！？.!?])", r"\1", s)              # comma right before a sentence end
    s = s.lstrip(PUN + " ")
    return s.strip()


def select(a):
    import pyarrow.parquet as pq
    from experiments.mist_w3po_v1.voicecond import wave14_data as W14
    if OUT.exists():
        sys.exit(f"拒绝覆盖已有输出: {OUT}")
    keys = set(l.strip() for l in open(REPO / "data/unist_simul_train_v2/keys.txt", encoding="utf-8"))
    used = {json.loads(l)["id"] for l in open(DATA / "train_zh2en.jsonl", encoding="utf-8")}
    rows = []
    for f in sorted(glob.glob(str(REPO / "data/unist_simul_v1/rows/*.parquet"))):
        rows += [r for r in pq.read_table(f, columns=COLS).to_pylist() if r["src_lang"] == "cmn"]
    cats, _ = W14.harvest(a.procs)
    allt = set().union(*cats.values())
    c, keep = collections.Counter(), []
    for r in rows:
        c["rows"] += 1
        if r["key"] not in keys:
            c["not_in_v2_index"] += 1; continue
        if r["id"] in used:
            c["in_train_zh2en"] += 1; continue
        if W14.held_out(r["id"]):
            c["heldout"] += 1; continue
        X = int(r["target_duration_ms"] or 0)
        if not 3000 <= X <= 15000:
            c["tgt_dur_out_of_3_15s"] += 1; continue
        if W14.norm(r["transcription"] or "") in allt or W14.norm(r["translation"] or "") in allt:
            c["overlap_test_text"] += 1; continue
        if not clean_ref(r["transcription"] or ""):
            c["empty_after_clean"] += 1; continue
        if not (RAW / r["source_parquet"]).exists():
            c["raw_shard_missing"] += 1; continue
        keep.append(r)
    keep.sort(key=lambda r: h(f"20261128|rev|{r['id']}"))
    c["eligible"] = len(keep)
    keep = keep[:N_ROWS]
    assert len(keep) == N_ROWS, c
    OUT.mkdir(parents=True)
    with open(OUT / "rows.jsonl", "x", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_used = set()
    with open(OUT / "slots.jsonl", "x", encoding="utf-8") as fh:
        for s in range(STEPS):
            for j in range(0, SLOTS, 2):
                k = h(f"{SEED}|risk|row|{s}|{j}") % N_ROWS
                n_used.add(k)
                fh.write(json.dumps({"s": s, "j": j, "row": keep[k]["id"]}) + "\n")
    c["rows_used_by_slots"] = len(n_used)
    json.dump(c, open(OUT / "select_stats.json", "x"), indent=1)
    print(json.dumps(c), flush=True)


def decode(a):
    import pyarrow.parquet as pq
    import torch
    from uniss.speech_tokenizer.bicodec.models.bicodec import BiCodec
    rows = {r["id"]: r for r in map(json.loads, open(OUT / "rows.jsonl", encoding="utf-8"))}
    need = sorted({json.loads(l)["row"] for l in open(OUT / "slots.jsonl")})
    cd = OUT / "clean"
    cd.mkdir(exist_ok=a.resume)
    by = collections.defaultdict(list)
    for i in need:
        if not (cd / f"{i}.wav").exists():
            by[rows[i]["source_parquet"]].append(i)
    dev = "cuda:0"
    m = BiCodec.load_from_checkpoint(str(REPO / "pretrained_models/UniSS/bicodec/BiCodec")).to(dev).eval()
    n, bad = 0, collections.Counter()
    for shard, ids in sorted(by.items()):
        pf = pq.ParquetFile(RAW / shard)
        want = {int(rows[i]["source_row_index"]): i for i in ids}
        base = 0
        for g in range(pf.num_row_groups):
            ng = pf.metadata.row_group(g).num_rows
            hit = [k for k in want if base <= k < base + ng]
            if hit:
                t = pf.read_row_group(g, columns=["id", "target_bicodec", "bicodec_global", "src_lang"])
                for k in hit:
                    r = t.slice(k - base, 1).to_pylist()[0]
                    i = want[k]
                    if r["id"] != i or r["src_lang"] != "cmn":
                        bad["id_mismatch"] += 1; continue
                    sem = torch.as_tensor([r["target_bicodec"]], dtype=torch.long, device=dev)
                    glo = torch.as_tensor([r["bicodec_global"]], dtype=torch.long, device=dev).unsqueeze(1)
                    with torch.no_grad():
                        w = m.detokenize(sem, glo).float().reshape(-1)[: len(r["target_bicodec"]) * 320]
                    w = w.cpu().numpy()
                    if not np.isfinite(w).all() or float(np.abs(w).max()) < 1e-3:
                        bad["bad_audio"] += 1; continue
                    sf.write(cd / f"{i}.wav", np.clip(w, -1, 1), 16000, subtype="PCM_16")
                    n += 1
            base += ng
        print(f"{shard}: decoded so far {n}, bad {dict(bad)}", flush=True)
    json.dump({"decoded": n, **bad}, open(OUT / f"decode_stats{'_resume' if a.resume else ''}.json", "w"), indent=1)


def _init():
    from experiments.sst_hsmm_v1.mix import Mixer
    _M["m"] = Mixer()


def _one(job):
    s, j, src, wav_p = job
    x = sf.read(src, dtype="float32")[0]
    y, env, _ = _M["m"].mix(x, f"{SEED}|risk|mix|{s}|{j}", p_clean=0.5)
    y = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    sf.write(wav_p, y, 16000, subtype="PCM_16")
    return s, j, len(y) // 16, env


def build(a):
    from transformers import AutoTokenizer
    if (OUT / "samples.jsonl").exists():
        sys.exit(f"拒绝覆盖已有输出: {OUT}/samples.jsonl")
    rows = {r["id"]: r for r in map(json.loads, open(OUT / "rows.jsonl", encoding="utf-8"))}
    slots = [json.loads(l) for l in open(OUT / "slots.jsonl")]
    tk = AutoTokenizer.from_pretrained(str(THINKER))
    refs, drop = {}, collections.Counter()
    for i in sorted({x["row"] for x in slots}):
        p = OUT / "clean" / f"{i}.wav"
        if not p.exists():
            drop["no_audio"] += 1; continue
        ref = clean_ref(rows[i]["transcription"])
        ids = tk(ref, add_special_tokens=False)["input_ids"]
        if len(ids) > 120:
            drop["too_long"] += 1; continue
        X = sf.info(str(p)).frames // 16
        refs[i] = {"id": i, "dir": "en2zh", "audio": str(p), "dur_ms": X, "src": rows[i]["translation"].strip(), "ref": ref,
                   "orig_transcription": rows[i]["transcription"], "tok": ids, "C": math.ceil(X / 1000), "X": X}
    (OUT / "audio" / "en2zh").mkdir(parents=True, exist_ok=True)
    jobs = [(x["s"], x["j"], refs[x["row"]]["audio"], str(OUT / "audio" / "en2zh" / f"s{x['s']:04d}_j{x['j']:02d}.wav"))
            for x in slots if x["row"] in refs]
    drop["slots_without_row"] = len(slots) - len(jobs)
    with mp.get_context("fork").Pool(a.procs, _init) as pool:
        res = {(s, j): (d, env) for s, j, d, env in pool.map(_one, jobs, chunksize=16)}
    row_of = {(x["s"], x["j"]): x["row"] for x in slots}
    p4 = [json.loads(l) for l in open(P4 / "samples.jsonl", encoding="utf-8")]
    with open(OUT / "samples.jsonl", "x", encoding="utf-8") as fh:
        for x in p4:
            if x["dir"] == "en2zh":
                k = (x["s"], x["j"])
                if k not in res:
                    continue
                d, env = res[k]
                x = {"s": x["s"], "j": x["j"], "dir": "en2zh", "row": row_of[k],
                     "wav": str((OUT / "audio" / "en2zh" / f"s{x['s']:04d}_j{x['j']:02d}.wav").relative_to(REPO)), "dur_ms": d, "env": env}
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(OUT / "train_en2zh.jsonl", "x", encoding="utf-8") as fh:
        for r in refs.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = {"rows_with_ref": len(refs), "en2zh_slots": len(res), **drop}
    json.dump(st, open(OUT / "build_stats.json", "x"), indent=1)
    print(json.dumps(st), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "decode", "build"))
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--resume", action="store_true", help="decode: continue into an existing clean/ (skips decoded rows)")
    a = ap.parse_args()
    {"select": select, "decode": decode, "build": build}[a.stage](a)


if __name__ == "__main__":
    main()
