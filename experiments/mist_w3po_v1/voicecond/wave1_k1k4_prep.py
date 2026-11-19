"""Robust wave 1, K1 / K4 prechecks (reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md K1, K4): items and voice tokens.

Items: cvt dev teacher items (data/open_voicecond_v1/idea2/cvt, Idea 2 e2e v1) that have usable diarizer-slot windows by
the inference rule (idea2_e2e_train.build_items). Per direction (zh2en_dev, en2zh_dev): speakers in sha1 order, each
speaker's items in sha1(id) order with a translation not used before, taken round-robin over speakers until 100 items
(P1a: 200 items). P1b / P4b / K7v: the first 60 of each direction in that order (120 items).

Voice tokens per item (32 BiCodec global ids; C = Idea 1 C, voicecond_idea1/m2/ctrl/encoder.pt, u = phi_C(y) as
idea1_tokens --mode ctrl):
  B      raw BiCodec tokens of the noisy prompt (the tokenizer's file path), noisy prompt = the item's mixture segment
         teacher_in/audio/<id>.wav (what si_infer uses as the voice prompt of a single-block item)
  C      C tokens of the noisy prompt
  neg    g-: C tokens of the stream's pure background bg.wav over the item's segment [start, end]
  s000 .. s100   P1a: C tokens of the speaker's clean prompt cvt/refs/<spk>/p0.wav mixed with bg.wav[start, start + len(p0)]
         at background energy share sigma in {0, .25, .5, .75, 1}: x = sqrt(1 - sigma) s / rms(s) + sqrt(sigma) b / rms(b)
lambda_bar: mean of the Idea 2 lambda head (shrink_fit_all/full.npz, sha1 25aaf6cd) over the item's slot windows.
rho (C's code residual ratio, once on ALL cvt dev teacher items, 800): mean ||u_C - u*|| / mean ||u_raw - u*|| of the
noisy prompts (u* = the speaker's clean centre, cvt/refs/ustar). sigma+ = (1 - lambda_bar) rho, w_ZNE = sigma+ / (1 - sigma+).

    <venvs/vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave1_k1k4_prep --out <dir> [--device cuda:0]
-> <dir>/items.jsonl, <dir>/prep.json. Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_e2e_train import (  # noqa: E402
    CV, DEV_SETS, FIT, FIT_SHA1, build_items, latest_refs)
from experiments.mist_w3po_v1.voicecond.idea2_shrink import ShrinkParams  # noqa: E402

C_CKPT = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt"
SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_P1A, N_P1B = 100, 60
START_CONTENT, END_CONTENT, START_SEMANTIC = 180386, 180390, 180388
SR = 16000


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def read64(p) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    assert sr == SR, (p, sr)
    return x[:, 0]


def seg_of(x: np.ndarray, a: float, n: int) -> np.ndarray:
    """n samples of x from a seconds on (wrapping around the stream end)"""
    i = int(round(a * SR))
    idx = (np.arange(n) + i) % len(x)
    return x[idx]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def select(items):
    """round-robin over speakers (sha1 order), unique translations, sha1(id) order within a speaker"""
    by = defaultdict(list)
    for it in sorted(items, key=lambda r: h(r["id"])):
        by[it["spk"]].append(it)
    spks = sorted(by, key=h)
    used, out = set(), []
    while len(out) < N_P1A:
        took = False
        for s in spks:
            lst = by[s]
            while lst and lst[0]["text"] in used:
                lst.pop(0)
            if lst and len(out) < N_P1A:
                it = lst.pop(0)
                used.add(it["text"])
                out.append(it)
                took = True
        if not took:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    log = lambda s: print(f"{time.strftime('%F %T')} {s}", flush=True)  # noqa: E731
    assert hashlib.sha1((FIT / "full.npz").read_bytes()).hexdigest() == FIT_SHA1
    pf = ShrinkParams.load(FIT / "full.npz")
    refs = latest_refs(CV)
    dv, dropped = build_items(DEV_SETS, pf, refs, log)
    # texts / segment times
    texts = {}
    for s in DEV_SETS:
        d, split = s.split("_")
        for r in map(json.loads, open(CV / f"teacher_in/items_{d}_{split}.jsonl", encoding="utf-8")):
            texts[r["id"]] = r
    metas = {}
    for it in dv:
        tr = texts[it["id"]]
        it["text"] = tr["translation"]
        it["noisy"] = tr["audio"]
        if it["sid"] not in metas:
            metas[it["sid"]] = json.load(open(CV / "streams" / it["sid"] / "meta.json"))
        seg = metas[it["sid"]]["segments"][int(it["id"].rsplit("__", 1)[1])]
        assert seg["spk"] == it["spk"]
        it["start"], it["end"] = float(seg["start"]), float(seg["end"])
    sel = {}
    for s in DEV_SETS:
        sel[s] = select([it for it in dv if it["set"] == s])
        log(f"[select] {s}: {len(sel[s])} items, {len({it['spk'] for it in sel[s]})} speakers "
            f"(from {sum(1 for it in dv if it['set'] == s)} with windows)")
    chosen = [it for s in DEV_SETS for it in sel[s]]
    p1b = {it["id"] for s in DEV_SETS for it in sel[s][:N_P1B]}

    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    P0 = BiCodecPhi(a.device)
    PC = load_encoder(BiCodecPhi(a.device), str(C_CKPT))
    for m in (P0.model, PC.model):
        m.eval()

    # rho on all cvt dev teacher items
    e_raw, e_c = [], []
    for s in DEV_SETS:
        d, split = s.split("_")
        for r in map(json.loads, open(CV / f"teacher_in/items_{d}_{split}.jsonl", encoding="utf-8")):
            us = refs[r["spk"]][0]
            ur = P0.phi_path(r["audio"]).reshape(-1).astype(np.float64)
            uc = PC.phi_wave(read64(r["audio"])).reshape(-1).astype(np.float64)
            e_raw.append(float(np.linalg.norm(ur - us)))
            e_c.append(float(np.linalg.norm(uc - us)))
    rho = float(np.mean(e_c) / np.mean(e_raw))
    log(f"[rho] n={len(e_raw)} mean||u_raw-u*||={np.mean(e_raw):.4f} mean||u_C-u*||={np.mean(e_c):.4f} rho={rho:.4f}")

    rows = []
    bgs = {}
    for it in chosen:
        if it["sid"] not in bgs:
            bgs = {it["sid"]: read64(CV / "streams" / it["sid"] / "bg.wav")}
        bg = bgs[it["sid"]]
        y = read64(it["noisy"])
        n_seg = int(round((it["end"] - it["start"]) * SR))
        tok = {"B": P0.tokens(P0.phi_path(it["noisy"])), "C": PC.tokens(PC.phi_wave(y)),
               "neg": PC.tokens(PC.phi_wave(seg_of(bg, it["start"], n_seg)))}
        cp = CV / "refs" / it["spk"].replace(":", "_") / "p0.wav"
        s0 = read64(cp)
        b0 = seg_of(bg, it["start"], len(s0))
        sn, bn = s0 / rms(s0), b0 / rms(b0)
        mixes = [np.sqrt(1 - sg) * sn + np.sqrt(sg) * bn for sg in SIGMAS]
        for sg, uu in zip(SIGMAS, PC.phi_wave(mixes)):
            tok[f"s{int(round(sg * 100)):03d}"] = PC.tokens(uu)
        lam = 1.0 / (1.0 + np.exp(-(pf.alpha + it["vt"].astype(np.float64) @ pf.beta)))
        lam_bar = float(lam.mean())
        sp = (1 - lam_bar) * rho
        ids = [int(v) for v in it["ids"]]
        lf, nc = int(it["lf"]), int(it["nc"])
        assert ids[lf - 1] == START_SEMANTIC, it["id"]
        n_text = ids.index(END_CONTENT) - ids.index(START_CONTENT) - 1
        d = it["set"].split("_")[0]
        rows.append({"id": it["id"], "set": it["set"], "dir": d, "tgt": "eng" if d == "zh2en" else "cmn",
                     "spk": it["spk"], "sid": it["sid"], "start": it["start"], "end": it["end"], "text": it["text"],
                     "noisy": str(it["noisy"]), "clean_prompt": str(cp),
                     "clean_refs": [str(p) for p in sorted((CV / "refs" / it["spk"].replace(":", "_")).glob("p*.wav"))],
                     "ids": ids, "lf": lf, "nc": nc, "n_text": n_text, "gstar": [int(v) for v in it["gstar"]],
                     "lam_bar": lam_bar, "n_win": int(len(lam)), "sigma_plus": sp, "w_zne": sp / (1 - sp),
                     "p1b": it["id"] in p1b, "tok": {k: [int(v) for v in t] for k, t in tok.items()}})
    out.mkdir(parents=True)
    with open(out / "items.jsonl", "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    w = np.array([r["w_zne"] for r in rows if r["p1b"]])
    ham = lambda k1, k2: float(np.mean([sum(x != z for x, z in zip(r["tok"][k1], r["tok"][k2])) for r in rows]))  # noqa: E731
    info = {"n_p1a": len(rows), "n_p1b": int(sum(r["p1b"] for r in rows)), "rho": rho, "rho_n": len(e_raw),
            "mean_err_raw": float(np.mean(e_raw)), "mean_err_C": float(np.mean(e_c)),
            "w_zne_p1b": {"median": float(np.median(w)), "mean": float(w.mean()), "min": float(w.min()), "max": float(w.max())},
            "lam_bar_p1b_median": float(np.median([r["lam_bar"] for r in rows if r["p1b"]])),
            "ham_C_vs_B": ham("C", "B"), "ham_C_vs_neg": ham("C", "neg"), "ham_s000_vs_s100": ham("s000", "s100"),
            "ham_s000_vs_s050": ham("s000", "s050"),
            "dropped_without_windows": dropped, "fit_sha1": FIT_SHA1, "c_ckpt": str(C_CKPT.relative_to(REPO)),
            "sec": round(time.time() - t0, 1)}
    (out / "prep.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(json.dumps(info))


if __name__ == "__main__":
    main()
