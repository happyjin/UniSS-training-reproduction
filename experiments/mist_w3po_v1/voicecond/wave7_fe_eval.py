"""Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): evaluation of the front-end arms.

process  apply a trained front-end (train_<arm>/final.pt) to every S1 item audio (wave2_S1/data/items_<dir>.jsonl, both
         sets, both conditions; the front-end is causal, so processing the whole file and cutting it later equals
         streaming) -> eval_<arm>/audio/<dir>/<id>.wav (float32) + eval_<arm>/items_<dir>.jsonl (same rows, audio
         replaced). The Thinker run is then si_thinker_stream --policy la2 --windows 1 on these items
         (wave7_fe_run.sh STAGE=thinker); the Talker run uses the ORIGINAL items (voice prompt = original audio, V1's
         C tokens, wave3 V tables) with the arm's blocks, 4 shards, seed 20261021 -- so speech metrics pair with
         wave 3's V1 run (same tokens, shards and seed; only the text differs).
rtf      real-time factor of the front-end on one GPU: offline (whole 60 s signal) and frame-by-frame streaming (one
         256-sample hop per call through the causal conv / GRU state), 200 calls; algorithmic latency = 512 samples.
score    text BLEU (S1 convention: normalize_for_bleu, sacrebleu zh / 13a) per direction x cell {a, b_noxt, b} x
         condition for base (S1 blocks), fe_clean, fe_pseudo; paired bootstrap (2000, seed 0; units = sentences (a) /
         single-speaker streams (b)) of corpus-BLEU differences pseudo-base, pseudo-clean, clean-base.
         Speech (if the talker runs exist): ASR-BLEU and LAAL D2 / D1 (laal_defs.sentence_defs) of each arm vs
         wave 3's V1 run. Verdicts: the unified primary criteria of the prereg's change (BLEU not worse in every
         direction x condition: point >= -0.3 and CI high >= 0, text and ASR-BLEU; at least one noisy cell with a
         significant text-BLEU gain, CI low > 0; prereg change 2: D2 LAAL diff (<= +0.10 s) reported, not a gate),
         then the original gates 1-4 as secondary.

    qwen3omni python -I experiments/mist_w3po_v1/voicecond/wave7_fe_eval.py process --arm pseudo|clean [--device cuda:0]
    qwen3omni python -I .../wave7_fe_eval.py rtf --arm pseudo [--device cuda:0]
    qwen3omni python -I .../wave7_fe_eval.py score --out eval_outputs/mist_w3po_v1/wave7/FE/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from experiments.mist_w3po_v1.voicecond.wave7_fe_common import OUT, SR, FrontEnd  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
V1 = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/v1"
ARMS = ("clean", "pseudo")
LANG = {"en2zh": "cmn", "zh2en": "eng"}
NB = 2000


def load_fe(arm, dev):
    st = torch.load(OUT / f"train_{arm}" / "final.pt", map_location=dev)
    fe = FrontEnd().to(dev)
    fe.load_state_dict(st["model"])
    fe.eval()
    return fe


def cmd_process(a):
    dev = a.device
    fe = load_fe(a.arm, dev)
    root = OUT / f"eval_{a.arm}"
    for d in ("zh2en", "en2zh"):
        p = root / f"items_{d}.jsonl"
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        rows = [json.loads(l) for l in open(S1 / "data" / f"items_{d}.jsonl", encoding="utf-8")]
        out = []
        t0 = time.time()
        for r in rows:
            x = sf.read(r["audio"], dtype="float32")[0]
            with torch.no_grad():
                y = fe(torch.from_numpy(x)[None].to(dev))[0].float().cpu().numpy()
            y = np.clip(y, -1.0, 1.0)
            q = root / "audio" / d / f"{r['id']}.wav"
            q.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(q), y.astype(np.float32), SR, subtype="FLOAT")
            out.append(dict(r, audio=str(q)))
        with open(p, "x", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{a.arm} {d}: {len(out)} items in {time.time() - t0:.0f} s", flush=True)


def cmd_rtf(a):
    dev = a.device
    fe = load_fe(a.arm, dev)
    x = torch.randn(1, 60 * SR, device=dev) * 0.05
    with torch.no_grad():
        fe(x)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            fe(x)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        off = (time.time() - t0) / 5 / 60.0
        # streaming: one hop per call, the conv context and GRU state carried
        hop, nfft = fe.HOP, fe.NFFT
        win = torch.zeros(1, nfft, device=dev)
        hist = torch.zeros(1, fe.c1.in_channels, 2, device=dev)
        hist2 = torch.zeros(1, fe.CH, 2, device=dev)
        hgru = None
        ts = []
        for i in range(200):
            new = x[:, i * hop: (i + 1) * hop]
            t1 = time.time()
            win = torch.cat([win[:, hop:], new], 1)
            spec = torch.fft.rfft(win * fe.win)[..., None]                     # [1, F, 1]
            lm = torch.log(spec.abs() + 1e-5)
            c1 = torch.nn.functional.gelu(fe.c1(torch.cat([hist, lm], 2)))
            hist = torch.cat([hist, lm], 2)[..., 1:]
            c2 = torch.nn.functional.gelu(fe.c2(torch.cat([hist2, c1], 2)))
            hist2 = torch.cat([hist2, c1], 2)[..., 1:]
            o, hgru = fe.gru(c2.transpose(1, 2), hgru)
            m = 2.0 * torch.sigmoid(fe.out(o)).transpose(1, 2)
            fr = torch.fft.irfft((spec * m)[..., 0], n=nfft) * fe.win
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append(time.time() - t1)
            _ = fr
    hop_s = hop / SR
    res = {"arm": a.arm, "device": dev, "offline_rtf": off, "stream_ms_per_hop_median": 1000 * float(np.median(ts[20:])),
           "stream_ms_per_hop_p95": 1000 * float(np.percentile(ts[20:], 95)),
           "stream_rtf": float(np.median(ts[20:])) / hop_s, "algorithmic_latency_ms": 1000 * nfft / SR,
           "params_M": sum(p.numel() for p in fe.parameters()) / 1e6}
    p = OUT / f"rtf_{a.arm}.json"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    json.dump(res, open(p, "x"), indent=1)
    print(json.dumps(res), flush=True)


# ------------------------------------------------------------------------------------------------------------ score
def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cmd_score(a):
    from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu, laal_defs
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    LD = laal_defs()
    res = {"bootstrap": f"{NB} paired resamples, seed 0; units = sentences (a) / single-speaker streams (b)"}
    verdict = {"bleu_not_worse": True, "laal_ok": True, "gain": False, "notes": []}
    for d in ("zh2en", "en2zh"):
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        items = {r["id"]: r for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        hyp = {"base": {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in jl(S1 / f"thinker_{d}/blocks_{d}_w1.jsonl")}}
        for arm in ARMS:
            p = OUT / f"eval_{arm}" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl"
            if p.exists():
                hyp[arm] = {r["id"]: "".join(b["text"] for b in r["blocks"]) for r in jl(p)}
        # speech: talker runs (arm) and V1 (reference)
        spk = {}
        for arm, run in [("v1", V1 / f"talker_{d}")] + [(arm, OUT / f"eval_{arm}" / f"talker_{d}") for arm in ARMS]:
            if (run / "asr_speech.jsonl").exists() and (run / "results.jsonl").exists():
                recs = {r["id"]: r for r in jl(run / "results.jsonl")}
                asr = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
                sd = {}
                for i, r in recs.items():
                    x = LD.sentence_defs(r)
                    sd[i] = (x["D2"] / 1000, x["D1"] / 1000) if x else (np.nan, np.nan)
                spk[arm] = {"asr": asr, "lat": sd}
        bl = Bleu(LANG[d])
        ovl = {m["pair"] for m in meta.values() if m["set"] == "b" and m["ovl"]}
        R = {}
        for cell, keep in (("a", lambda m: m["set"] == "a"), ("b", lambda m: m["set"] == "b"),
                           ("b_noxt", lambda m: m["set"] == "b" and m["pair"] not in ovl)):
            for cond in ("clean", "noisy"):
                ids = sorted(i for i, m in meta.items() if keep(m) and (m["cond"] == "clean") == (cond == "clean"))
                unit = [meta[i]["stream"] if meta[i]["set"] == "b" else i for i in ids]
                uix = {u: k for k, u in enumerate(sorted(set(unit)))}
                uid = np.array([uix[u] for u in unit])
                rng = np.random.default_rng(0)
                draws = [rng.integers(0, len(uix), len(uix)) for _ in range(NB)]
                refs = [items[i]["translation"] for i in ids]

                def unit_stats(h):
                    S = bl.stats([h.get(i, "") for i in ids], refs)
                    U = np.zeros((len(uix), S.shape[1]), np.int64)
                    np.add.at(U, uid, S)
                    return U
                C = {"n_sentences": len(ids), "n_units": len(uix)}
                US = {arm: unit_stats(h) for arm, h in hyp.items()}
                for arm, U in US.items():
                    C[f"bleu_{arm}"] = bl.score(U)
                for x, y in (("pseudo", "base"), ("pseudo", "clean"), ("clean", "base")):
                    if x in US and y in US:
                        bd = np.array([bl.score(US[x][j]) - bl.score(US[y][j]) for j in draws])
                        C[f"text_{x}-{y}"] = [bl.score(US[x]) - bl.score(US[y]), float(np.percentile(bd, 2.5)),
                                              float(np.percentile(bd, 97.5))]
                # speech vs V1 (same voice tokens, shards, seed)
                if "v1" in spk:
                    UV = None
                    for arm in ARMS:
                        if arm not in spk:
                            continue
                        if UV is None:
                            UV = unit_stats(spk["v1"]["asr"])
                        UA = unit_stats(spk[arm]["asr"])
                        bd = np.array([bl.score(UA[j]) - bl.score(UV[j]) for j in draws])
                        C[f"asr_{arm}-v1"] = [bl.score(UA) - bl.score(UV), float(np.percentile(bd, 2.5)),
                                              float(np.percentile(bd, 97.5))]
                        for k, nm in ((0, "d2"), (1, "d1")):
                            dif = np.array([spk[arm]["lat"].get(i, (np.nan,) * 2)[k] - spk["v1"]["lat"].get(i, (np.nan,) * 2)[k]
                                            for i in ids])
                            ok = ~np.isnan(dif)
                            us = np.zeros(len(uix))
                            uc = np.zeros(len(uix))
                            np.add.at(us, uid[ok], dif[ok])
                            np.add.at(uc, uid[ok], 1)
                            bd = np.array([us[j].sum() / max(1, uc[j].sum()) for j in draws])
                            C[f"{nm}_{arm}-v1"] = [float(us.sum() / max(1, uc.sum())), float(np.percentile(bd, 2.5)),
                                                   float(np.percentile(bd, 97.5))]
                R[f"{cell}_{cond}"] = C
                # primary verdict for fe_pseudo (main cells a and b_noxt)
                if cell in ("a", "b_noxt"):
                    for key in ("text_pseudo-base", "asr_pseudo-v1"):
                        if key in C and not (C[key][0] >= -0.3 and C[key][2] >= 0):
                            verdict["bleu_not_worse"] = False
                            verdict["notes"].append(f"{d} {cell} {cond} {key} {C[key]}")
                    if "d2_pseudo-v1" in C and C["d2_pseudo-v1"][0] > 0.10:
                        verdict["laal_ok"] = False
                        verdict["notes"].append(f"{d} {cell} {cond} d2 {C['d2_pseudo-v1']}")
                    if cond == "noisy" and "text_pseudo-base" in C and C["text_pseudo-base"][1] > 0:
                        verdict["gain"] = True
                        verdict["notes"].append(f"gain {d} {cell} noisy {C['text_pseudo-base']}")
        res[d] = R
    g = {}
    try:
        g["1_en2zh_bnoxt_noisy_vs_clean+1_and_base+1"] = (res["en2zh"]["b_noxt_noisy"]["text_pseudo-clean"][0] >= 1.0
                                                         and res["en2zh"]["b_noxt_noisy"]["text_pseudo-base"][0] >= 1.0)
        g["2_zh2en_a_noisy_base+1"] = res["zh2en"]["a_noisy"]["text_pseudo-base"][0] >= 1.0
        g["3_a_clean_both_ge_base-0.3"] = all(res[d]["a_clean"]["text_pseudo-base"][0] >= -0.3 for d in ("en2zh", "zh2en"))
    except KeyError as e:
        g["missing"] = str(e)
    for arm in ARMS:
        p = OUT / f"rtf_{arm}.json"
        if p.exists():
            res[f"rtf_{arm}"] = json.load(open(p))
    if "rtf_pseudo" in res:
        g["4_latency_32ms_rtf<0.05"] = res["rtf_pseudo"]["algorithmic_latency_ms"] <= 32 and res["rtf_pseudo"]["stream_rtf"] < 0.05
    # prereg change 2: LAAL is reported only (verdict["laal_ok"]), not a gate
    verdict["primary_pass"] = verdict["bleu_not_worse"] and verdict["gain"]
    res["primary_verdict_fe_pseudo"] = verdict
    res["original_gates_fe_pseudo"] = g
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"primary": verdict, "original": g}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("rtf")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("score")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"process": cmd_process, "rtf": cmd_rtf, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
