"""Robust wave 4, diagnosis #3 (reports/mist_w3po_v1/robust_wave4_diag_prereg.zh-CN.md): attention sinks / outlier tokens
in the base Thinker (original Qwen3-Omni-30B-A3B-Instruct, no adapter, HF bf16, si_thinker_stream's base prompt as in
voicecond/wave3_T_thinker.py), forward only, one setting.

Per CoVoST 2 validation clean / noisy twin (wave-1 manifest, 300 sentences per direction; the twins have the same length,
so they share one batch of two rows with identical token layout): one teacher-forced forward on the reference translation
(prompt with the audio, then the reference tokens). Attention is the eager implementation, so every layer's softmax
weights are exact; a pre-hook on each of the 48 decoder layers reads the residual stream entering the layer.
  outlier audio token (layer l): ||h_l|| at an audio position > 10 x the median ||h_l|| over that row's audio positions
  out_frac[l]   share of the row's audio positions that are outliers (l = 0 is the input of layer 0, i.e. the audio
                encoder output after projection, the "encoder output layer")
  mass_out[l]   attention mass from the reference (target) positions to the outlier audio positions, summed over keys,
                averaged over heads and target positions
  mass_audio[l] the same to all audio positions (context)
  ratio_max[l]  max / median audio-position norm
Writes <out>/<dir>.jsonl (one row per sentence id with both conditions). Refuses an existing output file.

    CUDA_VISIBLE_DEVICES=<card> <qwen3omni python> -I experiments/mist_w3po_v1/voicecond/wave4_sink.py run --dir zh2en \
        --out eval_outputs/mist_w3po_v1/wave4_diag/sink [--limit 2]
    <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/wave4_sink.py summary --out eval_outputs/mist_w3po_v1/wave4_diag/sink
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
THRESH = 10.0
N_LAYERS = 48
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
TRUNS = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/T/runs"
MAN = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"


def log(m: str) -> None:
    print(f"{time.strftime('%F %T')} {m}", flush=True)


def run(a) -> None:
    import torch
    from experiments.mist_w3po_v1.voicecond import wave3_T_thinker as T
    from experiments.mist_w3po_v1.voicecond import wave1_thinker as W1
    out = Path(a.out) / f"{a.dir}{'_smoke' if a.limit else ''}.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    eng = T.BaseThinker(a.max_gib)
    tm = eng.thinker
    tm.model.config._attn_implementation = "eager"
    if hasattr(tm.config, "text_config"):
        tm.config.text_config._attn_implementation = "eager"
    layers = tm.model.layers
    assert len(layers) == N_LAYERS, len(layers)
    AUDIO_ID = int(tm.config.audio_token_id)
    ctx = {}

    def pre(li):
        def fn(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            n = h.float().norm(dim=-1)                      # [B, S]
            am = ctx["audio"]
            outl = torch.zeros_like(am)
            for b in range(n.shape[0]):
                v = n[b][am[b]]
                med = v.median()
                outl[b, am[b]] = v > THRESH * med
                ctx["ratio"][b][li] = float(v.max() / med)
            ctx["outl"][li] = outl
            ctx["frac"][:, li] = (outl.sum(1).float() / am.sum(1).float()).cpu()
        return fn

    def attn(li):
        def fn(module, args, output):
            w = output[1]
            assert w is not None, "eager attention weights missing"
            tq = ctx["tgt"]                                  # [B, S] bool
            for b in range(w.shape[0]):
                wb = w[b][:, tq[b]].float()                  # [H, T, S]
                ctx["mout"][b, li] = float(wb[..., ctx["outl"][li][b]].sum(-1).mean())
                ctx["maud"][b, li] = float(wb[..., ctx["audio"][b]].sum(-1).mean())
        return fn

    for li, L in enumerate(layers):
        L.register_forward_pre_hook(pre(li), with_kwargs=True)
        L.self_attn.register_forward_hook(attn(li))
    M = W1.manifest()[a.dir]
    if a.limit:
        M = M[: a.limit]
    fh = open(str(out) + ".partial", "w", encoding="utf-8")
    t0 = time.time()
    for k, x in enumerate(M):
        wc, wn = W1.wav(x["clean"]), W1.wav(x["noisy"])
        assert len(wc) == len(wn), x["id"]
        head = eng.head[a.dir]
        with torch.no_grad():
            ph = eng.proc(text=[head], audio=[wc], return_tensors="pt", padding=True)
            n_head = ph["input_ids"].shape[1]
            inp = eng.proc(text=[head + x["ref"] + "<|im_end|>"] * 2, audio=[wc, wn], return_tensors="pt", padding=True)
            ids = inp["input_ids"]
            assert torch.equal(ids[0, :n_head], ph["input_ids"][0]), "prompt tokenisation differs with the target"
            assert torch.equal(ids[0], ids[1])
            S = ids.shape[1]
            audio = (ids == AUDIO_ID).to(eng.dev)
            tgt = torch.zeros_like(audio)
            tgt[:, n_head: S - 1] = True                     # the reference tokens (not <|im_end|>)
            ctx.update(audio=audio, tgt=tgt, outl={}, frac=torch.zeros(2, N_LAYERS), mout=np.zeros((2, N_LAYERS)),
                       maud=np.zeros((2, N_LAYERS)), ratio=[[0.0] * N_LAYERS for _ in range(2)])
            inputs = {kk: v.to(eng.dev) for kk, v in inp.items()}
            inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
            tm(**inputs, use_cache=False)
        row = {"id": x["id"], "dir": a.dir, "n_audio": int(audio[0].sum()), "n_tgt": int(tgt[0].sum()),
               "bvr_db": x.get("bvr_db")}
        for b, c in enumerate(("clean", "noisy")):
            row[c] = {"out_frac": [round(float(v), 5) for v in ctx["frac"][b]],
                      "mass_out": [round(float(v), 6) for v in ctx["mout"][b]],
                      "mass_audio": [round(float(v), 6) for v in ctx["maud"][b]],
                      "ratio_max": [round(v, 3) for v in ctx["ratio"][b]]}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        if k % 25 == 0:
            log(f"{a.dir} {k + 1}/{len(M)} {time.time() - t0:.0f}s")
    fh.close()
    Path(str(out) + ".partial").rename(out)
    log(f"done {a.dir} {len(M)} -> {out}")


def summary(a) -> None:
    from scipy.stats import spearmanr, wilcoxon, t as tdist
    from sacrebleu.metrics import CHRF
    from evaluation.text_metrics import normalize_for_bleu
    root = Path(a.out)
    dst = root / "summary.json"
    if dst.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {dst}")
    man = {r["id"]: r for r in map(json.loads, open(MAN, encoding="utf-8"))}
    chrf = CHRF()

    def pspear(x, y, z):
        from scipy.stats import rankdata
        rx, ry, rz = (rankdata(v) for v in (x, y, z))
        A = np.c_[np.ones_like(rz), rz]
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        r = float(np.corrcoef(ex, ey)[0, 1])
        n = len(x)
        tt = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
        return r, float(2 * tdist.sf(abs(tt), n - 3))

    res = {"threshold_x_median": THRESH}
    for d in ("zh2en", "en2zh"):
        NL = "cmn" if d == "en2zh" else "eng"
        nz = lambda s: normalize_for_bleu(s, NL)   # noqa: E731
        rows = [json.loads(l) for l in open(root / f"{d}.jsonl", encoding="utf-8")]
        la2 = {r["id"]: r["translation"] for r in map(json.loads, open(S1 / f"thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8"))}
        whole = {}
        for c in ("clean", "noisy"):
            for p in sorted((TRUNS / f"base_{c}_{d}").glob("s*.jsonl")):
                for r in map(json.loads, open(p, encoding="utf-8")):
                    whole[(c, r["id"])] = r["translation"]

        def cs(h, i):
            return chrf.sentence_score(nz(h), [nz(man[i]["ref"])]).score
        drop_main = np.array([cs(la2[f"a_clean__{r['id']}"], r["id"]) - cs(la2[f"a_noisy__{r['id']}"], r["id"]) for r in rows])
        drop_whole = np.array([cs(whole[("clean", r["id"])], r["id"]) - cs(whole[("noisy", r["id"])], r["id"]) for r in rows])
        bvr = np.array([man[r["id"]]["bvr_db"] for r in rows], float)
        L = {c: {k: np.array([r[c][k] for r in rows], float) for k in ("out_frac", "mass_out", "mass_audio", "ratio_max")}
             for c in ("clean", "noisy")}
        R = {"n": len(rows)}
        for k in ("out_frac", "mass_out", "mass_audio", "ratio_max"):
            xc, xn = L["clean"][k].mean(1), L["noisy"][k].mean(1)
            dd = xn - xc
            nzero = int((dd != 0).sum())
            w = wilcoxon(xn, xc, alternative="greater", zero_method="wilcox") if nzero else None
            e = {"clean_mean": float(xc.mean()), "noisy_mean": float(xn.mean()), "delta_median": float(np.median(dd)),
                 "delta_mean": float(dd.mean()), "n_nonzero_delta": nzero,
                 "wilcoxon_greater_p": float(w.pvalue) if w else None}
            for nm, y in (("main", drop_main), ("whole", drop_whole)):
                if np.all(dd == dd[0]):
                    e[f"spearman_{nm}"] = None
                    continue
                rho, p = spearmanr(dd, y)
                pr, pp = pspear(dd, y, bvr)
                e[f"spearman_{nm}"] = {"rho": float(rho), "p": float(p), "partial_bvr_rho": pr, "partial_bvr_p": pp}
            # per layer
            lay = []
            for li in range(N_LAYERS):
                dl = L["noisy"][k][:, li] - L["clean"][k][:, li]
                rho = spearmanr(dl, drop_main)[0] if not np.all(dl == dl[0]) else None
                lay.append({"layer": li, "clean": float(L["clean"][k][:, li].mean()), "noisy": float(L["noisy"][k][:, li].mean()),
                            "delta_median": float(np.median(dl)), "spearman_main": None if rho is None else float(rho)})
            e["layers"] = lay
            R[k] = e
        R["encoder_output_out_frac"] = {"clean": float(L["clean"]["out_frac"][:, 0].mean()),
                                        "noisy": float(L["noisy"]["out_frac"][:, 0].mean())}
        m = R["mass_out"]
        R["gate"] = {"wilcoxon_p_lt_0.01": m["wilcoxon_greater_p"] is not None and m["wilcoxon_greater_p"] < 0.01,
                     "spearman_main_ge_0.2": m["spearman_main"] is not None and m["spearman_main"]["rho"] >= 0.2}
        res[d] = R
    res["gate_pass"] = all(res[d]["gate"]["wilcoxon_p_lt_0.01"] and res[d]["gate"]["spearman_main_ge_0.2"]
                           for d in ("zh2en", "en2zh"))
    json.dump(res, open(dst, "w", encoding="utf-8"), indent=1)
    for d in ("zh2en", "en2zh"):
        R = res[d]
        print(d, "n", R["n"], "gate", R["gate"])
        for k in ("out_frac", "mass_out", "mass_audio", "ratio_max"):
            e = R[k]
            print(f"  {k}: clean {e['clean_mean']:.5f} noisy {e['noisy_mean']:.5f} dmed {e['delta_median']:.5f} "
                  f"wil_p {e['wilcoxon_greater_p']} sp_main {e['spearman_main']} sp_whole {e['spearman_whole']}")
    print("gate_pass", res["gate_pass"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "summary"))
    ap.add_argument("--dir", choices=("zh2en", "en2zh"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-gib", type=int, default=120)
    a = ap.parse_args()
    run(a) if a.mode == "run" else summary(a)


if __name__ == "__main__":
    main()
