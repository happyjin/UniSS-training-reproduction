"""Robust wave 3, group G scoring (robust_wave3_single_prereg.zh-CN.md, G + change 1: one TTS seed 20261021, paired
directly). Arms: w1 (guided) vs w0 (same implementation, control), set (a), per direction x condition (noisy = gate,
clean = reported).

check   the w = 0 backend check on the 20-item dev smoke: per item, total codes HF w0 / si_infer vLLM (same C voice),
        median ratio in [0.98, 1.02].
          python .../wave3_g_score.py check --hf <dir> --vllm <dir> --out <json>
spksim  (venvs/nemo_diar) TitaNet-large cosine between each output's speech-only wav and the sentence's clean source
        (wave-1 manifest `clean`) -> <run>/spksim.jsonl (refuses existing)
          python .../wave3_g_score.py spksim --run <dir> [--device cuda:0]
gate    per direction x condition: D2 (and D1) LAAL mean diff w1 - w0 with a 95 % CI; corpus ASR-BLEU diff; PESQ (SQUIM)
        and similarity paired median diff with a 95 % CI; 2000 paired bootstrap resamples over sentences (seed 0);
        gates of the prereg on the noisy condition. qwen3omni python.
          python .../wave3_g_score.py gate --root eval_outputs/mist_w3po_v1/wave3_combos/G --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
MANIFEST = REPO / "eval_outputs/mist_w3po_v1/wave1_K2K3K5/data/manifest.jsonl"
NEMO = ("/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--nvidia--speakerverification_en_titanet_large/snapshots/"
        "0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo")
LANG = {"en2zh": "cmn", "zh2en": "eng"}
B = 2000


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def results(run: Path):
    rows = []
    for p in sorted(run.glob("part_*.jsonl")):
        rows += jl(p)
    return {r["id"]: r for r in rows}


def check(a) -> None:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    for hf, vl in zip(a.hf, a.vllm):
        H, V = results(Path(hf)), results(Path(vl))
        ids = sorted(set(H) & set(V))
        r = [H[i]["codes"] / max(V[i]["codes"], 1) for i in ids]
        res[hf] = {"vllm": vl, "n": len(ids), "median_ratio": round(float(np.median(r)), 4),
                   "mean_ratio": round(float(np.mean(r)), 4), "ratios": [round(x, 3) for x in r]}
    allr = [x for v in res.values() for x in v["ratios"]]
    res["pooled"] = {"n": len(allr), "median_ratio": round(float(np.median(allr)), 4),
                     "pass_0.98_1.02": bool(0.98 <= np.median(allr) <= 1.02)}
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["pooled"]), {k: v["median_ratio"] for k, v in res.items() if k != "pooled"})


def spksim(a) -> None:
    run = Path(a.run)
    out = run / "spksim.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()
    clean = {r["id"]: REPO / r["clean"] for r in jl(MANIFEST)}

    @torch.no_grad()
    def emb(p):
        x, sr = sf.read(str(p), dtype="float32")
        assert sr == 16000
        x = x if x.ndim == 1 else x.mean(1)
        if len(x) < 8000:
            return None
        t = torch.from_numpy(x)[None].to(a.device)
        _, e = m.forward(input_signal=t, input_signal_length=torch.tensor([len(x)], device=a.device))
        return torch.nn.functional.normalize(e[0].float(), dim=0)

    cache = {}
    with open(out, "x", encoding="utf-8") as fh:
        for i, r in sorted(results(run).items()):
            cv = i.split("__", 1)[1]
            if cv not in cache:
                cache[cv] = emb(clean[cv])
            e = emb(r["speech_path"]) if r.get("codes") else None
            s = float((e * cache[cv]).sum()) if e is not None and cache[cv] is not None else None
            fh.write(json.dumps({"id": i, "sim": s}) + "\n")
    print(out)


def laal_defs():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "laal_defs", REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gate(a) -> None:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
    LD = laal_defs()
    root = Path(a.root)
    res = {"arms": "w1 (l+ + 1*(l+ - l-)) vs w0 (same HF implementation)", "seed": 20261021,
           "bootstrap": f"{B} paired resamples over sentences, seed 0", "diff": "w1 - w0"}
    for d in ("en2zh", "zh2en"):
        bl = Bleu(LANG[d])
        per = {}
        for arm in ("w0", "w1"):
            run = root / f"{arm}_{d}"
            recs = results(run)
            heard = {r["id"]: r.get("asr_text", "") for r in jl(run / "asr_speech.jsonl")}
            sq = {r["id"]: r.get("pesq") for r in jl(run / "squim.jsonl")}
            sim = {r["id"]: r.get("sim") for r in jl(run / "spksim.jsonl")}
            for i, r in recs.items():
                sd = LD.sentence_defs(r)
                per.setdefault(i, {"ref": r["translation_ref"]})[arm] = {
                    "asr": heard.get(i, ""), "d1": sd["D1"] / 1000 if sd else None, "d2": sd["D2"] / 1000 if sd else None,
                    "pesq": sq.get(i), "sim": sim.get(i), "codes": r["codes"]}
        res[d] = {}
        for cond in ("noisy", "clean"):
            ids = sorted(i for i in per if i.startswith(f"a_{cond}__") and "w0" in per[i] and "w1" in per[i])
            n = len(ids)
            st = {arm: bl.stats([per[i][arm]["asr"] for i in ids], [per[i]["ref"] for i in ids]) for arm in ("w0", "w1")}
            rng = np.random.default_rng(0)
            W = [np.ones(n)] + [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(B)]

            def vec(k, arm):
                return np.array([np.nan if per[i][arm][k] is None else per[i][arm][k] for i in ids], dtype=float)

            r = {"n": n}
            # ASR-BLEU (corpus): diff of the two corpus scores
            bb = [bl.score((st["w1"] * w[:, None]).sum(0)) - bl.score((st["w0"] * w[:, None]).sum(0)) for w in W]
            r["asr_bleu"] = {"w0": round(bl.score(st["w0"].sum(0)), 2), "w1": round(bl.score(st["w1"].sum(0)), 2),
                             "diff": round(bb[0], 2), "ci95": [round(float(np.percentile(bb[1:], q)), 2) for q in (2.5, 97.5)]}
            for k in ("d2", "d1"):           # mean diff
                x0, x1 = vec(k, "w0"), vec(k, "w1")
                ok = ~np.isnan(x0) & ~np.isnan(x1)
                dd = (x1 - x0)[ok]
                bs = [float(np.mean(dd[rng.integers(0, len(dd), len(dd))])) for _ in range(B)]
                r[f"laal_{k}"] = {"w0": round(float(np.mean(x0[ok])), 3), "w1": round(float(np.mean(x1[ok])), 3),
                                  "mean_diff": round(float(np.mean(dd)), 3), "n": int(ok.sum()),
                                  "ci95": [round(float(np.percentile(bs, q)), 3) for q in (2.5, 97.5)]}
            for k in ("pesq", "sim"):        # paired median diff
                x0, x1 = vec(k, "w0"), vec(k, "w1")
                ok = ~np.isnan(x0) & ~np.isnan(x1)
                dd = (x1 - x0)[ok]
                bs = [float(np.median(dd[rng.integers(0, len(dd), len(dd))])) for _ in range(B)]
                r[k] = {"w0_median": round(float(np.median(x0[ok])), 4), "w1_median": round(float(np.median(x1[ok])), 4),
                        "median_diff": round(float(np.median(dd)), 4), "n": int(ok.sum()),
                        "ci95": [round(float(np.percentile(bs, q)), 4) for q in (2.5, 97.5)]}
            c0, c1 = vec("codes", "w0"), vec("codes", "w1")
            r["speech_len_ratio_median"] = round(float(np.median(c1 / np.maximum(c0, 1))), 4)
            r["gate"] = {
                "laal_d2_lt0_ci_upper_lt0": bool(r["laal_d2"]["mean_diff"] < 0 and r["laal_d2"]["ci95"][1] < 0),
                "asr_bleu_ge_-0.5_ci_upper_ge0": bool(r["asr_bleu"]["diff"] >= -0.5 and r["asr_bleu"]["ci95"][1] >= 0),
                "pesq_median_ge_-0.05": bool(r["pesq"]["median_diff"] >= -0.05),
                "sim_median_ge_-0.02": bool(r["sim"]["median_diff"] >= -0.02)}
            r["gate"]["all"] = all(r["gate"].values())
            res[d][cond] = r
    res["pass_noisy_both_dirs"] = all(res[d]["noisy"]["gate"]["all"] for d in ("en2zh", "zh2en"))
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check"); p.add_argument("--hf", nargs="+", required=True); p.add_argument("--vllm", nargs="+", required=True)
    p.add_argument("--out", required=True); p.set_defaults(f=check)
    p = sub.add_parser("spksim"); p.add_argument("--run", required=True); p.add_argument("--device", default="cuda:0"); p.set_defaults(f=spksim)
    p = sub.add_parser("gate"); p.add_argument("--root", required=True); p.add_argument("--out", required=True); p.set_defaults(f=gate)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
