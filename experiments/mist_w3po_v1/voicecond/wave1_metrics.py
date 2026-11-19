"""Robust wave 1, K1 P1b / K4 P4b / K7v: output-speech metrics and the preregistered gates (real output metrics only).

measure  per generated wav (wave1_talker gen rows.jsonl): SQUIM PESQ / STOI / SI-SDR, flatness / contrast and the WavLM-SV
         similarity to the speaker's clean prompts (cvt/refs/<spk>/p*.wav, mean embedding) -- bench_eval._item_job, CPU
asrin    the ASR input rows for evaluation.asr_transcribe (Whisper-large-v3 for English output, Paraformer-zh for Chinese)
gate     arms vs C (P1b, K7v). Per item, each metric averaged over the two TTS seeds; paired item differences arm - C:
         PESQ median diff > 0 with the bootstrap 95 % CI lower bound > 0 (10000 resamples, seed 0); WER (English, words) /
         CER (Chinese, characters) of the ASR transcript against the CoVoST human translation: median diff <= 0;
         similarity median diff >= -0.02; duration median ratio arm / C in [0.98, 1.02].
p4b      duration CV across the voice conditions {B, C, ZNE, CFG1} per item (duration averaged over the seeds, population
         std / mean), mean over items, without vs with the rate clamp: reduction = 1 - CV_clamp / CV_plain >= 50 %; WER:
         per (item, condition), seed-averaged, clamp - plain, median <= 0.
    <uniss-eval python> -m experiments.mist_w3po_v1.voicecond.wave1_metrics measure --gen <dir> --items <items> [--procs 32]
    ... asrin --gen <dir> --items <items>
    ... gate --items <items> --base <C dir> --arms <dir> [<dir> ...] --out <json>
    ... p4b --items <items> --plain <B> <C> <ZNE> <CFG1> --clamp <B> <C> <ZNE> <CFG1> --out <json>
measure / asrin write <gen>/metrics.jsonl and <gen>/asr_in.jsonl; every output refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def norm_tokens(text: str, tgt: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text or "").lower()
    if tgt == "eng":
        return re.sub(r"[^a-z0-9']+", " ", t).split()
    return [c for c in t if not c.isspace() and not unicodedata.category(c).startswith(("P", "S"))]


def edit(a, b) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def wer(ref: str, hyp: str, tgt: str) -> float:
    r, h = norm_tokens(ref, tgt), norm_tokens(hyp, tgt)
    return edit(r, h) / max(1, len(r))


def cmd_measure(a):
    import multiprocessing as mp
    from experiments.mist_w3po_v1.voicecond.bench_eval import _item_job
    G = Path(a.gen)
    out = G / "metrics.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    items = {r["id"]: r for r in jl(a.items)}
    rows = [r for r in jl(G / "rows.jsonl") if r.get("wav")]
    jobs = [(r["key"], r["wav"], items[r["id"]]["clean_refs"]) for r in rows]
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    with mp.get_context("fork").Pool(a.procs) as P:
        res = P.map(_item_job, jobs, chunksize=2)
    with open(out, "x", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"gen": str(G), "n": len(res), "errors": sum("error" in r for r in res),
                      "pesq_median": float(np.median([r["pesq"] for r in res if "pesq" in r]))}))


def cmd_asrin(a):
    G = Path(a.gen)
    out = G / "asr_in.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    items = {r["id"]: r for r in jl(a.items)}
    with open(out, "x", encoding="utf-8") as fh:
        for r in jl(G / "rows.jsonl"):
            if r.get("wav"):
                fh.write(json.dumps({"id": r["key"], "mode": "speech", "audio_path": r["wav"],
                                     "tgt_lang": items[r["id"]]["tgt"], "audio_duration_seconds": r["dur_s"]}) + "\n")


def load_run(G, items):
    G = Path(G)
    rows = {r["key"]: r for r in jl(G / "rows.jsonl")}
    met = {r["id"]: r for r in jl(G / "metrics.jsonl")}
    asr = {r["id"]: r for r in jl(G / "asr.jsonl")}
    per = defaultdict(lambda: defaultdict(list))
    arm, flags = None, {"rows": len(rows), "no_wav": 0, "forced": 0, "no_closing": 0, "asr_missing": 0}
    for k, r in rows.items():
        arm = r["arm"]
        it = items[r["id"]]
        flags["forced"] += int(bool(r.get("forced")))
        flags["no_closing"] += int(r.get("closing") not in (180392, 180395))
        if not r.get("wav"):
            flags["no_wav"] += 1
            per[r["id"]]["wer"].append(1.0)
            continue
        m = met.get(k, {})
        for q in ("pesq", "sim", "sisdr", "stoi"):
            if m.get(q) is not None:
                per[r["id"]][q].append(float(m[q]))
        per[r["id"]]["dur"].append(float(r["dur_s"]))
        if k in asr:
            per[r["id"]]["wer"].append(wer(it["text"], asr[k].get("asr_text", ""), it["tgt"]))
        else:
            flags["asr_missing"] += 1
    return arm, {i: {q: float(np.mean(v)) for q, v in d.items() if v} for i, d in per.items()}, flags


def boot_ci(d, fn, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    v = fn(d[idx], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired(A, B, q, ratio=False, ids=None):
    from scipy.stats import wilcoxon
    ids = sorted(i for i in (ids or set(A) & set(B)) if q in A.get(i, {}) and q in B.get(i, {}))
    x = np.array([A[i][q] for i in ids])
    y = np.array([B[i][q] for i in ids])
    d = y / x if ratio else y - x
    try:
        p = float(wilcoxon(y - x).pvalue)
    except ValueError:
        p = 1.0
    return {"n": len(ids), "base_median": float(np.median(x)), "arm_median": float(np.median(y)),
            "base_mean": float(x.mean()), "arm_mean": float(y.mean()),
            ("median_ratio" if ratio else "median_diff"): float(np.median(d)),
            "median_ci": boot_ci(d, np.median), "mean_diff": float((y - x).mean()), "mean_ci": boot_ci(y - x, np.mean),
            "wilcoxon_p": p, "share_arm_gt_base": float(np.mean(y > x))}


def cmd_gate(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    items = {r["id"]: r for r in jl(a.items)}
    _, C, fc = load_run(a.base, items)
    res = {"base": a.base, "base_flags": fc, "arms": {}}
    for g in a.arms:
        arm, X, fx = load_run(g, items)
        r = {"dir": g, "flags": fx, "pesq": paired(C, X, "pesq"), "wer": paired(C, X, "wer"), "sim": paired(C, X, "sim"),
             "dur": paired(C, X, "dur", ratio=True), "sisdr": paired(C, X, "sisdr"), "stoi": paired(C, X, "stoi"),
             "per_dir": {}}
        for d in ("zh2en", "en2zh"):
            ids = {i for i in items if items[i]["dir"] == d}
            r["per_dir"][d] = {q: paired(C, X, q, ratio=(q == "dur"), ids=ids) for q in ("pesq", "wer", "sim", "dur")}
        r["gates"] = {"pesq": bool(r["pesq"]["median_diff"] > 0 and r["pesq"]["median_ci"][0] > 0),
                      "wer": bool(r["wer"]["median_diff"] <= 0), "sim": bool(r["sim"]["median_diff"] >= -0.02),
                      "dur": bool(0.98 <= r["dur"]["median_ratio"] <= 1.02)}
        r["pass"] = all(r["gates"].values())
        res["arms"][arm] = r
        print(f"{arm}: PESQ {r['pesq']['base_median']:.3f}->{r['pesq']['arm_median']:.3f} median diff "
              f"{r['pesq']['median_diff']:+.3f} {r['pesq']['median_ci']} | WER {r['wer']['base_mean']:.3f}->"
              f"{r['wer']['arm_mean']:.3f} median diff {r['wer']['median_diff']:+.3f} | SIM median diff "
              f"{r['sim']['median_diff']:+.4f} | dur ratio {r['dur']['median_ratio']:.4f} | gates {r['gates']} -> {r['pass']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)


def cmd_p4b(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    items = {r["id"]: r for r in jl(a.items)}
    runs = {}
    for kind, dirs in (("plain", a.plain), ("clamp", a.clamp)):
        for g in dirs:
            arm, X, fl = load_run(g, items)
            runs[(kind, arm)] = (X, fl, g)
    conds = sorted({arm for _, arm in runs})
    assert all(("plain", c) in runs and ("clamp", c) in runs for c in conds)
    res = {"conds": conds, "runs": {f"{k}_{c}": {"dir": v[2], "flags": v[1]} for (k, c), v in runs.items()}}
    ids = sorted(set.intersection(*[set(v[0]) for v in runs.values()]))
    cv = {}
    for kind in ("plain", "clamp"):
        D = np.array([[runs[(kind, c)][0][i]["dur"] for c in conds] for i in ids])
        cv[kind] = D.std(1) / D.mean(1)
        res[f"dur_mean_{kind}"] = {c: float(D[:, n].mean()) for n, c in enumerate(conds)}
    res["n_items"] = len(ids)
    res["cv_plain_mean"], res["cv_clamp_mean"] = float(cv["plain"].mean()), float(cv["clamp"].mean())
    res["cv_reduction"] = 1 - res["cv_clamp_mean"] / res["cv_plain_mean"]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(10000):
        j = rng.integers(0, len(ids), len(ids))
        bs.append(1 - cv["clamp"][j].mean() / cv["plain"][j].mean())
    res["cv_reduction_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["cv_median_plain"], res["cv_median_clamp"] = float(np.median(cv["plain"])), float(np.median(cv["clamp"]))
    dw = []
    for c in conds:
        P, Q = runs[("plain", c)][0], runs[("clamp", c)][0]
        for i in ids:
            if "wer" in P[i] and "wer" in Q[i]:
                dw.append(Q[i]["wer"] - P[i]["wer"])
        res[f"wer_{c}"] = paired(P, Q, "wer")
        res[f"dur_ratio_{c}"] = paired(P, Q, "dur", ratio=True)
        res[f"pesq_{c}"] = paired(P, Q, "pesq")
    dw = np.array(dw)
    res["wer_diff_clamp_minus_plain"] = {"n": len(dw), "median": float(np.median(dw)), "mean": float(dw.mean()),
                                         "mean_ci": boot_ci(dw, np.mean)}
    res["gates"] = {"cv_reduction_ge_0.5": bool(res["cv_reduction"] >= 0.5), "wer": bool(np.median(dw) <= 0)}
    res["pass"] = all(res["gates"].values())
    print(json.dumps({k: res[k] for k in ("n_items", "cv_plain_mean", "cv_clamp_mean", "cv_reduction", "cv_reduction_ci",
                                          "wer_diff_clamp_minus_plain", "gates", "pass")}))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("measure")
    s.add_argument("--gen", required=True)
    s.add_argument("--items", required=True)
    s.add_argument("--procs", type=int, default=32)
    s = sub.add_parser("asrin")
    s.add_argument("--gen", required=True)
    s.add_argument("--items", required=True)
    s = sub.add_parser("gate")
    s.add_argument("--items", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--arms", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("p4b")
    s.add_argument("--items", required=True)
    s.add_argument("--plain", nargs="+", required=True)
    s.add_argument("--clamp", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"measure": cmd_measure, "asrin": cmd_asrin, "gate": cmd_gate, "p4b": cmd_p4b}[a.cmd](a)


if __name__ == "__main__":
    main()
