"""voice_onset_v1 P0 (reports/voice_onset_v1/P0_prereg.zh-CN.md): does waiting longer add identity evidence more under
noise than when clean? CPU only, no training. S1 dev twins; tau = first block write_ms of the base LA2 Thinker; prompt =
source[: max(tau, 1000) + delta], delta in {0, 500, 1000, 1500, 2000} ms; identity = cos(TitaNet(prompt),
TitaNet(clean twin tail not overlapping any prompt)) (prereg change 1).
    nemo_diar: python -m experiments.voice_onset_v1.p0_prompt_length embed --out <dir> [--workers 24]
               python -m experiments.voice_onset_v1.p0_prompt_length score --out <dir>
Refuses existing outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
DIRS = ("en2zh", "zh2en")
DELTAS = (0, 500, 1000, 1500, 2000)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jobs():
    """one job per (direction, set, pair) with both twins"""
    out = []
    for d in DIRS:
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        audio = {r["id"]: r["audio"] for r in jl(S1 / "data" / f"items_{d}.jsonl")}
        first = {r["id"]: r["blocks"][0]["write_ms"] for r in jl(S1 / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}
        by = {}
        for i, m in meta.items():
            by.setdefault((m["set"], m["pair"]), {})[m["cond"]] = i
        for (st, pr), v in by.items():
            if "clean" not in v or "noisy" not in v:
                continue
            c, n = v["clean"], v["noisy"]
            out.append({"dir": d, "set": st, "pair": pr, "stream": meta[n]["stream"], "ovl": bool(meta[n].get("ovl")),
                        "clean": audio[c], "noisy": audio[n], "tau_c": int(first[c]), "tau_n": int(first[n])})
    return out


def worker(args):
    chunk, wid = args
    import soundfile as sf
    import torch
    torch.set_num_threads(4)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location="cpu").eval()

    @torch.no_grad()
    def emb(x):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        _, e = m.forward(input_signal=t, input_signal_length=torch.tensor([t.shape[1]]))
        e = e[0].float().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def read(p):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == 16000, p
        return x.mean(1)

    rows = []
    for j in chunk:
        xc, xn = read(j["clean"]), read(j["noisy"])
        dur = min(len(xc), len(xn)) // 16
        L1 = max(j["tau_c"], j["tau_n"], 1000) + 1000
        r = {k: j[k] for k in ("dir", "set", "pair", "stream", "ovl", "tau_c", "tau_n")} | {"dur_ms": dur, "L1": L1}
        for tag, L in (("ref1", L1), ("ref2", L1 + 1000)):
            if dur - L < 1000:
                continue
            ref = emb(xc[L * 16:])
            nd = 3 if tag == "ref1" else 5
            for cond, x, tau in (("clean", xc, j["tau_c"]), ("noisy", xn, j["tau_n"])):
                base = max(tau, 1000)
                r[f"{tag}_{cond}"] = [float(emb(x[: (base + dl) * 16]) @ ref) for dl in DELTAS[:nd]]
        rows.append(r)
    return rows


def embed(a):
    out = Path(a.out)
    f = out / "rows.jsonl"
    if f.exists():
        sys.exit(f"拒绝覆盖已有输出: {f}")
    out.mkdir(parents=True, exist_ok=True)
    J = jobs()
    n = a.workers
    chunks = [J[k::n] for k in range(n)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(n) as pool, open(f, "x", encoding="utf-8") as fh:
        for rows in pool.imap_unordered(worker, [(c, k) for k, c in enumerate(chunks)]):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[p0] +{len(rows)}", flush=True)
    print(f"[p0] done {len(J)} pairs", flush=True)


def boot_ci(dg, streams, rng):
    units = sorted(set(streams)); up = {u: k for k, u in enumerate(units)}
    pu = np.array([up[u] for u in streams]); nU = len(units)
    s = np.zeros(nU); c = np.zeros(nU); np.add.at(s, pu, dg); np.add.at(c, pu, 1)
    b = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, nU, nU), minlength=nU)
        b.append(float((s * w).sum() / (c * w).sum()))
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)], nU


def score(a):
    out = Path(a.out)
    f = out / "summary.json"
    if f.exists():
        sys.exit(f"拒绝覆盖已有输出: {f}")
    R = jl(out / "rows.jsonl")
    res = {}
    rng = np.random.default_rng(0)
    for d in DIRS:
        res[d] = {}
        for name, tag, keep in (("main_ab_ref1", "ref1", lambda r: True),
                                ("b_noxt_ref1", "ref1", lambda r: r["set"] == "b" and not r["ovl"]),
                                ("curve_ab_ref2", "ref2", lambda r: True)):
            P = [r for r in R if r["dir"] == d and f"{tag}_clean" in r and keep(r)]
            if not P:
                continue
            sn = np.array([r[f"{tag}_noisy"] for r in P]); sc = np.array([r[f"{tag}_clean"] for r in P])
            gn = sn[:, 2] - sn[:, 0]; gc = sc[:, 2] - sc[:, 0]; dg = gn - gc
            ci, nU = boot_ci(dg, [r["stream"] for r in P], rng)
            gap = sc[:, 0] - sn[:, 0]
            res[d][name] = {
                "n_pairs": len(P), "n_units": nU,
                "tau_ms_median": {"clean": float(np.median([r["tau_c"] for r in P])), "noisy": float(np.median([r["tau_n"] for r in P]))},
                "sim_noisy_by_delta": [round(float(x), 4) for x in sn.mean(0)],
                "sim_clean_by_delta": [round(float(x), 4) for x in sc.mean(0)],
                "g_noisy_1s": round(float(gn.mean()), 4), "g_clean_1s": round(float(gc.mean()), 4),
                "g_diff_1s": round(float(dg.mean()), 4), "g_diff_ci95": ci,
                "gap_at_0": round(float(gap.mean()), 4),
                "gap_closed_by_delta": [round(float((sn[:, k] - sn[:, 0]).mean() / gap.mean()), 3) for k in range(sn.shape[1])]}
        m = res[d]["main_ab_ref1"]
        res[d]["rule"] = {"g_noisy_ge_0.02": m["g_noisy_1s"] >= 0.02, "diff_ci_low_gt0": m["g_diff_ci95"][0] > 0}
    res["PASS"] = all(all(res[d]["rule"].values()) for d in DIRS)
    with open(f, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("embed", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    embed(a) if a.cmd == "embed" else score(a)


if __name__ == "__main__":
    main()
