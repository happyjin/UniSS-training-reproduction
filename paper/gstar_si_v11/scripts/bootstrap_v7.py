"""Paper v7 statistics (prereg_v7 §0): paired comparisons of cpBLEU_sess and of whole-meeting tcpBLEU.

For each pair (a vs b) on the windows both systems decoded, as bootstrap_v6.py, except:
  perm      exact paired permutation test by meeting: all 2^M sign flips of the meeting-level outputs (M <= 20),
            two-sided (|flipped difference| >= |observed|)
  noise     decoder noise sigma_dec(n) = max(12.3 / sqrt(n), N-check estimate scaled to n) for n scored windows;
            "ci95_noise" = diff +- 1.96 sqrt(var(meeting bootstrap) + sigma_dec(n)^2)
  lomo      min / max of the difference with one meeting left out, and the meeting whose removal moves it most
  verdict   "a better" / "b better" only if the (Holm-adjusted, for the registered family) p < 0.05 AND ci95_noise
            excludes 0; else "tie within noise and meeting sampling"
Seed pairs ("seeds:<dir>:<a1>+<a2>+..|<b>") average the difference over the seeds; their bootstrap resamples meetings and
seeds (two levels); the permutation flips each meeting's outputs for all seeds at once.
Holm over the registered family (prereg_v7: E-on1..3, E-cas1..3, E-seed-en, E-seed-zh).

    python scripts/bootstrap_v7.py           (writes generated/bootstrap_v7.json; existing keys are kept unless --redo)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_v6 as b6  # noqa: E402
from bootstrap_v6 import smap_of, stats, tok, window_data  # noqa: E402
from systems import ASR_FILE, SPEECH_DIR  # noqa: E402
from systems_v7 import SC7, SP6, V6, V7, hyp_dir, windows  # noqa: E402

B = 1000
GEN = Path(__file__).resolve().parents[1] / "generated"
SIG0 = 12.3   # sigma_dec(n) = SIG0 / sqrt(n), from v6's T13 (96 windows: 0.46)


def bleu_vec(S):
    """corpus BLEU (sacrebleu defaults: exp smoothing, 4-grams) of each row of S (N x 10 sacrebleu statistics)."""
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    hyp, ref, cor, tot = S[:, 0], S[:, 1], S[:, 2:6], S[:, 6:10]
    k = np.cumsum(cor == 0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cor > 0, 100.0 * cor / np.maximum(tot, 1), 100.0 / (np.power(2.0, k) * np.maximum(tot, 1)))
        p = np.where(tot > 0, p, 0.0)
        logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
        bp = np.where(hyp < ref, np.exp(1 - ref / np.maximum(hyp, 1e-9)), 1.0)
        out = np.where(hyp > 0, bp * np.exp(logp.mean(1)), 0.0)
    return np.nan_to_num(out, nan=0.0, neginf=0.0)


def sig_dec(n):
    s0 = SIG0
    f = GEN / "v7_nchk.json"
    if f.exists():
        s0 = max(s0, json.load(open(f))["sigma0"])
    return float(s0 / np.sqrt(max(n, 1)))


def perm_exact(Sa, Sb):
    """Sa, Sb: (M x 10) per-meeting statistics; returns two-sided exact permutation p."""
    M = len(Sa)
    obs = abs(bleu_vec(Sa.sum(0))[0] - bleu_vec(Sb.sum(0))[0])
    D = Sb - Sa
    cnt = tot = 0
    base = Sa.sum(0)
    for chunk in range(0, 2 ** M, 1 << 16):
        idx = np.arange(chunk, min(chunk + (1 << 16), 2 ** M))
        F = ((idx[:, None] >> np.arange(M)[None, :]) & 1).astype(np.float64)
        A = base + F @ D
        Bm = Sa.sum(0) + Sb.sum(0) - A
        d = np.abs(bleu_vec(A) - bleu_vec(Bm))
        cnt += int((d >= obs - 1e-9).sum())
        tot += len(idx)
    return cnt / tot


def compare(Xa, Xb, meet, rng, nwin):
    ms = np.unique(meet)
    idx = {m: np.flatnonzero(meet == m) for m in ms}
    Sa = np.stack([Xa[idx[m]].sum(0) for m in ms])
    Sb = np.stack([Xb[idx[m]].sum(0) for m in ms])
    full = float(bleu_vec(Sa.sum(0))[0] - bleu_vec(Sb.sum(0))[0])
    J = rng.integers(0, len(ms), (B, len(ms)))
    ds = np.array([bleu_vec(Sa[j].sum(0))[0] - bleu_vec(Sb[j].sum(0))[0] for j in J])
    sd = float(np.sqrt(ds.var() + sig_dec(nwin) ** 2))
    r = {"diff": round(full, 2), "windows": int(nwin), "meetings": int(len(ms)),
         "ci95": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
         "p_boot_two_sided": b6.two_sided(ds), "sigma_dec": round(sig_dec(nwin), 3),
         "ci95_noise": [round(full - 1.96 * sd, 2), round(full + 1.96 * sd, 2)],
         "perm_p_exact": round(perm_exact(Sa, Sb), 5)}
    lomo = {str(m): float(bleu_vec(np.delete(Sa, i, 0).sum(0))[0] - bleu_vec(np.delete(Sb, i, 0).sum(0))[0])
            for i, m in enumerate(ms)}
    worst = max(lomo, key=lambda m: abs(lomo[m] - full))
    r["lomo"] = [round(min(lomo.values()), 2), round(max(lomo.values()), 2), worst]
    r["per_meeting"] = {str(m): round(float(bleu_vec(Sa[i])[0] - bleu_vec(Sb[i])[0]), 2) for i, m in enumerate(ms)}
    return r


def compare_seeds(Xs, Xb, meet, rng, nwin):
    """Xs: list of (windows x 10) for the seeds of system a; Xb: system b; difference averaged over seeds."""
    ms = np.unique(meet)
    idx = {m: np.flatnonzero(meet == m) for m in ms}
    Ss = [np.stack([X[idx[m]].sum(0) for m in ms]) for X in Xs]
    Sb = np.stack([Xb[idx[m]].sum(0) for m in ms])
    each = [float(bleu_vec(S.sum(0))[0] - bleu_vec(Sb.sum(0))[0]) for S in Ss]
    full = float(np.mean(each))
    ds = []
    for _ in range(B):
        j = rng.integers(0, len(ms), len(ms))
        ks = rng.integers(0, len(Ss), len(Ss))
        ds.append(np.mean([bleu_vec(Ss[k][j].sum(0))[0] - bleu_vec(Sb[j].sum(0))[0] for k in ks]))
    ds = np.array(ds)
    sd = float(np.sqrt(ds.var() + sig_dec(nwin) ** 2 / len(Ss)))
    M = len(ms)
    obs = abs(full)
    cnt = tot = 0
    for chunk in range(0, 2 ** M, 1 << 16):
        ii = np.arange(chunk, min(chunk + (1 << 16), 2 ** M))
        F = ((ii[:, None] >> np.arange(M)[None, :]) & 1).astype(np.float64)
        dd = np.zeros(len(ii))
        for S in Ss:
            A = S.sum(0) + F @ (Sb - S)
            Bm = S.sum(0) + Sb.sum(0) - A
            dd += bleu_vec(A) - bleu_vec(Bm)
        cnt += int((np.abs(dd / len(Ss)) >= obs - 1e-9).sum())
        tot += len(ii)
    return {"diff": round(full, 2), "per_seed": [round(x, 2) for x in each], "seed_sd": round(float(np.std(each, ddof=1)), 2) if len(each) > 1 else None,
            "windows": int(nwin), "meetings": int(M), "seeds": len(Ss),
            "ci95": [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)],
            "p_boot_two_sided": b6.two_sided(ds), "ci95_noise": [round(full - 1.96 * sd, 2), round(full + 1.96 * sd, 2)],
            "perm_p_exact": round(cnt / tot, 5)}


def data_of(key):
    speech = key.startswith("sp:")
    dr, name = key.removeprefix("sp:").split(":")
    d = hyp_dir(dr, name)
    if d is None or not (d / "hyp.jsonl").exists():
        return None
    asr = None
    if speech:
        sp = SPEECH_DIR(dr, name)
        cand = SP6 / f"{dr}_{name}" / "talker/asr_speech.jsonl"
        asr = cand if cand.exists() else (ASR_FILE(sp) if sp else None)
        if asr is None or not asr.exists():
            return None
    return dr, window_data(dr, d, windows(dr, name), asr)


def online_stats(f):
    P = json.load(open(f))["per_meeting"]
    return {m: np.array(v["tcp_stats"], dtype=np.float64) for m, v in P.items()}


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = round(run, 5)
    return out


def main() -> None:
    from pairs_v7 import ONLINE, PAIRS, REGISTERED, SEEDS
    redo = "--redo" in sys.argv
    f = GEN / "bootstrap_v7.json"
    res = {} if redo or not f.exists() else json.load(open(f))
    rng = np.random.default_rng(20261101)
    cache = {}

    def get(k):
        if k not in cache:
            cache[k] = data_of(k)
        return cache[k]

    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        if key in res or get(a) is None or get(b) is None:
            print("skip" if key not in res else "kept", key, flush=True)
            continue
        dr, da = get(a)
        _, db = get(b)
        ids = sorted(set(da) & set(db))
        Xa, Xb = stats(dr, da, ids, smap_of(da, ids)), stats(dr, db, ids, smap_of(db, ids))
        meet = np.array([da[i]["meeting"] for i in ids])
        res[key] = {"a": a, "b": b, **compare(Xa, Xb, meet, rng, len(ids))}
        print(key, json.dumps({k: v for k, v in res[key].items() if k != "per_meeting"}), flush=True)
        f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for dr, na, nb, fa, fb, nwin in ONLINE:
        key = f"on:{dr}:{na}_vs_{nb}"
        if key in res or not (Path(fa).exists() and Path(fb).exists()):
            continue
        A, Bm = online_stats(fa), online_stats(fb)
        ms = sorted(set(A) & set(Bm))
        res[key] = {"a": na, "b": nb, **compare(np.stack([A[m] for m in ms]), np.stack([Bm[m] for m in ms]), np.array(ms), rng, nwin)}
        print(key, json.dumps({k: v for k, v in res[key].items() if k != "per_meeting"}), flush=True)
        f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for dr, seeds, b in SEEDS:
        key = f"seeds:{dr}:{'+'.join(seeds)}_vs_{b}"
        if key in res or any(get(f"{dr}:{s}") is None for s in seeds) or get(f"{dr}:{b}") is None:
            continue
        ds_ = [get(f"{dr}:{s}")[1] for s in seeds]
        db = get(f"{dr}:{b}")[1]
        ids = sorted(set(db).intersection(*[set(x) for x in ds_]))
        Xs = [stats(dr, x, ids, smap_of(x, ids)) for x in ds_]
        Xb = stats(dr, db, ids, smap_of(db, ids))
        meet = np.array([db[i]["meeting"] for i in ids])
        res[key] = {"seeds": seeds, "b": b, **compare_seeds(Xs, Xb, meet, rng, len(ids))}
        print(key, json.dumps(res[key]), flush=True)
        f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    ps = {k: res[k]["perm_p_exact"] for k in REGISTERED if k in res}
    if len(ps) == len(REGISTERED):
        H = holm(ps)
        for k, lab in REGISTERED.items():
            r = res[k]
            r["label"], r["p_holm"] = lab, H[k]
            lo, hi = r["ci95_noise"]
            r["verdict"] = ("a better" if lo > 0 else "b better") if (H[k] < 0.05 and (lo > 0 or hi < 0)) else "tie"
    for k, r in res.items():
        if k not in REGISTERED and "perm_p_exact" in r:
            lo, hi = r["ci95_noise"]
            r["verdict"] = ("a better" if lo > 0 else "b better") if (r["perm_p_exact"] < 0.05 and (lo > 0 or hi < 0)) else "tie"
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("registered:", {REGISTERED[k]: (res[k].get("p_holm"), res[k].get("verdict")) for k in REGISTERED if k in res})


if __name__ == "__main__":
    main()
