"""Idea 2 E2.2 evaluation on the synthetic streams (prereg §4.3-4.7): error / token recovery vs seconds observed.

    python -I experiments/mist_w3po_v1/voicecond/idea2_eval.py --which pilot|all --split test --e20 <dir> --em <dir> [--device cuda:0]

Methods (all causal; a window is usable from its end + 1.04 s diarizer delay):
  first8   phi(first <= 8 s of the slot's solo runs (p_j > .5, others < .5, runs >= 0.4 s)); < 1 s: the last 2 s of mix
  runmean  pi-weighted mean of the slot's window codes -> M          median   coordinate-wise median -> M
  shrink   Kalman with A = 0 and no v -> M                          kal_nov  full params, observation 2 dropped -> M
  kal_full full linear CA-Kalman -> M
  ref_clean (oracle labels, clean twin windows, no delay) and ref_vhat (HDemucs vocals windows, slot weights): references.
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/eval_<which>_<split>/ {units.npz, summary.json}; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea2_cakalman import CAKalmanStream, Params  # noqa: E402
from experiments.mist_w3po_v1.voicecond.idea2_common import (DIAR_DELAY, I2, OUTD, T_EVAL, apply_map,  # noqa: E402
                                                             cluster_boot, load_map, load_refs, load_stream, sids,
                                                             slot_for_speaker, speech_clock)

METHODS = ["first8", "runmean", "median", "shrink", "kal_nov", "kal_full", "ref_clean", "ref_vhat"]
SR = 16000


def solo_runs(post, s):
    alone = (post[:, s] > 0.5) & (np.delete(post, s, axis=1).max(1) < 0.5)
    runs, i = [], 0
    while i < len(alone):
        if alone[i]:
            j = i
            while j < len(alone) and alone[j]:
                j += 1
            if (j - i) * 0.08 >= 0.4:
                runs.append((i * 0.08, j * 0.08))
            i = j
        else:
            i += 1
    return runs


def first8_prompt(mix, runs, until):
    parts, tot = [], 0.0
    for a, b in runs:
        if b > until:
            break
        take = min(b - a, 8.0 - tot)
        parts.append(mix[int(a * SR): int((a + take) * SR)])
        tot += take
        if tot >= 8.0 - 1e-9:
            break
    if tot < 1.0:
        e = max(0.0, until)
        return mix[int(max(0.0, e - 2.0) * SR): max(int(e * SR), int(max(0.0, e - 2.0) * SR) + SR // 10)], True
    return np.concatenate(parts), False


def run_stream(sid, L, p, Mk, refs, P, dev):
    d = load_stream(sid, L)
    meta = d["meta"]
    K = len(d["end"])
    avail = d["end"] + DIAR_DELAY
    post = torch.load(I2 / "pre/diar" / f"{sid}.post.pt", weights_only=False)["post80"].float().numpy()
    mix = sf.read(str(I2 / "streams" / sid / "mix.wav"), dtype="float32")[0]
    J = len(meta["speakers"])
    unit_slot = [slot_for_speaker(d, j) for j in range(J)]
    slots = sorted(set(unit_slot))
    # filters
    filt = {"kal_full": CAKalmanStream(p, dev=dev), "kal_nov": CAKalmanStream(p, dev=dev, use_v=False),
            "shrink": CAKalmanStream(p, dev=dev, use_v=False, nuisance=False)}
    hist = {m: {s: np.zeros((K, 192)) for s in slots} for m in filt}
    trp = {s: np.zeros(K) for s in slots}
    for k in range(K):
        sl, pi = int(d["slot"][k]), float(d["pi"][k])
        z = d["z_mix"][k] if sl >= 0 else None
        for m, f in filt.items():
            f.step(z, sl, pi, d["v"][k])
            for s in slots:
                hist[m][s][k] = f.mean(s)
        for s in slots:
            trp[s][k] = filt["kal_full"].trP(s)
    prior = p.mu0
    out = []
    for j in range(J):
        s = unit_slot[j]
        spk = meta["speakers"][j]
        u_star, g_star = refs[spk]
        tau, tot = speech_clock(meta, j)
        runs = solo_runs(post, s)
        mine = np.nonzero(d["slot"] == s)[0]
        oracle = np.nonzero(d["single"] == j)[0]
        rec = {"sid": sid, "spk": spk, "lang": meta["lang"], "tot": tot, "j": j, "slot": s,
               "err": {m: np.full(len(T_EVAL), np.nan) for m in METHODS},
               "tok": {m: np.full(len(T_EVAL), np.nan) for m in METHODS}}
        est = {m: [] for m in METHODS}
        for ti, t in enumerate(T_EVAL):
            tt = tau(t)
            if tt is None:
                continue
            kk = int(np.searchsorted(avail, tt, side="right")) - 1
            sel = mine[mine <= kk]
            w = d["pi"][sel]
            e = {}
            e["runmean"] = (w[:, None] * d["z_mix"][sel]).sum(0) / w.sum() if len(sel) else prior
            e["median"] = np.median(d["z_mix"][sel], 0) if len(sel) else prior
            e["ref_vhat"] = (w[:, None] * d["z_vhat"][sel]).sum(0) / w.sum() if len(sel) else prior
            oc = oracle[d["end"][oracle] <= tt + 1e-9]
            e["ref_clean"] = d["z_speech"][oc].mean(0) if len(oc) else prior
            for m in filt:
                e[m] = hist[m][s][kk] if kk >= 0 else prior
            for m in e:
                est[m].append((ti, apply_map(Mk, e[m])))
            pr, fb = first8_prompt(mix, runs, tt - DIAR_DELAY)
            est["first8"].append((ti, P.phi_wave(pr).reshape(-1)))
        for m in METHODS:
            if not est[m]:
                continue
            idx = [ti for ti, _ in est[m]]
            U = np.stack([u for _, u in est[m]])
            rec["err"][m][idx] = np.linalg.norm(U - u_star[None], axis=1)
            g = P.tokens(U.reshape(-1, 32, 6).astype(np.float32))
            rec["tok"][m][idx] = (g == g_star[None]).mean(1)
        # switching policy (§4.7) on oracle turn ends of this speaker
        ends = sorted(tr["end"] for tr in meta["turns"] if tr["spk"] == spk)
        g_old, tr_old, nsw, emit60 = None, None, 0, np.nan
        t60 = tau(60)
        for te in ends:
            kk = int(np.searchsorted(avail, te, side="right")) - 1
            m_ = hist["kal_full"][s][kk] if kk >= 0 else prior
            g_new = P.tokens(apply_map(Mk, m_).reshape(32, 6).astype(np.float32))
            trn = trp[s][kk] if kk >= 0 else float(np.trace(p.Sig0))
            if g_old is None:
                g_old, tr_old = g_new, trn
            elif (tr_old - trn) / tr_old >= 0.2 and int((g_new != g_old).sum()) >= 4:
                g_old, tr_old, nsw = g_new, trn, nsw + 1
            if t60 is not None and te <= t60:
                emit60 = float((g_old == g_star).mean())
        rec["switches"], rec["emit_tok60"] = nsw, emit60
        out.append(rec)
    return out


def floor_fit(units, m, ts=(8, 10, 15, 20, 30, 40, 50, 60)):
    ti = [T_EVAL.index(t) for t in ts]
    E = np.array([[u["err"][m][i] ** 2 for i in ti] for u in units])
    y = np.nanmean(E, 0)
    X = np.c_[np.ones(len(ts)), 1.0 / np.array(ts)]
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def summarize(units, n_boot=2000):
    by = {}
    for u in units:
        by.setdefault(u["sid"], []).append(u)
    clusters = list(by.values())
    flat = lambda cl: [u for c in cl for u in c]
    S = {"n_units": len(units), "n_streams": len(clusters), "curves": {}, "tok": {}}
    for m in METHODS:
        S["curves"][m] = {str(t): [float(np.sqrt(np.nanmean([u["err"][m][i] ** 2 for u in units]))),
                                   int(np.sum([~np.isnan(u["err"][m][i]) for u in units]))] for i, t in enumerate(T_EVAL)}
        S["tok"][m] = {str(t): float(np.nanmean([u["tok"][m][i] for u in units])) for i, t in enumerate(T_EVAL)}
    i60 = T_EVAL.index(60)
    u60 = [c for c in ([u for u in c if u["tot"] >= 60] for c in clusters) if c]
    S["n_units_60"] = sum(len(c) for c in u60)

    def pdiff(m1, m2, i=i60):
        return lambda cl: float(np.nanmean([u["err"][m1][i] - u["err"][m2][i] for u in flat(cl)]))
    S["paired60"] = {f"kal_full-{m}": cluster_boot(u60, pdiff("kal_full", m), n_boot)
                     for m in ("runmean", "first8", "median", "shrink", "kal_nov", "ref_vhat")}
    S["paired60_tok"] = {f"kal_full-{m}": cluster_boot(
        u60, (lambda m_: lambda cl: float(np.nanmean([u["tok"]["kal_full"][i60] - u["tok"][m_][i60] for u in flat(cl)])))(m), n_boot)
        for m in ("runmean", "first8")}
    first = [T_EVAL.index(1), T_EVAL.index(2)]
    S["first_sentence"] = {m: [float(np.sqrt(np.nanmean([u["err"][m][i] ** 2 for u in units]))) for i in first] for m in METHODS}
    S["paired2_full_vs_first8"] = cluster_boot(clusters, pdiff("kal_full", "first8", T_EVAL.index(2)), n_boot)
    S["floor"] = {m: floor_fit(flat(u60), m) for m in METHODS}
    fa = lambda m, k: (lambda cl: floor_fit(flat(cl), m)[k])
    S["floor_ci"] = {m: cluster_boot(u60, fa(m, 0), n_boot) for m in ("kal_nov", "kal_full", "ref_clean", "runmean")}
    Dfn = lambda cl: (floor_fit(flat(cl), "kal_full")[0] - floor_fit(flat(cl), "ref_clean")[0]) \
        - 0.5 * (floor_fit(flat(cl), "kal_nov")[0] - floor_fit(flat(cl), "ref_clean")[0])
    S["D_c"] = cluster_boot(u60, Dfn, n_boot)
    a_n, b_n = S["floor"]["kal_nov"]
    S["G2.2"] = {
        "a": bool(S["paired60"]["kal_full-runmean"][2] < 0 and S["paired60"]["kal_full-first8"][2] < 0),
        "b": bool(S["floor_ci"]["kal_nov"][1] > 0 and a_n >= b_n / 180),
        "c": bool(S["D_c"][2] < 0)}
    S["G2.2"]["pass"] = bool(all(S["G2.2"].values()))
    S["switches_mean"] = float(np.mean([u["switches"] for u in units]))
    S["emit_tok60_mean"] = float(np.nanmean([u["emit_tok60"] for u in units]))
    for lang in ("en", "zh"):
        ul = [u for u in units if u["lang"] == lang]
        if ul:
            S[f"curves_{lang}"] = {m: {str(t): float(np.sqrt(np.nanmean([u["err"][m][i] ** 2 for u in ul])))
                                       for i, t in enumerate(T_EVAL) if t in (2, 8, 20, 60)} for m in METHODS}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pilot", "all"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--e20", required=True)
    ap.add_argument("--em", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="", help="output dir override (scratch tests)")
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these stream ids (scratch tests)")
    a = ap.parse_args()
    od = Path(a.out) if a.out else OUTD / f"eval_{a.which}_{a.split}{a.tag}"
    if (od / "summary.json").exists():
        sys.exit(f"refuse to overwrite {od}/summary.json")
    od.mkdir(parents=True, exist_ok=True)
    e20 = json.load(open(Path(a.e20) / "e20.json"))
    L = float(e20["L_chosen"])
    Mk = load_map(Path(a.e20) / f"M_L{L:g}.npz")
    p = Params.load(Path(a.em) / "params.npz")
    refs = load_refs()
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    P = BiCodecPhi(a.device)
    units = []
    t0 = time.time()
    ss = [s for s in sids(a.which, a.split) if not a.only or s in a.only]
    for i, sid in enumerate(ss, 1):
        units += run_stream(sid, L, p, Mk, refs, P, a.device)
        print(f"[eval] {i}/{len(ss)} {sid} units {len(units)} elapsed {time.time() - t0:.0f}s", flush=True)
    np.savez(od / "units.npz", units=np.array(units, dtype=object))
    S = summarize(units)
    S.update({"which": a.which, "split": a.split, "L": L, "M": e20["L"][f"{L:g}"]["M_chosen"], "r": p.r,
              "e20": str(a.e20), "em": str(a.em), "T_EVAL": T_EVAL})
    (od / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("n_units", "n_units_60", "paired60", "floor", "D_c", "G2.2")}, indent=1))
    for m in METHODS:
        print(f"{m:10s}", " ".join(f"{S['curves'][m][str(t)][0]:.3f}" for t in T_EVAL), "| tok60", f"{S['tok'][m]['60']:.3f}")


if __name__ == "__main__":
    main()
