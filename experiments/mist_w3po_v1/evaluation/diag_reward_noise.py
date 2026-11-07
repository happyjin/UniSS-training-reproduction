"""CPU diagnostics (user request 2026-11-07 16:10 UTC), on existing AMI dev decodes only (no new decode):

(i) reward validity: per online dev window, the STAMP reward the RL uses (rl/stamp.credit, the same function as in
    training) and its parts, against the evaluation metric. Window references: the reference utterances
    (windows_validation_ref.jsonl, absolute times) whose overlap with the online window is >= half their length,
    moved to window time. Per window: R, RQ (speaker factor 1), mean unit score S, latency, spurious segments,
    duplicate segments (leave-one-out delta < 0), and the window's sequence cpBLEU (stamp.seq_cpbleu) as the per-window
    metric. Per meeting: mean R vs the meeting's tcpBLEU all / ov (score_text.json per-meeting stats). Paired against
    v8 ctx0: per-meeting dR vs d tcpBLEU. System level: rank of mean R vs rank of tcpBLEU all / ov.
(ii) evaluation noise: per-meeting tcpBLEU spread, the meeting-bootstrap SD of system differences, and a Monte Carlo
    estimate of P(a model equal to v8 passes the gate): paired per-meeting statistics of a near-identical pair
    (pilot step 20 vs v8 ctx0, KL 0.008; speech: the closest available pair), centred to zero true difference,
    meetings resampled, the gate rule applied with the better of v8 ctx0/ctx2 as reference per item.

    python -m experiments.mist_w3po_v1.evaluation.diag_reward_noise --out reports/mist_w3po_v1/diag_reward_noise.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation.compare import bleu_of

D = "eval_outputs/mist_w3po_v1/decode"
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
REF = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"
SYSTEMS = ["dev_v8_ctx0", "dev_v8_ctx2", "dev_pilot20_ctx0", "dev_pilot50_ctx0", "dev_s0all_ctx0", "dev_s0all_ctx2"]


def ref_utts():
    by = defaultdict(list)
    for r in map(json.loads, open(REF, encoding="utf-8")):
        for u in r["utts"]:
            a, b = r["start_s"] + u["begin_s"], r["start_s"] + u["end_s"]
            by[r["meeting"]].append({**u, "abs_a": a, "abs_b": b})
    for m in by:   # ref windows may overlap: keep one copy per record
        seen, out = set(), []
        for u in sorted(by[m], key=lambda u: u["abs_a"]):
            if u["record_id"] not in seen:
                seen.add(u["record_id"])
                out.append(u)
        by[m] = out
    return by


def window_utts(w, by):
    s0, s1 = w["start_s"], w["start_s"] + w["dur_ms"] / 1000
    out = []
    for u in by.get(w["meeting"], []):
        ov = min(s1, u["abs_b"]) - max(s0, u["abs_a"])
        if ov > 0 and ov >= 0.5 * (u["abs_b"] - u["abs_a"]):
            out.append({**u, "begin_s": max(0.0, u["abs_a"] - s0), "end_s": min(s1, u["abs_b"]) - s0})
    return out


def one(args):
    from experiments.mist_w3po_v1.rl import stamp as S
    w, utts, h = args
    units = S.units_of(utts)
    dur = w["dur_ms"] / 1000
    reads = [rd["t"] / 1000 for rd in h.get("read_log", [])] or [dur]
    segs = []
    for s in h.get("segments", []):
        t = s.get("write_ms", w["dur_ms"]) / 1000
        r = next((i for i, x in enumerate(reads) if x >= t - 1e-6), len(reads) - 1)
        segs.append({"spk": s["spk"], "a": s["start_s"], "b": s["end_s"], "text": s["text"], "r": r, "t": t})
    wait_reads = [i for i, rd in enumerate(h.get("read_log", [])[:-1]) if not rd["new"]]
    post = np.load(w["post"]["lowlat"]).astype(np.float32)
    c = S.credit(segs, units, dur, wait_reads, reads, post=post)
    pi = S.speaker_map(segs, units)
    asg = [(j, u) for j, u in enumerate(c.unit_of) if u >= 0]
    mis = sum(1 for j, u in asg if pi.get(segs[j]["spk"]) != units[u]["spk"])
    return {"asg_n": len(asg), "mis_n": mis, "id": w["id"], "meeting": w["meeting"], "n_units": len(units), "n_segs": len(segs), "R": c.R, "RQ": c.RQ,
            "S_mean": float(np.mean(c.S)) if len(units) else None, "lat": c.lat, "spurious": c.spurious,
            "dup": sum(1 for d in c.delta if d < 0), "cpbleu": S.seq_cpbleu(segs, units) if units else None}


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return None
    r = spearmanr(x[ok], y[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 else None


def meeting_bleu(score, reg):
    return {m: bleu_of(np.array(v[f"{reg}_stats"] or [0] * 10, dtype=float)) for m, v in score["per_meeting"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    by = ref_utts()
    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    U = {w["id"]: window_utts(w, by) for w in W}
    res = {"window_units": {"windows": len(W), "with_units": sum(1 for w in W if U[w["id"]])}}
    per = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for sysn in SYSTEMS:
            H = {h["id"]: h for h in map(json.loads, open(f"{D}/{sysn}/hyp.jsonl", encoding="utf-8"))}
            jobs = [(w, U[w["id"]], H.get(w["id"], {})) for w in W if U[w["id"]]]
            per[sysn] = list(ex.map(one, jobs, chunksize=16))
            print(sysn, len(per[sysn]), flush=True)
    scores = {s: json.load(open(f"{D}/{s}/score_text.json")) for s in SYSTEMS}
    # (i) per window
    win = {}
    for s, rows in per.items():
        R = [r["R"] for r in rows]
        cp = [r["cpbleu"] for r in rows]
        win[s] = {"mean_R": float(np.mean(R)), "mean_RQ": float(np.mean([r["RQ"] for r in rows])),
                  "mean_S": float(np.nanmean([r["S_mean"] for r in rows])), "mean_lat": float(np.mean([r["lat"] for r in rows])),
                  "spurious_per_win": float(np.mean([r["spurious"] for r in rows])),
                  "dup_per_win": float(np.mean([r["dup"] for r in rows])), "mean_cpbleu": float(np.mean(cp)),
                  "spearman_R_cpbleu": spearman(R, cp), "pearson_R_cpbleu": pearson(R, cp)}
    res["per_window"] = win
    pooled_R = [r["R"] for s in per for r in per[s]]
    pooled_cp = [r["cpbleu"] for s in per for r in per[s]]
    res["per_window_pooled"] = {"spearman": spearman(pooled_R, pooled_cp), "pearson": pearson(pooled_R, pooled_cp)}
    # paired per window vs v8 ctx0: dR vs d cpBLEU
    base = {r["id"]: r for r in per["dev_v8_ctx0"]}
    pw = {}
    for s in SYSTEMS[1:]:
        dR, dC = [], []
        for r in per[s]:
            b = base.get(r["id"])
            if b:
                dR.append(r["R"] - b["R"])
                dC.append(r["cpbleu"] - b["cpbleu"])
        ch = [(x, y) for x, y in zip(dR, dC) if abs(x) > 1e-9 or abs(y) > 1e-9]
        pw[s] = {"spearman_dR_dcp_changed": spearman(*zip(*ch)) if len(ch) > 2 else None, "n_changed": len(ch),
                 "sign_agree": float(np.mean([np.sign(x) == np.sign(y) for x, y in ch if abs(x) > 1e-9 and abs(y) > 1e-9]))
                 if ch else None}
    res["per_window_paired_vs_v8ctx0"] = pw
    # per meeting
    mt = {}
    for s in SYSTEMS:
        mR = defaultdict(list)
        for r in per[s]:
            mR[r["meeting"]].append(r["R"])
        mb_all, mb_ov = meeting_bleu(scores[s], "all"), meeting_bleu(scores[s], "ov")
        ms = sorted(set(mR) & set(mb_all))
        mt[s] = {"spearman_R_all": spearman([np.mean(mR[m]) for m in ms], [mb_all[m] for m in ms]),
                 "spearman_R_ov": spearman([np.mean(mR[m]) for m in ms], [mb_ov[m] for m in ms]),
                 "meanR": {m: float(np.mean(mR[m])) for m in ms}, "bleu_all": {m: mb_all[m] for m in ms},
                 "bleu_ov": {m: mb_ov[m] for m in ms}}
    res["per_meeting"] = {s: {k: v for k, v in d.items() if k.startswith("spearman")} for s, d in mt.items()}
    pm = {}
    for s in SYSTEMS[1:]:
        ms = sorted(set(mt[s]["meanR"]) & set(mt["dev_v8_ctx0"]["meanR"]))
        dR = [mt[s]["meanR"][m] - mt["dev_v8_ctx0"]["meanR"][m] for m in ms]
        dA = [mt[s]["bleu_all"][m] - mt["dev_v8_ctx0"]["bleu_all"][m] for m in ms]
        dO = [mt[s]["bleu_ov"][m] - mt["dev_v8_ctx0"]["bleu_ov"][m] for m in ms]
        pm[s] = {"spearman_dR_dall": spearman(dR, dA), "spearman_dR_dov": spearman(dR, dO)}
    res["per_meeting_paired_vs_v8ctx0"] = pm
    # system level
    sysl = [{"system": s, "mean_R": win[s]["mean_R"], "tcpBLEU_all": scores[s]["tcpBLEU_all"], "tcpBLEU_ov": scores[s]["tcpBLEU_ov"],
             "dup_rate": scores[s]["duplicate_rate"]} for s in SYSTEMS]
    res["system_level"] = {"rows": sysl,
                           "spearman_R_all": spearman([x["mean_R"] for x in sysl], [x["tcpBLEU_all"] for x in sysl]),
                           "spearman_R_ov": spearman([x["mean_R"] for x in sysl], [x["tcpBLEU_ov"] for x in sysl])}
    # (ii) noise
    rng = np.random.default_rng(0)
    noise = {}
    for s in SYSTEMS:
        mb = meeting_bleu(scores[s], "all")
        noise.setdefault("per_meeting_all_sd", {})[s] = float(np.std(list(mb.values())))

    def stats(sc, reg):
        return {m: np.array(v[f"{reg}_stats"] or [0] * 10, dtype=float) for m, v in sc["per_meeting"].items()}

    def boot_sd(A, B):
        ms = sorted(set(A) & set(B))
        idx = rng.integers(0, len(ms), size=(2000, len(ms)))
        Am = np.array([A[m] for m in ms])
        Bm = np.array([B[m] for m in ms])
        d = [bleu_of(Bm[i].sum(0)) - bleu_of(Am[i].sum(0)) for i in idx]
        return float(np.std(d)), float(bleu_of(Bm.sum(0)) - bleu_of(Am.sum(0)))
    pairs = {}
    for b, c in (("dev_v8_ctx0", "dev_pilot20_ctx0"), ("dev_v8_ctx0", "dev_v8_ctx2"), ("dev_v8_ctx0", "dev_pilot50_ctx0")):
        for reg in ("all", "ov"):
            sd, d = boot_sd(stats(scores[b], reg), stats(scores[c], reg))
            pairs[f"{c}-{b}:{reg}"] = {"diff": d, "boot_sd": sd}
    sp = {}
    for s in ("dev_v8_ctx0", "dev_v8_ctx2", "dev_pilot50_ctx0", "dev_s0all_ctx0", "dev_s0all_ctx2"):
        f = Path(f"{D}/{s}/score_asr.json")
        if f.exists():
            sp[s] = json.load(open(f))
    if "dev_v8_ctx0" in sp and "dev_v8_ctx2" in sp:
        sd, d = boot_sd(stats(sp["dev_v8_ctx0"], "all"), stats(sp["dev_v8_ctx2"], "all"))
        pairs["speech dev_v8_ctx2-dev_v8_ctx0:all"] = {"diff": d, "boot_sd": sd}
    noise["pairs"] = pairs
    # Monte Carlo: an equal-to-v8 model = v8 ctx0 per-meeting stats + the centred per-meeting noise of a near-identical
    # pair (pilot20 - v8 ctx0 for text; v8 ctx2 - v8 ctx0 for speech, the only speech pair without RL), meetings
    # resampled; reference per item = better of v8 ctx0 / ctx2 point scores (as the gate)
    A_all, A_ov = stats(scores["dev_v8_ctx0"], "all"), stats(scores["dev_v8_ctx0"], "ov")
    N_all, N_ov = stats(scores["dev_pilot20_ctx0"], "all"), stats(scores["dev_pilot20_ctx0"], "ov")
    ms = sorted(set(A_all) & set(N_all))
    refs = {"all": max(scores["dev_v8_ctx0"]["tcpBLEU_all"], scores["dev_v8_ctx2"]["tcpBLEU_all"]),
            "ov": max(scores["dev_v8_ctx0"]["tcpBLEU_ov"], scores["dev_v8_ctx2"]["tcpBLEU_ov"])}
    have_sp = "dev_v8_ctx0" in sp and "dev_v8_ctx2" in sp
    if have_sp:
        SA, SN = stats(sp["dev_v8_ctx0"], "all"), stats(sp["dev_v8_ctx2"], "all")
        refs["speech"] = max(sp["dev_v8_ctx0"]["tcpBLEU_all"], sp["dev_v8_ctx2"]["tcpBLEU_all"])
    # noise draw: resample meetings of the pair difference d_m and add it (sign-symmetrised) to v8 ctx0 per meeting
    passes = {"all": 0, "ov": 0, "speech": 0, "joint": 0}
    T = 4000
    for _ in range(T):
        sel = rng.integers(0, len(ms), size=len(ms))
        sign = rng.choice([-1.0, 1.0])
        ok = {}
        for reg, A, N in (("all", A_all, N_all), ("ov", A_ov, N_ov)):
            base_m = np.array([A[m] for m in ms])
            noise_m = np.array([N[m] - A[m] for m in ms])
            noise_m = noise_m - noise_m.mean(0)   # centred: zero systematic difference
            cand = base_m + sign * noise_m[sel] * 1.0
            cand = np.clip(cand, 0, None)
            pt = bleu_of(cand.sum(0))
            ok[reg] = pt >= refs[reg] - 0.2
        if have_sp:
            msp = sorted(set(SA) & set(SN))
            sel2 = rng.integers(0, len(msp), size=len(msp))
            base_m = np.array([SA[m] for m in msp])
            noise_m = np.array([SN[m] - SA[m] for m in msp])
            noise_m = noise_m - noise_m.mean(0)
            pt = bleu_of(np.clip(base_m + sign * noise_m[sel2], 0, None).sum(0))
            ok["speech"] = pt >= refs["speech"] - 0.2
        for k, v in ok.items():
            passes[k] += v
        passes["joint"] += all(ok.values())
    noise["mc_equal_model_pass"] = {k: v / T for k, v in passes.items()}
    noise["mc_note"] = ("estimate: per-meeting noise taken from pilot20 vs v8 ctx0 (text) and v8 ctx2 vs v8 ctx0 (speech); "
                        "point criterion only (P(diff<=0)<0.95 is almost always met at zero true difference); refs = better "
                        "of v8 ctx0/ctx2 per item; to be replaced by the measured decode noise of dev_v8_ctx0_rep8")
    res["noise"] = noise
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "per_meeting"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
