"""T3's pre-registered approximation checks and the size / CPU study (reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md
§T3, thresholds there; CPU only, dev outputs read-only, nothing trained or tuned on dev).

  A1     A' + PL-EM dev blocks (seed 1 talker results): the pi-weighted expected duration sum_v pi(v | bin(b')) D(B, v)
         (b' = the queue left at the block's online availability, from the actual durations) vs the actual dur_ms:
         R^2 >= 0.6 and |duration-weighted relative bias| = |sum(pred - actual) / sum(actual)| <= 10 %.
  A2     train: mean over blocks of KL(q(v | b_n) || pi(v | bin(b_n))) <= 0.15 nats (the fit's saved block table).
  A1+A2+A4  the A' + PL-EM dev run with every block's duration replaced by the model's (expected duration at the model's
         own backlog, queue end on the absolute 80 ms grid), speech_latency_lit unchanged: |per-stream speech LAAL (all)
         - actual| <= 0.30 s and the unit-level Pearson r >= 0.8.
  A3     dev5 (B-2 targets, T0's label path vs S_prop's): the per-word speech-latency proxy difference
         Delta = L(T0) - L(S_prop) with availability t_r (training) vs the online clamp: |Delta_t - Delta_clamp| <=
         0.25 |Delta_clamp|.
  A4     A' + PL-EM dev streams, actual durations: the grid queue vs the exact queue, mean |per-word proxy difference|
         <= 0.05 s.
  A5     (--sizes) share of the 3000-step plan's windows whose augmented chain needed a coarser grid <= 2 %.
  A6     report only: share of (window, read, speaker) blocks with >= 2 pieces on the T0 label path.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import joint_speech_crf as JS

DEV_RUN = "eval_outputs/mist_w3po_v1/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06__plem_c1b1s2"
DEV_TALKER = "eval_outputs/mist_w3po_v1/speech/dev_crf_v2p_s3000_finalpost_s8_c2g06__plem_c1b1s2/talker"
DEV_WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
B2_TARGETS = "eval_outputs/mist_w3po_v1/crf/b2_earliest_write_dev5/cpu/targets.jsonl"
T0_PIECES = "eval_outputs/mist_w3po_v1/crf/t0_agree_dev5/analysis/pieces.jsonl"
TH = {"A1_r2": 0.6, "A1_bias": 0.10, "A2_kl": 0.15, "A124_laal": 0.30, "A124_r": 0.8, "A3_rel": 0.25, "A4_s": 0.05,
      "A5_share": 0.02}


def online(w: dict, t: float) -> float:
    c = max(t, w["open_detect_ms"])
    if t >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return float(c)


def exp_dur(tm: JS.TalkerModel, text: str, bp: float) -> float:
    pi = np.exp(tm.logpi_of(np.asarray([bp]))[0])
    return float((pi * tm.dur(tm.n(text))).sum())


def grid_end(a: float, bp: float, d: float, delta: int = JS.DELTA_MS) -> float:
    return float(np.rint((a + bp + d) / delta) * delta)


def dev_streams(path: str = DEV_TALKER, win: str = DEV_WIN) -> list:
    W = {w["id"]: w for w in map(json.loads, open(win, encoding="utf-8"))}
    out = []
    for l in open(Path(path) / "results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("error") or not r["blocks"]:
            continue
        out.append((r, W[r["id"].split("__S")[0]]))
    return out


def check_a1(tm: JS.TalkerModel, streams: list) -> dict:
    pred, act = [], []
    for r, w in streams:
        end = None
        for b in r["blocks"]:
            a = online(w, b["write_ms"])
            bp = 0.0 if end is None else max(0.0, end - a)
            pred.append(exp_dur(tm, b["text"], bp))
            act.append(float(b["dur_ms"]))
            end = max(a, end if end is not None else a) + b["dur_ms"]
    p, y = np.asarray(pred), np.asarray(act)
    r2 = 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    bias = float((p - y).sum() / y.sum())
    return {"blocks": len(y), "r2": round(r2, 4), "rel_bias": round(bias, 4), "mae_ms": round(float(np.abs(p - y).mean()), 1),
            "mean_actual_ms": round(float(y.mean()), 1), "pass": bool(r2 >= TH["A1_r2"] and abs(bias) <= TH["A1_bias"])}


def check_a2(fit_path: str) -> dict:
    tab = np.load(fit_path + ".blocks.npy")
    nV = (tab.shape[1] - 1) // 2
    qv, bins = tab[:, :nV], tab[:, -1].astype(int)
    pi = np.asarray(json.loads(Path(fit_path).read_text())["pi"])[bins]
    q = np.clip(qv, 1e-300, None)
    kl = (qv * (np.log(q) - np.log(pi))).sum(1)
    return {"blocks": len(kl), "kl_mean": round(float(kl.mean()), 4), "kl_p90": round(float(np.percentile(kl, 90)), 4),
            "pass": bool(kl.mean() <= TH["A2_kl"])}


def _laal_runs(run: str, talkers: list, win: str) -> list:
    """speech_latency_lit.main on each talker dir (outputs to scratch), with the per-unit speech LAALs (online, per
    stream) captured in call order."""
    from experiments.mist_w3po_v1.evaluation import speech_latency_lit as SL

    orig_swd, orig_laal = SL.speech_word_delays, SL.laal
    outs = []
    for tdir, out in talkers:
        cap = {"sd": [], "vals": []}

        def swd(pw, a, k0_="on_start", k1_="on_end"):
            d = orig_swd(pw, a, k0_, k1_)
            if k0_ == "on_start":
                cap["sd"].append(d)
            return d

        def laal(d, u):
            v = orig_laal(d, u)
            if any(d is x for x in cap["sd"][-1:]):
                cap["vals"].append(v)
            return v
        SL.speech_word_delays, SL.laal = swd, laal
        argv = sys.argv
        sys.argv = ["speech_latency_lit", "--run", run, "--talker", tdir, "--out", out, "--win", win]
        try:
            SL.main()
        finally:
            sys.argv = argv
            SL.speech_word_delays, SL.laal = orig_swd, orig_laal
        res = json.loads(Path(out).read_text())
        outs.append({"mean": res["unit"]["speech_laal_online"]["all"]["mean"], "units": cap["vals"]})
    return outs


def check_a124(tm: JS.TalkerModel, scratch: Path) -> dict:
    """the dev run's block durations replaced by the model's (grid queue), then speech_latency_lit."""
    W = {w["id"]: w for w in map(json.loads, open(DEV_WIN, encoding="utf-8"))}
    tdir = scratch / "a124_talker"
    tdir.mkdir(parents=True)
    with open(tdir / "results.jsonl", "x", encoding="utf-8") as fh:
        for l in open(Path(DEV_TALKER) / "results.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error") and r["blocks"]:
                w = W[r["id"].split("__S")[0]]
                end = None
                for b in r["blocks"]:
                    a = online(w, b["write_ms"])
                    bp = 0.0 if end is None else max(0.0, end - a)
                    d = exp_dur(tm, b["text"], bp)
                    e2 = grid_end(a, bp, d)
                    b["dur_ms"] = max(0.0, e2 - (a + bp))
                    end = e2
            fh.write(json.dumps(r) + "\n")
    runs = _laal_runs(DEV_RUN, [(DEV_TALKER, str(scratch / "a124_actual.json")),
                                (str(tdir), str(scratch / "a124_model.json"))], DEV_WIN)
    x, y = np.asarray(runs[0]["units"]), np.asarray(runs[1]["units"])
    assert len(x) == len(y)
    r = float(np.corrcoef(x, y)[0, 1])
    d = runs[1]["mean"] - runs[0]["mean"]
    return {"actual_speech_laal": runs[0]["mean"], "model_speech_laal": runs[1]["mean"], "diff_s": round(d, 3),
            "units": len(x), "pearson_r": round(r, 4), "unit_mean_actual": round(float(x.mean()), 3),
            "unit_mean_model": round(float(y.mean()), 3),
            "pass": bool(abs(d) <= TH["A124_laal"] and r >= TH["A124_r"])}


def _proxy(rec: dict, reads: list[int], tm: JS.TalkerModel, clamp: bool) -> tuple[float, int]:
    """sum over words of (spoken time - piece source end) / 1000 for one dev5 window under a write-read schedule."""
    P = rec["pieces"]
    t = rec["t_ms"]
    tot, n = 0.0, 0
    for k in sorted({p["slot"] for p in P}):
        end = None
        for r in sorted({reads[j] for j, p in enumerate(P) if p["slot"] == k}):
            cs = [j for j, p in enumerate(P) if p["slot"] == k and reads[j] == r]
            a = online(rec, t[r]) if clamp else float(t[r])
            bp = 0.0 if end is None else max(0.0, end - a)
            d = exp_dur(tm, " ".join(P[j]["text"] for j in cs), bp)
            Wb = sum(P[j]["n_words"] for j in cs)
            tot += sum(P[j]["n_words"] * (a - P[j]["src_end_ms"]) for j in cs) / 1000.0
            tot += Wb * bp / 1000.0 + (Wb + 1) / 2.0 * d / 1000.0
            n += Wb
            end = grid_end(a, bp, d)
    return tot, n


def check_a3(tm: JS.TalkerModel) -> dict:
    recs = {r["id"]: r for r in map(json.loads, open(B2_TARGETS, encoding="utf-8"))}
    e = defaultdict(dict)
    u = defaultdict(dict)
    for p in map(json.loads, open(T0_PIECES, encoding="utf-8")):
        e[p["id"]][p["j"]] = p["e"]
        u[p["id"]][p["j"]] = p["u"]
    L = {(s, c): [0.0, 0] for s in ("T0", "S_prop") for c in (False, True)}
    for i, rec in recs.items():
        J = len(rec["pieces"])
        sch = {"T0": [e[i][j] for j in range(J)], "S_prop": [u[i][j] for j in range(J)]}
        for s, rd in sch.items():
            for c in (False, True):
                tot, n = _proxy(rec, rd, tm, c)
                L[(s, c)][0] += tot
                L[(s, c)][1] += n
    per = {f"{s}_{'clamp' if c else 't_r'}": round(v[0] / v[1], 4) for (s, c), v in L.items()}
    d_t = per["T0_t_r"] - per["S_prop_t_r"]
    d_c = per["T0_clamp"] - per["S_prop_clamp"]
    return {**per, "delta_t_r": round(d_t, 4), "delta_clamp": round(d_c, 4),
            "rel_change": round(abs(d_t - d_c) / max(abs(d_c), 1e-9), 4),
            "pass": bool(abs(d_t - d_c) <= TH["A3_rel"] * abs(d_c))}


def check_a4(streams: list) -> dict:
    diff, words = 0.0, 0
    for r, w in streams:
        ex = gr = None
        for b in r["blocks"]:
            a = online(w, b["write_ms"])
            Wb = max(1, len(b["text"].split()))
            d = float(b["dur_ms"])
            bx = 0.0 if ex is None else max(0.0, ex - a)
            bg = 0.0 if gr is None else max(0.0, gr - a)
            diff += abs(Wb * (bg - bx)) / 1000.0
            words += Wb
            ex = a + bx + d
            gr = grid_end(a, bg, d)
    v = diff / max(words, 1)
    return {"words": words, "mean_abs_diff_s_per_word": round(v, 4), "pass": bool(v <= TH["A4_s"])}


def check_a6(sched: str) -> dict:
    n = m = 0
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            rec = json.loads(l)
            if rec.get("label_src") != "t0":
                continue
            cnt = defaultdict(int)
            for c in rec["chunks"]:
                cnt[(c["e_read"], c["slot"])] += 1
            n += len(cnt)
            m += sum(1 for v in cnt.values() if v >= 2)
    return {"blocks": n, "share_multi_piece": round(m / max(n, 1), 4)}


def run_checks(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    scratch = Path(str(out) + ".work")
    scratch.mkdir(parents=True)
    tm = JS.TalkerModel.load(args.talker_model)
    t0 = time.time()
    streams = dev_streams()
    res = {"thresholds": TH, "talker_model": args.talker_model, "feature": tm.feature}
    res["A1"] = check_a1(tm, streams)
    res["A2"] = check_a2(args.talker_model)
    res["A4"] = check_a4(streams)
    res["A3"] = check_a3(tm)
    res["A1_A2_A4"] = check_a124(tm, scratch)
    if Path(args.sched).exists():
        res["A6"] = check_a6(args.sched)
    res["cpu_s"] = round(time.time() - t0, 1)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ sizes
def _size_one(job) -> dict:
    rec, tm_d = job
    from experiments.mist_w3po_v1.crf import latent_write_crf as LW
    from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

    class W_:
        def __init__(self, rec):
            self.idx, self.rec = 0, rec
    tm = JS.TalkerModel(tm_d)
    t0 = time.perf_counter()
    pl = LW.plan_window(rec, LW.lattice(rec, "t2"))
    for q in pl["reqs"]:
        q["scored"] = True
    R = pl["lat"]["R"]
    res = {(0, k): V.dummy_scores(q["r"], q["r"] == R - 1, q["hist_ids"], q["forced_ids"], q["proto"])
           for k, q in enumerate(pl["reqs"])}
    t1 = time.perf_counter()
    parts = JS.window_parts(W_(rec), pl, res, tm, JS.TAU)
    t2 = time.perf_counter()
    JS.forward_backward(parts["aug"], 0.5)
    t3 = time.perf_counter()
    JS.posterior_from_parts(W_(rec), pl, parts, JS.TAU, 0.5)
    t4 = time.perf_counter()
    a = parts["aug"]
    return {"id": rec["id"], "R": R, "states_max": int(max(a["states"])), "states_sum": int(sum(a["states"])),
            "edges": a["edges"], "coarsened": bool(a["coarsened"]), "delta": a["delta"],
            "thinker_states_max": max(len(s) for s in pl["lat"]["states"]),
            "t_plan": t1 - t0, "t_parts": t2 - t1, "t_fb": t3 - t2, "t_post": t4 - t3}


def run_sizes(args) -> None:
    from multiprocessing import Pool

    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    tm_d = json.loads(Path(args.talker_model).read_text())
    recs = []
    for f in sorted(Path(args.sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    plan = T.plan_steps([r["n_read"] for r in recs], 20261109, args.steps, 64)
    used = sorted({i for s in plan for i in s})
    with Pool(args.procs) as pool:
        rows = pool.map(_size_one, [(recs[i], tm_d) for i in used], chunksize=16)
    by = {i: r for i, r in zip(used, rows)}
    n_evals = 56     # lam = 0, ~5 doublings, 50 bisections (T2's solve; the trainer logs the actual count)
    step_cpu = [sum(by[i]["t_parts"] + n_evals * by[i]["t_fb"] + by[i]["t_post"] for i in s) for s in plan]

    def st(v):
        v = np.asarray(v, np.float64)
        return {"mean": round(float(v.mean()), 4), "p50": round(float(np.median(v)), 4),
                "p95": round(float(np.percentile(v, 95)), 4), "p99": round(float(np.percentile(v, 99)), 4),
                "max": round(float(v.max()), 4)}
    co = sum(r["coarsened"] for r in rows)
    res = {"windows": len(rows), "coarsened": co, "coarsened_share": round(co / len(rows), 4),
           "A5_pass": bool(co / len(rows) <= TH["A5_share"]),
           "delta_hist": {str(d): sum(1 for r in rows if r["delta"] == d) for d in sorted({r["delta"] for r in rows})},
           "aug_states_max_per_window": st([r["states_max"] for r in rows]),
           "thinker_states_max_per_window": st([r["thinker_states_max"] for r in rows]),
           "aug_edges_per_window": st([r["edges"] for r in rows]),
           "t_parts_per_window_s": st([r["t_parts"] for r in rows]), "t_fb_per_window_s": st([r["t_fb"] for r in rows]),
           "step_cpu_s_est (parts + 56 FB + posterior, one process)": st(step_cpu),
           "steps": len(plan), "cpu_s": round(time.time() - t0, 1)}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(str(out) + ".windows.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(res, indent=1), flush=True)
