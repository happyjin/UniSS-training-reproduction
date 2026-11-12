"""PMC CPU precheck (reports/mist_w3po_v1/thinker_pmc_prereg_v1.zh-CN.md §2): P1 / P2 / P5 by crf/test_pmc.py (subprocess),
P3 (the e-path budget below VE1's trained posterior; the window opens earlier), P4 (sizes and CPU time per step on the
VE1 plan, pmc vs v2p), and the DART manifest (mode pmc) checked against the trainer's plan exactly as train_crf_sft does.

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.pmc_precheck \
        --out eval_outputs/mist_w3po_v1/crf/pmc_precheck_v1.json [--procs 48]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run: transformers' model modules probe the accelerator at import
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

C = "eval_outputs/mist_w3po_v1/crf"
SCHED = f"{C}/sched_t5_merge3_w10_v1"
VE1_METRICS = f"{C}/train/ve1_20261111T214454Z/metrics.jsonl"
DART = f"{C}/pmc_dart_merge3_w10_v1"
VE1_DART = f"{C}/ve1_dart_merge3_w10_v1"
SEED, STEPS, ROWS = 20261109, 3000, 64
VE1_STEP_S = {"t_pass1": 0.78, "t_pass2": 3.01, "t_step": 3.81}   # VE1 step 3000 (metrics.jsonl); replaced by the mean


def _stat(v) -> dict:
    v = np.asarray(v, np.float64)
    return {"mean": round(float(v.mean()), 3), "p50": float(np.median(v)), "p95": round(float(np.percentile(v, 95)), 3),
            "max": float(v.max())}


def _sizes(rec: dict) -> dict:
    out = {"R": rec["n_read"], "n_words": sum(LW.n_words(c) for c in rec["chunks"])}
    e = LW.bounds(rec, "v2p")[0]
    out["lat_e"] = LW.e_path_lat(rec, e)
    for mode in ("v2p", "pmc"):
        lat = LW.lattice(rec, mode)
        pl = LW.plan_window(rec, lat, skip_forced=False, mode=mode)
        forced = {rd["r"] for rd in pl["reads"] if rd["forced"]}
        sc = [q for q in pl["reqs"] if q["r"] not in forced]
        out[mode] = {"p1": len(sc), "p2": len(pl["reqs"]),
                     "tok1": sum(len(q["hist_ids"]) + len(q["forced_ids"]) for q in sc),
                     "tok2": sum(len(q["hist_ids"]) + len(q["forced_ids"]) for q in pl["reqs"]),
                     "max_states": max(len(s) for s in lat["states"]),
                     "earliest": sum(float(Lr[0, 0]) for Lr in pl["L"])}   # state 0 path: {l_i <= r}
        if mode == "v2p":   # the latest path of v2p: every chunk at e + 1 (state -1 of every read)
            out[mode]["latest"] = LW.e_path_lat(rec, LW.bounds(rec, "v2p")[1])
    return out


def run(a) -> None:
    from multiprocessing import Pool

    from experiments.mist_w3po_v1.crf import dart_aug as DA
    from experiments.mist_w3po_v1.crf import latent_write_train as LWT
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    rep: dict = {"prereg": "reports/mist_w3po_v1/thinker_pmc_prereg_v1.zh-CN.md", "sched": SCHED}
    # ---------------- P1 / P2 / P5: the unit tests
    env = dict(os.environ, PYTHONPATH=".:third_party/Megatron-LM", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "experiments.mist_w3po_v1.crf.test_pmc"], env=env,
                       capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith(("  P", "PASS", "ALL"))]
    rep["P1_P2_P5_tests"] = {"rc": p.returncode, "lines": lines, "pass": p.returncode == 0 and any(
        l.startswith("ALL 3 PASSED") for l in lines)}
    if p.returncode:
        rep["P1_P2_P5_tests"]["stderr_tail"] = p.stderr[-3000:]
    # P2 (regression): the existing tests, run as __main__ with the CPU accelerator patch (as rc5_precheck does)
    boot = ("import torch, runpy\nif not torch.cuda.is_available():\n"
            "    torch.accelerator.current_accelerator = lambda *a, **k: None\n"
            "runpy.run_module('experiments.mist_w3po_v1.crf.{}', run_name='__main__')")
    reg = {}
    for t in ("test_latent_write_crf", "test_train_crf_sft", "test_dart_aug"):
        q = subprocess.run([sys.executable, "-W", "ignore", "-c", boot.format(t)], env=env, capture_output=True,
                           text=True, timeout=3600)
        reg[t] = {"rc": q.returncode, "tail": [l for l in q.stdout.splitlines() if l.startswith("ALL")][-1:]}
    rep["P2_regression"] = {"tests": reg, "pass": all(v["rc"] == 0 for v in reg.values())}
    # ---------------- P3 (a): VE1's trained posterior vs the e-path budget (word-weighted, steps 2501-3000)
    m = [json.loads(l) for l in open(VE1_METRICS, encoding="utf-8")]
    m = [x for x in m if "Eq_lat_T2" in x and 2501 <= x["step"] <= 3000]
    sw = sum(x["sum_words"] for x in m)
    eq = sum(x["Eq_lat_T2"] * x["sum_words"] for x in m) / sw
    lab = sum(x["lat_label"] * x["sum_words"] for x in m) / sw
    VE1_STEP_S.update({k: float(np.mean([x[k] for x in m])) for k in ("t_pass1", "t_pass2", "t_step")})
    rep["P3a"] = {"steps": len(m), "ve1_Eq_lat_per_word": round(eq, 4), "e_path_budget_per_word": round(lab, 4),
                  "gap": round(eq - lab, 4), "pass": eq - lab >= 0.1}
    # ---------------- plan + sizes (P3 (b), P4)
    wins = T.load_windows(SCHED, True)
    plan = T.plan_steps([w.R for w in wins], SEED, STEPS, ROWS)
    used = sorted({i for s in plan for i in s})
    with Pool(a.procs) as pool:
        res = pool.map(_sizes, [wins[i].rec for i in used], chunksize=32)
    by = dict(zip(used, res))
    first = sorted({i for s in plan[:200] for i in s})
    nw = sum(by[i]["n_words"] for i in first)
    lat3 = {"pmc_earliest": sum(by[i]["pmc"]["earliest"] for i in first) / nw,
            "e_path": sum(by[i]["lat_e"] for i in first) / nw,
            "v2p_earliest": sum(by[i]["v2p"]["earliest"] for i in first) / nw,
            "v2p_latest": sum(by[i]["v2p"]["latest"] for i in first) / nw}
    rep["P3b"] = {"windows": len(first), "per_word_s": {k: round(v, 4) for k, v in lat3.items()},
                  "pass": lat3["pmc_earliest"] < lat3["e_path"] and abs(lat3["v2p_earliest"] - lat3["e_path"]) < 1e-9}
    per = {mode: {k: [sum(by[i][mode][k] for i in s) for s in plan] for k in ("p1", "p2", "tok1", "tok2")}
           for mode in ("v2p", "pmc")}
    p4 = {"windows_used": len(used),
          "per_step": {mode: {k: _stat(v) for k, v in per[mode].items()} for mode in per},
          "per_window_max_states": {mode: _stat([by[i][mode]["max_states"] for i in used]) for mode in per}}
    r_tok1 = float(np.mean(per["pmc"]["tok1"]) / np.mean(per["v2p"]["tok1"]))
    r_tok2 = float(np.mean(per["pmc"]["tok2"]) / np.mean(per["v2p"]["tok2"]))
    p95r = float(np.percentile(per["pmc"]["tok2"], 95) / np.percentile(per["v2p"]["tok2"], 95))
    # CPU time per step on dummy scores: plan, parts, lam*, q_lam, q^pol, PMC coefficients (30 seeded steps)
    rng = np.random.default_rng(SEED)
    tm, lams, bind, early_q, early_p, nch = [], [], 0, 0.0, 0.0, 0
    for si in sorted(rng.choice(len(plan), size=30, replace=False).tolist()):
        sw_ = [wins[i] for i in plan[si]]
        t1 = time.time()
        pls = {w.idx: LWT.plan(w, "pmc") for w in sw_}
        res_ = {}
        for w in sw_:
            for k, (v, q) in enumerate(zip(V.dummy_res(pls[w.idx]), pls[w.idx]["reqs"])):
                if q["scored"]:
                    res_[(w.idx, k)] = v
        t_sc = time.time()
        parts = {w.idx: LW.window_parts(w, pls[w.idx], res_, LW.TAU) for w in sw_}
        lam, info = LW.solve_lambda(list(parts.values()))
        posts = {w.idx: LW.posterior_from_parts(w, pls[w.idx], parts[w.idx], LW.TAU, lam) for w in sw_}
        pols = {w.idx: LW.posterior_from_parts(w, pls[w.idx], LW.pol_parts(w, pls[w.idx], res_, parts[w.idx]),
                                               LW.TAU, lam) for w in sw_}
        D = float(sum(w.h.sum() + w.R for w in sw_ if w.feasible))   # step_totals' D (its |A| needs set_reach)
        from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T

        rows = V2T.pass2_rows(sw_, pls, None, D)
        LWT.set_coefs_pmc(rows, pls, posts, pols, D)
        t_end = time.time()
        tm.append(round((t_sc - t1) + (t_end - t_sc), 3))
        lams.append(lam)
        bind += bool(info["binding"])
        early_q += sum(LW.exp_early(pls[w.idx], posts[w.idx]["fb"]) for w in sw_)
        early_p += sum(LW.exp_early(pls[w.idx], pols[w.idx]["fb"]) for w in sw_)
        nch += sum(len(w.rec["chunks"]) for w in sw_)
    est = VE1_STEP_S["t_pass1"] * r_tok1 + (VE1_STEP_S["t_pass2"]) * r_tok2 + \
        max(0.0, VE1_STEP_S["t_step"] - VE1_STEP_S["t_pass1"] - VE1_STEP_S["t_pass2"])
    p4.update({"tok1_ratio_mean": round(r_tok1, 3), "tok2_ratio_mean": round(r_tok2, 3), "tok2_ratio_p95": round(p95r, 3),
               "cpu_s_per_step_dummy": _stat(tm), "dummy_lam": _stat(lams), "dummy_binding_steps": bind,
               "dummy_exp_early_share_q": round(early_q / max(nch, 1), 4),
               "dummy_exp_early_share_pol": round(early_p / max(nch, 1), 4),
               "ve1_step_s_mean_2501_3000": {k: round(v, 3) for k, v in VE1_STEP_S.items()},
               "est_step_s": round(est, 2), "est_hours_3000": round(est * STEPS / 3600.0, 2),
               "pass": p95r <= 2.5 and float(np.percentile(tm, 95)) <= 2.0})
    rep["P4"] = p4
    # ---------------- P5 (trainer args + manifest): the launch's EXTRA parses; the DART manifest of mode pmc vs the
    # trainer's plan (load_manifest, as train_crf_sft does) and VE1's manifest (h is v2p's: same draws, same rows)
    ta = T.build_args(["--objective", "latent_write", "--lw-mode", "pmc", "--sched", SCHED, "--dart", DART,
                       "--out", "/nonexistent/pmc_args_check"])
    rep["P5_args"] = {"objective": ta.objective, "lw_mode": ta.lw_mode, "seed": ta.seed, "steps": ta.steps,
                      "rows_per_step": ta.rows_per_step,
                      "pass": (ta.objective, ta.lw_mode, ta.seed, ta.steps, ta.rows_per_step) ==
                      ("latent_write", "pmc", SEED, STEPS, ROWS)}
    if Path(DART).exists():
        dm = DA.load_manifest(DART, wins, plan, SCHED, SEED, STEPS, ROWS)
        a_ = [json.loads(l) for l in open(Path(DART) / "dart.jsonl", encoding="utf-8")]
        b_ = [json.loads(l) for l in open(Path(VE1_DART) / "dart.jsonl", encoding="utf-8")]
        same = sorted((x["step"], x["idx"], x["delay"], x["rows"]) for x in a_) == \
            sorted((x["step"], x["idx"], x["delay"], x["rows"]) for x in b_)
        rep["P5_manifest"] = {"mode": DA.manifest_mode(DART), "entries": sum(len(v) for v in dm.values()),
                              "same_draws_and_rows_as_ve1": same,
                              "pass": DA.manifest_mode(DART) == "pmc" and same}
    else:
        rep["P5_manifest"] = {"pass": False, "missing": DART}
    rep["pass"] = all(rep[k]["pass"] for k in ("P1_P2_P5_tests", "P2_regression", "P3a", "P3b", "P4", "P5_args",
                                               "P5_manifest"))
    rep["cpu_s"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
