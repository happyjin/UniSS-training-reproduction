"""PMC-noC CPU precheck (reports/mist_w3po_v1/thinker_pmc_noc_prereg_v1.zh-CN.md §2):
  tests     crf/test_pmc_noc.py (N1-N3) and crf/test_pmc.py (P1 / P2 / P5 with the new code), subprocesses
  regress   test_latent_write_crf, test_train_crf_sft, test_dart_aug (run as __main__ with the CPU accelerator patch)
  plan      30 seeded steps of the PMC-1 plan (sched_t5_merge3_w10_v1, seed 20261109, 3000 steps, 64 rows), real windows,
            dummy scores: every window's lattice and plan of PMC-noC are PMC-1's (the same --lw-mode pmc call, compared
            field by field against LW.lattice / LW.plan_window); the w_pol = 0 coefficients == V2T.set_coefs (VE1's M
            step) on the same pmc posterior bit for bit, and differ from PMC-1's (w = 1); lam* and the posterior do not
            depend on w
  args      the launch's EXTRA parses (latent_write / pmc / weight 0 / seed 20261109 / 3000 steps / 64 rows); the DART
            manifest crf/pmc_dart_merge3_w10_v1 (mode pmc) passes load_manifest against the plan, as train_crf_sft does

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.pmc_noc_precheck \
        --out eval_outputs/mist_w3po_v1/crf/pmc_noc_precheck_v1.json
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
DART = f"{C}/pmc_dart_merge3_w10_v1"
SEED, STEPS, ROWS = 20261109, 3000, 64


def _tests(env: dict) -> dict:
    out = {}
    for t, want in (("test_pmc_noc", "ALL 3 PASSED"), ("test_pmc", "ALL 3 PASSED")):
        p = subprocess.run([sys.executable, "-W", "ignore", "-m", f"experiments.mist_w3po_v1.crf.{t}"], env=env,
                           capture_output=True, text=True, timeout=3600)
        lines = [l for l in p.stdout.splitlines() if l.startswith(("  N", "  P", "PASS", "ALL"))]
        out[t] = {"rc": p.returncode, "lines": lines, "pass": p.returncode == 0 and any(l.startswith(want) for l in lines)}
        if p.returncode:
            out[t]["stderr_tail"] = p.stderr[-3000:]
    boot = ("import torch, runpy\nif not torch.cuda.is_available():\n"
            "    torch.accelerator.current_accelerator = lambda *a, **k: None\n"
            "runpy.run_module('experiments.mist_w3po_v1.crf.{}', run_name='__main__')")
    for t in ("test_latent_write_crf", "test_train_crf_sft", "test_dart_aug"):
        q = subprocess.run([sys.executable, "-W", "ignore", "-c", boot.format(t)], env=env, capture_output=True,
                           text=True, timeout=3600)
        out[t] = {"rc": q.returncode, "tail": [l for l in q.stdout.splitlines() if l.startswith("ALL")][-1:],
                  "pass": q.returncode == 0}
    return out


def run(a) -> None:
    from experiments.mist_w3po_v1.crf import dart_aug as DA
    from experiments.mist_w3po_v1.crf import latent_write_train as LWT
    from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    rep: dict = {"prereg": "reports/mist_w3po_v1/thinker_pmc_noc_prereg_v1.zh-CN.md", "sched": SCHED}
    env = dict(os.environ, PYTHONPATH=".:third_party/Megatron-LM", CUDA_VISIBLE_DEVICES="")
    tst = _tests(env)
    rep["tests"] = {**tst, "pass": all(v["pass"] for v in tst.values())}
    # ---------------- plan: 30 seeded steps of the PMC-1 plan, real windows, dummy scores
    wins = T.load_windows(SCHED, True)
    plan = T.plan_steps([w.R for w in wins], SEED, STEPS, ROWS)
    rng = np.random.default_rng(SEED)
    n_win = n_rows = 0
    lat_same = coef_same = differ = post_same = True
    lams, bind, l1_diff = [], 0, []
    for si in sorted(rng.choice(len(plan), size=30, replace=False).tolist()):
        sw = [wins[i] for i in plan[si]]
        pls = {w.idx: LWT.plan(w, "pmc") for w in sw}
        for w in sw:   # the lattice / plan PMC-noC trains on == PMC-1's (same mode, same call)
            lt = LW.lattice(w.rec, "pmc")
            pw = LW.plan_window(w.rec, lt, skip_forced=False, mode="pmc")
            lat_same &= pw["reqs"] == [{k: v for k, v in q.items() if k != "scored"} for q in pls[w.idx]["reqs"]]
            lat_same &= pw["reads"] == pls[w.idx]["reads"]
            n_win += 1
        res = {}
        for w in sw:
            for k, (v, q) in enumerate(zip(V.dummy_res(pls[w.idx]), pls[w.idx]["reqs"])):
                if q["scored"]:
                    res[(w.idx, k)] = v
        parts = {w.idx: LW.window_parts(w, pls[w.idx], res, LW.TAU) for w in sw}
        lam, info = LW.solve_lambda(list(parts.values()))
        lam2, _ = LW.solve_lambda(list(parts.values()))
        post_same &= lam == lam2
        posts = {w.idx: LW.posterior_from_parts(w, pls[w.idx], parts[w.idx], LW.TAU, lam) for w in sw}
        pols = {w.idx: LW.posterior_from_parts(w, pls[w.idx], LW.pol_parts(w, pls[w.idx], res, parts[w.idx]),
                                               LW.TAU, lam) for w in sw}
        D = float(sum(w.h.sum() + w.R for w in sw if w.feasible))
        ra, rb, rc = (V2T.pass2_rows(sw, pls, None, D) for _ in range(3))
        V2T.set_coefs(ra, pls, posts, D)
        LWT.set_coefs_pmc(rb, pls, posts, pols, D, 0.0)
        LWT.set_coefs_pmc(rc, pls, posts, pols, D)
        d = 0.0
        for x, y, z in zip(ra, rb, rc):
            coef_same &= np.array_equal(x.c_tok, y.c_tok) and np.array_equal(x.c_end, y.c_end)
            d += float(np.abs(z.c_tok - x.c_tok).sum())
        differ &= d > 0
        l1_diff.append(d)
        n_rows += len(ra)
        lams.append(lam)
        bind += bool(info["binding"])
    rep["plan"] = {"steps": 30, "windows": n_win, "pass2_rows": n_rows,
                   "lattice_and_plan_same_as_pmc": bool(lat_same),
                   "w0_coefs_equal_set_coefs_bitwise": bool(coef_same),
                   "pmc1_coefs_differ_every_step": bool(differ),
                   "sum_abs_coef_diff_pmc1_vs_w0_per_step": {"mean": round(float(np.mean(l1_diff)), 6),
                                                             "min": round(float(np.min(l1_diff)), 6)},
                   "lam_deterministic": bool(post_same),
                   "dummy_lam": {"mean": round(float(np.mean(lams)), 4), "binding_steps": bind},
                   "note": "the PMC term's total mass equals q_lam's (sum_t h_t each), so its share of the content "
                           "weight is 0.5 by construction; PMC-noC removes it",
                   "pass": bool(lat_same and coef_same and differ and post_same)}
    # ---------------- args + manifest
    ta = T.build_args(["--objective", "latent_write", "--lw-mode", "pmc", "--pmc-content-weight", "0", "--sched", SCHED,
                       "--dart", DART, "--out", "/nonexistent/pmcnoc_args_check"])
    rep["args"] = {"objective": ta.objective, "lw_mode": ta.lw_mode, "pmc_content_weight": ta.pmc_content_weight,
                   "seed": ta.seed, "steps": ta.steps, "rows_per_step": ta.rows_per_step,
                   "pass": (ta.objective, ta.lw_mode, ta.pmc_content_weight, ta.seed, ta.steps, ta.rows_per_step) ==
                   ("latent_write", "pmc", 0.0, SEED, STEPS, ROWS)}
    dm = DA.load_manifest(DART, wins, plan, SCHED, SEED, STEPS, ROWS)
    rep["manifest"] = {"dir": DART, "mode": DA.manifest_mode(DART), "entries": sum(len(v) for v in dm.values()),
                       "pass": DA.manifest_mode(DART) == "pmc"}
    rep["pass"] = all(rep[k]["pass"] for k in ("tests", "plan", "args", "manifest"))
    rep["cpu_s"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
