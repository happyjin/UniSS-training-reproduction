"""VE1's CPU precheck (reports/mist_w3po_v1/thinker_v2p_evidence_ve1_prereg_v1.zh-CN.md §3):
  V1  the merge3 1.0 s data: builder checks (n_infeasible, n_bad 0), words per block >= 4.5, final-read label word
      share and label-path latency reported
  V2  on the first 200 plan steps' windows: --lw-mode v2p's plan == serial_v2p_train.plan field for field, its lam = 0
      posterior == serial_v2p_train.posterior bit for bit (dummy scores), and the step-level lam* (T1 target) is 0
  V3  the trainer's arguments (run_crf_v2p.sh + EXTRA) parse to latent_write / v2p; the DART manifest loads against the
      plan and was made in mode v2p
    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.ve1_precheck --out <new json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if not torch.cuda.is_available():
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import dart_aug as DA
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import train_crf_sft as T

C = "eval_outputs/mist_w3po_v1/crf"
SCHED = f"{C}/sched_t5_merge3_w10_v1"
DART = f"{C}/ve1_dart_merge3_w10_v1"
EXTRA = f"--sched {SCHED} --objective latent_write --lw-mode v2p --dart {DART}"
BASE = ("--objective serial_v2p --tau 0.5 --final-read-full-post 1 --out /nonexistent --steps 3000 --save-every 500 "
        "--feat-per-row 1 --fast-moe-bwd 1 --prefetch 1 --empty-cache 0 --tok-budget 32000 --mem-cap-gib 120")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    # V1
    sm = json.loads(Path(SCHED, "summary.json").read_text())
    wins = T.load_windows(SCHED)
    nw = [c["n_words"] for w in wins for c in w.rec["chunks"]]
    fin = sum(c["n_words"] for w in wins for c in w.rec["chunks"] if c["e_read"] == w.R - 1)
    res["V1"] = {"windows": len(wins), "n_infeasible": sm["n_infeasible"], "n_bad": sm["n_bad"],
                 "words_per_block": round(float(np.mean(nw)), 3), "blocks": len(nw),
                 "final_read_label_word_share": round(fin / sum(nw), 4),
                 "label_path_lat_heur_mean": sm["L_ref_mean_lat_heur_windows"]}
    res["V1"]["pass"] = bool(sm["n_infeasible"] == 0 and sm["n_bad"] == 0 and res["V1"]["words_per_block"] >= 4.5)
    # V2
    plan = T.plan_steps([w.R for w in wins], 20261109, 3000, 64)
    n = eqp = eqq = 0
    lams = []
    for s in range(a.steps):
        parts = []
        for i in plan[s]:
            w = wins[i]
            pa, pb = V2T.plan(w), LWT.plan(w, "v2p")
            eqp += pa["reqs"] == pb["reqs"] and pa["reads"] == pb["reads"]
            res_ = {(w.idx, k): v for k, (v, q) in enumerate(zip(V.dummy_res(pa), pa["reqs"])) if q["scored"]}
            x, y = V2T.posterior(w, pa, res_, V2T.TAU), LW.posterior(w, pb, res_, V2T.TAU, 0.0)
            eqq += (x["logZ"] == y["logZ"] and all(np.array_equal(c1, c2) for c1, c2 in zip(x["cend"], y["cend"])))
            parts.append(LW.window_parts(w, pb, res_, V2T.TAU))
            n += 1
        lams.append(LW.solve_lambda(parts)[0])
    res["V2"] = {"windows": n, "plan_equal": eqp, "posterior_bit_equal": eqq, "steps": a.steps,
                 "lam_nonzero_steps": int(sum(1 for x in lams if x != 0.0)), "lam_max": float(max(lams))}
    res["V2"]["pass"] = bool(eqp == n and eqq == n and res["V2"]["lam_nonzero_steps"] == 0)
    # V3
    args = T.build_args((BASE + " " + EXTRA).split())
    d = DA.load_manifest(args.dart, wins, plan, args.sched, args.seed, args.steps, args.rows_per_step)
    res["V3"] = {"objective": args.objective, "lw_mode": args.lw_mode, "dart_entries": sum(len(v) for v in d.values()),
                 "manifest_mode": DA.manifest_mode(args.dart)}
    res["V3"]["pass"] = bool(args.objective == "latent_write" and args.lw_mode == "v2p"
                             and res["V3"]["manifest_mode"] == "v2p" and res["V3"]["dart_entries"] > 0)
    res["pass"] = res["V1"]["pass"] and res["V2"]["pass"] and res["V3"]["pass"]
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
