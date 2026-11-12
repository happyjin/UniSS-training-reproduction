"""Held-out loss of a SerialCRF v2' checkpoint (train_crf_sft --objective serial_v2p) on dev sched windows: the
near-free secondary signal of the En->Zh checkpoint selection (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1
.zh-CN.md §3.2; recorded, never a selection criterion). One GPU, no gradient.

Windows: the dev sched windows (crf/build_sched_windows --split validation; en2zh: sched_en2zh_dev_v1) of the --meetings
(default: the 5 precheck meetings), feasible ones (set_reach, W 128), in id order. Model: the trainer's own (v8 merged
base + the checkpoint's LoRA / projector / token rows; train_crf_sft.load_thinker / CrfThinker / load_ckpt). Inputs:
the trainer's rows (serial_v2p_train.plan / pass2_rows / collate_all) with perturbation 0 and the final read's full
posteriors (--final-read-full-post 1, as the training run), <wait> barred at the final read as in pass 1.

Per window, from the log-probs of EVERY plan request (forced reads included):
  label_nll   the C0 loss terms of the label path (serial_v2p_train.set_coefs' c_lab: h-weighted token NLL + end NLL)
  full_ll     log sum over the v2' lattice paths of exp(sum of path log-probs): serial_v2p_train.posterior(tau = 1)'s
              logZ over the scored reads + the forced reads' own (single-path) log-probs
  D           sum_j h_j + R (the trainer's normaliser), n = N + R (tokens + end tokens)
Output json: ce_label = sum label_nll / sum D (= the trainer's sft_loss on these windows), crf_nll = -sum full_ll / sum n,
q_label_mean (tau 0.5 posterior mass of the label path, the training statistic), counts, ckpt, timings.

    python -m experiments.mist_w3po_v1.crf.heldout_loss --base <v8 merged> --ckpt <run>/ckpt/step_NNNNN \\
        --sched <dev sched dir> --dir en2zh --out <json> [--meetings ES2011a ...] [--tok-budget 12000]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

DEV5 = ("ES2011a", "IB4003", "IB4011", "IS1008c", "TS3004b")


def window_terms(w, pl: dict, res: dict) -> dict:
    """label_nll / full_ll / D / n / q_label of one window from res {(w.idx, k): (tok_lp [m], end_lp [m + 1])} holding
    every request of its plan (CPU, float64)."""
    from experiments.mist_w3po_v1.crf import serial_v2p_train as V

    rows = V.pass2_rows([w], {w.idx: pl}, None, 1.0)
    post1 = V.posterior(w, pl, res, 1.0)
    V.set_coefs(rows, {w.idx: pl}, {w.idx: post1}, 1.0)
    lab = 0.0
    for rw in rows:
        tl, ec = res[rw.key]
        m = len(tl)
        lab -= float(np.dot(rw.c_lab_tok[:m], tl) + np.dot(rw.c_lab_end, ec))
    forced = 0.0
    for k, q in enumerate(pl["reqs"]):
        if not q["scored"]:
            tl, ec = res[(w.idx, k)]
            forced += float(tl.sum() + ec[len(tl)])
    q_label = V.posterior(w, pl, res, V.TAU)["q_label"]
    return {"label_nll": lab, "full_ll": float(post1["logZ"]) + forced, "D": float(w.h.sum() + w.R),
            "n": int(w.N + w.R), "q_label": float(q_label)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sched", required=True)
    ap.add_argument("--dir", default="en2zh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meetings", nargs="*", default=list(DEV5))
    ap.add_argument("--tok-budget", type=int, default=12000)
    ap.add_argument("--lp-chunk", type=int, default=1024)
    ap.add_argument("--chunk-windows", type=int, default=32)
    ap.add_argument("--mem-cap-gib", type=float, default=0.0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if not (Path(a.ckpt) / "DONE").exists():
        raise SystemExit(f"no DONE in {a.ckpt}")
    import torch

    from experiments.mist_w3po_v1.core.engine import Prompter
    from experiments.mist_w3po_v1.crf import serial_v2p_train as V
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    t0 = time.time()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    if a.mem_cap_gib > 0:
        tot = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gib * 2**30 / tot), dev)
    wins = [w for w in T.load_windows(a.sched, True) if w.rec["meeting"] in set(a.meetings)]
    T.set_reach(wins, 128)
    wins = [w for w in wins if w.feasible]
    prompter = Prompter(a.dir)
    thinker = T.load_thinker(a.base, dev)
    sd_state = torch.load(Path(a.base) / "sd_proj.pt", map_location="cpu")
    model = T.CrfThinker(thinker, sd_state, 32, 64.0, None)
    T.load_ckpt(Path(a.ckpt), model, None, [])
    model.eval()
    ctx = {"audio": T.AudioCache(), "proc": prompter.proc, "prompter": prompter, "perturb": 0.0, "seed": 0,
           "tok_budget": a.tok_budget, "lp_chunk": a.lp_chunk, "final_bar_wait": True, "final_full_post": True,
           "n_prompt0": len(prompter.a_ids) + len(prompter.b_ids) - 1, "feat_cache": {}}
    t1 = time.time()
    acc = {"label_nll": 0.0, "full_ll": 0.0, "D": 0.0, "n": 0}
    ql, nrows = [], 0
    for i in range(0, len(wins), a.chunk_windows):
        sw = wins[i: i + a.chunk_windows]
        plans = {w.idx: V.plan(w) for w in sw}
        rows = V.pass2_rows(sw, plans, None, 1.0)   # every request (forced reads too) is scored
        nrows += len(rows)
        res, _ = V.score_batches(model, V.collate_all(rows, ctx, 0, "score", False), ctx)
        for w in sw:
            t = window_terms(w, plans[w.idx], res)
            for k in acc:
                acc[k] += t[k]
            ql.append(t["q_label"])
        ctx["audio"] = T.AudioCache()
        ctx["feat_cache"] = {}
        print(f"[heldout] {i + len(sw)}/{len(wins)} windows {time.time() - t1:.0f}s", flush=True)
    rep = {"ckpt": a.ckpt, "base": a.base, "sched": a.sched, "dir": a.dir, "meetings": a.meetings, "windows": len(wins),
           "rows": nrows, "ce_label": acc["label_nll"] / acc["D"], "crf_nll": -acc["full_ll"] / acc["n"],
           "q_label_mean": float(np.mean(ql)) if ql else math.nan, "sum": acc,
           "t_load_s": round(t1 - t0, 1), "t_score_s": round(time.time() - t1, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep), flush=True)
    print("HELDOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
