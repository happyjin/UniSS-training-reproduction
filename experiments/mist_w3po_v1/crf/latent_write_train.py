"""T2 training step: crf/train_crf_sft.py --objective latent_write (crf/latent_write_crf.py; pre-registration
reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T2).

serial_v2p_train.run_step with the T2 lattice: the same pass-1 rows / split / collation / scoring / all-gather / pass-2
rows / coefficients / prefetch (serial_v2p_train's functions, imported, unchanged); between pass 1 and pass 2 every rank
computes every window's lam-free parts, solves the step-level posterior-regularisation dual lam* (sum_w E_{q_lam}[Lat] =
sum_w E_{q_T1}[Lat], T1 = v2' on the same labels, same model), and the lam* posterior's sequence weights (identical inputs,
deterministic, so every rank gets the same lam* and the same coefficients).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import train_crf_sft as T

TAU = LW.TAU


def plan(w: T.Window, mode: str = "t2") -> dict:
    """The T2 plan of window w, every read included; req["scored"] is False for the requests of forced reads."""
    pl = LW.plan_window(w.rec, LW.lattice(w.rec, mode), skip_forced=False, mode=mode)
    forced = {rd["r"] for rd in pl["reads"] if rd["forced"]}
    for q in pl["reqs"]:
        q["scored"] = q["r"] not in forced
    return pl


def run_step(model: T.CrfThinker, sw: list[T.Window], D: float, ctx: dict, step: int, tau: float, rk: int, nw: int,
             prefetch: bool = False, mode: str = "t2", dart: dict | None = None) -> dict:
    """Pass 1 + lam* + posterior + pass-2 rows of this rank (the grad pass is the trainer's grad_pass).
    dart (T4, crf/dart_aug.py): {w.idx: delay} of this step; its augmentation rows (coefficients fixed) are split over
    the ranks on their own and appended to this rank's pass-2 rows (None: T2 exactly)."""
    t0 = time.time()
    hook = model._post_transform is not None
    plans = {w.idx: plan(w, mode) for w in sw}
    t_plan = time.time() - t0
    p1 = V2T.split(V2T.pass1_rows(sw, plans), nw, ctx["n_prompt0"])[rk]
    p2 = V2T.split(V2T.pass2_rows(sw, plans, None, D), nw, ctx["n_prompt0"])[rk]
    p2all, n_aug = p2, 0
    if dart is not None:
        from experiments.mist_w3po_v1.crf import dart_aug as DA

        aug = DA.aug_rows(sw, dart, D)
        n_aug = len(aug)
        p2all = p2 + V2T.split(aug, nw, ctx["n_prompt0"])[rk]
    b1 = V2T.collate_all(p1, ctx, step, "score", hook)
    th, box = None, {}
    if prefetch:
        th = threading.Thread(target=lambda: box.__setitem__("b2", V2T.collate_all(p2all, ctx, step, "train", hook)),
                              daemon=True)
        th.start()
    t1 = time.time()
    res, s1 = V2T.score_batches(model, b1, ctx)
    t_score = time.time() - t1
    t2 = time.time()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gat = [None] * nw
        torch.distributed.all_gather_object(gat, res)
        res = {k: v for g in gat for k, v in g.items()}
    t_gather = time.time() - t2
    t3 = time.time()
    parts = {w.idx: LW.window_parts(w, plans[w.idx], res, tau) for w in sw}
    lam, info = LW.solve_lambda([parts[w.idx] for w in sw])
    t_lam = time.time() - t3
    posts = {w.idx: LW.posterior_from_parts(w, plans[w.idx], parts[w.idx], tau, lam) for w in sw}
    V2T.set_coefs(p2, plans, posts, D)
    t_post = time.time() - t3
    t4 = time.time()
    batches = None
    if th is not None:
        th.join()
        batches = box["b2"]
    t_join = time.time() - t4
    sum_w = float(sum(p["sum_words"] for p in posts.values()))
    st = {"q_label": float(np.mean([p["q_label"] for p in posts.values()])),
          "exp_nf": float(sum(p["exp_nf"] for p in posts.values())),
          "exp_defer": float(sum(p["exp_defer"] for p in posts.values())),
          "n_real": int(sum(p["n_real"] for p in posts.values())),
          "logZ_mean": float(np.mean([p["logZ"] for p in posts.values()])),
          "rows_pass1": int(sum(1 for w in sw for q in plans[w.idx]["reqs"] if q["scored"])),
          "rows_pass2": int(sum(len(plans[w.idx]["reqs"]) for w in sw)),
          "lam": float(lam), "lam_bracketed": bool(info["bracketed"]), "lam_capped": bool(info["capped"]),
          "lam_binding": bool(info["binding"]), "lam_evals": int(info["evals"]),
          "Eq_lat_T2": float(sum(p["Eq_lat"] for p in posts.values()) / max(sum_w, 1e-9)),
          "Eq_lat_T1_target": float(info["target"] / max(sum_w, 1e-9)),
          "Eq_lat_lam0": float(info["Eq_lat_lam0"] / max(sum_w, 1e-9)),
          "lat_label": float(sum(p["lat_label"] for p in posts.values()) / max(sum_w, 1e-9)),
          "sum_words": sum_w, "t_lambda_solve": round(t_lam, 3),
          "edges": int(sum(int(x.sum()) for w in sw for x in plans[w.idx]["edge"]))}
    if dart is not None:
        st["dart_rows"] = n_aug
        st["dart_windows"] = int(sum(1 for w in sw if dart.get(w.idx) is not None))
    return {"rows": p2all, "batches": batches, "t_pass1": t3 - t0, "t_post": t_post, "stats": st,
            "local": {"tok_pass1": s1["tok"], "mbs_pass1": s1["mbs"], "t_plan": t_plan,
                      "t_collate1": sum(getattr(b, "t_collate", 0.0) for _, b in b1),
                      "t_score1": t_score, "t_gather": t_gather, "t_join": t_join}}
