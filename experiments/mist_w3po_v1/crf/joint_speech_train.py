"""T3 training step: crf/train_crf_sft.py --objective joint_speech --talker-model <json> (crf/joint_speech_crf.py;
pre-registration reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T3).

latent_write_train.run_step with the joint chain: T2's plan (mode "t2"), pass-1 rows / split / collation / scoring /
all-gather / pass-2 rows / coefficients / prefetch (serial_v2p_train's functions, imported, unchanged); between pass 1
and pass 2 every rank builds every window's augmented (Thinker state, backlog vector) chain, solves the step-level dual
lam* (sum_w E_{q_lam}[l] = sum_w E_{q_T1}[l], speech-timeline l), and the lam* posterior's sequence weights (identical
inputs, deterministic: every rank gets the same lam* and coefficients).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import train_crf_sft as T

TAU = JS.TAU


def run_step(model: T.CrfThinker, sw: list[T.Window], D: float, ctx: dict, step: int, tau: float, rk: int, nw: int,
             prefetch: bool = False) -> dict:
    """Pass 1 + augmented chains + lam* + posterior + pass-2 rows of this rank. ctx["talker_model"]: JS.TalkerModel."""
    tm = ctx["talker_model"]
    t0 = time.time()
    hook = model._post_transform is not None
    plans = {w.idx: LWT.plan(w, "t2") for w in sw}
    t_plan = time.time() - t0
    p1 = V2T.split(V2T.pass1_rows(sw, plans), nw, ctx["n_prompt0"])[rk]
    p2 = V2T.split(V2T.pass2_rows(sw, plans, None, D), nw, ctx["n_prompt0"])[rk]
    b1 = V2T.collate_all(p1, ctx, step, "score", hook)
    th, box = None, {}
    if prefetch:
        th = threading.Thread(target=lambda: box.__setitem__("b2", V2T.collate_all(p2, ctx, step, "train", hook)),
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
    parts = {w.idx: JS.window_parts(w, plans[w.idx], res, tm, tau) for w in sw}
    t_aug = time.time() - t3
    lam, info = JS.solve_lambda([parts[w.idx] for w in sw])
    t_lam = time.time() - t3 - t_aug
    posts = {w.idx: JS.posterior_from_parts(w, plans[w.idx], parts[w.idx], tau, lam) for w in sw}
    V2T.set_coefs(p2, plans, posts, D)
    t_post = time.time() - t3
    t4 = time.time()
    batches = None
    if th is not None:
        th.join()
        batches = box["b2"]
    t_join = time.time() - t4
    sum_w = float(sum(p["sum_words"] for p in posts.values()))
    st = {"exp_nf": float(sum(p["exp_nf"] for p in posts.values())),
          "exp_defer": float(sum(p["exp_defer"] for p in posts.values())),
          "n_real": int(sum(p["n_real"] for p in posts.values())),
          "logZ_mean": float(np.mean([p["logZ"] for p in posts.values()])),
          "rows_pass1": int(sum(1 for w in sw for q in plans[w.idx]["reqs"] if q["scored"])),
          "rows_pass2": int(sum(len(plans[w.idx]["reqs"]) for w in sw)),
          "lam": float(lam), "lam_bracketed": bool(info["bracketed"]), "lam_capped": bool(info["capped"]),
          "lam_binding": bool(info["binding"]), "lam_evals": int(info["evals"]),
          "Eq_speech_lat_T3": float(sum(p["Eq_l"] for p in posts.values()) / max(sum_w, 1e-9)),
          "Eq_speech_lat_T1_target": float(info["target"] / max(sum_w, 1e-9)),
          "Eq_speech_lat_lam0": float(info["Eq_l_lam0"] / max(sum_w, 1e-9)),
          "sum_words": sum_w, "aug_states_max": int(max(p["aug_states_max"] for p in posts.values())),
          "aug_edges": int(sum(p["aug_edges"] for p in posts.values())),
          "aug_coarsened": int(sum(p["coarsened"] for p in posts.values())),
          "t_aug": round(t_aug, 3), "t_lambda_solve": round(t_lam, 3),
          "edges": int(sum(int(x.sum()) for w in sw for x in plans[w.idx]["edge"]))}
    return {"rows": p2, "batches": batches, "t_pass1": t3 - t0, "t_post": t_post, "stats": st,
            "local": {"tok_pass1": s1["tok"], "mbs_pass1": s1["mbs"], "t_plan": t_plan,
                      "t_collate1": sum(getattr(b, "t_collate", 0.0) for _, b in b1),
                      "t_score1": t_score, "t_gather": t_gather, "t_join": t_join}}
