"""SerialCRF v2' training objective: crf/train_crf_sft.py --objective serial_v2p (crf_novel_ideas.zh-CN.md §13.3, §13.10,
§13.11; precheck reports/mist_w3po_v1/serial_crf_v2p_precheck.zh-CN.md; training pre-registration
reports/mist_w3po_v1/serial_v2p_train.zh-CN.md). Additive: the lattice, the forward plan, the psi assembly and the
forward recursion are the precheck's own functions (crf/precheck_serial_v2.py, crf/precheck_serial_v2p.py; imported,
unchanged), so training optimises exactly the lattice the precheck scored.

Per window (a sched_v1 record: the same windows, steps, perturbation draws and normaliser D as C0)
  lattice   v2' (VP module docstring): items = the chunks, slack one read (final-read items not deferrable), set state
            D_r, kappa history, kap write order, order latent o_r in {canon, newfirst} (VP.newfirst; a real choice only
            when the pending set and N_r are both non-empty).
  plan      VP.plan_window_p(rec, skip_forced=False): per (read, context) the canonical cover forwards (V.covers: one
            forward per maximal write set, every kap-prefix read off at its item boundary) and the newfirst forwards
            (item-prefix sharing). Every request is one token sequence hist_ids ⊕ forced_ids, then e_r.
  pass 1    (no grad, model.eval) every request of every non-forced read, through the trainer's own rows (collate:
            v8's row layout, the C0 perturbation draw of (seed, step, window, read), --final-read-full-post as given):
            tok_lp [m] and end_lp [m + 1] at the m + 1 positions from before the first forced token to after the last;
            at the final read <wait> barred before the softmax (--final-bar-wait 1: Engine.read, the precheck's
            score_reqs_v2). Forced reads (one context, one write set) add the same constant to every path: not scored
            (psi := 0; the posterior is unchanged, as in the precheck).
  posterior psi cells by VP.assemble_p; joint tempering q_tau(rho, o) ∝ exp(tau sum_r psi_{r,o_r}), tau = 0.5
            (pre-registered); forward-backward over the set states with the per-edge logsumexp_o (VP.edge_terms,
            V.forward_backward); edge-order marginals q(r, i -> j, o) = exp(alpha_{r-1}(i) + tau psi_{r,o}(i,j) + beta_r(j)
            - log Z). CPU float64.
  pass 2    (grad, model.train) the DAEM / EM M-step surrogate (§13.3), q without gradient:
              L = -(1/D) sum_{r, i, j, o} q(r, i->j, o) [ sum_t h_t log p(W_{r,o}(i,j)_t | x, kappa(A_{r-1}), W_<t)
                                                           + log p(e_r | x, kappa(A_{r-1}), W_{r,o}(i,j)) ]
            plain softmax at every read (C0 / v8's cross-entropy); h = v8's hierarchical weights (a function of the
            token id); D = C0's sum over the step's feasible windows of (sum_j h_j + R): every path writes every item
            once and ends every read once, so the expected total weight per window equals C0's (tested).
            Sequences sharing a forward (prefixes) form one row whose positions carry the summed coefficients:
            c_end[b] = the q of the sequences ending at item boundary b, c_tok[t] = h_t sum_{b > t} c_end[b].
            Every plan request is a row (all positions kept, also where q underflows to 0), so the rows are a function of
            the plan only. On the label path (q = 1) the rows are C0's rows (tested: loss and gradients bit-identical).
  ranks     pass-1 requests and pass-2 rows are each split over the ranks by estimated padded length (LPT, deterministic);
            the pass-1 log-probs are all-gathered, every rank computes every window's posterior (identical inputs,
            deterministic), gradients are all-reduced as before. Only the packing depends on the world size / budget.

Logged per step (metrics.jsonl, serial_v2p only): sft_loss = C0's loss (label path, h, D) of the same windows at the
current parameters (from the pass-2 forward), q_label (posterior mass of the label path), exp_nf (expected newfirst reads),
exp_defer (expected deferred items), n_real (reads with a real order choice), rows / tokens / micro-batches of both
passes, collate / gather / posterior times.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_serial_v2p_train
    python -m experiments.mist_w3po_v1.crf.serial_v2p_train --check-all --out <new json> [--procs 64]
        (CPU: the plan of every sched_v1 window used by the 3000 steps: label path == C0's rows, sizes per step)
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP
from experiments.mist_w3po_v1.crf import train_crf_sft as T

TAU = 0.5
assert TAU == VP.TAU


# ------------------------------------------------------------------------------------------------ plan and rows
def plan(w: T.Window) -> dict:
    """The v2' plan of window w, every read included (forced reads too: their single row is trained); req["scored"]
    is False for the requests of forced reads (not scored in pass 1)."""
    lat = V.lattice(w.rec)
    pl = VP.plan_window_p(w.rec, lat, skip_forced=False)
    forced = {rd["r"] for rd in pl["reads"] if rd["forced"]}
    for q in pl["reqs"]:
        q["scored"] = q["r"] not in forced
    return pl


def _row(w: T.Window, q: dict, key: tuple) -> T.Row:
    seq = list(q["hist_ids"]) + list(q["forced_ids"])
    h0, m = len(q["hist_ids"]), len(q["forced_ids"])
    return T.Row(w, q["r"], len(seq), np.arange(h0, h0 + m + 1), seq=seq, end=w.end_ids[q["r"]], key=key)


def pass1_rows(wins: list[T.Window], plans: dict) -> list[T.Row]:
    return [_row(w, q, (w.idx, k)) for w in wins for k, q in enumerate(plans[w.idx]["reqs"]) if q["scored"]]


def row_len(rw: T.Row, n_prompt0: int) -> int:
    """micro_batches' length estimate (prompt + audio tokens + interleaved speaker tokens + fed tokens)."""
    n = math.ceil(13 * rw.win.t_ms[rw.r] / 1000.0) + 1
    return n_prompt0 + n + math.ceil(n / 5) + rw.K


def split(rows: list[T.Row], n: int, n_prompt0: int) -> list[list[T.Row]]:
    """Longest-processing-time split of rows over n ranks by row_len (deterministic; rows keep their order)."""
    if n == 1:
        return [list(rows)]
    ln = [row_len(rw, n_prompt0) for rw in rows]
    load, part = [0] * n, [[] for _ in range(n)]
    for i in sorted(range(len(rows)), key=lambda i: (-ln[i], i)):
        k = min(range(n), key=lambda k: (load[k], k))
        part[k].append(i)
        load[k] += ln[i]
    return [[rows[i] for i in sorted(p)] for p in part]


# ------------------------------------------------------------------------------------------------ pass 1
def score_batches(model: T.CrfThinker, batches, ctx: dict) -> tuple[dict, dict]:
    """{row key: (tok_lp [m], end_lp [m + 1]) float64 numpy} of collated pass-1 micro-batches; the final read's pair has
    <wait> barred before the softmax when ctx["final_bar_wait"] (the decoding distribution, as the precheck)."""
    out, st = {}, {"tok": 0, "mbs": 0, "t_collate": 0.0}
    for mb, b in batches:
        st["t_collate"] += b.t_collate if hasattr(b, "t_collate") else 0.0
        with torch.no_grad():
            lt, le, lm = model.pos_logprobs(b, ctx["lp_chunk"], masked_final=ctx["final_bar_wait"])
        lt, le = lt.double().cpu().numpy(), le.double().cpu().numpy()
        if lm is not None:
            lm = (lm[0].double().cpu().numpy(), lm[1].double().cpu().numpy())
        k = 0
        for rw in mb:
            n = len(rw.pos)
            if lm is not None and rw.r == rw.win.R - 1:
                a, e = lm[0][k: k + n], lm[1][k: k + n]
            else:
                a, e = lt[k: k + n], le[k: k + n]
            out[rw.key] = (a[: n - 1].copy(), e.copy())
            k += n
        st["tok"] += int(b.input_ids.numel())
        st["mbs"] += 1
    return out, st


def collate_all(rows: list[T.Row], ctx: dict, step: int, phase: str, hook: bool) -> list:
    """[(micro-batch rows, Batch)] with micro_batches' packing; Batch.t_collate = its CPU time."""
    out = []
    for mb in T.micro_batches(rows, ctx["tok_budget"], ctx["n_prompt0"]):
        t0 = time.time()
        b = T.collate(mb, ctx["audio"], ctx["proc"], ctx["prompter"], hook, ctx["perturb"], ctx["seed"], step, phase,
                      final_full=ctx.get("final_full_post", False), feat_cache=ctx.get("feat_cache"))
        b.t_collate = time.time() - t0
        out.append((mb, b))
    return out


# ------------------------------------------------------------------------------------------------ posterior
def posterior(w: T.Window, pl: dict, res: dict, tau: float = TAU) -> dict:
    """Edge-order posterior of one window from the pass-1 log-probs res {(w.idx, k): (tl, ec)}.
    Returns per request k the summed end weights cend[k] [m + 1] (the q of the sequences ending at each boundary) and
    the window's statistics."""
    full = []
    for k, q in enumerate(pl["reqs"]):
        m = len(q["forced_ids"])
        full.append(res[(w.idx, k)] if q["scored"] else (np.zeros(m), np.zeros(m + 1)))
    reads = VP.assemble_p(pl, full)
    E = VP.edge_terms(reads, tau, "joint")
    fb = V.forward_backward(E, 1.0)
    cend = [np.zeros(len(q["forced_ids"]) + 1) for q in pl["reqs"]]
    exp_nf, n_real, worst = 0.0, 0, 0.0
    for r, (rd, pr) in enumerate(zip(reads, pl["reads"])):
        C, Nf = VP.mats(rd)
        a = fb["alpha"][r - 1] if r > 0 else np.zeros(1)
        qe = np.exp(a[:, None] + E[r] + fb["beta"][r][None, :] - fb["logZ"])
        qC = qe * np.exp(tau * C - E[r])
        if Nf is not None and not np.isnan(Nf).all():
            real = ~np.isnan(Nf)
            qN = np.where(real, qe * np.exp(tau * np.where(real, Nf, 0.0) - E[r]), 0.0)
            n_real += 1
            exp_nf += float(qN.sum())
        else:
            qN = np.zeros_like(qC)
        worst = max(worst, abs(float(qC.sum() + qN.sum()) - 1.0))
        for cx in pr["ctx"]:
            i = cx["i"]
            for j, (k, b) in cx["cells"]["new"].items():
                cend[k][b] += qC[i, j]
            for j, (k, b, _bn, _s, _nl) in cx["nf"].items():
                cend[k][b] += qN[i, j]
    if worst > 1e-6:
        raise RuntimeError(f"v2' posterior of {w.rec['id']}: per-read mass off by {worst:.2e}")
    R = pl["lat"]["R"]
    q_label = math.exp(tau * VP.path_score_p(reads, [0] * R, [0] * R) - fb["logZ"])
    defer = V.deferral_marginals(pl["lat"], fb)
    return {"cend": cend, "logZ": fb["logZ"], "q_label": q_label, "exp_nf": exp_nf, "n_real": n_real,
            "exp_defer": float(sum(defer.values())), "fb": fb, "reads": reads}


def pass2_rows(wins: list[T.Window], plans: dict, posts: dict | None, D: float) -> list[T.Row]:
    """One row per plan request (forced reads included); coefficients from the posterior (posts None: structure only,
    coefficients attached later by set_coefs). The C0 label-path coefficients ride along for logging."""
    rows = []
    for w in wins:
        for k, q in enumerate(plans[w.idx]["reqs"]):
            rows.append(_row(w, q, (w.idx, k)))
    if posts is not None:
        set_coefs(rows, plans, posts, D)
    return rows


def token_weights(w: T.Window, ids) -> np.ndarray:
    """h of the given token ids: the window's own h (v8's hierarchical weights, a function of the token id; all 1 with
    --hier-weights 0); every written token is a token of S."""
    hm = dict(zip(w.S, w.h.tolist()))
    return np.asarray([hm[t] for t in ids], np.float64)


def set_coefs(rows: list[T.Row], plans: dict, posts: dict, D: float) -> None:
    """c_end[b] = cend[b] / D, c_tok[t] = h_t sum_{b > t} cend[b] / D (positions 0..m of the row's write part); the
    label row (context 0 -> state 0 of its read) also carries C0's coefficients (h / D, 1 / D) for the logged C0 loss."""
    for rw in rows:
        widx, k = rw.key
        q = plans[widx]["reqs"][k]
        ce = posts[widx]["cend"][k]
        m = len(q["forced_ids"])
        h = token_weights(rw.win, q["forced_ids"])
        suf = np.cumsum(ce[::-1])[::-1]              # suf[t] = sum_{b >= t} ce[b]
        ct = np.zeros(m + 1)
        ct[:m] = h * suf[1:] / D
        rw.c_tok, rw.c_end, rw.c_state = ct, ce / D, np.zeros(m + 1)
        lt, le = np.zeros(m + 1), np.zeros(m + 1)
        rd = plans[widx]["reads"][q["r"]]
        if rd["ctx"][0]["cells"]["new"][0][0] == k:    # context 0 -> state 0 = the label path: its block is the row
            assert rd["ctx"][0]["i"] == 0 and rd["ctx"][0]["cells"]["new"][0][1] == m
            lt[:m], le[m] = h / D, 1.0 / D
        rw.c_lab_tok, rw.c_lab_end = lt, le


# ------------------------------------------------------------------------------------------------ one step
def run_step(model: T.CrfThinker, sw: list[T.Window], D: float, ctx: dict, step: int, tau: float, rk: int, nw: int,
             prefetch: bool = False) -> dict:
    """Pass 1 + posterior + pass-2 rows of this rank (the grad pass is the trainer's grad_pass). With prefetch the
    pass-2 micro-batches are collated in a background thread while pass 1 runs (same packing, same tensors)."""
    t0 = time.time()
    hook = model._post_transform is not None
    plans = {w.idx: plan(w) for w in sw}
    t_plan = time.time() - t0
    p1 = split(pass1_rows(sw, plans), nw, ctx["n_prompt0"])[rk]
    p2 = split(pass2_rows(sw, plans, None, D), nw, ctx["n_prompt0"])[rk]
    b1 = collate_all(p1, ctx, step, "score", hook)
    th, box = None, {}
    if prefetch:
        th = threading.Thread(target=lambda: box.__setitem__("b2", collate_all(p2, ctx, step, "train", hook)),
                              daemon=True)
        th.start()
    t1 = time.time()
    res, s1 = score_batches(model, b1, ctx)
    t_score = time.time() - t1
    t2 = time.time()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gat = [None] * nw
        torch.distributed.all_gather_object(gat, res)
        res = {k: v for g in gat for k, v in g.items()}
    t_gather = time.time() - t2
    t3 = time.time()
    posts = {w.idx: posterior(w, plans[w.idx], res, tau) for w in sw}
    set_coefs(p2, plans, posts, D)
    t_post = time.time() - t3
    t4 = time.time()
    if th is not None:
        th.join()
        batches = box["b2"]
    else:
        batches = None
    t_join = time.time() - t4
    t_pass1 = t3 - t0
    st = {"q_label": float(np.mean([p["q_label"] for p in posts.values()])),
          "exp_nf": float(sum(p["exp_nf"] for p in posts.values())),
          "exp_defer": float(sum(p["exp_defer"] for p in posts.values())),
          "n_real": int(sum(p["n_real"] for p in posts.values())),
          "logZ_mean": float(np.mean([p["logZ"] for p in posts.values()])),
          "rows_pass1": int(sum(1 for w in sw for q in plans[w.idx]["reqs"] if q["scored"])),
          "rows_pass2": int(sum(len(plans[w.idx]["reqs"]) for w in sw))}
    return {"rows": p2, "batches": batches, "t_pass1": t_pass1, "t_post": t_post,
            "stats": st, "local": {"tok_pass1": s1["tok"], "mbs_pass1": s1["mbs"], "t_plan": t_plan,
                                   "t_collate1": sum(getattr(b, "t_collate", 0.0) for _, b in b1),
                                   "t_score1": t_score, "t_gather": t_gather, "t_join": t_join}}


def reduce_logs(v2p: dict, lg: dict, dev) -> None:
    """Sums (loss_label, pass-1 tokens / micro-batches) and maxima (timings) over the ranks, into v2p["red"]."""
    loc = v2p["local"]
    s = torch.tensor([lg["loss_label"], loc["tok_pass1"], loc["mbs_pass1"]], dtype=torch.float64, device=dev)
    mx = torch.tensor([loc["t_plan"], loc["t_collate1"], loc["t_score1"], loc["t_gather"], loc["t_join"],
                       lg["t_collate"]], dtype=torch.float64, device=dev)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(s)
        torch.distributed.all_reduce(mx, op=torch.distributed.ReduceOp.MAX)
    a, b = s.tolist(), mx.tolist()
    v2p["red"] = {"sft_loss": a[0], "tok_pass1": int(a[1]), "micro_batches_pass1": int(a[2]),
                  "t_plan": round(b[0], 3), "t_collate1": round(b[1], 3), "t_score1": round(b[2], 3),
                  "t_gather": round(b[3], 3), "t_join2": round(b[4], 3), "t_collate2": round(b[5], 3)}


def metrics_fields(v2p: dict, lg: dict, rec: dict) -> dict:
    """the serial_v2p fields of the metrics record (sft_loss = C0's loss of the step's windows, from pass 2)."""
    out = dict(v2p["red"])
    out.update(v2p["stats"])
    return out


# ------------------------------------------------------------------------------------------------ --fast-moe-bwd
class FastGroupedMM(torch.autograd.Function):
    """transformers' grouped_mm fallback (integrations/moe.py: one torch.mm per expert, the path taken with torch 2.6,
    which has no grouped_mm kernel), with the same forward and input-gradient calls in the same order (bit-identical),
    but the expert-weight gradient only when the weight requires grad: the fallback's registered backward always
    allocates zeros_like(weight) and runs one extra mm per expert for the frozen experts, whose gradient is discarded.
    The offsets are read once in forward (the fallback reads them again in backward: one GPU sync less)."""

    @staticmethod
    def forward(ctx, inp, weight, offs):
        out = torch.zeros(inp.size(0), weight.size(2), device=inp.device, dtype=inp.dtype)
        ol = offs.tolist()
        start = 0
        for i, end in enumerate(ol):
            if start == end:
                continue
            torch.mm(inp[start:end], weight[i], out=out[start:end])
            start = end
        ctx.save_for_backward(inp, weight)
        ctx.ol = ol
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        gi = torch.zeros_like(inp) if ctx.needs_input_grad[0] else None
        gw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        start = 0
        for i, end in enumerate(ctx.ol):
            if start == end:
                continue
            if gi is not None:
                torch.mm(grad_output[start:end], weight[i].T, out=gi[start:end])
            if gw is not None:
                torch.mm(inp[start:end].T, grad_output[start:end], out=gw[i])
            start = end
        return gi, gw, None


_ORIG_GROUPED_MM = None
FAST_MOE_CALLS = [0]


def _fast_grouped_mm(inp, weight, offs):
    from transformers.integrations import moe

    if moe._can_use_grouped_mm(inp, weight, offs):
        return _ORIG_GROUPED_MM(inp, weight, offs)
    FAST_MOE_CALLS[0] += 1
    return FastGroupedMM.apply(inp.to(weight.dtype), weight, offs)


def patch_fast_moe_bwd(on: bool) -> None:
    """Route transformers.integrations.moe._grouped_mm's fallback through FastGroupedMM (on) or restore it (off)."""
    global _ORIG_GROUPED_MM
    from transformers.integrations import moe

    if _ORIG_GROUPED_MM is None:
        _ORIG_GROUPED_MM = moe._grouped_mm
    moe._grouped_mm = _fast_grouped_mm if on else _ORIG_GROUPED_MM


# ------------------------------------------------------------------------------------------------ CPU check of all windows
def _check_one(rec: dict) -> dict:
    """Plan of one window: the label path's rows are C0's rows (tokens and positions) and the sizes."""
    w = T.make_window(0, rec, True)
    pl = plan(w)
    beta = [0] + [int(b) for b in w.beta]
    ok = True
    for rd in pl["reads"]:
        r = rd["r"]
        k, b = rd["ctx"][0]["cells"]["new"][0]
        q = pl["reqs"][k]
        ok &= (rd["ctx"][0]["i"] == 0 and list(q["hist_ids"]) == w.S[: beta[r]]
               and list(q["forced_ids"])[:b] == w.S[beta[r]: beta[r + 1]] and b == beta[r + 1] - beta[r]
               and b == len(q["forced_ids"]))
    reqs = pl["reqs"]
    return {"id": rec["id"], "ok": bool(ok), "R": w.R, "p1": sum(1 for q in reqs if q["scored"]), "p2": len(reqs),
            "tok1": sum(len(q["hist_ids"]) + len(q["forced_ids"]) for q in reqs if q["scored"]),
            "tok2": sum(len(q["hist_ids"]) + len(q["forced_ids"]) for q in reqs),
            "nf": sum(1 for q in reqs if q.get("kind") == "nf"),
            "real_reads": sum(1 for rd in pl["reads"] for cx in rd["ctx"] if cx["nf"]) > 0}


def check_all(args) -> None:
    from multiprocessing import Pool

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    recs = []
    for f in sorted(Path(T.SCHED).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r["n_tok"] > 0]
    recs.sort(key=lambda r: r["id"])
    wins = [T.make_window(i, r, True) for i, r in enumerate(recs)]
    plan_ = T.plan_steps([w.R for w in wins], args.seed, args.steps, args.rows_per_step)
    used = sorted({i for s in plan_ for i in s})
    with Pool(args.procs) as pool:
        res = pool.map(_check_one, [recs[i] for i in used], chunksize=64)
    by = {i: r for i, r in zip(used, res)}
    bad = [r["id"] for r in res if not r["ok"]]
    per_step = {k: [sum(by[i][k] for i in s) for s in plan_] for k in ("p1", "p2", "tok1", "tok2", "R")}
    rep = {"windows_used": len(used), "label_rows_equal_c0": len(used) - len(bad), "bad": bad[:20],
           "windows_with_real_order_choice": sum(1 for r in res if r["real_reads"]),
           "per_step": {k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                            "max": int(np.max(v))} for k, v in per_step.items()},
           "steps": args.steps, "seed": args.seed, "rows_per_step": args.rows_per_step, "cpu_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if bad:
        raise SystemExit(f"LABEL_PATH_MISMATCH in {len(bad)} windows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-all", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rows-per-step", type=int, default=64)
    args = ap.parse_args()
    if args.check_all:
        check_all(args)


if __name__ == "__main__":
    main()
