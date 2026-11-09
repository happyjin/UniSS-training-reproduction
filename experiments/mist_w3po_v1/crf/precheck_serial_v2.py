"""SerialCRF v2 precheck (crf_novel_ideas.zh-CN.md §13.10, pre-registered 2026-11-09 21:40 UTC; report:
reports/mist_w3po_v1/serial_crf_v2_precheck.zh-CN.md). Precheck only: no training code, no training.
Same windows as crf/precheck_serial.py (train 500 = sched_v1 sample, dev 300 = sched_dev_v1 sample, seed 20261109);
dev (never trained on) is the primary evidence, train is secondary.

Lattice (§13.10, exactly as pre-registered)
  items     the chunks of the sched records (crf/precheck_serial.py: every chunk a closed G-STAR segment); chunks of the
            same speaker (slot; 1-1 with the speaker in every window) keep their label order.
  slack     rho(c) in {rho_lab(c), rho_lab(c) + 1}; the items of the final read cannot be deferred.
  state     the committed set A. With slack 1, after read r: A_r = {rho_lab <= r-1} ∪ D_r, D_r ⊆ B_r = {rho_lab = r} the
            items committed on time, per speaker a prefix of its label order (the per-speaker lower set). D_{R-1} = B_{R-1}.
            So the lattice is a linear chain over D_0, ..., D_{R-1} (states(r) = all such D_r; index 0 = B_r, the label).
  history   h_r = kappa(A_{r-1}), kappa = the corrected key (rho_lab, source end, source start) of precheck_serial.kappa
            (a function of the set: a deferred item sits at its LABEL position in later histories).
  kap       the sort used here: (rho_lab, commit_ms, source start, chunk), S's own build key (build_sched_windows:
            commit_ms = min(source end + 1 s, window end)). It EQUALS the pre-registered kappa on every chunk outside the
            final read (commit = end + 1 s there), hence on every history (asserted in context() for every state) and every
            non-final write set. Only inside the final read, where the build clamps commit_ms to the window end, S orders
            the read's own items by start; the pre-registered key would reorder them away from S / v8's training target
            in 4.3 % (dev) / 4.1 % (train) of the windows. Found by the S3 (a) write-set test before any GPU run; kap
            keeps the label path == S (and == v8 / C0) at every read.
  read r    writes W_r = (B_{r-1} \\ D_{r-1}) ∪ D_r (the pending items must be written now) in the canonical kappa order
            (= the pending items, then D_r: their rho_lab is smaller), then e_r (<wait>; <eos> at the final read, <wait>
            barred there before the softmax, as Engine.read).
  potential psi_r(D_{r-1} -> D_r) = log p_theta(kappa-ordered W_r ⊕ e_r | x <= t_r, c = "", kappa(A_{r-1})).
  posterior q_tau(path) ∝ exp(tau * sum_r psi_r); log Z by the forward recursion over the set states (exact), checked
            against brute-force enumeration of all rho assignments (S3). tau = 0.5 pre-registered, tau = 1 reported.
  forced reads  reads with one context and one write set (no choice: e.g. B_{r-1} = B_r = ∅) add the same constant to
            every path; they are not scored (the posterior is unchanged; the reported log Z is that of the free reads,
            tested equal to the full one minus the constants).

Scoring protocol = the corrected baseline: at the FINAL read the diarizer posteriors P are fully visible
  train  the whole window's P (training windows have no close-detect time);
  dev    evaluation/online_decode_finalpost.final_cut_ms (the --final-read-post close_detect rule,
         cut = max(t - 1.04 s, min(t, close_detect - 1.04 s))); the reference windows have no detector time either, so the
         rule's no-later-than-t cap is taken (close_detect - 1.04 s >= t, the 93.6 % case of the online dev windows):
         cut = t, i.e. every complete 80 ms frame visible, the partial frame straddling the window end masked.
  non-final reads unchanged (P zeroed after t - 1.04 s). The OLD protocol (final read masked like the others) is also
  scored at the final read (report-only S2 difference).
  Engine path of precheck_serial.score_reqs (Engine.from_merged, audio_embeds, Prompter.ids, left padding + mask).
  Efficiency: per (read, context) ONE forward with the canonical continuation (pending ⊕ all of B_r) gives every write set
  whose on-time part is a kappa-prefix of B_r (log p(e_r) is read at every item boundary); only the non-prefix write sets
  get extra forwards (a greedy cover by maximal sets).

PRE-REGISTERED DECISION RULES (constants below; committed before any GPU run of this script)
  S3 exactness  (a) on the label path (D_r = B_r for all r) the history ids == S_ids[:beta_{r-1}] and the write-set ids ==
                S_ids[beta_{r-1}:beta_r] for every read of every window of sched_v1 and sched_dev_v1 (plus
                precheck_serial's own kappa test); (b) forward log Z, the path masses, the state marginals and the
                deferral marginals == brute-force enumeration on random synthetic instances and on real windows with dummy
                log-probs (every sample window with <= 10 deferrable items), including the forced-read skipping.
                FAIL -> fix the code, no GPU (every scoring run re-checks (a) on its windows before loading the model).
  S1 coverage   (CPU, v8's formal reference decode dev_v8_ctx0_guard_s8, the scorer state of check A
                (crf/diag_order_attr.py), whose short-recall counts are asserted equal to score_text.json).
                Population: every dev scored utterance with b - a <= 1 s, ov50 (others cover >= 50 %), NOT recalled
                (the scorer's recalled() False; check A's 295).
                due window w / due read t_due: check A's (the online window with the largest time overlap; the first read
                with t >= (b - w.start) + 1 s, else w's last read).
                "u written at t_due": a segment of w committed at t_due (write_ms == t_due) with non-empty text, its
                source span widened by 2 s overlapping u, whose normalised text starts or ends with u's normalised
                reference, and not explained away by another scored utterance of the segment's mapped speaker (check A's
                text-match rule, under ANY speaker label).
                DEFERRAL(u) iff  t_due is NOT w's final read  AND  some segment of w with non-empty text was committed at
                t_due ("another item")  AND  u was not written at t_due.
                S1 = #DEFERRAL / #population.  S1 < 0.30 -> STOP.
  S2 signal     (GPU, dev) items c with rho_lab(c) non-final (deferrable) and |B_{rho_lab(c)}| >= 2 (a label multi-item
                read): S2 = mean over these items of q_{tau=0.5}(rho(c) = rho_lab(c) + 1) (the deferred-then-written mass:
                with slack 1 every deferral is written at the next read). S2 < 0.05 -> STOP.
                (the paths compared write exactly the same tokens, so the absolute content probability cancels; unlike
                P2's P_complete this is a schedule posterior.)
  GO iff S3 passes, S1 >= 0.30 and S2 >= 0.05.
  REPORT ONLY: tau = 1 S2; S2 under the old final-read protocol; in a deferred state (context with pending items) the
  structural opening probability of m = the first pending item (first in canonical order) at the start of the next read,
  speaker token alone and speaker x start-time token, with p(e_r) there, vs the on-label-path opening of the same m at its
  label read; S2 subgroups; lattice sizes; S1 on the final-read-post decode (dev_v8_ctx0_guard_finalpost_s8).

    # CPU stage: S3 (a)+(b), tokenisation of off-label histories, lattice sizes -> <root>/cpu (root must be new)
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --cpu-stage --root <new root>
    # S1 (CPU) -> <root>/s1
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --s1 --root <root>
    # scoring (GPU), one model load per shard -> <root>/<split>_shard_KofN
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --score --root <root> --shard K/N --splits dev,train
    # smoke: the first n dev windows with a non-final multi-speaker multi-item read
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --score --root <work dir> --splits dev --limit 2
    # alignment of scored dirs vs the v1 precheck's stored label-order psi (smoke gate)
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --align <scored dir>[,<dir>...]
    # S2, report-only items, decision (CPU) -> <root>/analysis
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --analyse --root <root>
    # plan + analysis on dummy scores, nothing written
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2 --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S

P = S.P
TAU = 0.5
TAUS = (0.5, 1.0)
S1_STOP = 0.30
S2_STOP = 0.05
SHORT_S = 1.0
BRUTE_MAX_DEFERRABLE = 10
DECODE_REF = "eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0_guard_s8"
DECODE_CAND = "eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0_guard_finalpost_s8"
V1_ROOT = "eval_outputs/mist_w3po_v1/crf/serial_precheck_v1"
ALIGN_MED_MAX, ALIGN_MAX = 0.5, 5.0          # smoke gate (run check, not a decision rule): v1 vs v1 noise was 0.26 / 2.97


# ------------------------------------------------------------------------------------------------ lattice
def kap(rec: dict, rho: list[int], H) -> list[int]:
    """the canonical order (module docstring): (rho_lab, commit_ms, source start, chunk index); a function of the set."""
    ch = rec["chunks"]
    return sorted(set(H), key=lambda c: (rho[c], ch[c]["commit_ms"], ch[c]["src_start_ms"], c))


def read_states(rec: dict, rho: list[int], Br, final: bool) -> list[tuple]:
    """The on-time subsets D ⊆ B_r: per speaker (slot) a prefix of its chunks in label order; kappa-ordered tuples.
    Index 0 = all of B_r (the label). The final read: [B_r] only (no deferral)."""
    Br = kap(rec, rho, Br)
    if final:
        return [tuple(Br)]
    by = defaultdict(list)
    for c in Br:                                   # kappa order within a read = label order within a speaker
        by[rec["chunks"][c]["slot"]].append(c)
    ks = list(by)
    out = []
    for js in itertools.product(*[range(len(by[k]), -1, -1) for k in ks]):
        out.append(tuple(kap(rec, rho, [c for k, j in zip(ks, js) for c in by[k][:j]])))
    return out


def lattice(rec: dict) -> dict:
    rho = S.chunk_reads(rec)
    R = rec["n_read"]
    B = [kap(rec, rho, b) for b in S.read_items(rec, rho)]
    return {"rho": rho, "R": R, "B": B, "states": [read_states(rec, rho, B[r], r == R - 1) for r in range(R)]}


def prev_states(lat: dict, r: int) -> list[tuple]:
    return lat["states"][r - 1] if r > 0 else [()]


def context(rec: dict, lat: dict, r: int, i: int) -> tuple[list[int], list[int]]:
    """(history chunks = kappa(A_{r-1}), pending chunks = B_{r-1} \\ D_{r-1} in kappa order) of context i at read r."""
    rho = lat["rho"]
    D = prev_states(lat, r)[i]
    base = [c for c in range(len(rec["chunks"])) if rho[c] <= r - 2]
    hist = kap(rec, rho, base + list(D))
    assert hist == S.kappa(rec, rho, hist), (rec["id"], r, i)       # == the pre-registered key on every history
    pend = [c for c in lat["B"][r - 1] if c not in set(D)] if r > 0 else []
    return hist, pend


def covers(states: list[tuple]) -> list[tuple[tuple, list[tuple[int, int]]]]:
    """A cover of the write sets by forwards: [(the forced on-time items D_max, [(state j, k): state j = D_max[:k]])];
    every state appears exactly once; the first cover is the full B_r (all kappa-prefixes)."""
    idx = {s: j for j, s in enumerate(states)}
    done, out = set(), []
    for s in sorted(states, key=lambda s: (-len(s), idx[s])):
        if s in done:
            continue
        ent = [(idx[s[:k]], k) for k in range(len(s) + 1) if s[:k] in idx and s[:k] not in done]
        done.update(states[j] for j, _ in ent)
        out.append((s, ent))
    return out


def plan_window(rec: dict, lat: dict | None = None, old_final: bool = True, skip_forced: bool = True) -> dict:
    """The forwards of one window. reqs: dicts (r, proto, hist_ids, forced_ids, items, offsets); per read and context
    the cells (j -> (req, boundary)) of the new protocol and, at the final read, of the old one."""
    lat = lat or lattice(rec)
    R = lat["R"]
    reads, reqs = [], []
    for r in range(R):
        ps, cs = prev_states(lat, r), lat["states"][r]
        forced = len(ps) == 1 and len(cs) == 1
        rd = {"r": r, "final": r == R - 1, "forced": forced, "n_prev": len(ps), "n_cur": len(cs), "ctx": []}
        reads.append(rd)
        if forced and skip_forced:
            continue
        cov = covers(cs)
        protos = ("new", "old") if (r == R - 1 and old_final) else ("new",)
        for i in range(len(ps)):
            hist, pend = context(rec, lat, r, i)
            h_ids = S.ids_of(rec, hist)
            n_p = len(S.ids_of(rec, pend))
            cx = {"i": i, "hist": hist, "pend": pend, "hist_len": len(h_ids), "cells": {pr: {} for pr in protos},
                  "first_req": None}
            for s, ent in cov:
                items = list(pend) + list(s)
                lens = [rec["chunks"][c]["tok1"] - rec["chunks"][c]["tok0"] for c in items]
                offs = [int(x) for x in np.cumsum([0] + lens)]
                for pr in protos:
                    q = len(reqs)
                    reqs.append({"r": r, "proto": pr, "hist_ids": h_ids, "forced_ids": S.ids_of(rec, items),
                                 "items": items, "offsets": offs})
                    if cx["first_req"] is None and pr == "new":
                        cx["first_req"] = q
                    for j, k in ent:
                        cx["cells"][pr][j] = (q, offs[len(pend) + k])
            assert all(len(v) == len(cs) for v in cx["cells"].values())
            assert n_p == offs[len(pend)]
            rd["ctx"].append(cx)
    return {"lat": lat, "reads": reads, "reqs": reqs}


def assemble(plan: dict, res: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Per read: psi [n_prev, n_cur] (None for a skipped forced read), psi_old (final read), and per context the
    opener log-probs of every item of its first (full-continuation) forward: [chunk, lp(spk), lp(t_start | spk),
    lp(e_r) just before the item]."""
    out = []
    reqs = plan["reqs"]
    for rd in plan["reads"]:
        o = {"r": rd["r"], "final": rd["final"], "forced": rd["forced"], "psi": None, "psi_old": None, "ctx": []}
        if rd["ctx"]:
            for pr in ("new", "old"):
                if pr not in rd["ctx"][0]["cells"]:
                    continue
                M = np.zeros((rd["n_prev"], rd["n_cur"]))
                for cx in rd["ctx"]:
                    for j, (q, b) in cx["cells"][pr].items():
                        tl, ec = res[q]
                        M[cx["i"], j] = float(np.sum(tl[:b]) + ec[b])
                o["psi" if pr == "new" else "psi_old"] = M.tolist()
            for cx in rd["ctx"]:
                q = cx["first_req"]
                tl, ec = res[q]
                rq = reqs[q]
                op = [[c, float(tl[a]), float(tl[a + 1]), float(ec[a])] for c, a in zip(rq["items"], rq["offsets"])]
                o["ctx"].append({"i": cx["i"], "pend": cx["pend"], "hist_len": cx["hist_len"], "open": op})
        out.append(o)
    return out


# ------------------------------------------------------------------------------------------------ inference
def lse(x, axis=None):
    x = np.asarray(x, np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(y, axis=axis) if axis is not None else float(y.reshape(()))


def psi_list(reads: list[dict], proto: str = "new") -> list[np.ndarray]:
    """psi matrices of the stored reads (a skipped forced read: [[0]]); proto old: the final read's old protocol."""
    out = []
    for rd in reads:
        M = rd["psi_old"] if (proto == "old" and rd["final"] and rd.get("psi_old") is not None) else rd["psi"]
        out.append(np.zeros((1, 1)) if M is None else np.asarray(M, np.float64))
    return out


def forward_backward(psis: list[np.ndarray], tau: float = 1.0) -> dict:
    """Linear chain over the per-read states: psis[r] [n_prev(r), n_cur(r)], n_prev(0) = 1, n_cur(R-1) = 1."""
    R = len(psis)
    alphas, a = [], np.zeros(1)
    for M in psis:
        assert M.shape[0] == len(a), (M.shape, len(a))
        a = lse(a[:, None] + tau * M, axis=0)
        alphas.append(np.atleast_1d(a))
    assert len(a) == 1
    logZ = float(a[0])
    betas = [None] * R
    b = np.zeros(1)
    betas[R - 1] = b
    for r in range(R - 1, 0, -1):
        b = np.atleast_1d(lse(tau * psis[r] + b[None, :], axis=1))
        betas[r - 1] = b
    marg = [np.exp(alphas[r] + betas[r] - logZ) for r in range(R)]
    return {"logZ": logZ, "alpha": alphas, "beta": betas, "marg": marg}


def deferral_marginals(lat: dict, fb: dict) -> dict[int, float]:
    """q(rho(c) = rho_lab(c) + 1) for every deferrable chunk c."""
    out = {}
    for c, r in enumerate(lat["rho"]):
        if r < lat["R"] - 1:
            out[c] = float(sum(q for s, q in zip(lat["states"][r], fb["marg"][r]) if c not in s))
    return out


def path_states(lat: dict, a: list[int]) -> list[int]:
    """the state index per read of an assignment a (chunk -> commit read)."""
    return [lat["states"][r].index(tuple(c for c in lat["B"][r] if a[c] == r)) for r in range(lat["R"])]


def path_score(psis: list[np.ndarray], idx: list[int]) -> float:
    prev, s = 0, 0.0
    for M, j in zip(psis, idx):
        s += float(M[prev, j])
        prev = j
    return s


# ------------------------------------------------------------------------------------------------ brute force (S3 b)
def assignments(rec: dict, lat: dict):
    """every rho with rho(c) in {rho_lab(c), rho_lab(c)+1}, <= the final read, nondecreasing along each speaker's
    label order (enumerated independently of the lattice states)."""
    rho, R = lat["rho"], lat["R"]
    n = len(rec["chunks"])
    free = [c for c in range(n) if rho[c] < R - 1]
    chains = defaultdict(list)
    for c in range(n):
        chains[rec["chunks"][c]["slot"]].append(c)
    pairs = [(x, y) for ch in chains.values() for x, y in zip(ch, ch[1:])]
    for bits in itertools.product((0, 1), repeat=len(free)):
        a = list(rho)
        for c, x in zip(free, bits):
            a[c] += x
        if all(a[x] <= a[y] for x, y in pairs):
            yield a


def brute_paths(rec: dict, lat: dict, psi_fn) -> tuple[list[list[int]], np.ndarray]:
    """psi_fn(r, final, hist_ids, written_ids) -> psi. Histories and write sets built directly from each assignment."""
    rho, R = lat["rho"], lat["R"]
    n = len(rec["chunks"])
    paths, scores = [], []
    for a in assignments(rec, lat):
        sc = 0.0
        for r in range(R):
            hist = kap(rec, rho, [c for c in range(n) if a[c] <= r - 1])
            w = kap(rec, rho, [c for c in range(n) if a[c] == r])
            sc += psi_fn(r, r == R - 1, S.ids_of(rec, hist), S.ids_of(rec, w))
        paths.append(a)
        scores.append(sc)
    return paths, np.asarray(scores, np.float64)


def brute_force(rec: dict, lat: dict, paths: list[list[int]], scores: np.ndarray, tau: float = 1.0) -> dict:
    """the tempered posterior of the enumerated paths: log Z, path masses, deferral and state marginals."""
    rho, R = lat["rho"], lat["R"]
    logZ = lse(tau * scores)
    q = np.exp(tau * scores - logZ)
    dm = {c: 0.0 for c in range(len(rec["chunks"])) if rho[c] < R - 1}
    sm = [defaultdict(float) for _ in range(R)]
    for a, qq in zip(paths, q):
        for c in dm:
            if a[c] == rho[c] + 1:
                dm[c] += qq
        for r in range(R):
            sm[r][tuple(c for c in lat["B"][r] if a[c] == r)] += qq
    return {"logZ": logZ, "q": q, "defer": dm, "state_marg": sm}


def dummy_scores(r: int, final: bool, hist_ids, forced_ids, proto: str = "new") -> tuple[np.ndarray, np.ndarray]:
    """A deterministic dummy 'LM' (a function of the whole prefix, via a rolling hash): tok_lp [m], end_col [m + 1]."""
    def val(h, t):
        return -4.0 * ((hash((h, t)) % 1000003) / 1000003.0)
    h = hash((r, bool(final), proto == "new", tuple(int(x) for x in hist_ids)))
    tl, ec = [], []
    for t in forced_ids:
        ec.append(val(h, -1))
        tl.append(val(h, int(t)))
        h = hash((h, int(t)))
    ec.append(val(h, -1))
    return np.asarray(tl, np.float64), np.asarray(ec, np.float64)


def dummy_psi(r, final, hist_ids, w_ids, proto="new") -> float:
    tl, ec = dummy_scores(r, final, hist_ids, w_ids, proto)
    return float(tl.sum() + ec[-1])


def dummy_res(plan: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    R = plan["lat"]["R"]
    return [dummy_scores(q["r"], q["r"] == R - 1, q["hist_ids"], q["forced_ids"], q["proto"]) for q in plan["reqs"]]


def check_brute(rec: dict, taus=TAUS, tol: float = 1e-9) -> dict:
    """S3 (b) on one window with the dummy LM: forward (all reads scored, and forced reads skipped) vs brute force."""
    lat = lattice(rec)
    full = plan_window(rec, lat, skip_forced=False)
    free = plan_window(rec, lat, skip_forced=True)
    rd_full = assemble(full, dummy_res(full))
    rd_free = assemble(free, dummy_res(free))
    ps_full, ps_free = psi_list(rd_full), psi_list(rd_free)
    const = sum(float(M[0, 0]) for M, rd in zip(ps_full, free["reads"]) if rd["forced"])
    st = Counter()
    worst = 0.0
    paths, scores = brute_paths(rec, lat, dummy_psi)
    idxs = [path_states(lat, a) for a in paths]
    n_paths = math.exp(forward_backward([np.zeros_like(M) for M in ps_full], 1.0)["logZ"])
    for tau in taus:
        bf = brute_force(rec, lat, paths, scores, tau)
        fa, fr = forward_backward(ps_full, tau), forward_backward(ps_free, tau)
        d = [abs(fa["logZ"] - bf["logZ"]), abs(fr["logZ"] + tau * const - bf["logZ"])]
        for idx, qb, sb in zip(idxs, bf["q"], scores):
            d.append(abs(path_score(ps_full, idx) - sb))
            d.append(abs(math.exp(tau * path_score(ps_free, idx) - fr["logZ"]) - qb))
        for which in (fa, fr):
            dm = deferral_marginals(lat, which)
            d += [abs(dm[c] - bf["defer"][c]) for c in bf["defer"]]
            for r in range(lat["R"]):                       # state marginals
                d += [abs(which["marg"][r][j] - bf["state_marg"][r].get(s, 0.0))
                      for j, s in enumerate(lat["states"][r])]
                assert set(bf["state_marg"][r]) <= set(lat["states"][r])
        st["n_paths_equal"] += round(n_paths) == len(paths)
        st["checks"] += 1
        worst = max(worst, max(d))
    st["pass"] = int(worst < tol and st["n_paths_equal"] == st["checks"])
    return {"stats": dict(st), "max_abs_diff": worst, "n_paths": len(paths),
            "deferrable": sum(1 for c, r in enumerate(lat["rho"]) if r < lat["R"] - 1)}


def synthetic_rec(rng: np.random.Generator, n_read: int, n_chunk: int, n_spk: int) -> dict:
    """A random window built like build_sched_windows: commit_ms = min(source end + 1 s, window end), S = the chunks in
    (commit read, commit_ms, source start) order, beta = the committed token count after each read."""
    dur = 1000 * n_read - int(rng.integers(0, 500))
    t_ms = [1000 * (r + 1) for r in range(n_read - 1)] + [dur]
    raw = []
    for _ in range(n_chunk):
        end = int(rng.integers(100, dur + 1))
        raw.append({"slot": int(rng.integers(1, n_spk + 1)), "src_end_ms": end,
                    "src_start_ms": max(0, end - int(rng.integers(80, 1500))), "commit_ms": min(end + 1000, dur)})
    for c in raw:
        c["r"] = next(r for r, t in enumerate(t_ms) if t >= c["commit_ms"])
    raw.sort(key=lambda c: (c["r"], c["commit_ms"], c["src_start_ms"]))
    chunks, ids = [], []
    for ci, c in enumerate(raw):
        p = [S.SPK0 + c["slot"] - 1, S.NEW0 + ci % 200, 1000 + ci, 2000 + int(rng.integers(0, 50)), S.NEW0 + ci % 200 + 1]
        chunks.append({"utt": ci, "slot": c["slot"], "tok0": len(ids), "tok1": len(ids) + len(p), "piece": f"<c{ci}>",
                       "src_start_ms": c["src_start_ms"], "src_end_ms": c["src_end_ms"], "commit_ms": c["commit_ms"]})
        ids += p
    beta = [sum(c["tok1"] - c["tok0"] for c, x in zip(chunks, raw) if x["r"] <= r) for r in range(n_read)]
    return {"id": f"syn{n_read}_{n_chunk}_{n_spk}", "chunks": chunks, "S_ids": ids, "n_read": n_read, "t_ms": t_ms,
            "dur_ms": dur, "beta_heur": beta, "n_tok": len(ids),
            "units": [{"utt": c["utt"], "a_ms": c["src_start_ms"]} for c in chunks]}


# ------------------------------------------------------------------------------------------------ S3 (a)
def label_exact(rec: dict, lat: dict | None = None) -> Counter:
    """Label path: the context of D_{r-1} = B_{r-1} has history ids == S[:beta_{r-1}], no pending item, and the label write
    set (state 0 = B_r) has ids == S[beta_{r-1}:beta_r]; state 0 is the label B_r."""
    lat = lat or lattice(rec)
    beta = [0] + list(rec["beta_heur"])
    st = Counter()
    for r in range(lat["R"]):
        hist, pend = context(rec, lat, r, 0)
        st["reads"] += 1
        st["label_state0"] += list(lat["states"][r][0]) == list(lat["B"][r]) == S.read_items(rec, lat["rho"])[r]
        st["hist_ok"] += S.ids_of(rec, hist) == rec["S_ids"][: beta[r]]
        st["pend_empty"] += not pend
        st["write_ok"] += S.ids_of(rec, pend + list(lat["states"][r][0])) == rec["S_ids"][beta[r]: beta[r + 1]]
    return st


def label_exact_ok(st) -> bool:
    return st["reads"] == st["label_state0"] == st["hist_ok"] == st["pend_empty"] == st["write_ok"]


def check_label_or_die(recs: list[dict]) -> dict:
    st = Counter()
    for rec in recs:
        st.update(label_exact(rec))
    ex = S.check_exact_or_die(recs)
    if not label_exact_ok(st):
        raise SystemExit(f"S3 (a) FAILED on the label path, no scoring: {dict(st)}")
    return {"v2_label_path": dict(st), "v1_kappa": ex}


# ------------------------------------------------------------------------------------------------ S1 (CPU)
def deferral_class(due_final: bool, u_written_at_due: bool, n_other_at_due: int) -> str:
    """S1's pre-registered classification of a missed unit with a due window (DEFERRAL is the only counted class)."""
    if due_final:
        return "DUE_FINAL"
    if u_written_at_due:
        return "U_WRITTEN_AT_DUE"
    if n_other_at_due == 0:
        return "NOTHING_ELSE_WRITTEN_AT_DUE"
    return "DEFERRAL"


def s1_rows(decode: str) -> tuple[list[dict], dict]:
    """the deferral classification of every dev missed ov50 <= 1 s unit of an online decode (module docstring)."""
    from experiments.mist_w3po_v1.crf import diag_order_attr as D

    W = [json.loads(l) for l in open(D.WINDOWS, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(decode) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(D.REF_DEV, encoding="utf-8")]
    W = [w for w in W if w["id"] in H]
    Wid = {w["id"]: w for w in W}
    res, cap, segs = D.run_scorer(W, H, RW)
    ref = json.load(open(Path(decode) / "score_text.json", encoding="utf-8"))
    assert res["short_recall"] == ref["short_recall"], "scorer state does not reproduce score_text.json"
    meta = D.seg_meta(W, H, segs)
    wins_of = defaultdict(list)
    for w in W:
        wins_of[w["meeting"]].append(w)
    rows = []
    for m in sorted(cap):
        st = cap[m]
        sc, mp = st["sc"], st["mp"]
        for rid, u in sc.items():
            d = u["b"] - u["a"]
            if d > SHORT_S or st["rec"][rid]:
                continue
            if not D.ov50_of(u, [v for r2, v in sc.items() if r2 != rid and v["spk"] != u["spk"]]):
                continue
            r = D.norm(u["ref"])
            cls = D.classify_missed(u, rid, st, segs[m], meta[m], wins_of[m], H)["cls"]
            row = {"meeting": m, "rid": rid, "dur": round(d, 3), "le05": d <= 0.5, "cls": cls}

            def explained(spk, x):
                if spk is None:
                    return False
                return any(v["spk"] == spk and rid2 != rid and D.ov(x["a"] - D.COLLAR_S, x["b"] + D.COLLAR_S, v["a"],
                                                                     v["b"]) > 0 and D.contains(D.norm(v["ref"]), r)
                           for rid2, v in sc.items())
            due = D.due_of(u, wins_of[m], H)
            if due["window"] is None:
                row.update({"window": None, "why": "NO_WINDOW", "deferral": False})
                rows.append(row)
                continue
            w = due["window"]
            reads = [x["t"] for x in H[w]["read_log"]]
            i = D.due_read(reads, Wid[w]["start_s"], u["b"])
            t_due = reads[i]
            at = [k for k, x in enumerate(segs[m]) if meta[m][k]["wid"] == w and meta[m][k]["write_ms"] == t_due
                  and x["text"].strip()]
            u_at = [k for k in at if D.ov(segs[m][k]["a"] - D.COLLAR_S, segs[m][k]["b"] + D.COLLAR_S, u["a"], u["b"]) > 0
                    and D.edge_match(D.norm(segs[m][k]["text"]), r) and not explained(mp.get(segs[m][k]["spk"]), segs[m][k])]
            other = [k for k in at if k not in u_at]
            final = i == len(reads) - 1
            why = deferral_class(final, bool(u_at), len(other))
            row.update({"window": w, "due_read": i, "n_reads": len(reads), "t_due_ms": t_due, "due_final": final,
                        "n_written_at_due": len(at), "u_written_at_due": bool(u_at), "n_other_at_due": len(other),
                        "why": why, "deferral": why == "DEFERRAL"})
            rows.append(row)
    return rows, {"short_recall_reproduced": True, "decode": decode}


def s1_summary(rows: list[dict]) -> dict:
    def g(sel):
        x = [r for r in rows if sel(r)]
        n = len(x)
        return {"n": n, "deferral": sum(r["deferral"] for r in x), "share": sum(r["deferral"] for r in x) / max(n, 1),
                "why": dict(Counter(r["why"] for r in x)),
                "deferral_by_cls": dict(Counter(r["cls"] for r in x if r["deferral"])),
                "cls": dict(Counter(r["cls"] for r in x))}
    out = {"all_ov50_le1_missed": g(lambda r: True), "ov50_le05_missed": g(lambda r: r["le05"]),
           "ORDER_OMISSION": g(lambda r: r["cls"] == "ORDER_OMISSION")}
    for c in sorted({r["cls"] for r in rows}):
        out[f"cls_{c}"] = g(lambda r, c=c: r["cls"] == c)
    return out


def run_s1(args) -> None:
    root = Path(args.root)
    out = root / "s1"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    rep = {"rule": f"S1 = share of DEFERRAL among the dev missed ov50 <= 1 s units of {DECODE_REF}; < {S1_STOP} -> STOP"}
    per = {}
    for name, dec in (("ref", DECODE_REF), ("cand_finalpost", DECODE_CAND)):
        if not (Path(dec) / "score_text.json").exists():
            rep[name] = {"missing": dec}
            continue
        rows, info = s1_rows(dec)
        per[name] = rows
        rep[name] = {**info, **s1_summary(rows)}
        print(f"[S1 {name}] {json.dumps(rep[name]['all_ov50_le1_missed'])}", flush=True)
    # consistency with check A's stored rows (the reference decode)
    ca = [json.loads(l) for l in open(S.CHECK_A, encoding="utf-8")]
    want = {(x["rid"], x["cls"]) for x in ca if x["ov50"]}
    got = {(x["rid"], x["cls"]) for x in per["ref"]}
    rep["ref_matches_check_a_rows"] = want == got
    assert want == got, (len(want), len(got))
    s1 = rep["ref"]["all_ov50_le1_missed"]["share"]
    rep["S1"] = s1
    rep["S1_stop"] = s1 < S1_STOP
    rep["elapsed_s"] = time.time() - t0
    out.mkdir(parents=True)
    (out / "s1.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, rows in per.items():
        with open(out / f"s1_rows_{name}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"S1_DONE {out} S1={s1:.4f} STOP={rep['S1_stop']}", flush=True)


# ------------------------------------------------------------------------------------------------ scoring (GPU)
def final_read_post(post: np.ndarray, t_ms: int, split: str) -> np.ndarray:
    """The corrected baseline's final-read P (module docstring): train = the whole window; dev = the --final-read-post
    rule with the detector time unknown (close_detect - 1.04 s >= t): frames >= t // 80 zeroed."""
    from experiments.mist_w3po_v1.core.engine import LOOKAHEAD_MS
    from experiments.mist_w3po_v1.evaluation.online_decode_finalpost import final_cut_ms

    p = post.copy()
    if split == "dev":
        p[final_cut_ms(t_ms, t_ms + LOOKAHEAD_MS) // 80:] = 0.0
    elif split != "train":
        raise ValueError(split)
    return p


def make_items(reqs: list[tuple], wavs: dict, split: str) -> list:
    """ReadItems of (rec, r, hist ids, forced ids, proto) requests: the new protocol's final read = final_read_post with
    full=True (audio_embeds applies no second mask); everything else = the engine default (P zeroed after t - 1.04 s).
    The akey includes the protocol at the final read (items sharing an akey share the speaker-stream embedding)."""
    from experiments.mist_w3po_v1.core.engine import ReadItem
    from experiments.mist_w3po_v1.rl.rollout import load_window

    its = []
    for rec, r, hist, forced, proto in reqs:
        if rec["id"] not in wavs:
            wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
        wav, post = wavs[rec["id"]]
        t = rec["t_ms"][r]
        final = r == rec["n_read"] - 1
        if final and proto == "new":
            its.append(ReadItem(wav=wav, post=final_read_post(post, t, split), t_ms=t, final=True,
                                committed=list(hist) + list(forced), ctx="", akey=f"{rec['id']}|{t}|new", full=True))
        else:
            its.append(ReadItem(wav=wav, post=post, t_ms=t, final=final, committed=list(hist) + list(forced), ctx="",
                                akey=f"{rec['id']}|{t}|{'old' if final else 'nf'}"))
    return its


def score_reqs_v2(eng, reqs: list[tuple], batch: int, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """precheck_serial.score_reqs with the protocol per request (make_items); same forward, slicing and <wait> bar."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD

    wavs, out = {}, []
    emb_dtype = eng.emb_layer.weight.dtype
    for s0 in range(0, len(reqs), batch):
        part = reqs[s0: s0 + batch]
        its = make_items(part, wavs, split)
        ms = [len(q[3]) for q in part]
        with torch.no_grad():
            ae = eng.audio_embeds(its, emb_dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            B, L, M = len(its), max(len(p) for p in prompts), max(ms)
            ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            for i, p in enumerate(prompts):
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(eng.dev), mask.to(eng.dev)
            emb = eng.emb_layer(ids)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            h = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                  return_dict=True).last_hidden_state[:, L - M - 1:]
            lg = eng.logits(h).to(eng.dev)
            for i, it in enumerate(its):
                if it.final:
                    lg[i, :, S.WAIT_ID] = -1e9                      # Engine.read line 246
            lsm = torch.log_softmax(lg, -1)
            for i, it in enumerate(its):
                tl, ec = S.slice_scores(lsm[i], ms[i], part[i][3], S.EOS_ID if it.final else S.WAIT_ID)
                out.append((tl.double().cpu().numpy(), ec.double().cpu().numpy()))
        del h, lg, lsm
    return out


def smoke_select(recs: list[dict], n: int) -> list[dict]:
    """the first n windows with a non-final label read of >= 2 items of >= 2 speakers."""
    out = []
    for rec in recs:
        lat = lattice(rec)
        if any(len({rec["chunks"][c]["slot"] for c in lat["B"][r]}) >= 2 for r in range(lat["R"] - 1)):
            out.append(rec)
        if len(out) >= n:
            break
    return out


def score_split(eng, recs: list[dict], args, out: Path, split: str) -> dict:
    t0 = time.time()
    n_fw = 0
    with open(out / "windows.jsonl", "x", encoding="utf-8") as fh:
        for k, rec in enumerate(recs):
            plan = plan_window(rec)
            reqs = [(rec, q["r"], q["hist_ids"], q["forced_ids"], q["proto"]) for q in plan["reqs"]]
            res = score_reqs_v2(eng, reqs, args.batch, split) if reqs else []
            n_fw += len(reqs)
            fh.write(json.dumps({"id": rec["id"], "R": plan["lat"]["R"], "n_fw": len(reqs),
                                 "states": [[list(s) for s in st] for st in plan["lat"]["states"]],
                                 "reads": assemble(plan, res)}) + "\n")
            if (k + 1) % 20 == 0:
                print(f"[{split}] {k + 1}/{len(recs)} windows, {n_fw} forwards, {time.time() - t0:.0f}s", flush=True)
    return {"windows": len(recs), "forwards": n_fw, "scoring_s": time.time() - t0}


def run_score(args) -> None:
    import torch

    from experiments.mist_w3po_v1.core.engine import Engine

    root = Path(args.root)
    sh = P.parse_shard(args.shard)
    jobs = []
    for split in [s for s in args.splits.split(",") if s]:
        recs = S.sample(split)
        if sh:
            recs = P.shard_select(recs, *sh)
        if args.limit:
            recs = smoke_select(recs, args.limit)
        name = f"{split}_shard_{sh[0]}of{sh[1]}" if sh else (f"{split}_limit{args.limit}" if args.limit else split)
        out = root / name
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        ex = check_label_or_die(recs)                     # S3 (a) on these windows, before any GPU use
        jobs.append((split, recs, out, ex))
    for split, recs, out, ex in jobs:
        out.mkdir(parents=True)
        plans = [plan_window(r) for r in recs]
        (out / "meta.json").write_text(json.dumps({"split": split, "sched": S.SCHEDS[split], "n": S.NS[split],
                                                   "seed": S.SEED, "ids": [r["id"] for r in recs], "exactness": ex,
                                                   "forwards_planned": sum(len(p["reqs"]) for p in plans),
                                                   "args": vars(args)}, indent=1))
    t0 = time.time()
    eng = Engine.from_merged(args.thinker, "zh2en", args.max_gib)
    load_s = time.time() - t0
    for split, recs, out, ex in jobs:
        bad = [r["id"] for r in recs[:50] if eng.tok(r["S"], add_special_tokens=False)["input_ids"] != r["S_ids"]]
        if bad:
            raise SystemExit(f"tokenizer mismatch between the sched data and the engine: {bad[:5]}")
        st = score_split(eng, recs, args, out, split)
        peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
        st.update({"load_s": load_s, "peak_alloc_gib": peak})
        (out / "done.json").write_text(json.dumps(st, indent=1))
        print(f"SERIAL_V2_SCORED {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ alignment (run check)
def v1_label_psi(split: str) -> dict:
    """(id, r) -> the v1 precheck's stored psi of the label order (old protocol at every read)."""
    out = {}
    for d in sorted(Path(V1_ROOT).glob(f"{split}_shard_*of*")):
        for l in open(d / "reads.jsonl", encoding="utf-8"):
            x = json.loads(l)
            assert x["orders"][0]["pi"] == x["label"]
            out[(x["id"], x["r"])] = x["orders"][0]["psi"]
    return out


def align_rows(rows: list[dict], v1: dict) -> list[dict]:
    """label-path psi (context 0 -> state 0) vs v1: non-final reads (protocol unchanged) and final reads (old protocol)."""
    out = []
    for w in rows:
        for rd in w["reads"]:
            if (w["id"], rd["r"]) not in v1 or rd["psi"] is None:
                continue
            new = rd["psi_old"][0][0] if rd["final"] else rd["psi"][0][0]
            out.append({"id": w["id"], "r": rd["r"], "final": rd["final"], "v2": new, "v1": v1[(w["id"], rd["r"])],
                        "abs": abs(new - v1[(w["id"], rd["r"])])})
    return out


def run_align(args) -> None:
    res = {}
    for d in [Path(x) for x in args.align.split(",") if x]:
        split = json.loads((d / "meta.json").read_text())["split"]
        rows = [json.loads(l) for l in open(d / "windows.jsonl", encoding="utf-8")]
        al = align_rows(rows, v1_label_psi(split))
        a = [x["abs"] for x in al]
        med = float(np.median(a)) if a else None
        ok = bool(a) and med < ALIGN_MED_MAX and max(a) < ALIGN_MAX
        fin = all(np.isfinite(v).all() for w in rows for rd in w["reads"] for k in ("psi", "psi_old")
                  if rd.get(k) is not None for v in [np.asarray(rd[k])])
        res[str(d)] = {"reads": len(a), "median_abs": med, "max_abs": max(a) if a else None, "finite": fin,
                       "pass": bool(ok and fin)}
    print(json.dumps(res, indent=1))
    print(f"ALIGN {'PASS' if all(v['pass'] for v in res.values()) else 'FAIL'}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def window_analysis(rec: dict, row: dict) -> dict:
    lat = lattice(rec)
    assert [[list(s) for s in st] for st in lat["states"]] == row["states"], rec["id"]
    ui = S.unit_info(rec)
    R = lat["R"]
    q = {}
    for proto in ("new", "old"):
        ps = psi_list(row["reads"], proto)
        for tau in TAUS:
            fb = forward_backward(ps, tau)
            q[(proto, tau)] = {"defer": deferral_marginals(lat, fb), "logZ_free": fb["logZ"],
                               "q_label": math.exp(tau * path_score(ps, [0] * R) - fb["logZ"])}
    items = []
    for c, r in enumerate(lat["rho"]):
        if r >= R - 1:
            continue
        ch = rec["chunks"][c]
        Br = lat["B"][r]
        u = ui[ch["utt"]]
        items.append({"c": c, "r": r, "n_items": len(Br), "n_spk": len({rec["chunks"][x]["slot"] for x in Br}),
                      "short_ov": u["dur_s"] <= SHORT_S and u["ov50"], "ov50": u["ov50"], "dur_s": u["dur_s"],
                      **{f"q_{p}_{t}": q[(p, t)]["defer"][c] for p in ("new", "old") for t in TAUS}})
    # opening probabilities: deferred-state contexts (pending items) vs the on-label-path opening of the same m
    on_path = {}
    for rd in row["reads"]:
        for cx in rd["ctx"]:
            if cx["i"] == 0 and not cx["pend"]:
                for c, a, b, e in cx["open"]:
                    on_path[c] = (a, b, e)
    opens = []
    for rd in row["reads"]:
        r = rd["r"]
        for cx in rd["ctx"]:
            if not cx["pend"]:
                continue
            m = cx["pend"][0]
            c, a, b, e = cx["open"][0]
            assert c == m
            ctl = on_path.get(m)
            mu = ui[rec["chunks"][m]["utt"]]
            opens.append({"r": r, "m": m, "n_pend": len(cx["pend"]), "final": rd["final"],
                          "m_multi": len(lat["B"][lat["rho"][m]]) >= 2, "m_short_ov": mu["dur_s"] <= SHORT_S and mu["ov50"],
                          "spk_lp": a, "t_lp": b, "e0_lp": e,
                          "ctl_spk_lp": None if ctl is None else ctl[0], "ctl_t_lp": None if ctl is None else ctl[1],
                          "ctl_e_lp": None if ctl is None else ctl[2]})
    sizes = {"reads": R, "forced_reads": sum(rd["forced"] for rd in row["reads"]),
             "states": sum(len(s) for s in lat["states"]), "max_states": max(len(s) for s in lat["states"]),
             "deferrable": len(items), "n_fw": row["n_fw"],
             "log10_paths": forward_backward([np.zeros_like(M) for M in psi_list(row["reads"])], 1.0)["logZ"] / math.log(10)}
    return {"id": rec["id"], "items": items, "opens": opens, "sizes": sizes,
            "q": {f"{p}_{t}": {k: v for k, v in q[(p, t)].items() if k != "defer"} for p in ("new", "old") for t in TAUS}}


def s2_of(items: list[dict], key: str) -> dict:
    x = np.asarray([it[key] for it in items], np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "share_ge_0.05": float((x >= 0.05).mean()), "share_ge_0.1": float((x >= 0.1).mean()),
            "share_ge_0.5": float((x >= 0.5).mean())}


def split_report(per: list[dict]) -> dict:
    items = [it for w in per for it in w["items"]]
    pop = [it for it in items if it["n_items"] >= 2]
    rep = {"windows": len(per), "deferrable_items": len(items), "S2_population": len(pop)}
    for p in ("new", "old"):
        for t in TAUS:
            k = f"q_{p}_{t}"
            rep[f"S2[{p},tau={t}]"] = s2_of(pop, k)
    k = f"q_new_{TAU}"
    rep["S2_subgroups(new, tau=0.5)"] = {
        "multi_speaker_read": s2_of([it for it in pop if it["n_spk"] >= 2], k),
        "single_speaker_read": s2_of([it for it in pop if it["n_spk"] < 2], k),
        "short_ov_items": s2_of([it for it in pop if it["short_ov"]], k),
        "not_short_ov_items": s2_of([it for it in pop if not it["short_ov"]], k),
        "all_deferrable_items(single-item reads too)": s2_of(items, k),
        "single_item_reads": s2_of([it for it in items if it["n_items"] < 2], k)}
    opens = [o for w in per for o in w["opens"]]

    def og(sel):
        g = [o for o in opens if sel(o)]
        e = lambda f: S.qstats([math.exp(f(o)) for o in g if f(o) is not None])
        paired = [o for o in g if o["ctl_spk_lp"] is not None]
        return {"contexts": len(g),
                "p_open_spk": e(lambda o: o["spk_lp"]), "p_open_spk_t": e(lambda o: o["spk_lp"] + o["t_lp"]),
                "p_e_at_start": e(lambda o: o["e0_lp"]),
                "ctl_on_label_path_p_open_spk": e(lambda o: o["ctl_spk_lp"]),
                "ctl_on_label_path_p_open_spk_t": e(lambda o: None if o["ctl_spk_lp"] is None else o["ctl_spk_lp"] + o["ctl_t_lp"]),
                "ctl_on_label_path_p_e_before": e(lambda o: o["ctl_e_lp"]),
                "paired_spk_deferred_minus_ctl": S.qstats([math.exp(o["spk_lp"]) - math.exp(o["ctl_spk_lp"]) for o in paired]),
                "paired_spk_t_deferred_minus_ctl": S.qstats([math.exp(o["spk_lp"] + o["t_lp"])
                                                             - math.exp(o["ctl_spk_lp"] + o["ctl_t_lp"]) for o in paired])}
    rep["deferred_state_opening"] = {"all": og(lambda o: True), "m_from_multi_item_read": og(lambda o: o["m_multi"]),
                                     "m_short_ov": og(lambda o: o["m_short_ov"]), "single_pending": og(lambda o: o["n_pend"] == 1),
                                     "next_read_final": og(lambda o: o["final"]), "next_read_nonfinal": og(lambda o: not o["final"])}
    sz = [w["sizes"] for w in per]
    rep["lattice_sizes"] = {k: S.qstats([s[k] for s in sz]) for k in ("reads", "forced_reads", "states", "max_states",
                                                                       "deferrable", "n_fw", "log10_paths")}
    rep["lattice_sizes"]["total_forwards"] = int(sum(s["n_fw"] for s in sz))
    rep["lattice_sizes"]["total_states"] = int(sum(s["states"] for s in sz))
    rep["lattice_sizes"]["max_states_any_read"] = int(max(s["max_states"] for s in sz))
    rep["q_label_path"] = {f"{p}_{t}": S.qstats([w["q"][f"{p}_{t}"]["q_label"] for w in per]) for p in ("new", "old")
                           for t in TAUS}
    rep["expected_deferrals_per_window(new, tau=0.5)"] = S.qstats([sum(it[k] for it in w["items"]) for w in per])
    return rep


def run_analyse(args) -> None:
    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cpu = json.loads((root / "cpu" / "cpu_stage.json").read_text())
    s1 = json.loads((root / "s1" / "s1.json").read_text())
    rep = {"S3_pass": cpu["S3_pass"], "S1": s1["S1"], "S1_stop": s1["S1_stop"]}
    per_all = {}
    for split in ("dev", "train"):
        dirs = sorted(root.glob(f"{split}_shard_*of*"))
        if not dirs:
            continue
        rows, ids = [], []
        for d in dirs:
            if not (d / "done.json").exists():
                raise SystemExit(f"shard not finished: {d}")
            ids += json.loads((d / "meta.json").read_text())["ids"]
            rows += [json.loads(l) for l in open(d / "windows.jsonl", encoding="utf-8")]
        recs = {r["id"]: r for r in S.sample(split)}
        per = [window_analysis(recs[w["id"]], w) for w in rows]
        per_all[split] = per
        a = split_report(per)
        a["complete_sample"] = sorted(ids) == sorted(recs)
        al = align_rows(rows, v1_label_psi(split))
        a["alignment_vs_v1_label_psi"] = {"nonfinal": S.qstats([x["abs"] for x in al if not x["final"]]),
                                          "final_old_protocol": S.qstats([x["abs"] for x in al if x["final"]])}
        # the alternative population (final-read items of multi-item reads included with mass 0), report only
        pop = [it for w in per for it in w["items"] if it["n_items"] >= 2]
        n_fin = 0
        for w in per:
            lat = lattice(recs[w["id"]])
            n_fin += len(lat["B"][-1]) if len(lat["B"][-1]) >= 2 else 0
        a["S2_including_final_read_items_as_0(new, tau=0.5)"] = {
            "n": len(pop) + n_fin, "mean": float(sum(it[f"q_new_{TAU}"] for it in pop) / max(len(pop) + n_fin, 1))}
        rep[split] = a
    s2 = rep["dev"][f"S2[new,tau={TAU}]"]["mean"]
    rep["decision"] = {"S3_pass": cpu["S3_pass"], "S1": s1["S1"], "S1_stop": s1["S1"] < S1_STOP,
                       "S2_dev": s2, "S2_stop": s2 < S2_STOP,
                       "GO": bool(cpu["S3_pass"] and s1["S1"] >= S1_STOP and s2 >= S2_STOP)}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for split, per in per_all.items():
        with open(out / f"per_window_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(w) + "\n" for w in per)
    print(json.dumps(rep["decision"], indent=1))
    print(f"SERIAL_V2_ANALYSED {out} GO={rep['decision']['GO']}", flush=True)


# ------------------------------------------------------------------------------------------------ CPU stage
def run_cpu_stage(args) -> None:
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {root}")
    t0 = time.time()
    rep = {"S3a": {}, "S3b": {}, "tokenisation": {}, "sizes": {}}
    ok = True
    for split in ("dev", "train"):                                # (a) every window of both scheds
        st, ex = Counter(), Counter()
        for rec in S.load_all(S.SCHEDS[split]):
            st.update(label_exact(rec))
            ex.update(S.exactness_window(rec)["stats"])
        rep["S3a"][split] = {"v2_label_path": dict(st), "v2_pass": label_exact_ok(st),
                             "v1_kappa_reads_ok": ex["kappa_ids_ok"], "v1_reads": ex["reads"],
                             "v1_pass": S.exactness_ok(ex)}
        ok &= label_exact_ok(st) and S.exactness_ok(ex)
        print(f"[S3a {split}] {json.dumps(rep['S3a'][split])}", flush=True)
    rng = np.random.default_rng(20261109)                        # (b) synthetic
    syn = Counter()
    worst = 0.0
    for k in range(300):
        rec = synthetic_rec(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)), int(rng.integers(1, 4)))
        c = check_brute(rec)
        syn["instances"] += 1
        syn["pass"] += c["stats"]["pass"]
        syn["paths_total"] += c["n_paths"]
        worst = max(worst, c["max_abs_diff"])
    rep["S3b"]["synthetic"] = {**syn, "max_abs_diff": worst}
    ok &= syn["pass"] == syn["instances"]
    tok = tokenizer()
    for split in ("dev", "train"):                                # (b) real windows, dummy LM
        st = Counter()
        worst = 0.0
        tk = Counter()
        sizes = []
        for rec in S.sample(split):
            lat = lattice(rec)
            plan = plan_window(rec, lat)
            sizes.append({"reads": lat["R"], "states": sum(len(s) for s in lat["states"]),
                          "max_states": max(len(s) for s in lat["states"]), "forwards": len(plan["reqs"]),
                          "forwards_new": sum(q["proto"] == "new" for q in plan["reqs"]),
                          "forced_reads": sum(rd["forced"] for rd in plan["reads"]),
                          "deferrable": sum(1 for c, r in enumerate(lat["rho"]) if r < lat["R"] - 1)})
            for rd in plan["reads"]:                             # off-label histories tokenise as their chunk ids
                for cx in rd["ctx"]:
                    tk["contexts"] += 1
                    tk["ok"] += tok(S.string_of(rec, cx["hist"]), add_special_tokens=False)["input_ids"] == \
                        S.ids_of(rec, cx["hist"])
                    w = cx["pend"] + list(lat["B"][rd["r"]])
                    tk["write_ok"] += tok(S.string_of(rec, w), add_special_tokens=False)["input_ids"] == S.ids_of(rec, w)
            if sizes[-1]["deferrable"] > BRUTE_MAX_DEFERRABLE:
                st["skipped_too_big"] += 1
                continue
            c = check_brute(rec)
            st["windows"] += 1
            st["pass"] += c["stats"]["pass"]
            st["paths_total"] += c["n_paths"]
            worst = max(worst, c["max_abs_diff"])
        rep["S3b"][split] = {**st, "max_abs_diff": worst}
        rep["tokenisation"][split] = dict(tk)
        ok &= st["pass"] == st["windows"] and tk["ok"] == tk["write_ok"] == tk["contexts"]
        rep["sizes"][split] = {"windows": len(sizes), **{k: S.qstats([s[k] for s in sizes]) for k in sizes[0]},
                               "total_forwards": sum(s["forwards"] for s in sizes),
                               "total_forwards_new": sum(s["forwards_new"] for s in sizes),
                               "total_states": sum(s["states"] for s in sizes)}
        print(f"[S3b {split}] {json.dumps(rep['S3b'][split])} tok {dict(tk)}", flush=True)
    rep["S3_pass"] = bool(ok)
    rep["elapsed_s"] = time.time() - t0
    out = root / "cpu"
    out.mkdir(parents=True)
    (out / "cpu_stage.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep["sizes"], indent=1))
    print(f"SERIAL_V2_CPU_STAGE_DONE {out} S3_pass={ok}", flush=True)
    if not ok:
        raise SystemExit("S3 FAILED: fix the code before any GPU run")


def run_dry(args) -> None:
    """plan sizes + the analysis on dummy scores (normalised to look like log-probs), nothing written."""
    for split in ("dev", "train"):
        recs = S.sample(split)[:40]
        per = []
        for rec in recs:
            plan = plan_window(rec)
            reads = assemble(plan, dummy_res(plan))
            row = {"id": rec["id"], "R": plan["lat"]["R"], "n_fw": len(plan["reqs"]),
                   "states": [[list(s) for s in st] for st in plan["lat"]["states"]], "reads": reads}
            per.append(window_analysis(rec, row))
        rep = split_report(per)
        print(split, json.dumps({k: rep[k] for k in ("windows", "deferrable_items", "S2_population",
                                                      f"S2[new,tau={TAU}]")}), flush=True)
    print("DRY_RUN_OK (dummy scores; nothing written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu-stage", action="store_true")
    g.add_argument("--s1", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--align", default="")
    g.add_argument("--analyse", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default="")
    ap.add_argument("--thinker", default=P.V8)
    ap.add_argument("--splits", default="dev,train")
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-gib", type=int, default=80)
    args = ap.parse_args()
    if not (args.dry_run or args.align) and not args.root:
        raise SystemExit("--root is required")
    if args.cpu_stage:
        run_cpu_stage(args)
    elif args.s1:
        run_s1(args)
    elif args.score:
        run_score(args)
    elif args.align:
        run_align(args)
    elif args.analyse:
        run_analyse(args)
    else:
        run_dry(args)


if __name__ == "__main__":
    main()
