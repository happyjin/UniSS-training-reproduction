"""SerialCRF v2' precheck (crf_novel_ideas.zh-CN.md §13.11; report reports/mist_w3po_v1/serial_crf_v2p_precheck.zh-CN.md).
Precheck only: no training code, no training. Extends crf/precheck_serial_v2.py (imported, unchanged) by ONE binary
within-read order latent; everything else (items, slack 1, set state, kappa history, kap write order, final-read P
protocol, tau = 0.5, the 300 dev / 500 train sample windows) is v2's.

Order latent (pre-registered)
  o_r in {canon, newfirst} per read r.
  canon     v2's write order of W_r = pending ⊕ D_r (kap order).
  newfirst  first N_r = the items of D_r whose speaker (slot) shares no slot with any pending (carried-over) item, in kap
            order, then the canonical (kap) order of the rest (pending, then D_r \\ N_r). Per-speaker label order is kept.
  o_r is a real choice only when the pending set and N_r are both non-empty; otherwise newfirst == canon and the read
  counts once (one order, no double counting).
  edge      psi_r,o(D_{r-1} -> D_r) = log p_theta(W_r in order o ⊕ e_r | x <= t_r, kappa(A_{r-1})); the marginal edge
            potential psi_r = logsumexp_o psi_r,o (the specified v2' potential, = the tau = 1 edge term).
  tempering q_tau(rho, o) ∝ exp(tau * sum_r psi_r,o_r) on the JOINT latent (the posterior a brute-force enumeration over
            (rho assignment x orders) defines); the forward's edge term is logsumexp_o(tau * psi_r,o), equal to
            tau * psi_r at tau = 1. Report only: the alternative "logsumexp, then temper" (q ∝ exp(tau sum_r psi_r), order
            conditional untempered).
  exactness kappa depends on the set only, so o_r enters read r's own term only: the forward recursion over the set
            states is v2's with the per-edge logsumexp. The latent is legitimate because the scorer stamps every item
            of a read with the same write time (evaluation is invariant to the within-read order).
  label     all D_r = B_r and o_r = canon (no pending item on the label path, so no order choice): == S at every read.

PRE-REGISTERED DECISION RULES (constants below; report §1 committed before any S1' / S2' result)
  S3' exactness (CPU)  (a) v2's label-path check on every read of every dev and train window (label path == S), and
                no order choice on the label path; (b) forward log Z, path masses, state marginals, deferral marginals
                and the order marginals q(o_r = newfirst), q(o_r real) == a brute-force enumeration over (rho assignment
                x orders) built independently of the lattice, tol 1e-9 (synthetic instances; real sample windows with
                <= 10 deferrable items, dummy log-probs); negative tests (v2 lattice without the order latent, a wrong
                newfirst order, an off-by-one boundary) must be caught. FAIL -> fix the code, no GPU.
  S1' coverage (CPU)   population = v2 S1's (dev <= 1 s ov50 missed units) of the final-read causal P decode
                dev_v8_ctx0_guard_finalpost_s8 (primary; v2 S1 there: 263 / 121 DEFERRAL); dev_v8_ctx0_guard_s8 report
                only. v2's S1 code path (the scorer state asserted to reproduce score_text.json; v2's fields asserted
                equal to v2's stored S1 rows). CHAIN(u) iff (a) v2's DEFERRAL holds at t_due; (b) in read t_due + 1 of w
                at least one non-empty segment that is not u is committed; (c) u is not written in read t_due + 1 nor
                in any later read of w (v2 S1's text-match rule, any speaker label). S1' = #CHAIN / #population.
                S1' < 0.15 -> STOP.
  S2' signal (GPU)     S2'a = on dev reads where o_r is a real choice (some lattice edge of read r has two orders), the
                mean over reads of q_{tau=0.5}(o_r = newfirst) marginalised over the lattice. S2'a < 0.05 -> STOP.
                S2'b = over every dev newfirst context (unique (window, read, context, N_r): the teacher-forced prefix
                history ⊕ N_r with m = the first pending item still pending), the mean of p(e_r) right after N_r
                (<wait>; <eos> at the final read, <wait> barred). S2'b < 0.02 -> STOP. Also its median and the matching
                label-path p(e_r) at the end of the same read (nothing pending).
  GO iff S3' passes, S1' >= 0.15, S2'a >= 0.05 and S2'b >= 0.02 (one setting each, no sweeps).
  REPORT ONLY: tau = 1; q(o_r = newfirst | o_r real); the alternative tempering; p(open m) right after N_r; the v2 S2
  quantities on the v2' lattice; train; alignment of the label-path psi (and every canonical cell) with v2's saved psi
  (smoke gate: median |dpsi| < 0.5, max < 5 nats).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_serial_v2p
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --cpu-stage --root <new root> [--procs 48]
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --s1 --root <root>
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --score --root <root> --shard K/8 --splits dev,train  # GPU
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --score --root <work dir> --splits dev --limit 2   # smoke
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --align <scored dir>[,<dir>...]
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --analyse --root <root>
    python -m experiments.mist_w3po_v1.crf.precheck_serial_v2p --dry-run
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
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

P = S.P
TAU = 0.5
TAUS = (0.5, 1.0)
S1P_STOP = 0.15
S2A_STOP = 0.05
S2B_STOP = 0.02
SHORT_S = 1.0
BRUTE_MAX_DEFERRABLE = 10
BRUTE_MAX_PATHS = 400_000           # (rho x order) paths; larger real windows are skipped and counted
DECODE_PRIMARY = V.DECODE_CAND      # dev_v8_ctx0_guard_finalpost_s8 (the adopted final-read causal P protocol)
DECODE_REPORT = V.DECODE_REF        # dev_v8_ctx0_guard_s8
V2_ROOT = "eval_outputs/mist_w3po_v1/crf/serial_v2_precheck"
ALIGN_MED_MAX, ALIGN_MAX = 0.5, 5.0


# ------------------------------------------------------------------------------------------------ lattice
def slot(rec: dict, c: int) -> int:
    return rec["chunks"][c]["slot"]


def newfirst(rec: dict, rho: list[int], pend, D) -> tuple[tuple, tuple] | None:
    """(N, rest): N = the items of D whose slot is no pending item's slot (kap order), rest = kap(pend ∪ D \\ N).
    None when pend or N is empty (no real choice: newfirst == canon)."""
    if not pend:
        return None
    ps = {slot(rec, c) for c in pend}
    N = [c for c in V.kap(rec, rho, D) if slot(rec, c) not in ps]
    if not N:
        return None
    n = set(N)
    return tuple(N), tuple(V.kap(rec, rho, list(pend) + [c for c in D if c not in n]))


def plan_window_p(rec: dict, lat: dict | None = None, nf_fn=newfirst, skip_forced: bool = True) -> dict:
    """v2's plan (new protocol only: the canonical cells) plus, per context, the newfirst forwards: cx["nf"] =
    {j: (req, end boundary, after-N boundary, sequence, |N|)}. A newfirst sequence that is an item-prefix of a longer
    one of the same context shares its forward (the longest first)."""
    lat = lat or V.lattice(rec)
    plan = V.plan_window(rec, lat, old_final=False, skip_forced=skip_forced)
    reqs, rho = plan["reqs"], lat["rho"]
    for rd in plan["reads"]:
        r = rd["r"]
        cs = lat["states"][r]
        for cx in rd["ctx"]:
            seqs = []
            for j, D in enumerate(cs):
                nf = nf_fn(rec, rho, cx["pend"], D)
                if nf is not None:
                    seqs.append((j, tuple(nf[0]) + tuple(nf[1]), len(nf[0])))
            h_ids = S.ids_of(rec, cx["hist"])
            cells, done = {}, set()
            for j, s, nl in sorted(seqs, key=lambda x: (-len(x[1]), x[0])):
                if j in done:
                    continue
                items = list(s)
                lens = [rec["chunks"][c]["tok1"] - rec["chunks"][c]["tok0"] for c in items]
                offs = [int(x) for x in np.cumsum([0] + lens)]
                q = len(reqs)
                reqs.append({"r": r, "proto": "new", "kind": "nf", "hist_ids": h_ids, "forced_ids": S.ids_of(rec, items),
                             "items": items, "offsets": offs})
                for j2, s2, nl2 in seqs:
                    if j2 not in done and s[: len(s2)] == s2:
                        cells[j2] = (q, offs[len(s2)], offs[nl2], s2, nl2)
                        done.add(j2)
            assert len(cells) == len(seqs)
            cx["nf"] = cells
        if rd["forced"]:
            assert not rd["ctx"] or not any(cx["nf"] for cx in rd["ctx"])
    return plan


def assemble_p(plan: dict, res: list) -> list[dict]:
    """v2's assemble (psi = the canonical order) plus per read: psi_nf [n_prev, n_cur] (None where no real choice),
    nf_ctx = the unique newfirst contexts (i, N): m = the first pending item, lp(e_r), lp(spk of m), lp(t of m | spk)
    right after N, and label_e_lp = lp(e_r) at the end of the read on the label path (context 0 -> state 0)."""
    out = V.assemble(plan, res)
    for o, rd in zip(out, plan["reads"]):
        o["psi_nf"], o["nf_ctx"], o["label_e_lp"] = None, [], None
        if not rd["ctx"]:
            continue
        q0, b0 = rd["ctx"][0]["cells"]["new"][0]
        assert rd["ctx"][0]["i"] == 0
        o["label_e_lp"] = float(res[q0][1][b0])
        if not any(cx["nf"] for cx in rd["ctx"]):
            continue
        M = [[None] * rd["n_cur"] for _ in range(rd["n_prev"])]
        nfc = {}
        for cx in rd["ctx"]:
            for j, (q, b, bn, s, nl) in cx["nf"].items():
                tl, ec = res[q]
                M[cx["i"]][j] = float(np.sum(tl[:b]) + ec[b])
                key = (cx["i"], tuple(s[:nl]))
                if nl >= len(s):                               # only a negative-control order can end with N
                    continue
                if key not in nfc:
                    nfc[key] = {"i": cx["i"], "N": list(s[:nl]), "m": s[nl], "pend": list(cx["pend"]),
                                "e_lp": float(ec[bn]), "spk_lp": float(tl[bn]), "t_lp": float(tl[bn + 1]), "js": []}
                nfc[key]["js"].append(j)
        o["psi_nf"] = M
        o["nf_ctx"] = list(nfc.values())
    return out


# ------------------------------------------------------------------------------------------------ inference
def mats(rd: dict) -> tuple[np.ndarray, np.ndarray | None]:
    C = np.zeros((1, 1)) if rd["psi"] is None else np.asarray(rd["psi"], np.float64)
    if rd.get("psi_nf") is None:
        return C, None
    Nf = np.asarray([[np.nan if v is None else v for v in row] for row in rd["psi_nf"]], np.float64)
    return C, Nf


def edge_terms(reads: list[dict], tau: float, mode: str = "joint") -> list[np.ndarray]:
    """per read the tempered edge term: joint = logsumexp_o(tau psi_o); lse_temper = tau logsumexp_o psi_o."""
    out = []
    for rd in reads:
        C, Nf = mats(rd)
        if Nf is None:
            out.append(tau * C)
            continue
        Nn = np.where(np.isnan(Nf), -np.inf, Nf)
        if mode == "joint":
            out.append(np.logaddexp(tau * C, tau * Nn))
        elif mode == "lse_temper":
            out.append(tau * np.logaddexp(C, Nn))
        else:
            raise ValueError(mode)
    return out


def forward_p(reads: list[dict], tau: float, mode: str = "joint") -> dict:
    """v2's forward-backward over the set states with the per-edge order logsumexp; fb["order"][r] = None or
    {q_real: q(o_r is a real choice), q_nf: q(o_r = newfirst)} (edge marginal x the order conditional)."""
    E = edge_terms(reads, tau, mode)
    fb = V.forward_backward(E, 1.0)
    order = []
    for r, rd in enumerate(reads):
        C, Nf = mats(rd)
        if Nf is None or np.isnan(Nf).all():
            order.append(None)
            continue
        a = fb["alpha"][r - 1] if r > 0 else np.zeros(1)
        qe = np.exp(a[:, None] + E[r] + fb["beta"][r][None, :] - fb["logZ"])
        real = ~np.isnan(Nf)
        Nn = np.where(real, Nf, 0.0)
        if mode == "joint":
            cond = np.where(real, np.exp(tau * Nn - E[r]), 0.0)
        else:
            cond = np.where(real, np.exp(Nn - np.logaddexp(C, np.where(real, Nf, -np.inf))), 0.0)
        order.append({"q_real": float(qe[real].sum()), "q_nf": float((qe * cond)[real].sum())})
    fb["order"] = order
    return fb


def n_paths_lattice(reads: list[dict]) -> float:
    """number of (rho x order) paths: the forward with psi = 0 (an edge with a real choice counts twice)."""
    E = []
    for rd in reads:
        C, Nf = mats(rd)
        E.append(np.zeros_like(C) if Nf is None else np.where(np.isnan(Nf), 0.0, math.log(2.0)))
    return math.exp(V.forward_backward(E, 1.0)["logZ"])


def path_score_p(reads: list[dict], idx: list[int], orders) -> float:
    prev, s = 0, 0.0
    for rd, j, o in zip(reads, idx, orders):
        C, Nf = mats(rd)
        v = C[prev, j] if o == 0 else (np.nan if Nf is None else Nf[prev, j])
        if not np.isfinite(v):
            raise KeyError("order not in the lattice")
        s += float(v)
        prev = j
    return s


# ------------------------------------------------------------------------------------------------ brute force (S3' b)
def brute_orders(rec: dict, rho: list[int], r: int, W: list[int]) -> list[tuple[int, list[int]]]:
    """the distinct within-read orders of write set W at read r, built without the lattice: canon = sorted by the kap
    key; newfirst = a stable sort that puts first the on-time items (rho_lab = r) whose slot no carried-over item
    (rho_lab = r - 1) has."""
    ch = rec["chunks"]
    key = lambda c: (rho[c], ch[c]["commit_ms"], ch[c]["src_start_ms"], c)     # noqa: E731
    canon = sorted(W, key=key)
    carried = {ch[c]["slot"] for c in W if rho[c] == r - 1}
    first = [c for c in W if rho[c] == r and ch[c]["slot"] not in carried]
    out = [(0, canon)]
    if carried and first:
        fs = set(first)
        out.append((1, sorted(W, key=lambda c: (0 if c in fs else 1,) + key(c))))
    return out


def brute_paths_p(rec: dict, lat: dict, psi_fn, max_paths: int = BRUTE_MAX_PATHS):
    """every (rho assignment x per-read order): psi_fn(r, final, hist_ids, written_ids in order). None if too many."""
    rho, R = lat["rho"], lat["R"]
    n = len(rec["chunks"])
    paths, scores = [], []
    for a in V.assignments(rec, lat):
        opts = []
        for r in range(R):
            hist = V.kap(rec, rho, [c for c in range(n) if a[c] <= r - 1])
            W = [c for c in range(n) if a[c] == r]
            h = S.ids_of(rec, hist)
            opts.append([(o, psi_fn(r, r == R - 1, h, S.ids_of(rec, w))) for o, w in brute_orders(rec, rho, r, W)])
        for combo in itertools.product(*opts):
            paths.append((a, tuple(o for o, _ in combo)))
            scores.append(sum(s for _, s in combo))
            if len(paths) > max_paths:
                return None
    return paths, np.asarray(scores, np.float64)


def brute_force_p(rec: dict, lat: dict, paths, scores: np.ndarray, tau: float) -> dict:
    rho, R = lat["rho"], lat["R"]
    logZ = V.lse(tau * scores)
    q = np.exp(tau * scores - logZ)
    dm = {c: 0.0 for c in range(len(rec["chunks"])) if rho[c] < R - 1}
    sm = [defaultdict(float) for _ in range(R)]
    om = [{"q_real": 0.0, "q_nf": 0.0, "any": False} for _ in range(R)]
    for (a, os_), qq in zip(paths, q):
        for c in dm:
            if a[c] == rho[c] + 1:
                dm[c] += qq
        for r in range(R):
            sm[r][tuple(c for c in lat["B"][r] if a[c] == r)] += qq
            W = [c for c in range(len(a)) if a[c] == r]
            if len(brute_orders(rec, rho, r, W)) == 2:
                om[r]["q_real"] += qq
                om[r]["any"] = True
            if os_[r] == 1:
                om[r]["q_nf"] += qq
    return {"logZ": logZ, "q": q, "defer": dm, "state_marg": sm, "order": om}


def check_brute_p(rec: dict, taus=TAUS, tol: float = 1e-9, nf_fn=newfirst, max_paths: int = BRUTE_MAX_PATHS) -> dict:
    """S3' (b) on one window with v2's dummy LM: lattice (all reads scored, and forced reads skipped) vs brute force."""
    lat = V.lattice(rec)
    full = plan_window_p(rec, lat, nf_fn=nf_fn, skip_forced=False)
    free = plan_window_p(rec, lat, nf_fn=nf_fn, skip_forced=True)
    rd_full = assemble_p(full, V.dummy_res(full))
    rd_free = assemble_p(free, V.dummy_res(free))
    n_lat = n_paths_lattice(rd_full)
    if n_lat > max_paths:
        return {"skipped": True, "n_paths": n_lat}
    bp = brute_paths_p(rec, lat, V.dummy_psi, max_paths)
    if bp is None:
        return {"skipped": True, "n_paths": n_lat}
    paths, scores = bp
    const = sum(float(mats(rd)[0][0, 0]) for rd, pr in zip(rd_full, free["reads"]) if pr["forced"])
    worst, st = 0.0, Counter()
    try:
        idxs = [V.path_states(lat, a) for a, _ in paths]
        lat_scores = [path_score_p(rd_full, idx, os_) for idx, (_, os_) in zip(idxs, paths)]
    except KeyError:
        return {"skipped": False, "stats": {"pass": 0, "order_missing": 1}, "max_abs_diff": float("inf"),
                "n_paths": len(paths)}
    n_real_reads = sum(1 for x in brute_force_p(rec, lat, paths, scores, 1.0)["order"] if x["any"])
    for tau in taus:
        bf = brute_force_p(rec, lat, paths, scores, tau)
        fa, fr = forward_p(rd_full, tau), forward_p(rd_free, tau)
        d = [abs(fa["logZ"] - bf["logZ"]), abs(fr["logZ"] + tau * const - bf["logZ"])]
        d += [abs(ls - sb) for ls, sb in zip(lat_scores, scores)]
        for idx, (_, os_), qb in zip(idxs, paths, bf["q"]):
            d.append(abs(math.exp(tau * path_score_p(rd_free, idx, os_) - fr["logZ"]) - qb))
        for which in (fa, fr):
            dm = V.deferral_marginals(lat, which)
            d += [abs(dm[c] - bf["defer"][c]) for c in bf["defer"]]
            for r in range(lat["R"]):
                d += [abs(which["marg"][r][j] - bf["state_marg"][r].get(s, 0.0)) for j, s in enumerate(lat["states"][r])]
                if not set(bf["state_marg"][r]) <= set(lat["states"][r]):
                    d.append(float("inf"))
                lo, bo = which["order"][r], bf["order"][r]
                if lo is None:
                    d += [bo["q_real"], bo["q_nf"]] + ([float("inf")] if bo["any"] else [])
                else:
                    d += [abs(lo["q_real"] - bo["q_real"]), abs(lo["q_nf"] - bo["q_nf"])]
        st["checks"] += 1
        worst = max(worst, max(d))
    st["n_paths_equal"] = int(round(n_lat) == len(paths))
    st["pass"] = int(worst < tol and st["n_paths_equal"] == 1)
    return {"skipped": False, "stats": dict(st), "max_abs_diff": worst, "n_paths": len(paths),
            "real_choice_reads": n_real_reads,
            "deferrable": sum(1 for c, r in enumerate(lat["rho"]) if r < lat["R"] - 1)}


def wrong_nf_all_new_first(rec, rho, pend, D):
    """negative control (report only: equal to newfirst when N = D): ALL of D first (no slot condition), then pend."""
    if not pend or not D:
        return None
    return tuple(V.kap(rec, rho, D)), tuple(V.kap(rec, rho, pend))


def wrong_nf_pending_head(rec, rho, pend, D):
    """negative control (differs from newfirst at every real choice): the first pending item, then N, then the rest."""
    nf = newfirst(rec, rho, pend, D)
    if nf is None:
        return None
    N, rest = nf
    return (rest[0],) + tuple(N), tuple(rest[1:])


# ------------------------------------------------------------------------------------------------ S1' (CPU)
def s1p_class(deferral: bool, u_next: bool, u_next_same: bool, u_later: bool, u_later_same: bool,
              n_other_next: int) -> str:
    """the pre-registered S1' class of a missed unit (CHAIN is the only counted class)."""
    if not deferral:
        return "NOT_DEFERRAL"
    if u_next:
        return "U_WRITTEN_NEXT_SAME_LABEL" if u_next_same else "U_WRITTEN_NEXT_OTHER_LABEL"
    if u_later:
        return "U_WRITTEN_LATER_SAME_LABEL" if u_later_same else "U_WRITTEN_LATER_OTHER_LABEL"
    if n_other_next == 0:
        return "NOTHING_ELSE_WRITTEN_AT_NEXT"
    return "CHAIN"


def trace_order_check(H: dict) -> dict:
    """the within-read write order is in the traces: per window, the concatenated per-read text (read_log.new) == raw,
    and the segment list (the Parser's emission order) has non-decreasing write_ms."""
    st = Counter()
    for h in H.values():
        st["windows"] += 1
        st["raw_eq_concat_reads"] += "".join(x["new"] for x in h["read_log"]) == h["raw"]
        w = [s["write_ms"] for s in h["segments"]]
        st["segments_in_emission_order"] += w == sorted(w)
    return dict(st)


def s1p_rows(decode: str) -> tuple[list[dict], dict]:
    """v2's s1_rows (same scorer path, same rules, same assertion) plus the CHAIN fields."""
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

            def u_match(k):
                x = segs[m][k]
                return (D.ov(x["a"] - D.COLLAR_S, x["b"] + D.COLLAR_S, u["a"], u["b"]) > 0
                        and D.edge_match(D.norm(x["text"]), r) and not explained(mp.get(x["spk"]), x))
            due = D.due_of(u, wins_of[m], H)
            if due["window"] is None:
                row.update({"window": None, "why": "NO_WINDOW", "deferral": False})
                row.update({"chain": False, "why2": "NOT_DEFERRAL"})
                rows.append(row)
                continue
            w = due["window"]
            reads = [x["t"] for x in H[w]["read_log"]]
            i = D.due_read(reads, Wid[w]["start_s"], u["b"])
            t_due = reads[i]
            at = [k for k, x in enumerate(segs[m]) if meta[m][k]["wid"] == w and meta[m][k]["write_ms"] == t_due
                  and x["text"].strip()]
            u_at = [k for k in at if u_match(k)]
            other = [k for k in at if k not in u_at]
            final = i == len(reads) - 1
            why = V.deferral_class(final, bool(u_at), len(other))
            row.update({"window": w, "due_read": i, "n_reads": len(reads), "t_due_ms": t_due, "due_final": final,
                        "n_written_at_due": len(at), "u_written_at_due": bool(u_at), "n_other_at_due": len(other),
                        "why": why, "deferral": why == "DEFERRAL"})
            # ---- S1' (the chain), only defined after a DEFERRAL (then read i + 1 exists)
            if why == "DEFERRAL":
                nxt = reads[i + 1]
                mine = [k for k, x in enumerate(segs[m]) if meta[m][k]["wid"] == w and x["text"].strip()]
                at_n = [k for k in mine if meta[m][k]["write_ms"] == nxt]
                u_n = [k for k in at_n if u_match(k)]
                u_l = [k for k in mine if meta[m][k]["write_ms"] > nxt and u_match(k)]
                o_n = [k for k in at_n if k not in u_n]
                same = lambda ks: any(mp.get(segs[m][k]["spk"]) == u["spk"] for k in ks)   # noqa: E731
                why2 = s1p_class(True, bool(u_n), same(u_n), bool(u_l), same(u_l), len(o_n))
                u_else = [k for k, x in enumerate(segs[m]) if meta[m][k]["wid"] != w and x["text"].strip() and u_match(k)]
                first_next = None
                if at_n:
                    k0 = min(at_n, key=lambda k: meta[m][k]["si"])
                    first_next = "u" if k0 in u_n else "other"
                row.update({"next_read": i + 1, "next_final": i + 1 == len(reads) - 1, "n_written_at_next": len(at_n),
                            "n_other_at_next": len(o_n), "u_written_at_next": bool(u_n), "u_written_later": bool(u_l),
                            "u_written_in_other_window": bool(u_else), "first_written_at_next": first_next,
                            "why2": why2, "chain": why2 == "CHAIN"})
            else:
                row.update({"why2": "NOT_DEFERRAL", "chain": False})
            rows.append(row)
    return rows, {"short_recall_reproduced": True, "decode": decode, "trace_order": trace_order_check(H)}


V2_FIELDS = ("meeting", "rid", "dur", "le05", "cls", "window", "due_read", "n_reads", "t_due_ms", "due_final",
             "n_written_at_due", "u_written_at_due", "n_other_at_due", "why", "deferral")


def s1p_summary(rows: list[dict]) -> dict:
    def g(sel):
        x = [r for r in rows if sel(r)]
        n = len(x)
        ch = [r for r in x if r["chain"]]
        return {"n": n, "deferral": sum(r["deferral"] for r in x), "chain": len(ch), "share": len(ch) / max(n, 1),
                "deferral_why2": dict(Counter(r["why2"] for r in x if r["deferral"])),
                "chain_by_cls": dict(Counter(r["cls"] for r in ch)),
                "chain_next_final": sum(r["next_final"] for r in ch),
                "chain_u_in_other_window": sum(r["u_written_in_other_window"] for r in ch),
                "chain_n_other_at_next": S.qstats([r["n_other_at_next"] for r in ch])}
    out = {"all_ov50_le1_missed": g(lambda r: True), "ov50_le05_missed": g(lambda r: r["le05"])}
    for c in sorted({r["cls"] for r in rows}):
        out[f"cls_{c}"] = g(lambda r, c=c: r["cls"] == c)
    return out


def run_s1(args) -> None:
    root = Path(args.root)
    out = root / "s1"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    rep = {"rule": f"S1' = share of CHAIN among the dev missed ov50 <= 1 s units of {DECODE_PRIMARY}; < {S1P_STOP} -> STOP"}
    per = {}
    for name, dec, v2name in (("primary_finalpost", DECODE_PRIMARY, "cand_finalpost"), ("report_ref", DECODE_REPORT, "ref")):
        rows, info = s1p_rows(dec)
        v2rows = [json.loads(l) for l in open(Path(V2_ROOT) / "s1" / f"s1_rows_{v2name}.jsonl", encoding="utf-8")]
        mine = [{k: r[k] for k in V2_FIELDS if k in r} for r in rows]
        theirs = [{k: r[k] for k in V2_FIELDS if k in r} for r in v2rows]
        assert mine == theirs, f"v2 S1 fields differ from v2's stored rows ({name})"
        per[name] = rows
        rep[name] = {**info, "v2_rows_reproduced": True, **s1p_summary(rows)}
        print(f"[S1' {name}] {json.dumps(rep[name]['all_ov50_le1_missed'])}", flush=True)
    s1 = rep["primary_finalpost"]["all_ov50_le1_missed"]["share"]
    rep["S1p"] = s1
    rep["S1p_stop"] = s1 < S1P_STOP
    rep["elapsed_s"] = time.time() - t0
    out.mkdir(parents=True)
    (out / "s1.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, rows in per.items():
        with open(out / f"s1_rows_{name}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"S1P_DONE {out} S1'={s1:.4f} STOP={rep['S1p_stop']}", flush=True)


# ------------------------------------------------------------------------------------------------ scoring (GPU)
def smoke_select_p(recs: list[dict], n: int) -> list[dict]:
    """the first n windows with a real order choice somewhere in the lattice."""
    out = []
    for rec in recs:
        plan = plan_window_p(rec)
        if any(cx["nf"] for rd in plan["reads"] for cx in rd["ctx"]):
            out.append(rec)
        if len(out) >= n:
            break
    return out


def score_split(eng, recs: list[dict], args, out: Path, split: str) -> dict:
    t0 = time.time()
    n_fw = n_nf = 0
    with open(out / "windows.jsonl", "x", encoding="utf-8") as fh:
        for k, rec in enumerate(recs):
            plan = plan_window_p(rec)
            reqs = [(rec, q["r"], q["hist_ids"], q["forced_ids"], q["proto"]) for q in plan["reqs"]]
            res = V.score_reqs_v2(eng, reqs, args.batch, split) if reqs else []
            n_fw += len(reqs)
            n_nf += sum(q.get("kind") == "nf" for q in plan["reqs"])
            fh.write(json.dumps({"id": rec["id"], "R": plan["lat"]["R"], "n_fw": len(reqs),
                                 "states": [[list(s) for s in st] for st in plan["lat"]["states"]],
                                 "reads": assemble_p(plan, res)}) + "\n")
            if (k + 1) % 10 == 0:
                print(f"[{split}] {k + 1}/{len(recs)} windows, {n_fw} forwards, {time.time() - t0:.0f}s", flush=True)
    return {"windows": len(recs), "forwards": n_fw, "forwards_nf": n_nf, "scoring_s": time.time() - t0}


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
            recs = smoke_select_p(recs, args.limit)
        name = f"{split}_shard_{sh[0]}of{sh[1]}" if sh else (f"{split}_limit{args.limit}" if args.limit else split)
        out = root / name
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        ex = V.check_label_or_die(recs)                   # S3 (a) on these windows, before any GPU use
        jobs.append((split, recs, out, ex))
    for split, recs, out, ex in jobs:
        out.mkdir(parents=True)
        plans = [plan_window_p(r) for r in recs]
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
        print(f"SERIAL_V2P_SCORED {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ alignment (run check)
def v2_psi(split: str) -> dict:
    """id -> v2's saved reads (psi = the canonical cells, new protocol)."""
    out = {}
    for d in sorted(Path(V2_ROOT).glob(f"{split}_shard_*of*")):
        for l in open(d / "windows.jsonl", encoding="utf-8"):
            x = json.loads(l)
            out[x["id"]] = x
    return out


def align_rows(rows: list[dict], v2: dict) -> dict:
    lab, cells = [], []
    for w in rows:
        if w["id"] not in v2:
            continue
        old = v2[w["id"]]
        assert old["states"] == w["states"], w["id"]
        for a, b in zip(w["reads"], old["reads"]):
            if a["psi"] is None or b["psi"] is None:
                continue
            A, B = np.asarray(a["psi"]), np.asarray(b["psi"])
            lab.append({"final": a["final"], "abs": float(abs(A[0, 0] - B[0, 0]))})
            cells += list(np.abs(A - B).ravel())
    return {"label": lab, "cells": cells}


def run_align(args) -> None:
    res = {}
    for d in [Path(x) for x in args.align.split(",") if x]:
        split = json.loads((d / "meta.json").read_text())["split"]
        rows = [json.loads(l) for l in open(d / "windows.jsonl", encoding="utf-8")]
        al = align_rows(rows, v2_psi(split))
        a = [x["abs"] for x in al["label"]]
        med = float(np.median(a)) if a else None
        ok = bool(a) and med < ALIGN_MED_MAX and max(a) < ALIGN_MAX
        fin = True
        for w in rows:
            for rd in w["reads"]:
                C, Nf = mats(rd)
                fin &= bool(np.isfinite(C).all() and (Nf is None or np.isfinite(Nf[~np.isnan(Nf)]).all()))
        n_nf = sum(len(rd["nf_ctx"]) for w in rows for rd in w["reads"])
        res[str(d)] = {"label_reads": len(a), "median_abs": med, "max_abs": max(a) if a else None,
                       "all_cells": S.qstats(al["cells"]), "finite": fin, "nf_contexts": n_nf,
                       "pass": bool(ok and fin and n_nf > 0)}
    print(json.dumps(res, indent=1))
    print(f"ALIGN {'PASS' if all(v['pass'] for v in res.values()) else 'FAIL'}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def window_analysis(rec: dict, row: dict) -> dict:
    lat = V.lattice(rec)
    assert [[list(s) for s in st] for st in lat["states"]] == row["states"], rec["id"]
    ui = S.unit_info(rec)
    R = lat["R"]
    reads = row["reads"]
    res = {}
    for mode in ("joint", "lse_temper"):
        for tau in TAUS:
            fb = forward_p(reads, tau, mode)
            res[(mode, tau)] = fb
    orders = []
    for r, rd in enumerate(reads):
        if res[("joint", TAU)]["order"][r] is None:
            continue
        o = {"r": r, "final": rd["final"]}
        for (mode, tau), fb in res.items():
            x = fb["order"][r]
            o[f"q_nf_{mode}_{tau}"] = x["q_nf"]
            o[f"q_real_{mode}_{tau}"] = x["q_real"]
        orders.append(o)
    ctxs = []
    for rd in reads:
        for c in rd["nf_ctx"]:
            mu = ui[rec["chunks"][c["m"]]["utt"]]
            ctxs.append({"r": rd["r"], "final": rd["final"], "i": c["i"], "nN": len(c["N"]), "n_pend": len(c["pend"]),
                         "m_short_ov": mu["dur_s"] <= SHORT_S and mu["ov50"], "e_lp": c["e_lp"], "spk_lp": c["spk_lp"],
                         "t_lp": c["t_lp"], "label_e_lp": rd["label_e_lp"]})
    dms = {t: V.deferral_marginals(lat, res[("joint", t)]) for t in TAUS}
    items = []
    for c, r in enumerate(lat["rho"]):
        if r >= R - 1:
            continue
        Br = lat["B"][r]
        u = ui[rec["chunks"][c]["utt"]]
        items.append({"c": c, "r": r, "n_items": len(Br), "n_spk": len({slot(rec, x) for x in Br}),
                      "short_ov": u["dur_s"] <= SHORT_S and u["ov50"], **{f"q_{t}": dms[t][c] for t in TAUS}})
    q_label = {f"{t}": math.exp(t * path_score_p(reads, [0] * R, [0] * R) - res[("joint", t)]["logZ"]) for t in TAUS}
    sizes = {"reads": R, "states": sum(len(s) for s in lat["states"]), "n_fw": row["n_fw"],
             "real_choice_reads": len(orders), "nf_contexts": len(ctxs),
             "log10_paths": math.log10(n_paths_lattice(reads))}
    return {"id": rec["id"], "orders": orders, "ctxs": ctxs, "items": items, "q_label": q_label, "sizes": sizes}


def exps(x):
    return [math.exp(v) for v in x]


def split_report(per: list[dict]) -> dict:
    orders = [o for w in per for o in w["orders"]]
    ctxs = [c for w in per for c in w["ctxs"]]
    rep = {"windows": len(per), "real_choice_reads": len(orders), "nf_contexts": len(ctxs)}
    for mode in ("joint", "lse_temper"):
        for t in TAUS:
            rep[f"S2a[{mode},tau={t}]"] = S.qstats([o[f"q_nf_{mode}_{t}"] for o in orders])
            rep[f"q_real[{mode},tau={t}]"] = S.qstats([o[f"q_real_{mode}_{t}"] for o in orders])
            rep[f"q_nf_given_real[{mode},tau={t}](pooled)"] = (
                sum(o[f"q_nf_{mode}_{t}"] for o in orders) / max(sum(o[f"q_real_{mode}_{t}"] for o in orders), 1e-300))
    k = f"q_nf_joint_{TAU}"
    rep["S2a_subgroups"] = {"final_read": S.qstats([o[k] for o in orders if o["final"]]),
                            "nonfinal_read": S.qstats([o[k] for o in orders if not o["final"]])}

    def cg(sel):
        g = [c for c in ctxs if sel(c)]
        pe = exps([c["e_lp"] for c in g])
        le = exps([c["label_e_lp"] for c in g])
        return {"contexts": len(g), "p_e_after_N": S.qstats(pe), "label_path_p_e_end_of_read": S.qstats(le),
                "paired_minus_label": S.qstats([a - b for a, b in zip(pe, le)]),
                "share_p_e_ge_0.02": float(np.mean([p >= 0.02 for p in pe])) if pe else None,
                "p_open_m_spk_after_N": S.qstats(exps([c["spk_lp"] for c in g])),
                "p_open_m_spk_t_after_N": S.qstats(exps([c["spk_lp"] + c["t_lp"] for c in g])),
                "share_p_e_gt_p_open_spk": float(np.mean([c["e_lp"] > c["spk_lp"] for c in g])) if g else None}
    rep["S2b"] = {"all": cg(lambda c: True), "nonfinal_read(<wait>)": cg(lambda c: not c["final"]),
                  "final_read(<eos>)": cg(lambda c: c["final"]), "m_short_ov": cg(lambda c: c["m_short_ov"]),
                  "single_pending": cg(lambda c: c["n_pend"] == 1)}
    items = [it for w in per for it in w["items"]]
    pop = [it for it in items if it["n_items"] >= 2]
    rep["v2_S2_on_v2p_lattice"] = {f"tau={t}": V.s2_of(pop, f"q_{t}") for t in TAUS}
    rep["v2_S2_on_v2p_lattice"]["short_ov_items(tau=0.5)"] = V.s2_of([it for it in pop if it["short_ov"]], f"q_{TAU}")
    rep["q_label_path"] = {t: S.qstats([w["q_label"][t] for w in per]) for t in ("0.5", "1.0")}
    sz = [w["sizes"] for w in per]
    rep["lattice_sizes"] = {k: S.qstats([s[k] for s in sz]) for k in sz[0]} if sz else {}
    rep["lattice_sizes"]["total_forwards"] = int(sum(s["n_fw"] for s in sz))
    return rep


def run_analyse(args) -> None:
    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cpu = json.loads((root / "cpu" / "cpu_stage.json").read_text())
    s1 = json.loads((root / "s1" / "s1.json").read_text())
    rep = {"S3p_pass": cpu["S3p_pass"], "S1p": s1["S1p"]}
    per_all = {}
    v2pw = {}
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
        al = align_rows(rows, v2_psi(split))
        a["alignment_vs_v2_psi"] = {"label_nonfinal": S.qstats([x["abs"] for x in al["label"] if not x["final"]]),
                                    "label_final": S.qstats([x["abs"] for x in al["label"] if x["final"]]),
                                    "all_canonical_cells": S.qstats(al["cells"])}
        f = Path(V2_ROOT) / "analysis" / f"per_window_{split}.jsonl"   # v2's own S2 on the same items (report only)
        if f.exists():
            v2pw = {x["id"]: x for x in map(json.loads, open(f, encoding="utf-8"))}
            v2items = [it for x in v2pw.values() for it in x["items"] if it["n_items"] >= 2]
            a["v2_S2_saved(new,tau=0.5)"] = V.s2_of(v2items, f"q_new_{TAU}")
        rep[split] = a
    d = rep["dev"]
    s2a = d[f"S2a[joint,tau={TAU}]"].get("mean", 0.0)
    s2b = d["S2b"]["all"]["p_e_after_N"].get("mean", 0.0)
    rep["decision"] = {"S3p_pass": cpu["S3p_pass"], "S1p": s1["S1p"], "S1p_stop": s1["S1p"] < S1P_STOP,
                       "S2a_dev": s2a, "S2a_stop": s2a < S2A_STOP, "S2b_dev": s2b, "S2b_stop": s2b < S2B_STOP,
                       "GO": bool(cpu["S3p_pass"] and s1["S1p"] >= S1P_STOP and s2a >= S2A_STOP and s2b >= S2B_STOP)}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for split, per in per_all.items():
        with open(out / f"per_window_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(w) + "\n" for w in per)
    print(json.dumps(rep["decision"], indent=1))
    print(f"SERIAL_V2P_ANALYSED {out} GO={rep['decision']['GO']}", flush=True)


# ------------------------------------------------------------------------------------------------ CPU stage
_CACHE: dict = {}          # filled in the parent before the fork pool starts (workers inherit it)


def cached(kind: str, split: str) -> list[dict]:
    if (kind, split) not in _CACHE:
        _CACHE[(kind, split)] = S.load_all(S.SCHEDS[split]) if kind == "all" else S.sample(split)
    return _CACHE[(kind, split)]


def _tok():
    if "tok" not in _CACHE:
        from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer
        _CACHE["tok"] = tokenizer()
    return _CACHE["tok"]


def _s3a_chunk(args):
    split, lo, hi = args
    st, ex_ = Counter(), Counter()
    recs = cached("all", split)[lo:hi]
    for rec in recs:
        lat = V.lattice(rec)
        st.update(V.label_exact(rec, lat))
        ex_.update(S.exactness_window(rec)["stats"])
        for r in range(lat["R"]):                          # no order choice on the label path
            hist, pend = V.context(rec, lat, r, 0)
            st["label_nf_choice"] += newfirst(rec, lat["rho"], pend, lat["states"][r][0]) is not None
    return dict(st), dict(ex_)


def _brute_job(args):
    kind, payload = args
    if kind == "syn":
        seed, k = payload
        rng = np.random.default_rng([seed, k])
        rec = V.synthetic_rec(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)), int(rng.integers(1, 4)))
    else:
        split, idx = payload
        rec = cached("sample", split)[idx]
    c = check_brute_p(rec)
    return kind, payload, c


def _sample_job(args):
    """sizes + tokenisation of the newfirst write sequences of one sample window."""
    split, idx = args
    rec = cached("sample", split)[idx]
    tok = _tok()
    lat = V.lattice(rec)
    plan = plan_window_p(rec, lat)
    tk = Counter()
    for q in plan["reqs"]:
        if q.get("kind") == "nf":
            tk["nf_forwards"] += 1
            tk["nf_tok_ok"] += tok(S.string_of(rec, q["items"]), add_special_tokens=False)["input_ids"] == q["forced_ids"]
    size = {"reads": lat["R"], "states": sum(len(s) for s in lat["states"]), "forwards": len(plan["reqs"]),
            "forwards_nf": tk["nf_forwards"], "nf_contexts": sum(len({(cx["i"], v[3][:v[4]]) for v in cx["nf"].values()})
                                                                  for rd in plan["reads"] for cx in rd["ctx"]),
            "real_choice_reads": sum(any(cx["nf"] for cx in rd["ctx"]) for rd in plan["reads"]),
            "deferrable": sum(1 for c, r in enumerate(lat["rho"]) if r < lat["R"] - 1)}
    return split, idx, size, dict(tk)


def run_cpu_stage(args) -> None:
    from multiprocessing import get_context

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {root}")
    t0 = time.time()
    rep = {"S3a": {}, "S3b": {}, "tokenisation": {}, "sizes": {}}
    ok = True
    for split in ("dev", "train"):
        cached("all", split)
        cached("sample", split)
    _tok()
    pool = get_context("fork").Pool(args.procs)
    for split in ("dev", "train"):                                # (a) every window of both scheds
        n = len(cached("all", split))
        step = max(1, n // (args.procs * 2) + 1)
        st, ex_ = Counter(), Counter()
        for a, b in pool.map(_s3a_chunk, [(split, lo, min(n, lo + step)) for lo in range(0, n, step)]):
            st.update(a)
            ex_.update(b)
        rep["S3a"][split] = {"windows": n, "v2_label_path": dict(st), "v2_pass": V.label_exact_ok(st),
                             "label_nf_choice": st["label_nf_choice"], "v1_kappa_reads_ok": ex_["kappa_ids_ok"],
                             "v1_reads": ex_["reads"], "v1_pass": S.exactness_ok(ex_)}
        ok &= V.label_exact_ok(st) and S.exactness_ok(ex_) and st["label_nf_choice"] == 0
        print(f"[S3a' {split}] {json.dumps(rep['S3a'][split])}", flush=True)
    jobs = [("syn", (20261110, k)) for k in range(400)]
    jobs += [("real", (split, i)) for split in ("dev", "train") for i in range(S.NS[split])]
    agg = defaultdict(Counter)
    worst = defaultdict(float)
    for kind, payload, c in pool.imap_unordered(_brute_job, jobs, chunksize=2):
        key = "synthetic" if kind == "syn" else payload[0]
        if c["skipped"] or (kind == "real" and c.get("deferrable", 0) > BRUTE_MAX_DEFERRABLE):
            agg[key]["skipped_too_big"] += 1
            continue
        agg[key]["instances"] += 1
        agg[key]["pass"] += c["stats"]["pass"]
        agg[key]["paths_total"] += c["n_paths"]
        agg[key]["with_real_choice"] += c["real_choice_reads"] > 0
        agg[key]["real_choice_reads"] += c["real_choice_reads"]
        worst[key] = max(worst[key], c["max_abs_diff"])
    for key in ("synthetic", "dev", "train"):
        rep["S3b"][key] = {**agg[key], "max_abs_diff": worst[key]}
        ok &= agg[key]["pass"] == agg[key]["instances"] > 0 and agg[key]["with_real_choice"] > 0
        print(f"[S3b' {key}] {json.dumps(rep['S3b'][key])}", flush=True)
    neg = negative_controls()
    rep["S3b"]["negative_controls"] = neg
    ok &= neg["all_caught"]
    print(f"[S3b' negative] {json.dumps(neg)}", flush=True)
    sizes = defaultdict(list)
    tk = defaultdict(Counter)
    for split, idx, size, t in pool.imap_unordered(_sample_job, [(s, i) for s in ("dev", "train") for i in range(S.NS[s])],
                                                   chunksize=4):
        sizes[split].append(size)
        tk[split].update(t)
    pool.close()
    pool.join()
    for split in ("dev", "train"):
        rep["tokenisation"][split] = dict(tk[split])
        ok &= tk[split]["nf_tok_ok"] == tk[split]["nf_forwards"]
        sz = sizes[split]
        rep["sizes"][split] = {"windows": len(sz), **{k: S.qstats([s[k] for s in sz]) for k in sz[0]},
                               **{f"total_{k}": int(sum(s[k] for s in sz)) for k in sz[0]}}
    rep["S3p_pass"] = bool(ok)
    rep["elapsed_s"] = time.time() - t0
    out = root / "cpu"
    out.mkdir(parents=True)
    (out / "cpu_stage.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in rep["sizes"][s].items() if k.startswith("total")} for s in rep["sizes"]}))
    print(f"SERIAL_V2P_CPU_STAGE_DONE {out} S3p_pass={ok}", flush=True)
    if not ok:
        raise SystemExit("S3' FAILED: fix the code before any GPU run")


def negative_controls() -> dict:
    """the brute-force check must catch (1) v2's lattice without the order latent, (2) a wrong newfirst order,
    (3) an off-by-one newfirst boundary, on sample windows with a real order choice."""
    recs = []
    for split in ("dev", "train"):
        for rec in S.sample(split):
            lat = V.lattice(rec)
            if sum(1 for r in lat["rho"] if r < lat["R"] - 1) > 8:
                continue
            if any(cx["nf"] for rd in plan_window_p(rec, lat)["reads"] for cx in rd["ctx"]):
                recs.append(rec)
            if len(recs) >= 3 * (1 + (split == "train")):
                break
    st = Counter()
    for rec in recs:
        lat = V.lattice(rec)
        paths, scores = brute_paths_p(rec, lat, V.dummy_psi)
        bf = brute_force_p(rec, lat, paths, scores, 1.0)["logZ"]
        plan = plan_window_p(rec, lat, skip_forced=False)
        res = V.dummy_res(plan)
        good = forward_p(assemble_p(plan, res), 1.0)["logZ"]
        no_order = V.forward_backward(V.psi_list(V.assemble(plan, res)), 1.0)["logZ"]       # v2: canon only
        wrong = check_brute_p(rec, (1.0,), nf_fn=wrong_nf_pending_head)
        st["all_new_first_caught(report)"] += check_brute_p(rec, (1.0,), nf_fn=wrong_nf_all_new_first)["stats"]["pass"] == 0
        for rd in plan["reads"]:                                                          # off-by-one boundary
            for cx in rd["ctx"]:
                cx["nf"] = {j: (q, max(0, b - 1), bn, s, nl) for j, (q, b, bn, s, nl) in cx["nf"].items()}
        off = forward_p(assemble_p(plan, res), 1.0)["logZ"]
        st["windows"] += 1
        st["good_equal"] += abs(good - bf) < 1e-9
        st["no_order_caught"] += abs(no_order - bf) > 1e-6
        st["wrong_newfirst_caught"] += wrong["stats"]["pass"] == 0
        st["off_by_one_caught"] += abs(off - bf) > 1e-6
    st["all_caught"] = int(st["windows"] > 0 and st["windows"] == st["good_equal"] == st["no_order_caught"]
                           == st["wrong_newfirst_caught"] == st["off_by_one_caught"])
    return dict(st)


def run_dry(args) -> None:
    for split in ("dev", "train"):
        recs = S.sample(split)[:40]
        per = []
        for rec in recs:
            plan = plan_window_p(rec)
            row = {"id": rec["id"], "R": plan["lat"]["R"], "n_fw": len(plan["reqs"]),
                   "states": [[list(s) for s in st] for st in plan["lat"]["states"]],
                   "reads": assemble_p(plan, V.dummy_res(plan))}
            per.append(window_analysis(rec, row))
        rep = split_report(per)
        print(split, json.dumps({k: rep[k] for k in ("windows", "real_choice_reads", "nf_contexts",
                                                      f"S2a[joint,tau={TAU}]")}), flush=True)
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
    ap.add_argument("--procs", type=int, default=48)
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
