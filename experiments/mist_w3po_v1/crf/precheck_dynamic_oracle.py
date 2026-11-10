"""CRF dynamic oracle precheck (design: reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §14; pre-registration and results:
reports/mist_w3po_v1/crf_dynamic_oracle_precheck.zh-CN.md). Precheck only: no training code, no training.
Data: AMI dev only; the on-policy states are those of v8's final-read causal P online decode
eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0_guard_finalpost_s8 (3,050 windows, the v2' S1' decode), scored with v8.

State s = (window w, read i, end of the read's committed block, the model's own committed ids own(s)).
  own(s)       the concatenation over reads 0..i of tok(read_log[r].new) (the decode commits exactly the decoded ids of a
               read; ctx is "" for every window: --context 0). O1 asserts on every read of every window: the per-read
               ids decode back to read_log.new, their concatenation decodes to raw, and engine.Parser fed with them per
               read (write_ms = t) and closed as online_decode does reproduces the stored segments and blocks exactly.
               Report-only GPU check: greedy replay of the read block under teacher forcing.
  items        per window w: the scorer's scored utterances u (overlap_score.score, online=True, the state captured by
               diag_order_attr.run_scorer) whose due window is w (diag_order_attr.due_of: the online window with the
               largest time overlap), cut into chunks by v8's own label recipe (build_sched_windows.v8_chunks: lag 1 s,
               slot = slot_map.json lowlat of the meeting, piece = <|spk k|><|t a|>text<|t b|> on w's clock); the due
               read rho_lab(c) = the first read with t >= commit_ms (the builder's read loop; commit clamped to the
               window end).
  coverage     at the end of read i an utterance u (all its chunks) is COVERED iff some CLOSED segment x with non-empty
               text, written in w with write_ms <= t_i or written in an earlier window of the meeting (window start
               < w start; all its segments), satisfies one of
                 (a) TEXT  S1's text-match rule under any speaker label (serial_crf_v2_precheck §1.5: x's span widened by
                           2 s overlaps u, norm(x.text) starts or ends with norm(u.ref), not explained away by another
                           scored utterance of x's mapped speaker containing norm(u.ref));
                 (b) CREDIT the scorer credits x to u's unit within 1 s of u (= recalled() restricted to the written
                           segments; over all segments of the meeting it IS recalled(), asserted for every utterance);
                 (c) TIME  only for u longer than 1 s: u is the scorer's best-overlap annotated utterance of x (its own
                           first assignment step, collar 2 s), whatever x's label, and the shared normalised word types
                           are >= half of the smaller of the two word sets (mislabelled long items count as covered;
                           short units are decided by (a)/(b) only, so the S1 / CHAIN classes are untouched).
               A segment left open at the end of a read is not a segment yet (it covers nothing until it is closed);
               states whose committed ids end inside a segment are not structural states (excluded, counted).
  pending(s)   uncovered chunks with rho_lab <= i (due at this read or overdue).
Residual lattice L(s) (the v2' lattice on the uncovered chunks U(s), reads i..R-1 re-indexed 0..R-1-i)
  due read     rt(c) = max(rho_lab(c), i) - i: an overdue chunk gets the current read as its due read
  slack        rho(c) in {rt(c), rt(c) + 1}, <= the final read; final-read chunks cannot be deferred; per speaker the label
               order is kept (precheck_serial_v2.lattice on the residual record, unchanged)
  history      prompt of residual read r' = own(s) ⊕ kappa(lattice chunks committed at residual reads < r'); the read-0
               context is own(s) itself (the continuation of the read the model just ended)
  order        v2' order latent, generalised: carried = pending items of the previous read (v2') or, at residual read 0,
               the overdue chunks written there; N = on-time chunks whose slot no carried chunk has; newfirst = N ⊕ kap(rest)
  potentials   psi = log p_v8(ordered write set ⊕ e_r | x <= t_r, history) (final read: <eos>, <wait> barred, P by the
               finalpost rule with the window's close_detect_ms); joint tempering tau = 0.5 as v2'; exact forward
  oracle       q(a | s) = sum over (rho, o) of the residual-read-0 first action: wait (D_0 empty), or the slot of the first
               chunk written (canon: kap-first of D_0; newfirst: N[0])

PRE-REGISTERED DECISION RULES (constants below)
  O1 (CPU)  (i) trace rebuild asserted on every read of every dev window (above); (ii) rule (b) == recalled() for every
            scored dev utterance; (iii) on the 263 missed units of S1' (s1_rows_primary_finalpost.jsonl): the coverage
            function reproduces every S1 / S1' field it decides (u written at the due / next / later read, in another
            window, n_other at due / next) and hence the why / why2 classes, and (b) never fires (assert); the due window
            and the due read of the unit's last chunk equal S1's; (iv) hand check: N_REVIEW = 50 random O2-population
            states (seed SEED), every due-or-overdue utterance's coverage decision judged from the texts;
            agreement = correct decisions / all decisions. Any assertion failing or agreement < O1_STOP -> STOP.
  O3 (CPU)  residual forward vs brute force (independent enumeration of rho assignments x per-read orders, histories and
            orders built from each assignment) within 1e-9: log Z (all reads scored, and forced reads skipped + constant),
            every path score and mass, residual-read-0 action marginals q(a), state, deferral and order marginals, path
            counts; on random synthetic residual instances and on real O2 states (dummy LM; <= 400k paths). Negative tests,
            all must be caught: N1 the v2' order rule at residual read 0 (no overdue-carried latent) on states with a real
            read-0 choice; N2 the lattice without own(s) in its histories. Fail -> fix the code, no GPU.
  O2 (GPU)  population: non-final read ends with no loop/budget guard cut, a structural state, and pending(s) non-empty.
            D = share with q(wait | s) < 0.5 and p_v8(wait | s) >= 0.5 (full softmax at s, no bar). D < O2_STOP -> STOP.
            Budget: all states if their planned forwards sum <= FW_BUDGET, else a seeded uniform random prefix of the
            states up to FW_BUDGET planned forwards (D on it) plus every state where a missed unit is pending (D_263 on
            those). States above FW_STATE_CAP planned forwards are excluded and counted.
            REPORT ONLY: mean / median KL(q || p_bar), p_bar = p_v8 renormalised over {wait, eos, spk1..8}; D_263;
            CHAIN and NOTHING_ELSE_WRITTEN_AT_NEXT units with >= 1 disagreeing state while pending; replay of the read
            block; q / p distributions.
  GO iff O1 and O3 pass and D >= O2_STOP.

    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --o3 --root <new root> --procs 64
    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --o1 --root <root> --procs 64     # + states for O2
    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --score --root <root> --shard K/N   # GPU
    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --analyse --root <root>
"""
from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import precheck_serial_v2p as VP

P = S.P
TAU = 0.5
O1_STOP = 0.95
O2_STOP = 0.05
N_REVIEW = 50
SEED = 20261110
LONG_S = 1.0
LAG_MS = 1000
COLLAR_S, RECALL_COLLAR = 2.0, 1.0
FW_BUDGET = 120_000
FW_STATE_CAP = 3_000
BRUTE_MAX_PATHS = 400_000
DECODE = VP.DECODE_PRIMARY
S1P_ROWS = "eval_outputs/mist_w3po_v1/crf/serial_v2p_precheck/s1/s1_rows_primary_finalpost.jsonl"
ACTS = ["wait", "eos"] + [f"spk{k}" for k in range(1, 9)]


def ov(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ------------------------------------------------------------------------------------------------ coverage tables
def meeting_tables(st: dict, seglist: list[dict]) -> dict:
    """Per meeting, plain data: for every scored utterance the segment indices that cover it by rule a / b / c."""
    from experiments.mist_w3po_v1.crf import diag_order_attr as D

    sc, got, unit_of, mp, us = st["sc"], st["got"], st["unit_of"], st["mp"], st["us"]
    idx = {id(x): k for k, x in enumerate(seglist)}
    bu = []
    for x in seglist:                                     # the scorer's first assignment step (strict >, us order)
        if not x["text"]:
            bu.append(None)
            continue
        a, b = x["a"] - COLLAR_S, x["b"] + COLLAR_S
        best, bid = 0.0, None
        for u in us:
            o = min(b, u[3]) - max(a, u[2])
            if o > best:
                best, bid = o, u[0]
        bu.append(bid)
    nr = {rid: D.norm(v["ref"]) for rid, v in sc.items()}
    by_spk = defaultdict(list)
    for rid, v in sc.items():
        by_spk[v["spk"]].append(rid)
    out = {}
    for rid, u in sc.items():
        r = nr[rid]
        ra, rb, rc = [], [], []
        for x, t in got.get(unit_of[rid], []):
            if t.strip() and ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0:
                rb.append(idx[id(x)])
        for k, x in enumerate(seglist):
            if not x["text"] or ov(x["a"] - COLLAR_S, x["b"] + COLLAR_S, u["a"], u["b"]) <= 0:
                continue
            if D.edge_match(D.norm(x["text"]), r):
                spk = mp.get(x["spk"])
                expl = spk is not None and any(
                    rid2 != rid and ov(x["a"] - COLLAR_S, x["b"] + COLLAR_S, sc[rid2]["a"], sc[rid2]["b"]) > 0
                    and D.contains(nr[rid2], r) for rid2 in by_spk.get(spk, []))
                if not expl:
                    ra.append(k)
            if bu[k] == rid and u["b"] - u["a"] > LONG_S and word_overlap(D.norm(x["text"]), r):
                rc.append(k)
        out[rid] = {"a": sorted(set(ra)), "b": sorted(set(rb)), "c": sorted(set(rc))}
    return out


def word_overlap(xw: list[str], r: list[str]) -> bool:
    """rule (c)'s content guard: the shared word types are >= half of the smaller word set (a backchannel inside a long
    utterance of another speaker shares none)."""
    X, U = set(xw), set(r)
    return bool(X and U) and len(X & U) >= 0.5 * min(len(X), len(U))


def visible(k: int, meta_m: list[dict], wid: str, w_start: float, t_ms: int) -> bool:
    """segment k is written by the end of read t of window wid (or in an earlier window)."""
    x = meta_m[k]
    return (x["wid"] == wid and x["write_ms"] <= t_ms) or x["w_start"] < w_start - 1e-9


def covered_by(cov: dict, ks_ok) -> dict:
    """which rules cover an utterance given a predicate on segment indices."""
    return {r: any(ks_ok(k) for k in cov[r]) for r in ("a", "b", "c")}


# ------------------------------------------------------------------------------------------------ items, traces
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from experiments.mist_w3po_v1.crf import build_sched_windows as B
        _TOK = B.tokenizer()
    return _TOK


def window_items(w: dict, rids: list[str], sc: dict, smap_m: dict, t_reads: list[int]) -> list[dict]:
    """the chunks of w's due utterances by v8's label recipe, with ids and the due read."""
    from experiments.mist_w3po_v1.crf import build_sched_windows as B

    utts = [{"speaker_id": sc[r]["spk"], "begin_s": sc[r]["a"] - w["start_s"], "end_s": sc[r]["b"] - w["start_s"],
             "ref": sc[r]["ref"], "hesitation": False} for r in rids]
    ch = B.v8_chunks({"start_s": w["start_s"], "dur_ms": t_reads[-1], "utts": utts}, smap_m, LAG_MS)
    tk = tok()
    out = []
    for c in ch:
        ids = tk(c["piece"], add_special_tokens=False)["input_ids"]
        assert tk.decode(ids, skip_special_tokens=False) == c["piece"], c["piece"]
        out.append({"rid": rids[c["utt"]], "slot": c["slot"], "ids": ids, "piece": c["piece"],
                    "src_start_ms": c["src_start_ms"], "src_end_ms": c["src_end_ms"], "commit_ms": c["commit_ms"],
                    "rho": min(bisect.bisect_left(t_reads, c["commit_ms"]), len(t_reads) - 1)})
    return out


def rebuild(h: dict) -> dict:
    """O1 (i): per-read ids from the trace; asserts decode == read_log.new, concat == raw, ctx == "", Parser replay ==
    the stored segments and blocks. Returns ids per read, guard flags, the segment state at every read end."""
    from experiments.mist_w3po_v1.core.engine import Parser, seg_state

    tk = tok()
    reads = h["read_log"]
    assert h["ctx"] == "", h["id"]
    ids_per = []
    for rd in reads:
        ids = tk(rd["new"], add_special_tokens=False)["input_ids"] if rd["new"] else []
        assert tk.decode(ids, skip_special_tokens=False) == rd["new"], (h["id"], rd["t"])
        ids_per.append(ids)
    allids = [t for x in ids_per for t in x]
    assert tk.decode(allids, skip_special_tokens=False) == h["raw"], h["id"]
    p = Parser(tk)
    blocks = []
    for rd, ids in zip(reads, ids_per):
        blocks += p.feed(ids, rd["t"])
    p.close(reads[-1]["t"] / 1000, reads[-1]["t"])
    rt = lambda x: json.loads(json.dumps(x))   # noqa: E731
    assert rt(p.segments) == rt(h["segments"]), h["id"]
    assert rt(blocks) == rt(h["blocks"]), h["id"]
    states, cum = [], []
    for ids in ids_per:
        cum = cum + ids
        states.append(seg_state(cum))
    return {"ids": ids_per, "t": [rd["t"] for rd in reads], "guard": [bool(rd.get("guard")) for rd in reads],
            "seg_state": states}


# ------------------------------------------------------------------------------------------------ residual lattice
def slot(rec: dict, c: int) -> int:
    return rec["chunks"][c]["slot"]


def nf_gen(rec: dict, rho: list[int], pend, D) -> tuple[tuple, tuple] | None:
    """v2' newfirst, generalised: carried = pend ∪ (D ∩ overdue); N = the non-overdue chunks of D (kap order) whose slot
    no carried chunk has; newfirst = N ⊕ kap(carried ∪ D \\ N). None when carried or N is empty. Without overdue chunks
    this is exactly precheck_serial_v2p.newfirst."""
    ovd = set(rec.get("overdue", ()))
    carried = list(pend) + [c for c in D if c in ovd]
    if not carried:
        return None
    cs = {slot(rec, c) for c in carried}
    N = [c for c in V.kap(rec, rho, D) if c not in ovd and slot(rec, c) not in cs]
    if not N:
        return None
    n = set(N)
    return tuple(N), tuple(V.kap(rec, rho, carried + [c for c in D if c not in n and c not in set(carried)]))


def residual_rec(wid: str, items: list[dict], t_reads: list[int], i: int, unc: list[int], own: list[int]) -> dict:
    """the residual record of state (w, read i): uncovered chunks unc (indices into items), reads i..R-1 -> 0..R-1-i."""
    sel = sorted(unc, key=lambda c: (max(items[c]["rho"], i) - i, items[c]["commit_ms"], items[c]["src_start_ms"], c))
    chunks, ids = [], []
    for c in sel:
        it = items[c]
        chunks.append({"orig": c, "rid": it["rid"], "utt": c, "slot": it["slot"], "tok0": len(ids),
                       "tok1": len(ids) + len(it["ids"]), "src_start_ms": it["src_start_ms"],
                       "src_end_ms": it["src_end_ms"], "commit_ms": it["commit_ms"], "rho_lab": it["rho"],
                       "rt": max(it["rho"], i) - i, "piece": it["piece"]})
        ids += it["ids"]
    R = len(t_reads) - i
    beta = [sum(c["tok1"] - c["tok0"] for c in chunks if c["rt"] <= r) for r in range(R)]
    return {"id": f"{wid}|{i}", "wid": wid, "i": i, "chunks": chunks, "S_ids": ids, "n_read": R,
            "t_ms": list(t_reads[i:]), "dur_ms": t_reads[-1], "beta_heur": beta, "n_tok": len(ids),
            "units": [{"utt": c["utt"], "a_ms": c["src_start_ms"]} for c in chunks],
            "overdue": [k for k, c in enumerate(chunks) if c["rho_lab"] < i], "own": list(own)}


def plan_state(rec: dict, nf_fn=nf_gen, own: bool = True, skip_forced: bool = True) -> dict:
    """v2' plan on the residual record (order rule nf_fn); own(s) prepended to every history."""
    lat = V.lattice(rec)
    assert [c["rt"] for c in rec["chunks"]] == lat["rho"]
    plan = VP.plan_window_p(rec, lat, nf_fn=nf_fn, skip_forced=skip_forced)
    if own:
        for q in plan["reqs"]:
            q["hist_ids"] = list(rec["own"]) + q["hist_ids"]
    return plan


def first_actions(rec: dict, plan: dict, nf_fn=nf_gen) -> list[tuple[str, str | None]]:
    """per residual-read-0 state j: (canon first action, newfirst first action or None)."""
    lat = plan["lat"]
    out = []
    for D in lat["states"][0]:
        a = "wait" if not D else f"spk{slot(rec, D[0])}"
        nf = nf_fn(rec, lat["rho"], [], D)
        out.append((a, None if nf is None else f"spk{slot(rec, nf[0][0])}"))
    return out


def oracle(rec: dict, plan: dict, reads: list[dict], tau: float = TAU, nf_fn=nf_gen) -> dict:
    """the exact tempered posterior of the residual lattice: log Z, q(a | s) over ACTS (residual read 0), order marginals."""
    fb = VP.forward_p(reads, tau, "joint")
    C, Nf = VP.mats(reads[0])
    q = {a: 0.0 for a in ACTS}
    for j, (ac, an) in enumerate(first_actions(rec, plan, nf_fn)):
        base = fb["beta"][0][j] - fb["logZ"]
        q[ac] += math.exp(tau * C[0, j] + base)
        if an is not None:
            assert Nf is not None and np.isfinite(Nf[0, j])
            q[an] += math.exp(tau * Nf[0, j] + base)
        else:
            assert Nf is None or np.isnan(Nf[0, j])
    return {"logZ": fb["logZ"], "q": q, "fb": fb}


# ------------------------------------------------------------------------------------------------ brute force (O3)
def brute_orders_res(rec: dict, rho: list[int], r: int, W: list[int]) -> list[tuple[int, list[int]]]:
    """within-read orders built without the lattice: canon = the kap key; newfirst = a stable sort putting first the
    on-time non-overdue chunks (rt = r) whose slot no carried chunk (rt < r, or overdue) has."""
    ch = rec["chunks"]
    ovd = set(rec.get("overdue", ()))
    key = lambda c: (rho[c], ch[c]["commit_ms"], ch[c]["src_start_ms"], c)     # noqa: E731
    canon = sorted(W, key=key)
    carried = [c for c in W if rho[c] < r or c in ovd]
    cs = {ch[c]["slot"] for c in carried}
    first = [c for c in W if rho[c] == r and c not in ovd and ch[c]["slot"] not in cs]
    out = [(0, canon)]
    if carried and first:
        fs = set(first)
        out.append((1, sorted(W, key=lambda c: (0 if c in fs else 1,) + key(c))))
    return out


def brute_res(rec: dict, lat: dict, psi_fn, own: list[int], max_paths: int = BRUTE_MAX_PATHS):
    rho, R = lat["rho"], lat["R"]
    n = len(rec["chunks"])
    paths, scores, firsts = [], [], []
    for a in V.assignments(rec, lat):
        opts = []
        for r in range(R):
            hist = V.kap(rec, rho, [c for c in range(n) if a[c] <= r - 1])
            W = [c for c in range(n) if a[c] == r]
            h = list(own) + S.ids_of(rec, hist)
            opts.append([(o, psi_fn(r, r == R - 1, h, S.ids_of(rec, w)), w) for o, w in brute_orders_res(rec, rho, r, W)])
        for combo in itertools.product(*opts):
            paths.append((a, tuple(o for o, _, _ in combo)))
            scores.append(sum(s for _, s, _ in combo))
            w0 = combo[0][2]
            firsts.append("wait" if not w0 else f"spk{slot(rec, w0[0])}")
            if len(paths) > max_paths:
                return None
    return paths, np.asarray(scores, np.float64), firsts


def check_brute_res(rec: dict, taus=(0.5, 1.0), tol: float = 1e-9, nf_fn=nf_gen, use_own: bool = True,
                    max_paths: int = BRUTE_MAX_PATHS) -> dict:
    """O3 on one residual record with v2's dummy LM (a function of the whole prefix, own(s) included)."""
    lat = V.lattice(rec)
    full = plan_state(rec, nf_fn, own=use_own, skip_forced=False)
    free = plan_state(rec, nf_fn, own=use_own, skip_forced=True)
    rd_full = VP.assemble_p(full, V.dummy_res(full))
    rd_free = VP.assemble_p(free, V.dummy_res(free))
    n_lat = VP.n_paths_lattice(rd_full)
    if n_lat > max_paths:
        return {"skipped": True, "n_paths": n_lat}
    bp = brute_res(rec, lat, V.dummy_psi, rec["own"], max_paths)
    if bp is None:
        return {"skipped": True, "n_paths": n_lat}
    paths, scores, firsts = bp
    const = sum(float(VP.mats(rd)[0][0, 0]) for rd, pr in zip(rd_full, free["reads"]) if pr["forced"])
    try:
        idxs = [V.path_states(lat, a) for a, _ in paths]
        lat_scores = [VP.path_score_p(rd_full, idx, os_) for idx, (_, os_) in zip(idxs, paths)]
    except (KeyError, ValueError):
        return {"skipped": False, "pass": 0, "why": "path_missing", "max_abs_diff": float("inf"), "n_paths": len(paths)}
    worst = 0.0
    for tau in taus:
        bf = VP.brute_force_p(rec, lat, paths, scores, tau)
        logZ = V.lse(tau * scores)
        qp = np.exp(tau * scores - logZ)
        qa = Counter()
        for f, x in zip(firsts, qp):
            qa[f] += x
        fa, fr = oracle(rec, full, rd_full, tau, nf_fn), oracle(rec, free, rd_free, tau, nf_fn)
        d = [abs(fa["logZ"] - logZ), abs(fr["logZ"] + tau * const - logZ)]
        d += [abs(ls - sb) for ls, sb in zip(lat_scores, scores)]
        for idx, (_, os_), qb in zip(idxs, paths, qp):
            d.append(abs(math.exp(tau * VP.path_score_p(rd_free, idx, os_) - fr["logZ"]) - qb))
        for which in (fa, fr):
            d += [abs(which["q"][a] - qa.get(a, 0.0)) for a in ACTS]
            if set(qa) - set(ACTS):
                d.append(float("inf"))
            fb = which["fb"]
            dm = V.deferral_marginals(lat, fb)
            d += [abs(dm[c] - bf["defer"][c]) for c in bf["defer"]]
            for r in range(lat["R"]):
                d += [abs(fb["marg"][r][j] - bf["state_marg"][r].get(s, 0.0)) for j, s in enumerate(lat["states"][r])]
                if not set(bf["state_marg"][r]) <= set(lat["states"][r]):
                    d.append(float("inf"))
        # order marginals: brute with the generalised rule
        om = [{"q_real": 0.0, "q_nf": 0.0, "any": False} for _ in range(lat["R"])]
        for (a, os_), x in zip(paths, qp):
            for r in range(lat["R"]):
                W = [c for c in range(len(a)) if a[c] == r]
                if len(brute_orders_res(rec, lat["rho"], r, W)) == 2:
                    om[r]["q_real"] += x
                    om[r]["any"] = True
                if os_[r] == 1:
                    om[r]["q_nf"] += x
        for which in (fa, fr):
            for r in range(lat["R"]):
                lo, bo = which["fb"]["order"][r], om[r]
                if lo is None:
                    d += [bo["q_real"], bo["q_nf"]] + ([float("inf")] if bo["any"] else [])
                else:
                    d += [abs(lo["q_real"] - bo["q_real"]), abs(lo["q_nf"] - bo["q_nf"])]
        worst = max(worst, max(d))
    ok_paths = int(round(n_lat) == len(paths))
    return {"skipped": False, "pass": int(worst < tol and ok_paths == 1), "max_abs_diff": worst, "n_paths": len(paths),
            "n_paths_equal": ok_paths, "real_choice_read0": bool(any(o[0] == 1 for _, o in paths)),
            "n_overdue": len(rec.get("overdue", ())), "R": lat["R"]}


def synthetic_residual(rng: np.random.Generator, n_read: int, n_chunk: int, n_spk: int) -> dict:
    """a random window (V.synthetic_rec with n_read + 2 reads), a random non-final read i >= 1, a random uncovered subset
    of its due-or-overdue chunks (all later chunks uncovered), a random own prefix -> residual_rec."""
    w = V.synthetic_rec(rng, n_read + 2, n_chunk, n_spk)
    rho = S.chunk_reads(w)
    items = [{"rid": f"c{c}", "slot": ch["slot"], "ids": S.ids_of(w, [c]), "piece": ch["piece"],
              "src_start_ms": ch["src_start_ms"], "src_end_ms": ch["src_end_ms"], "commit_ms": ch["commit_ms"],
              "rho": rho[c]} for c, ch in enumerate(w["chunks"])]
    i = int(rng.integers(1, n_read + 1))
    unc = [c for c in range(len(items)) if rho[c] > i or rng.random() < 0.6]
    own = [int(x) for x in rng.integers(0, 5000, int(rng.integers(0, 6)))]
    return residual_rec(w["id"], items, w["t_ms"], i, unc, own)


# ------------------------------------------------------------------------------------------------ O3 stage (CPU)
def _o3_one(rec: dict) -> dict:
    """the main check and the two negative tests (N1 on a real read-0 order choice, N2 with a non-empty own(s))."""
    out = {"id": rec["id"], "main": check_brute_res(rec), "N1": None, "N2": None}
    if not out["main"].get("skipped"):
        if out["main"]["real_choice_read0"]:
            out["N1"] = check_brute_res(rec, nf_fn=VP.newfirst)
        if rec["own"]:
            out["N2"] = check_brute_res(rec, use_own=False)
    return out


def _o3_syn(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return _o3_one(synthetic_residual(rng, int(rng.integers(2, 5)), int(rng.integers(1, 7)), int(rng.integers(1, 4))))


def o3_summary(res: list[dict]) -> dict:
    ok = [x for x in res if not x["main"].get("skipped")]
    n1 = [x["N1"] for x in ok if x["N1"] is not None]
    n2 = [x["N2"] for x in ok if x["N2"] is not None]
    rep = {"n": len(res), "checked": len(ok), "skipped_large": len(res) - len(ok),
           "pass": sum(x["main"]["pass"] for x in ok),
           "real_choice_read0": sum(x["main"]["real_choice_read0"] for x in ok),
           "with_overdue": sum(x["main"]["n_overdue"] > 0 for x in ok),
           "paths": sum(x["main"]["n_paths"] for x in ok),
           "max_abs_diff": max((x["main"]["max_abs_diff"] for x in ok), default=None),
           "N1_cases": len(n1), "N1_caught": sum(x.get("pass", 0) == 0 for x in n1),
           "N2_cases": len(n2), "N2_caught": sum(x.get("pass", 0) == 0 for x in n2)}
    rep["pass_all"] = bool(rep["checked"] > 0 and rep["pass"] == rep["checked"] and rep["real_choice_read0"] > 0
                           and rep["N1_cases"] > 0 and rep["N1_caught"] == rep["N1_cases"]
                           and rep["N2_cases"] > 0 and rep["N2_caught"] == rep["N2_cases"])
    return rep


def run_o3(args, recs: list[dict] | None = None) -> dict:
    """synthetic instances (recs None) or the given real residual records."""
    from multiprocessing import get_context

    t0 = time.time()
    with get_context("fork").Pool(args.procs) as pool:
        if recs is None:
            res = pool.map(_o3_syn, range(SEED, SEED + 600), chunksize=4)
        else:
            res = pool.map(_o3_one, recs, chunksize=1)
    rep = o3_summary(res)
    rep["elapsed_s"] = time.time() - t0
    return rep


# ------------------------------------------------------------------------------------------------ O1 stage (CPU)
_G: dict = {}


def _window_job(wid: str) -> dict:
    """per window: trace rebuild (asserts), items, per-read coverage, states, residual plans of the population."""
    g = _G
    w, h = g["Wid"][wid], g["H"][wid]
    m = w["meeting"]
    tr = rebuild(h)
    t_reads = tr["t"]
    R = len(t_reads)
    rids = g["items_of"].get(wid, [])
    cov = g["cov"][m]
    meta_m = g["meta"][m]
    sc = g["cap"][m]["sc"]
    items = window_items(w, rids, sc, g["smap"][m], t_reads) if rids else []
    out = {"wid": wid, "R": R, "n_items": len(items), "n_utts": len(rids), "states": [], "dur_eq": t_reads[-1] == w["dur_ms"],
           "reads": R, "guard_reads": sum(tr["guard"]), "open_end_reads": sum(s != "out" for s in tr["seg_state"])}
    own = []
    for i in range(R):
        own = own + tr["ids"][i]
        cv = {r: covered_by(cov[r], lambda k: visible(k, meta_m, wid, w["start_s"], t_reads[i])) for r in rids}
        is_cov = {r: any(v.values()) for r, v in cv.items()}
        unc = [c for c, it in enumerate(items) if not is_cov[it["rid"]]]
        pend = [c for c in unc if items[c]["rho"] <= i]
        st = {"i": i, "final": i == R - 1, "guard": tr["guard"][i], "out": tr["seg_state"][i] == "out",
              "n_unc": len(unc), "pend": sorted({items[c]["rid"] for c in pend}),
              "cov_rules": {r: [k for k, v in cv[r].items() if v] for r in rids if is_cov[r]},
              "due_rids": sorted({it["rid"] for it in items if it["rho"] <= i})}
        st["pop"] = bool(not st["final"] and not st["guard"] and st["out"] and pend)
        if st["pop"]:
            rec = residual_rec(wid, items, t_reads, i, unc, own)
            plan = plan_state(rec)
            st["n_fw"] = len(plan["reqs"])
            st["rec"] = rec
            st["block"] = tr["ids"][i]
        out["states"].append(st)
    out["items"] = [{k: it[k] for k in ("rid", "slot", "rho", "commit_ms", "piece")} for it in items]
    return out


def load_dev_state(decode: str = DECODE) -> dict:
    from experiments.mist_w3po_v1.crf import diag_order_attr as D

    W = [json.loads(l) for l in open(D.WINDOWS, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(decode) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(D.REF_DEV, encoding="utf-8")]
    W = [w for w in W if w["id"] in H]
    res, cap, segs = D.run_scorer(W, H, RW)
    ref = json.load(open(Path(decode) / "score_text.json", encoding="utf-8"))
    assert res["short_recall"] == ref["short_recall"], "scorer state does not reproduce score_text.json"
    meta = D.seg_meta(W, H, segs)
    smap = json.load(open(D.SLOT_MAP, encoding="utf-8"))["lowlat"]
    wins_of = defaultdict(list)
    for w in W:
        wins_of[w["meeting"]].append(w)
    items_of, due_of = defaultdict(list), {}
    for m, st in cap.items():
        for rid, u in st["sc"].items():
            d = D.due_of(u, wins_of[m], H)
            due_of[rid] = d
            if d["window"] is not None:
                items_of[d["window"]].append(rid)
    Wid = {w["id"]: w for w in W}
    for w in items_of:
        sc = cap[Wid[w]["meeting"]]["sc"]
        items_of[w].sort(key=lambda r: (sc[r]["a"], r))
    return {"W": W, "Wid": Wid, "H": H, "cap": cap, "segs": segs, "meta": meta, "smap": smap,
            "items_of": dict(items_of), "due_of": due_of, "wins_of": wins_of}


def s1_check(g: dict, by_w: dict) -> dict:
    """O1 (iii): the coverage function reproduces the S1 / S1' fields of the 263 missed units."""
    from experiments.mist_w3po_v1.crf import diag_order_attr as D

    rows = [json.loads(l) for l in open(S1P_ROWS, encoding="utf-8")]
    st = Counter()
    extra = Counter()
    for row in rows:
        m, rid = row["meeting"], row["rid"]
        st["rows"] += 1
        cov, meta_m, seglist = g["cov"][m][rid], g["meta"][m], g["segs"][m]
        assert not cov["b"], rid                                  # not recalled -> no credit
        assert not g["cap"][m]["rec"][rid]
        assert not cov["c"], rid                                  # <= 1 s: rule c never applies
        if row["window"] is None:
            assert g["due_of"][rid]["window"] is None
            st["no_window"] += 1
            continue
        w = row["window"]
        assert g["due_of"][rid]["window"] == w and rid in g["items_of"][w]
        ws = by_w[w]
        last = max((it for it in ws["items"] if it["rid"] == rid), key=lambda it: it["commit_ms"])
        assert last["rho"] == row["due_read"], (rid, last["rho"], row["due_read"])
        t = ws["t"]
        A = set(cov["a"])
        mine = [k for k in range(len(seglist)) if meta_m[k]["wid"] == w and seglist[k]["text"].strip()]
        at = [k for k in mine if meta_m[k]["write_ms"] == row["t_due_ms"]]
        u_at = [k for k in at if k in A]
        why = V.deferral_class(row["due_final"], bool(u_at), len(at) - len(u_at))
        assert (bool(u_at), len(at) - len(u_at), why) == (row["u_written_at_due"], row["n_other_at_due"], row["why"]), rid
        if why == "DEFERRAL":
            nxt = t[row["due_read"] + 1]
            at_n = [k for k in mine if meta_m[k]["write_ms"] == nxt]
            u_n = [k for k in at_n if k in A]
            u_l = [k for k in mine if meta_m[k]["write_ms"] > nxt and k in A]
            u_else = [k for k in A if meta_m[k]["wid"] != w and seglist[k]["text"].strip()]
            same = lambda ks: any(g["cap"][m]["mp"].get(seglist[k]["spk"]) == g["cap"][m]["sc"][rid]["spk"] for k in ks)  # noqa: E731
            why2 = VP.s1p_class(True, bool(u_n), same(u_n), bool(u_l), same(u_l), len(at_n) - len(u_n))
            assert (bool(u_n), bool(u_l), bool(u_else), len(at_n) - len(u_n), why2) == (
                row["u_written_at_next"], row["u_written_later"], row["u_written_in_other_window"],
                row["n_other_at_next"], row["why2"]), rid
            st["chain"] += why2 == "CHAIN"
        # the coverage state along w's reads
        cs = ws["cov_by_read"][rid]
        for i, c in enumerate(cs):
            mine_c = any(meta_m[k]["wid"] == w and meta_m[k]["write_ms"] <= t[i] for k in A)
            earlier = any(meta_m[k]["w_start"] < g["Wid"][w]["start_s"] - 1e-9 for k in A)
            assert c == (mine_c or earlier), rid
        if row.get("chain"):   # S1' constrains the due / next / later reads only: no text match in w at or after t_due
            assert not any(meta_m[k]["wid"] == w and meta_m[k]["write_ms"] >= row["t_due_ms"] for k in A), rid
            extra["chain_covered_by_earlier_window"] += any(meta_m[k]["w_start"] < g["Wid"][w]["start_s"] - 1e-9
                                                            for k in A)
            extra["chain_covered_before_due_in_w"] += any(meta_m[k]["wid"] == w and meta_m[k]["write_ms"] < row["t_due_ms"]
                                                          for k in A)
            extra["chain_uncovered_all_reads_from_due"] += not any(cs[row["due_read"]:])
        extra[f"covered_before_due_{row['why']}"] += any(cs[: row["due_read"]])
        st["checked"] += 1
    st["asserted"] = True
    return {**dict(st), **dict(extra)}


def run_o1(args) -> None:
    from multiprocessing import get_context

    root = Path(args.root)
    out = root / "o1"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    g = load_dev_state(DECODE)
    tok()
    cov, rec_eq = {}, Counter()
    for m, st in g["cap"].items():
        cov[m] = meeting_tables(st, g["segs"][m])
        for rid in st["sc"]:                                  # O1 (ii): rule b over all segments == recalled()
            ok = bool(cov[m][rid]["b"]) == st["rec"][rid]
            rec_eq["utts"] += 1
            rec_eq["b_eq_recalled"] += ok
            assert ok, (m, rid)
    g["cov"] = cov
    _G.update(g)
    print(f"[O1] scorer + tables {time.time() - t0:.0f}s", flush=True)
    wids = [w["id"] for w in g["W"]]
    with get_context("fork").Pool(args.procs) as pool:
        res = pool.map(_window_job, wids, chunksize=4)
    print(f"[O1] windows {time.time() - t0:.0f}s", flush=True)
    by_w = {}
    for r in res:
        r["t"] = [g["H"][r["wid"]]["read_log"][i]["t"] for i in range(r["R"])]
        cbr = defaultdict(list)
        for st in r["states"]:
            covs = set(st["cov_rules"])
            for rid in g["items_of"].get(r["wid"], []):
                cbr[rid].append(rid in covs)
        r["cov_by_read"] = cbr
        by_w[r["wid"]] = r
    s1 = s1_check(g, by_w)
    print(f"[O1] S1 check {json.dumps(s1)}", flush=True)
    # population, rules, sizes
    pop = [(r["wid"], st) for r in res for st in r["states"] if st["pop"]]
    agg = Counter()
    for r in res:
        agg["windows"] += 1
        agg["reads"] += r["reads"]
        agg["guard_reads"] += r["guard_reads"]
        agg["open_end_reads"] += r["open_end_reads"]
        agg["items"] += r["n_items"]
        agg["utts_in_windows"] += r["n_utts"]
        agg["dur_eq_window_dur"] += r["dur_eq"]
        for st in r["states"]:
            agg["nonfinal_read_ends"] += not st["final"]
            agg["nonfinal_with_pending"] += (not st["final"]) and bool(st["pend"])
    rule_end = Counter()
    for r in res:
        last = r["states"][-1]
        for rid in g["items_of"].get(r["wid"], []):
            m = r["wid"].rsplit("_on_", 1)[0]
            rs = last["cov_rules"].get(rid, [])
            rec = g["cap"][m]["rec"][rid]
            rule_end[f"{'+'.join(rs) or 'none'}|recalled={int(rec)}"] += 1
    # 263-unit flags of the population
    rows = [json.loads(l) for l in open(S1P_ROWS, encoding="utf-8")]
    miss = {row["rid"]: row for row in rows}
    n_fw = [st["n_fw"] for _, st in pop]
    for wid, st in pop:
        st["miss263"] = sorted(set(st["pend"]) & set(miss))
    # O2 selection (pre-registered budget rule)
    keep = [(wid, st) for wid, st in pop if st["n_fw"] <= FW_STATE_CAP]
    excluded = len(pop) - len(keep)
    order = list(range(len(keep)))
    random.Random(SEED).shuffle(order)
    total = sum(st["n_fw"] for _, st in keep)
    if total <= FW_BUDGET:
        uni = set(order)
    else:
        uni, acc = set(), 0
        for j in order:
            if acc + keep[j][1]["n_fw"] > FW_BUDGET:
                break
            uni.add(j)
            acc += keep[j][1]["n_fw"]
    sel = [j for j in range(len(keep)) if j in uni or keep[j][1]["miss263"]]
    # review sample (O1 iv): N_REVIEW random population states
    rv = random.Random(SEED + 1).sample(range(len(pop)), min(N_REVIEW, len(pop)))
    out.mkdir(parents=True)
    write_review(out, g, by_w, [pop[j] for j in rv])
    with open(out / "states.jsonl", "x", encoding="utf-8") as fh:
        for j in sel:
            wid, st = keep[j]
            w = g["Wid"][wid]
            fh.write(json.dumps({"wid": wid, "i": st["i"], "uniform": j in uni, "miss263": st["miss263"],
                                 "pend": st["pend"], "n_fw": st["n_fw"], "rec": st["rec"], "block": st["block"],
                                 "audio": w["audio"], "post": w["post"]["lowlat"],
                                 "close_detect_ms": w["close_detect_ms"]}) + "\n")
    rep = {"decode": DECODE, "recalled_eq_rule_b": dict(rec_eq), "s1_check": s1, "counts": dict(agg),
           "coverage_rules_at_window_end": dict(rule_end),
           "population": {"states": len(pop), "excluded_over_cap": excluded,
                          "planned_forwards": {"sum": int(sum(n_fw)), "mean": float(np.mean(n_fw)) if n_fw else 0,
                                               "max": int(max(n_fw)) if n_fw else 0},
                          "budget": FW_BUDGET, "uniform_selected": len(uni), "selected_total": len(sel),
                          "selected_forwards": int(sum(keep[j][1]["n_fw"] for j in sel)),
                          "states_with_missed_unit_pending": sum(bool(st["miss263"]) for _, st in pop),
                          "states_with_overdue": sum(bool(st["rec"]["overdue"]) for _, st in pop)},
           "review_states": [f"{pop[j][0]}|{pop[j][1]['i']}" for j in rv], "elapsed_s": time.time() - t0}
    (out / "o1.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    # O3 on real states: those with <= 400k paths (checked inside), a deterministic subset with overdue chunks first
    small = [st["rec"] for _, st in pop if len(st["rec"]["chunks"]) <= 12]
    small.sort(key=lambda r: (-len(r["overdue"]), r["id"]))
    o3 = run_o3(args, small[: args.o3_real])
    (out / "o3_real_states.json").write_text(json.dumps(o3, indent=1), encoding="utf-8")
    print(f"O1_STAGE_DONE {out} states={len(pop)} selected={len(sel)} O3_real={json.dumps(o3)}", flush=True)


def write_review(out: Path, g: dict, by_w: dict, sample: list) -> None:
    """the hand-check file: per sampled state, every due-or-overdue utterance with its decision, and the model's
    written segments of the window (and of earlier windows within 3 s) with label, mapped speaker, time, read."""
    lines, js = [], []
    for n, (wid, st) in enumerate(sample):
        m = wid.rsplit("_on_", 1)[0]
        w = g["Wid"][wid]
        sc, mp = g["cap"][m]["sc"], g["cap"][m]["mp"]
        meta_m, seglist = g["meta"][m], g["segs"][m]
        t = by_w[wid]["t"][st["i"]]
        lines.append(f"\n## S{n:02d} {wid} read {st['i']} (t = {t} ms, window start {w['start_s']:.2f} s)\n")
        lines.append("Reference utterances due by this read (meeting clock):")
        dec = []
        for rid in st["due_rids"]:
            u = sc[rid]
            rules = st["cov_rules"].get(rid, [])
            lines.append(f"- [{rid[-8:]}] {u['spk']} {u['a']:.2f}-{u['b']:.2f} ({u['b'] - u['a']:.2f} s): \"{u['ref']}\" "
                         f"-> {'COVERED by ' + '+'.join(rules) if rules else 'NOT covered'}")
            dec.append({"rid": rid, "covered": bool(rules), "rules": rules})
        lo = min(sc[r]["a"] for r in st["due_rids"]) - 3.0
        hi = max(sc[r]["b"] for r in st["due_rids"]) + 3.0
        lines.append("Model segments written so far (this window; earlier windows only if within the range):")
        for k, x in enumerate(seglist):
            mk = meta_m[k]
            vis = visible(k, meta_m, wid, w["start_s"], t)
            if not vis or not x["text"] or (mk["wid"] != wid and (x["b"] < lo or x["a"] > hi)):
                continue
            lines.append(f"  * k{k} slot{x['spk']}->{mp.get(x['spk'])} {x['a']:.2f}-{x['b']:.2f} read@{mk['write_ms']}"
                         f"{'' if mk['wid'] == wid else ' (earlier window)'}: \"{x['text']}\"")
        js.append({"n": n, "wid": wid, "i": st["i"], "decisions": dec})
    (out / "review_states.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "review_states.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in js)


def review_agreement(root: Path) -> dict:
    """O1 (iv): judgments.jsonl lines {"n", "rid", "correct": bool, "note"} for every decision of review_states.jsonl."""
    dec = [(x["n"], d["rid"]) for x in map(json.loads, open(root / "o1" / "review_states.jsonl")) for d in x["decisions"]]
    jd = {(x["n"], x["rid"]): x for x in map(json.loads, open(root / "o1" / "review_judgments.jsonl"))}
    assert set(jd) == set(dec), "every decision must be judged exactly once"
    ok = sum(bool(jd[k]["correct"]) for k in dec)
    per = defaultdict(list)
    for k in dec:
        per[k[0]].append(bool(jd[k]["correct"]))
    return {"decisions": len(dec), "correct": ok, "agreement": ok / max(len(dec), 1), "states": len(per),
            "states_all_correct": sum(all(v) for v in per.values()),
            "wrong": [jd[k] for k in dec if not jd[k]["correct"]]}


# ------------------------------------------------------------------------------------------------ O2 scoring (GPU)
def positions(L: int, m: int, K: int) -> tuple[list[int], list[int]]:
    """In a left-padded row of length L whose last m tokens are the forced ones: the positions predicting forced[0..m-1]
    and then e (m + 1), and the K positions predicting the K prompt tokens just before the forced ones (the read block)."""
    pf = [L - m - 1 + j for j in range(m + 1)]
    pr = [L - m - K - 1 + j for j in range(K)]
    return pf, pr


def score_reqs_do(eng, reqs: list[dict], batch: int) -> list[dict]:
    """reqs: {wid, audio, post, close_detect_ms, t, final, hist, forced, act, block}. Per req: tl [m], ec [m + 1]
    (e = <eos> at the final read, where <wait> is barred before the softmax as Engine.read), and with act: the log-softmax
    at s over ACTS (no bar: s is a non-final read) and the greedy replay of the block and of the token after it."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.evaluation.online_decode_finalpost import final_cut_ms
    from experiments.mist_w3po_v1.rl.rollout import load_window

    act_ids = [S.WAIT_ID, S.EOS_ID] + [S.SPK0 + k for k in range(8)]
    wavs, out = {}, []
    emb_dtype = eng.emb_layer.weight.dtype
    for s0 in range(0, len(reqs), batch):
        part = reqs[s0: s0 + batch]
        its = []
        for q in part:
            if q["wid"] not in wavs:
                wavs[q["wid"]] = load_window({"audio": q["audio"], "post": {"lowlat": q["post"]}})
            wav, post = wavs[q["wid"]]
            if q["final"]:
                p = post.copy()
                p[final_cut_ms(q["t"], q["close_detect_ms"]) // 80:] = 0.0
                its.append(ReadItem(wav=wav, post=p, t_ms=q["t"], final=True, committed=list(q["hist"]) + list(q["forced"]),
                                    ctx="", akey=f"{q['wid']}|{q['t']}|fp", full=True))
            else:
                its.append(ReadItem(wav=wav, post=post, t_ms=q["t"], final=False,
                                    committed=list(q["hist"]) + list(q["forced"]), ctx="", akey=f"{q['wid']}|{q['t']}|nf"))
        with torch.no_grad():
            ae = eng.audio_embeds(its, emb_dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            B, L = len(its), max(len(p) for p in prompts)
            ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            for i, p in enumerate(prompts):
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(eng.dev), mask.to(eng.dev)
            emb = eng.emb_layer(ids)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            hs = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            for i, (q, it) in enumerate(zip(part, its)):
                m = len(q["forced"])
                K = len(q["block"]) if q["act"] else 0
                pf, pr = positions(L, m, K)
                lg = eng.logits(hs[i, pf + pr]).to(eng.dev)
                if it.final:
                    lg[:, S.WAIT_ID] = -1e9
                lsm = torch.log_softmax(lg, -1)
                f = torch.as_tensor(q["forced"], dtype=torch.long, device=eng.dev)
                tl = lsm[torch.arange(m, device=eng.dev), f] if m else lsm[:0, 0]
                ec = lsm[: m + 1, S.EOS_ID if it.final else S.WAIT_ID]
                r = {"tl": tl.double().cpu().numpy(), "ec": ec.double().cpu().numpy()}
                if q["act"]:
                    assert prompts[i][len(prompts[i]) - m - K: len(prompts[i]) - m] == list(q["block"])
                    r["act"] = lsm[0, act_ids].double().cpu().tolist()
                    r["argmax_s"] = int(lsm[0].argmax())
                    am = lsm[m + 1:].argmax(-1).cpu().tolist()
                    r["replay_ok"] = am == list(q["block"])
                    r["replay_first_bad"] = next((j for j, (x, y) in enumerate(zip(am, q["block"])) if x != y), None)
                out.append(r)
            del hs
    return out


def state_reqs(st: dict) -> tuple[dict, dict, list[dict]]:
    rec = st["rec"]
    plan = plan_state(rec)
    R = rec["n_read"]
    act_q = plan["reads"][0]["ctx"][0]["first_req"]
    reqs = []
    for k, q in enumerate(plan["reqs"]):
        reqs.append({"wid": st["wid"], "audio": st["audio"], "post": st["post"], "close_detect_ms": st["close_detect_ms"],
                     "t": rec["t_ms"][q["r"]], "final": q["r"] == R - 1, "hist": q["hist_ids"], "forced": q["forced_ids"],
                     "act": k == act_q, "block": st["block"] if k == act_q else []})
    return rec, plan, reqs


def state_result(st: dict, rec: dict, plan: dict, res: list[dict]) -> dict:
    reads = VP.assemble_p(plan, [(r["tl"], r["ec"]) for r in res])
    ora = oracle(rec, plan, reads, TAU)
    a = next(r for r in res if "act" in r)
    lp = np.asarray(a["act"])
    pbar = np.exp(lp - V.lse(lp))
    q = np.asarray([ora["q"][x] for x in ACTS])
    kl = float(sum(qi * (math.log(qi) - math.log(pi)) for qi, pi in zip(q, pbar) if qi > 0))
    psis = [x for rd in reads for x in ([] if rd["psi"] is None else np.ravel(rd["psi"]).tolist())]
    return {"wid": st["wid"], "i": st["i"], "uniform": st["uniform"], "miss263": st["miss263"], "pend": st["pend"],
            "n_fw": len(plan["reqs"]), "n_items": len(rec["chunks"]), "n_overdue": len(rec["overdue"]),
            "q": ora["q"], "logZ": ora["logZ"], "p_act": a["act"], "p_wait": math.exp(a["act"][0]),
            "kl": kl, "argmax_s": a["argmax_s"], "replay_ok": a["replay_ok"], "replay_first_bad": a["replay_first_bad"],
            "psi_finite": bool(np.all(np.isfinite(psis))), "q_sum": float(q.sum()),
            "D": bool(ora["q"]["wait"] < 0.5 and math.exp(a["act"][0]) >= 0.5),
            "reads": [{"psi": rd["psi"], "psi_nf": rd["psi_nf"]} for rd in reads]}


def run_score(args) -> None:
    import torch

    from experiments.mist_w3po_v1.core.engine import Engine

    root = Path(args.root)
    states = [json.loads(l) for l in open(root / "o1" / "states.jsonl", encoding="utf-8")]
    sh = P.parse_shard(args.shard)
    if sh:   # LPT by planned forwards, deterministic
        load = [0] * sh[1]
        mine = []
        for j in sorted(range(len(states)), key=lambda j: (-states[j]["n_fw"], j)):
            b = min(range(sh[1]), key=lambda x: (load[x], x))
            load[b] += states[j]["n_fw"]
            if b == sh[0]:
                mine.append(j)
        states = [states[j] for j in sorted(mine)]
    if args.limit:
        states = states[: args.limit]
    name = f"shard_{sh[0]}of{sh[1]}" if sh else (f"limit{args.limit}" if args.limit else "all")
    out = Path(args.out_root or root) / name
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    (out / "meta.json").write_text(json.dumps({"n_states": len(states), "forwards_planned": sum(s["n_fw"] for s in states),
                                               "args": vars(args)}, indent=1))
    t0 = time.time()
    eng = Engine.from_merged(args.thinker, "zh2en", args.max_gib)
    load_s = time.time() - t0
    bad = 0
    for s in states:                                           # the engine tokenizer == the CPU stage's on the pieces
        for c in s["rec"]["chunks"]:
            bad += eng.tok(c["piece"], add_special_tokens=False)["input_ids"] != s["rec"]["S_ids"][c["tok0"]: c["tok1"]]
    if bad:
        raise SystemExit(f"tokenizer mismatch on {bad} pieces")
    n_fw, t1 = 0, time.time()
    with open(out / "states.jsonl", "x", encoding="utf-8") as fh:
        for s0 in range(0, len(states), args.group):
            grp = states[s0: s0 + args.group]
            built = [state_reqs(st) for st in grp]
            allr = [q for _, _, rq in built for q in rq]
            order = sorted(range(len(allr)), key=lambda k: (allr[k]["wid"], allr[k]["t"], len(allr[k]["hist"])))
            res_o = score_reqs_do(eng, [allr[k] for k in order], args.batch)
            res = [None] * len(allr)
            for k, r in zip(order, res_o):
                res[k] = r
            pos = 0
            for st, (rec, plan, rq) in zip(grp, built):
                fh.write(json.dumps(state_result(st, rec, plan, res[pos: pos + len(rq)])) + "\n")
                pos += len(rq)
            n_fw += len(allr)
            fh.flush()
            print(f"{s0 + len(grp)}/{len(states)} states, {n_fw} forwards, {time.time() - t1:.0f}s", flush=True)
    peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
    st = {"states": len(states), "forwards": n_fw, "load_s": load_s, "scoring_s": time.time() - t1, "peak_alloc_gib": peak}
    (out / "done.json").write_text(json.dumps(st, indent=1))
    print(f"DO_SCORED {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def qs(x) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))}


def run_analyse(args) -> None:
    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    dirs = sorted(d for d in [*root.glob("shard_*of*"), *[Path(x) for x in args.extra_dirs]] if (d / "done.json").exists())
    rows, seen = [], set()
    for d in dirs:
        for r in map(json.loads, open(d / "states.jsonl", encoding="utf-8")):
            k = (r["wid"], r["i"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    sel = [json.loads(l) for l in open(root / "o1" / "states.jsonl", encoding="utf-8")]
    missing = {(s["wid"], s["i"]) for s in sel} - seen
    o1 = json.load(open(root / "o1" / "o1.json"))["population"]
    s1 = {x["rid"]: x for x in map(json.loads, open(S1P_ROWS, encoding="utf-8"))}
    uni = [r for r in rows if r["uniform"]]
    m263 = [r for r in rows if r["miss263"]]

    def block(rs):
        return {"n": len(rs), "D": sum(r["D"] for r in rs) / max(len(rs), 1), "n_D": sum(r["D"] for r in rs),
                "q_wait": qs([r["q"]["wait"] for r in rs]), "p_wait": qs([r["p_wait"] for r in rs]),
                "kl": qs([r["kl"] for r in rs]),
                "share_q_wait_lt_05": float(np.mean([r["q"]["wait"] < 0.5 for r in rs])) if rs else None,
                "share_p_wait_ge_05": float(np.mean([r["p_wait"] >= 0.5 for r in rs])) if rs else None,
                "replay_block_ok": float(np.mean([r["replay_ok"] for r in rs])) if rs else None,
                "argmax_s_is_wait": float(np.mean([r["argmax_s"] == S.WAIT_ID for r in rs])) if rs else None,
                "with_overdue": sum(r["n_overdue"] > 0 for r in rs),
                "D_with_overdue": (sum(r["D"] for r in rs if r["n_overdue"]) / max(sum(r["n_overdue"] > 0 for r in rs), 1)),
                "D_without_overdue": (sum(r["D"] for r in rs if not r["n_overdue"])
                                      / max(sum(r["n_overdue"] == 0 for r in rs), 1))}
    rep = {"rule": f"D = share of population states with q(wait) < 0.5 and p(wait) >= 0.5; < {O2_STOP} -> STOP",
           "dirs": [str(d) for d in dirs], "missing_states": len(missing), "uniform": block(uni), "miss263": block(m263),
           "all_scored": block(rows), "o1_population": o1, "run_checks": {"psi_finite": all(r["psi_finite"] for r in rows),
                                                     "q_sum_max_dev": max(abs(r["q_sum"] - 1) for r in rows)}}
    units = defaultdict(lambda: {"states": 0, "dis": 0})
    for r in m263:
        for rid in r["miss263"]:
            units[rid]["states"] += 1
            units[rid]["dis"] += r["D"]
    for cls in ("CHAIN", "NOTHING_ELSE_WRITTEN_AT_NEXT"):
        us = [rid for rid, x in s1.items() if x.get("why2") == cls]
        rep[f"units_{cls}"] = {"n": len(us), "with_state": sum(units[r]["states"] > 0 for r in us),
                               "with_disagreeing_state": sum(units[r]["dis"] > 0 for r in us)}
    rep["units_all263"] = {"n": len(s1), "with_state": sum(units[r]["states"] > 0 for r in s1),
                           "with_disagreeing_state": sum(units[r]["dis"] > 0 for r in s1)}
    D = rep["uniform"]["D"]
    o3 = bool(json.load(open(root / "o3" / "o3.json"))["pass_all"]
              and json.load(open(root / "o1" / "o3_real_states.json"))["pass_all"])
    rep["O3_pass"] = o3
    rep["O2_D"] = D
    rep["O2_stop"] = D < O2_STOP
    rep["note"] = "O1 (iv) agreement and the GO decision are written in the report from review_agreement()"
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"ANALYSE_DONE D={D:.4f} n={rep['uniform']['n']} STOP={rep['O2_stop']} O3={o3}", flush=True)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=False)
    ap.add_argument("--o3", action="store_true", help="O3 on synthetic instances -> <root>/o3")
    ap.add_argument("--o1", action="store_true", help="O1 + the O2 states + O3 on real states -> <root>/o1")
    ap.add_argument("--review", action="store_true", help="O1 (iv) agreement from <root>/o1/review_judgments.jsonl")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--o3-real", type=int, default=300)
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thinker", default=P.V8)
    ap.add_argument("--max-gib", type=int, default=80)
    ap.add_argument("--extra-dirs", nargs="*", default=[])
    args = ap.parse_args()
    if args.o3:
        out = Path(args.root) / "o3"
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        rep = run_o3(args, None)
        out.mkdir(parents=True)
        (out / "o3.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"O3_SYN_DONE {json.dumps(rep)}", flush=True)
    elif args.o1:
        run_o1(args)
    elif args.review:
        rep = review_agreement(Path(args.root))
        p = Path(args.root) / "o1" / "review_agreement.json"
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        p.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"REVIEW agreement={rep['agreement']:.4f} ({rep['correct']}/{rep['decisions']}) STOP={rep['agreement'] < O1_STOP}")
    elif args.score:
        run_score(args)
    elif args.analyse:
        run_analyse(args)
    else:
        ap.error("one of --o3 / --o1 / --review / --score / --analyse")


if __name__ == "__main__":
    main()
