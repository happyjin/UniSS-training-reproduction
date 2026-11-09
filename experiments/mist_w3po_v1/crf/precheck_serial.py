"""SerialCRF (scheme 1'-b) precheck: crf_novel_ideas.zh-CN.md §13.6 (P0-P3), on the windows of the scheme-1' precheck
(sched_precheck_v8.zh-CN.md: train 500 = sched_v1 sample n 500, dev 300 = sched_dev_v1 sample n 300, seed 20261109).
Precheck only: no training code. Dev (never trained on) is the primary evidence, train is secondary.

Items and orders (§13.2)
  items   the chunks of sched_v1 / sched_dev_v1 (crf/build_sched_windows.py: v8's chunks, every one a closed G-STAR
          segment <spk><t_a>text<t_b>). rho(c) = the heuristic (v8 label) commit read of chunk c = the first read whose
          beta_heur covers its last token. B_r = {c : rho(c) = r} in label (S) order.
  orders  Pi(B_r) = the linear extensions of B_r: chunks of the same unit keep their label order; a continuation chunk
          of a segment left open by the history goes first (never binds here: every history ends after a closing time
          token, counted). The label order is Pi[0]. |B_r| > CAP (4): the label order only (share reported).
  kappa   h_r = kappa(the chunks committed before r) = those chunks sorted by a key of the chunk alone (a function of
          the SET, not of the earlier orders). §13.2 writes the key as (rho, unit start a_u, chunk index): KEY_S13,
          tested and reported. It is NOT the S prefix (v8 / the precheck sort within a commit read by the chunk's
          commit time = source end + 1 s, not by the unit's start), so the scoring uses KEY = (rho, source end,
          source start, chunk index), also a set function; the exactness test must pass for KEY before any GPU use.
  Exactness test (CPU, run again by every scoring run on its windows before the model is loaded): for every read r of
          every window, ids(kappa(H_r)) == S_ids[:beta_heur[r-1]] (order level, all windows of both scheds) and, for the
          precheck samples, the kappa string tokenised jointly (the trainer's tokenizer) == that prefix and == the
          "prefix" string of v8's own row at that read (rows_train / rows_validation), with the label B_r string ==
          the row's "target".

Scoring (GPU; the engine path of crf/precheck_sched.score_window: Engine.from_merged, Engine.audio_embeds (P zeroed
  after t_r - 1.04 s), Prompter.ids with ctx "", Engine.read-style left padding + mask, <wait> barred at the final read
  at every position): per read with |B_r| >= 2 and per pi in Pi(B_r), one teacher-forced forward of
      prompt(audio <= t_r, ctx "", h_r) + B_r^pi + e_r          (e_r = <wait>, <eos> at the last read)
  stored per pi: psi_r(pi) = sum of the token log-probs of B_r^pi + log p(e_r | ..), the log-prob of the first item's
  full token sequence, the per-token log-probs, and log p(e_r) at every item boundary.

Metrics (§13.6; pre-registered stop rules unchanged)
  P0  coverage (CPU): the dev ORDER_OMISSION short units of check A (diag_order_attr_v1/check_a_missed.jsonl; primary:
      the overlapped (ov50) ones, 151), mapped by record id to their dev reference window (all of sched_dev_v1);
      due read = rho of the unit's last chunk (= v8's label rule, source end + 1 s at the next read). P0 = share whose
      due read has |B_r| >= 2. < 30 % -> STOP.
  P1  (report only) per read with |Pi| >= 2: q(pi) = softmax psi; share with q(non-label) >= 0.1; the tempered
      q* ∝ q^{1/(1+mu)}, mu >= 0 the smallest with H(q*) >= eta log|Pi| (eta 0.5, bisection), mean q*(non-label).
  P2  per read with |Pi| >= 2: f = argmax over the possible first items of log p(y_f | x, c, h_r);
      P_complete(f) = sum_{pi: pi_1 = f} e^{psi(pi)} / p(y_f | ..). Median / quartiles for f != label-first and
      f = label-first, all reads and reads with a <= 1 s ov50 unit. STOP if the dev median for f != label-first >= 0.8.
  P3  label path: psi(label) vs the earlier precheck's sum(tok_lp[r, beta_{r-1}:beta_r]) + end_lp[r, beta_r] (same
      reads, its logprobs/*.npz); max / median absolute difference (bf16 batch noise expected).
  plus: multi-item reads, the |B_r| and |Pi| distributions, the share of windows with any multi-item read.

    # CPU stage (exactness, P0, plan) -> <root>/cpu (root must be new)
    python -m experiments.mist_w3po_v1.crf.precheck_serial --cpu-stage --root <new root>
    # scoring (GPU), one model load per shard, splits in the given order -> <root>/<split>_shard_KofN
    python -m experiments.mist_w3po_v1.crf.precheck_serial --score --root <root> --shard K/N --splits dev,train
    # smoke: the first n windows (with a read of |Pi| >= 2) of the dev sample
    python -m experiments.mist_w3po_v1.crf.precheck_serial --score --root <work dir> --splits dev --limit 2
    # analysis (CPU) -> <root>/analysis
    python -m experiments.mist_w3po_v1.crf.precheck_serial --analyse --root <root>
    # plan + analysis on random scores, nothing written
    python -m experiments.mist_w3po_v1.crf.precheck_serial --dry-run
"""
from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_sched as P

SCHEDS = {"train": P.SCHED, "dev": "eval_outputs/mist_w3po_v1/crf/sched_dev_v1"}
NS = {"train": 500, "dev": 300}
SEED = 20261109
PREV = {"train": "eval_outputs/mist_w3po_v1/crf/precheck_v8_500", "dev": "eval_outputs/mist_w3po_v1/crf/precheck_v8_dev300"}
V8_ROWS = {"train": "data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_train.jsonl",
           "dev": "data/ami_multispk_si_v1/e2e_gstar_stream_zh2en_v1/rows_validation.jsonl"}
CHECK_A = "eval_outputs/mist_w3po_v1/crf/diag_order_attr_v1/check_a_missed.jsonl"
REF_DEV = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"
WAIT_ID, EOS_ID = P.WAIT_ID, P.EOS_ID
NEW0, N_TIME, N_SLOT = 151676, 251, 8
SPK0 = NEW0 + N_TIME
NEW_END = SPK0 + N_SLOT
CAP = 4
ETA = 0.5
P0_STOP = 0.30
P1_MASS = 0.1
P2_STOP = 0.8
SHORT_S = 1.0


# ------------------------------------------------------------------------------------------------ items, orders, kappa
def chunk_reads(rec: dict) -> list[int]:
    """rho per chunk: the first read whose heuristic committed count covers the chunk's last token."""
    beta = rec["beta_heur"]
    return [bisect.bisect_left(beta, c["tok1"]) for c in rec["chunks"]]


def read_items(rec: dict, rho: list[int] | None = None) -> list[list[int]]:
    """B_r per read: the chunk indices committed at read r, in label (S) order."""
    rho = chunk_reads(rec) if rho is None else rho
    B = [[] for _ in range(rec["n_read"])]
    for ci, r in enumerate(rho):
        B[r].append(ci)
    return B


def key_of(rec: dict, rho: list[int], ci: int, key: str = "srcend") -> tuple:
    """kappa's sort key of chunk ci (a function of the chunk and its commit read only)."""
    c = rec["chunks"][ci]
    if key == "srcend":                       # (rho, source end, source start, chunk index): exact, used
        return rho[ci], c["src_end_ms"], c["src_start_ms"], ci
    if key == "s13":                          # §13.2 as written: (rho, unit source start a_u, chunk index)
        a_u = next(u["a_ms"] for u in rec["units"] if u["utt"] == c["utt"])
        return rho[ci], a_u, ci
    raise ValueError(key)


def kappa(rec: dict, rho: list[int], H, key: str = "srcend") -> list[int]:
    """The canonical serialisation of the committed set H (chunk indices; any iterable, its order is ignored)."""
    return sorted(set(H), key=lambda ci: key_of(rec, rho, ci, key))


def ids_of(rec: dict, order) -> list[int]:
    S = rec["S_ids"]
    return [t for ci in order for t in S[rec["chunks"][ci]["tok0"]: rec["chunks"][ci]["tok1"]]]


def string_of(rec: dict, order) -> str:
    return "".join(rec["chunks"][ci]["piece"] for ci in order)


def seg_open(ids: list[int]) -> bool:
    """core/engine.seg_state(ids) != "out": a segment is left open (speaker token seen, closing time token not yet)."""
    st = "out"
    for i in ids:
        if SPK0 <= i < NEW_END:
            st = "hdr"
        elif NEW0 <= i < SPK0:
            st = "text" if st == "hdr" else "out"
    return st != "out"


def orders(rec: dict, B: list[int], open_first: int | None = None) -> list[tuple]:
    """Pi(B): the label order first, then the other linear extensions (permutation order): the chunks of one unit keep
    their label order; open_first (the chunk that continues a segment left open by the history) stays first."""
    lab = tuple(B)
    utt = {ci: rec["chunks"][ci]["utt"] for ci in B}
    seq = defaultdict(list)
    for ci in B:
        seq[utt[ci]].append(ci)
    out = [lab]
    for p in itertools.permutations(B):
        if p == lab or (open_first is not None and p[0] != open_first):
            continue
        sub = defaultdict(list)
        for ci in p:
            sub[utt[ci]].append(ci)
        if all(sub[u] == seq[u] for u in seq):
            out.append(p)
    return out


def n_orders(rec: dict, B: list[int]) -> int:
    """|Pi(B)| (multinomial; open_first never binds in this construction)."""
    cnt = Counter(rec["chunks"][ci]["utt"] for ci in B)
    n = math.factorial(len(B))
    for v in cnt.values():
        n //= math.factorial(v)
    return n


def history(rec: dict, rho: list[int], r: int) -> list[int]:
    return [ci for ci, x in enumerate(rho) if x < r]


def read_plan(rec: dict, cap: int = CAP) -> list[dict]:
    """Per read with |B_r| >= 2: the history ids (kappa), the orders to score (all of Pi when |B_r| <= cap, else the
    label order only)."""
    rho = chunk_reads(rec)
    out = []
    for r, B in enumerate(read_items(rec, rho)):
        if len(B) < 2:
            continue
        h = ids_of(rec, kappa(rec, rho, history(rec, rho, r)))
        op = B[0] if seg_open(h) else None
        capped = len(B) > cap
        pis = [tuple(B)] if capped else orders(rec, B, op)
        out.append({"r": r, "B": B, "hist": h, "orders": pis, "capped": capped, "open": op is not None,
                    "n_pi": n_orders(rec, B)})
    return out


# ------------------------------------------------------------------------------------------------ exactness (CPU)
def exactness_window(rec: dict, tok=None, rows: dict | None = None, keep: int = 0) -> dict:
    """The exactness test of one window; rows: {t_ms: v8 row} of the window (None: not checked)."""
    rho = chunk_reads(rec)
    assert all(a["tok1"] == b["tok0"] for a, b in zip(rec["chunks"], rec["chunks"][1:])), rec["id"]
    beta = [0] + list(rec["beta_heur"])
    st = Counter()
    bad13 = []
    B = read_items(rec, rho)
    for r in range(rec["n_read"]):
        H = history(rec, rho, r)
        want = rec["S_ids"][: beta[r]]
        k = kappa(rec, rho, H)
        st["reads"] += 1
        st["kappa_ids_ok"] += ids_of(rec, k) == want
        keys = [key_of(rec, rho, ci)[:3] for ci in H]
        st["kappa_key_ties"] += len(keys) != len(set(keys))
        st["hist_open"] += seg_open(want)
        if tok is not None:
            st["tok_checked"] += 1
            st["kappa_tok_ok"] += tok(string_of(rec, k), add_special_tokens=False)["input_ids"] == want
        if rows is not None and rec["t_ms"][r] in rows:
            row = rows[rec["t_ms"][r]]
            st["v8_rows"] += 1
            st["v8_prefix_ok"] += row["prefix"] == string_of(rec, k)
            st["v8_target_ok"] += row["target"] == string_of(rec, B[r])
            st["v8_end_ok"] += row["end"] == ("eos" if r == rec["n_read"] - 1 else "wait")
        k13 = kappa(rec, rho, H, "s13")
        if k13 == H:
            st["s13_ok"] += 1
        else:
            j = next(i for i, (a, b) in enumerate(zip(k13, H)) if a != b)
            x, y = rec["chunks"][H[j]], rec["chunks"][k13[j]]       # S puts x first, §13's key puts y first
            cont = any(c["utt"] == y["utt"] for c in rec["chunks"][: k13[j]])
            # y continues a unit that started earlier (a_u earlier, its own chunk ends later), or y is another unit
            # that starts earlier than x's unit but ends later (S orders by the end, §13's key by the start)
            cause = ("S13_FIRST_IS_CONTINUATION_CHUNK" if cont else "S13_FIRST_STARTS_EARLIER_ENDS_LATER")
            st["s13_bad"] += 1
            st["s13_bad_" + cause] += 1
            st["s13_bad_same_read"] += rho[H[j]] == rho[k13[j]]
            if len(bad13) < keep or keep < 0:
                bad13.append({"id": rec["id"], "r": r, "pos": j, "rho": rho[H[j]], "cause": cause,
                              "S_first": {"chunk": H[j], "utt": x["utt"], "src_start_ms": x["src_start_ms"],
                                          "src_end_ms": x["src_end_ms"], "commit_ms": x["commit_ms"], "piece": x["piece"]},
                              "s13_first": {"chunk": k13[j], "utt": y["utt"], "src_start_ms": y["src_start_ms"],
                                            "src_end_ms": y["src_end_ms"], "commit_ms": y["commit_ms"],
                                            "piece": y["piece"]}})
    st["windows"] = 1
    st["windows_s13_bad"] = int(st["s13_bad"] > 0)
    return {"stats": st, "s13_bad": bad13}


def load_all(sched: str) -> list[dict]:
    recs = []
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    return [r for r in recs if r["n_tok"] > 0]


def v8_rows_of(path: str, ids: set) -> dict:
    out = defaultdict(dict)
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("kind") != "read":
            continue
        wid = r["id"].split("|")[0]
        if wid in ids:
            out[wid][r["t_ms"]] = r
    return out


def exactness_ok(st: dict) -> bool:
    ok = st["kappa_ids_ok"] == st["reads"]
    if st.get("tok_checked"):
        ok &= st["kappa_tok_ok"] == st["tok_checked"]
    if st.get("v8_rows"):
        ok &= st["v8_prefix_ok"] == st["v8_target_ok"] == st["v8_end_ok"] == st["v8_rows"]
    return bool(ok)


def check_exact_or_die(recs: list[dict]) -> dict:
    """The order-level exactness test on the windows about to be scored (before any model is loaded)."""
    st = Counter()
    for rec in recs:
        st.update(exactness_window(rec)["stats"])
    if not exactness_ok(st):
        raise SystemExit(f"EXACTNESS FAILED (kappa != S prefix), no scoring: {dict(st)}")
    return dict(st)


# ------------------------------------------------------------------------------------------------ P0 (CPU)
def p0_coverage(check_a: str = CHECK_A, ref: str = REF_DEV, sched: str = SCHEDS["dev"],
                sample_ids: set | None = None) -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in open(check_a, encoding="utf-8")]
    oo = [r for r in rows if r["cls"] == "ORDER_OMISSION"]
    rid_of = {}
    for l in open(ref, encoding="utf-8"):
        w = json.loads(l)
        rid_of[w["id"]] = [u["record_id"] for u in w["utts"]]
    where = defaultdict(list)
    recs = {}
    for rec in load_all(sched):
        recs[rec["id"]] = rec
        for u in rec["units"]:
            where[rid_of[rec["id"]][u["utt"]]].append((rec["id"], u["utt"]))
    units = []
    for o in oo:
        x = {k: o[k] for k in ("meeting", "rid", "dur", "le05", "ov50", "spk", "a", "b", "due_final",
                                "order_same_read")}
        locs = where.get(o["rid"], [])
        x["ref_windows"] = len(locs)
        if locs:
            wid, ui = locs[0]
            rec = recs[wid]
            rho = chunk_reads(rec)
            B = read_items(rec, rho)
            mine = [ci for ci, c in enumerate(rec["chunks"]) if c["utt"] == ui]
            r = rho[mine[-1]]
            x.update({"window": wid, "utt": ui, "due_read": r, "due_is_final": r == rec["n_read"] - 1,
                      "n_items": len(B[r]), "n_pi": n_orders(rec, B[r]),
                      "other_unit_in_B": any(rec["chunks"][ci]["utt"] != ui for ci in B[r]),
                      "in_sample": sample_ids is not None and wid in sample_ids})
        units.append(x)

    def grp(sel):
        g = [u for u in units if sel(u)]
        found = [u for u in g if u["ref_windows"]]
        n = len(g)

        def sh(f):
            return sum(1 for u in found if f(u)) / max(n, 1)
        return {"n": n, "found_in_ref_windows": len(found),
                "multi_ref_windows": sum(1 for u in g if u["ref_windows"] > 1),
                "P0_share_B_ge2": sh(lambda u: u["n_items"] >= 2),
                "share_B_ge2_other_unit": sh(lambda u: u["n_items"] >= 2 and u["other_unit_in_B"]),
                "share_Pi_ge2": sh(lambda u: u["n_pi"] >= 2),
                "share_Pi_ge2_within_cap": sh(lambda u: u["n_pi"] >= 2 and u["n_items"] <= CAP),
                "share_due_final": sh(lambda u: u["due_is_final"]),
                "n_items_dist": dict(sorted(Counter(u["n_items"] for u in found).items())),
                "in_sample": sum(1 for u in found if u["in_sample"])}
    rep = {"source": check_a, "sched": sched, "rule": f"P0 = share with |B_r| >= 2 at the due read; < {P0_STOP} -> STOP",
           "le1_ov50": grp(lambda u: u["ov50"]), "le05_ov50": grp(lambda u: u["ov50"] and u["le05"]),
           "le1_iso": grp(lambda u: not u["ov50"]), "le1_all": grp(lambda u: True)}
    rep["P0"] = rep["le1_ov50"]["P0_share_B_ge2"]
    rep["P0_stop"] = rep["P0"] < P0_STOP
    return rep, units


# ------------------------------------------------------------------------------------------------ counts (CPU)
def unit_info(rec: dict) -> dict:
    from experiments.mist_w3po_v1.crf import diag_short_units as D

    return {u["utt"]: {"dur_s": (u["b_ms"] - u["a_ms"]) / 1000.0, "ov50": bool(D.ov50(rec["units"], u)),
                       "slot": u["slot"]} for u in rec["units"]}


def counts(recs: list[dict], cap: int = CAP) -> dict:
    sizes, pis = Counter(), Counter()
    reads = multi = windows_multi = capped = pi1 = final = forwards = 0
    for rec in recs:
        rho = chunk_reads(rec)
        any_m = False
        for r, B in enumerate(read_items(rec, rho)):
            reads += 1
            if len(B) < 2:
                continue
            any_m = True
            multi += 1
            sizes[len(B)] += 1
            n = n_orders(rec, B)
            pis[n] += 1
            capped += len(B) > cap
            pi1 += n == 1
            final += r == rec["n_read"] - 1
            forwards += 1 if len(B) > cap else n
        windows_multi += any_m
    return {"windows": len(recs), "reads": reads, "multi_item_reads": multi,
            "share_reads_multi": multi / max(reads, 1),
            "B_size_dist": dict(sorted(sizes.items())), "Pi_size_dist": dict(sorted(pis.items())),
            "windows_with_multi": windows_multi, "share_windows_with_multi": windows_multi / max(len(recs), 1),
            "multi_above_cap": capped, "share_multi_above_cap": capped / max(multi, 1),
            "multi_Pi_eq_1": pi1, "share_multi_Pi_eq_1": pi1 / max(multi, 1),
            "multi_at_final_read": final, "forwards": forwards}


# ------------------------------------------------------------------------------------------------ scoring (GPU)
def score_reqs(eng, reqs: list[tuple], batch: int, final_bar_wait: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """reqs: (rec, r, history ids, forced ids). Per req: (tok_lp [m]: log p(forced_t | ..), end_col [m+1]: log p(e_r | ..)
    at every position from before the first forced token to after the last; e_r = <eos> at the last read, else
    <wait>). The engine path of precheck_sched.score_window (left padding + mask as Engine.read, <wait> barred at the
    final read at every position, before the softmax)."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.rl.rollout import load_window

    wavs, out = {}, []
    emb_dtype = eng.emb_layer.weight.dtype
    for s0 in range(0, len(reqs), batch):
        part = reqs[s0: s0 + batch]
        its, ms = [], []
        for rec, r, hist, forced in part:
            if rec["id"] not in wavs:
                wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
            wav, post = wavs[rec["id"]]
            its.append(ReadItem(wav=wav, post=post, t_ms=rec["t_ms"][r], final=r == rec["n_read"] - 1,
                                committed=list(hist) + list(forced), ctx="", akey=f"{rec['id']}|{rec['t_ms'][r]}"))
            ms.append(len(forced))
        with torch.no_grad():
            ae = eng.audio_embeds(its, emb_dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            B, L, M = len(its), max(len(p) for p in prompts), max(ms)
            ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((B, L), dtype=torch.long)
            for i, p in enumerate(prompts):                       # Engine.read: left padding
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(eng.dev), mask.to(eng.dev)
            emb = eng.emb_layer(ids)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            h = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                  return_dict=True).last_hidden_state[:, L - M - 1:]
            lg = eng.logits(h).to(eng.dev)                                    # [B, M+1, V] float32
            for i, it in enumerate(its):
                if it.final and final_bar_wait:
                    lg[i, :, WAIT_ID] = -1e9                                  # Engine.read line 246
            lsm = torch.log_softmax(lg, -1)
            for i, it in enumerate(its):
                tl, ec = slice_scores(lsm[i], ms[i], part[i][3], EOS_ID if it.final else WAIT_ID)
                out.append((tl.double().cpu().numpy(), ec.double().cpu().numpy()))
        del h, lg, lsm
    return out


def slice_scores(lsm_row, m: int, forced: list[int], e_id: int):
    """lsm_row [M+1, V] (the last M+1 positions of a left-padded row); this row's m forced tokens are the last m of the
    sequence, so position M - m + t predicts forced[t] and position M predicts e_r."""
    import torch

    M = lsm_row.shape[0] - 1
    rows = lsm_row[M - m:]
    tl = rows[torch.arange(m, device=rows.device), torch.as_tensor(forced, dtype=torch.long, device=rows.device)] \
        if m else rows[:0, 0]
    return tl, rows[:, e_id]


def score_split(eng, recs: list[dict], args, out: Path, split: str) -> dict:
    """Scores the multi-item reads of recs into out/reads.jsonl; one line per read."""
    t0 = time.time()
    n_fw = 0
    with open(out / "reads.jsonl", "x", encoding="utf-8") as fh:
        for k, rec in enumerate(recs):
            plan = read_plan(rec, args.cap)
            if not plan:
                continue
            reqs, where = [], []
            for pi_, rd in enumerate(plan):
                for oi, pi in enumerate(rd["orders"]):
                    reqs.append((rec, rd["r"], rd["hist"], ids_of(rec, pi)))
                    where.append((pi_, oi))
            res = score_reqs(eng, reqs, args.batch, not args.no_final_bar_wait)
            n_fw += len(reqs)
            ui = unit_info(rec)
            per = [[None] * len(rd["orders"]) for rd in plan]
            for (pi_, oi), (tl, ec) in zip(where, res):
                rd = plan[pi_]
                pi = rd["orders"][oi]
                lens = [rec["chunks"][ci]["tok1"] - rec["chunks"][ci]["tok0"] for ci in pi]
                bnd = np.cumsum([0] + lens)
                per[pi_][oi] = {"pi": list(pi), "psi": float(tl.sum() + ec[-1]), "first_lp": float(tl[: lens[0]].sum()),
                                "tok_lp": [float(x) for x in tl], "end_at_bounds": [float(ec[b]) for b in bnd]}
            for pi_, rd in enumerate(plan):
                items = []
                for ci in rd["B"]:
                    c = rec["chunks"][ci]
                    items.append({"chunk": ci, "utt": c["utt"], "slot": c["slot"], "n_tok": c["tok1"] - c["tok0"],
                                  **{kk: ui[c["utt"]][kk] for kk in ("dur_s", "ov50")}})
                fh.write(json.dumps({"id": rec["id"], "r": rd["r"], "t_ms": rec["t_ms"][rd["r"]],
                                     "final": rd["r"] == rec["n_read"] - 1, "n_read": rec["n_read"],
                                     "label": rd["B"], "items": items, "capped": rd["capped"], "n_pi": rd["n_pi"],
                                     "hist_len": len(rd["hist"]), "orders": per[pi_]}) + "\n")
            if (k + 1) % 20 == 0:
                print(f"[{split}] {k + 1}/{len(recs)} windows, {n_fw} forwards, {time.time() - t0:.0f}s", flush=True)
    return {"windows": len(recs), "forwards": n_fw, "scoring_s": time.time() - t0}


def sample(split: str) -> list[dict]:
    return P.load_records(SCHEDS[split], NS[split], SEED)


def run_score(args) -> None:
    import torch

    from experiments.mist_w3po_v1.core.engine import Engine

    root = Path(args.root)
    sh = P.parse_shard(args.shard)
    splits = [s for s in args.splits.split(",") if s]
    jobs = []
    for split in splits:
        recs = sample(split)
        if sh:
            recs = P.shard_select(recs, *sh)
        if args.limit:
            recs = [r for r in recs if any(rd["n_pi"] >= 2 and not rd["capped"] for rd in read_plan(r, args.cap))]
            recs = recs[: args.limit]
        name = f"{split}_shard_{sh[0]}of{sh[1]}" if sh else (f"{split}_limit{args.limit}" if args.limit else split)
        out = root / name
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        ex = check_exact_or_die(recs)                       # before any GPU use
        jobs.append((split, recs, out, ex))
    for split, recs, out, ex in jobs:
        out.mkdir(parents=True)
        (out / "windows.json").write_text(json.dumps({"split": split, "sched": SCHEDS[split], "n": NS[split],
                                                      "seed": SEED, "ids": [r["id"] for r in recs],
                                                      "exactness": ex, "counts": counts(recs, args.cap),
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
        print(f"SERIAL_PRECHECK_SCORED {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def lse(x) -> float:
    x = np.asarray(x, np.float64)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum())) if np.isfinite(m) else float(m)


def entropy(logq: np.ndarray) -> float:
    q = np.exp(logq)
    return float(-(q * np.where(q > 0, logq, 0.0)).sum())


def tempered(psi, eta: float = ETA, iters: int = 200) -> tuple[np.ndarray, float]:
    """q* ∝ q^{1/(1+mu)} with the smallest mu >= 0 such that H(q*) >= eta log|Pi| (bisection on tau = 1/(1+mu); H is
    non-increasing in tau). Returns (q*, mu)."""
    psi = np.asarray(psi, np.float64)
    n = len(psi)
    target = eta * math.log(n)

    def logq(tau):
        z = tau * psi
        return z - lse(z)
    if n <= 1 or entropy(logq(1.0)) >= target - 1e-12:
        return np.exp(logq(1.0)), 0.0
    lo, hi = 0.0, 1.0                                    # H(lo) = log n >= target > H(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if entropy(logq(mid)) >= target:
            lo = mid
        else:
            hi = mid
    tau = max(lo, 1e-300)
    return np.exp(logq(tau)), 1.0 / tau - 1.0


def read_metrics(row: dict) -> dict | None:
    """P1 / P2 of one read with |Pi| >= 2 scored in full (orders[0] = the label order)."""
    od = row["orders"]
    if row["capped"] or len(od) < 2:
        return None
    psi = np.array([o["psi"] for o in od])
    logZ = lse(psi)
    q = np.exp(psi - logZ)
    qs, mu = tempered(psi)
    firsts = defaultdict(list)
    for k, o in enumerate(od):
        firsts[o["pi"][0]].append(k)
    lp_first = {f: float(np.mean([od[k]["first_lp"] for k in ks])) for f, ks in firsts.items()}
    first_spread = max(max(od[k]["first_lp"] for k in ks) - min(od[k]["first_lp"] for k in ks)
                       for ks in firsts.values())
    f_star = max(lp_first, key=lambda f: lp_first[f])
    lab0 = row["label"][0]

    def pc(f):
        return math.exp(lse([psi[k] for k in firsts[f]]) - lp_first[f])

    short_ov = any(it["dur_s"] <= SHORT_S and it["ov50"] for it in row["items"])
    end_after_first = {f: float(np.mean([od[k]["end_at_bounds"][1] for k in ks])) for f, ks in firsts.items()}
    return {"id": row["id"], "r": row["r"], "final": row["final"], "n_items": len(row["label"]), "n_pi": len(od),
            "logZ": logZ, "q_label": float(q[0]), "nonlabel": float(1 - q[0]), "qstar_nonlabel": float(1 - qs[0]),
            "mu": mu, "H_q": entropy(np.log(np.maximum(q, 1e-300))), "f_star": f_star, "label_first": lab0,
            "f_is_label": f_star == lab0, "P_complete_f": pc(f_star), "P_complete_label_first": pc(lab0),
            "lp_first_f": lp_first[f_star], "lp_first_label": lp_first[lab0],
            "p_end_after_f": math.exp(end_after_first[f_star]), "p_end_after_label_first": math.exp(end_after_first[lab0]),
            "first_lp_spread": first_spread, "short_ov": short_ov,
            "argmax_pi_is_label": int(np.argmax(psi)) == 0}


def qstats(x) -> dict:
    x = np.asarray([v for v in x if v is not None], np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def p3_row(row: dict, prev: tuple[np.ndarray, np.ndarray], rec: dict) -> dict:
    tok_lp, end_lp = prev
    beta = [0] + list(rec["beta_heur"])
    r = row["r"]
    b0, b1 = beta[r], beta[r + 1]
    old = float(tok_lp[r, b0:b1].sum() + end_lp[r, b1])
    lab = row["orders"][0]
    assert lab["pi"] == row["label"]
    new_tok = np.asarray(lab["tok_lp"])
    assert len(new_tok) == b1 - b0
    return {"psi_new": lab["psi"], "psi_old": old, "abs_diff": abs(lab["psi"] - old),
            "tok_abs_diff_max": float(np.abs(new_tok - tok_lp[r, b0:b1]).max()),
            "end_abs_diff": abs(lab["end_at_bounds"][-1] - float(end_lp[r, b1]))}


def analyse_split(rows: list[dict], prev_lp: dict | None, recs: dict) -> dict:
    mets = [m for m in (read_metrics(r) for r in rows) if m is not None]
    out = {"reads_scored": len(rows), "reads_full_Pi_ge2": len(mets),
           "reads_capped": sum(r["capped"] for r in rows), "reads_Pi_eq_1": sum(1 for r in rows if r["n_pi"] == 1)}
    # P1
    out["P1"] = {"n": len(mets), "share_nonlabel_ge_0.1": float(np.mean([m["nonlabel"] >= P1_MASS for m in mets]))
                 if mets else None,
                 "nonlabel_mass": qstats([m["nonlabel"] for m in mets]),
                 "qstar_eta0.5_nonlabel_mass": qstats([m["qstar_nonlabel"] for m in mets]),
                 "share_mu_gt_0": float(np.mean([m["mu"] > 0 for m in mets])) if mets else None,
                 "share_argmax_pi_is_label": float(np.mean([m["argmax_pi_is_label"] for m in mets])) if mets else None}
    # P2

    def p2(sel):
        g = [m for m in mets if sel(m)]
        ne = [m for m in g if not m["f_is_label"]]
        eq = [m for m in g if m["f_is_label"]]
        return {"reads": len(g), "share_f_ne_label": len(ne) / max(len(g), 1),
                "f_ne_label": qstats([m["P_complete_f"] for m in ne]),
                "f_eq_label": qstats([m["P_complete_f"] for m in eq]),
                "f_ne_label__P_complete_of_label_first_same_reads": qstats([m["P_complete_label_first"] for m in ne]),
                "f_ne_label__p_end_after_f": qstats([m["p_end_after_f"] for m in ne]),
                "f_ne_label__p_end_after_label_first": qstats([m["p_end_after_label_first"] for m in ne]),
                "f_eq_label__p_end_after_f": qstats([m["p_end_after_f"] for m in eq])}
    out["P2"] = {"all": p2(lambda m: True), "short_ov_reads": p2(lambda m: m["short_ov"]),
                 "nonfinal_reads": p2(lambda m: not m["final"]), "final_reads": p2(lambda m: m["final"]),
                 "first_lp_spread_max": max((m["first_lp_spread"] for m in mets), default=None)}
    # P3
    if prev_lp is not None:
        d = [p3_row(r, prev_lp[r["id"]], recs[r["id"]]) for r in rows]
        out["P3"] = {"reads": len(d), "abs_diff": qstats([x["abs_diff"] for x in d]),
                     "abs_diff_p90": float(np.percentile([x["abs_diff"] for x in d], 90)) if d else None,
                     "tok_abs_diff_max": max((x["tok_abs_diff_max"] for x in d), default=None),
                     "end_abs_diff_max": max((x["end_abs_diff"] for x in d), default=None)}
    return out, mets


def prev_logprobs(split: str) -> dict:
    root = Path(PREV[split])
    dirs = [str(root / f"shard_{k}of8") for k in range(8)]
    _, ids, lp_of = P.load_scored(dirs)
    out = {}
    for i in ids:
        z = np.load(lp_of[i] / f"{i}.npz")
        out[i] = (z["tok_lp"], z["end_lp"])
    return out


def run_analyse(args) -> None:
    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cpu = json.loads((root / "cpu" / "cpu_stage.json").read_text())
    rep = {"cpu": {"exactness_pass": cpu["exactness_pass"], "P0": cpu["P0"]["P0"], "P0_stop": cpu["P0"]["P0_stop"]}}
    per = {}
    for split in ("dev", "train"):
        dirs = sorted(root.glob(f"{split}_shard_*of*"))
        if not dirs:
            continue
        rows, ids = [], []
        for d in dirs:
            if not (d / "done.json").exists():
                raise SystemExit(f"shard not finished: {d}")
            ids += json.loads((d / "windows.json").read_text())["ids"]
            rows += [json.loads(l) for l in open(d / "reads.jsonl", encoding="utf-8")]
        full = [r["id"] for r in sample(split)]
        complete = sorted(ids) == sorted(full)
        recs = {r["id"]: r for r in sample(split)}
        a, mets = analyse_split(rows, prev_logprobs(split), recs)
        a["windows_scored"] = len(ids)
        a["complete_sample"] = complete
        a["counts"] = counts([recs[i] for i in ids])
        rep[split] = a
        per[split] = mets
    dev = rep.get("dev", {})
    med = dev.get("P2", {}).get("all", {}).get("f_ne_label", {}).get("median")
    p2_stop = med is None or med >= P2_STOP
    rep["decision"] = {"P0_dev": rep["cpu"]["P0"], "P0_stop": rep["cpu"]["P0_stop"],
                       "P2_dev_median_f_ne_label": med, "P2_stop": p2_stop,
                       "exactness_pass": rep["cpu"]["exactness_pass"],
                       "GO": bool(rep["cpu"]["exactness_pass"] and not rep["cpu"]["P0_stop"] and not p2_stop)}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for split, mets in per.items():
        with open(out / f"per_read_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(m) + "\n" for m in mets)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"SERIAL_PRECHECK_ANALYSED {out} GO={rep['decision']['GO']}", flush=True)


# ------------------------------------------------------------------------------------------------ CPU stage
def run_cpu_stage(args) -> None:
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {root}")
    t0 = time.time()
    tok = tokenizer()
    rep = {"exactness": {}, "counts": {}}
    bad_all = []
    ok = True
    for split in ("dev", "train"):
        smp = sample(split)
        sids = {r["id"] for r in smp}
        rows = v8_rows_of(V8_ROWS[split], sids)
        st_s, st_a = Counter(), Counter()
        for rec in smp:                                      # tokenised + v8 rows, the sample
            e = exactness_window(rec, tok, rows.get(rec["id"], {}), keep=-1)
            st_s.update(e["stats"])
            bad_all += [{"split": split, "in_sample": True, **b} for b in e["s13_bad"]]
        for rec in load_all(SCHEDS[split]):                  # order level, every window
            e = exactness_window(rec, keep=0)
            st_a.update(e["stats"])
        rep["exactness"][split] = {"sample": dict(st_s), "all_windows": dict(st_a),
                                   "sample_pass": exactness_ok(st_s), "all_pass": exactness_ok(st_a)}
        ok &= exactness_ok(st_s) and exactness_ok(st_a)
        rep["counts"][split] = counts(smp)
        print(f"[{split}] exactness sample {dict(st_s)}\n[{split}] exactness all {dict(st_a)}", flush=True)
    rep["exactness_pass"] = bool(ok)
    rep["kappa"] = {"used": "(rho, src_end_ms, src_start_ms, chunk index)",
                    "s13_as_written": "(rho, a_u, chunk index): reported, not used (not the S prefix)"}
    dev_ids = {r["id"] for r in sample("dev")}
    p0, p0_units = p0_coverage(sample_ids=dev_ids)
    rep["P0"] = p0
    rep["elapsed_s"] = time.time() - t0
    out = root / "cpu"
    out.mkdir(parents=True)
    (out / "cpu_stage.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "kappa_s13_mismatch_sample.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(b, ensure_ascii=False) + "\n" for b in bad_all)
    with open(out / "p0_units.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(u) + "\n" for u in p0_units)
    print(json.dumps({k: rep[k] for k in ("exactness_pass", "counts")}, indent=1))
    print(json.dumps(p0, indent=1))
    print(f"SERIAL_CPU_STAGE_DONE {out} exactness_pass={ok} P0={p0['P0']:.3f} P0_stop={p0['P0_stop']}", flush=True)


def run_dry(args) -> None:
    """Plan + the analysis on random scores for 20 windows of each sample; nothing written."""
    rng = np.random.default_rng(0)
    for split in ("dev", "train"):
        recs = sample(split)
        print(split, json.dumps(counts(recs, args.cap)), flush=True)
        rows = []
        for rec in recs[:40]:
            ui = unit_info(rec)
            for rd in read_plan(rec, args.cap):
                od = []
                for pi in rd["orders"]:
                    lens = [rec["chunks"][ci]["tok1"] - rec["chunks"][ci]["tok0"] for ci in pi]
                    tl = -rng.exponential(0.3, sum(lens))
                    ends = list(np.log(rng.uniform(1e-3, 1, len(lens) + 1)))
                    od.append({"pi": list(pi), "psi": float(tl.sum() + ends[-1]), "first_lp": float(tl[: lens[0]].sum()),
                               "tok_lp": list(tl), "end_at_bounds": ends})
                rows.append({"id": rec["id"], "r": rd["r"], "final": rd["r"] == rec["n_read"] - 1, "label": rd["B"],
                             "capped": rd["capped"], "n_pi": rd["n_pi"], "orders": od,
                             "items": [{"dur_s": ui[rec["chunks"][ci]["utt"]]["dur_s"],
                                        "ov50": ui[rec["chunks"][ci]["utt"]]["ov50"]} for ci in rd["B"]]})
        a, _ = analyse_split(rows, None, {r["id"]: r for r in recs})
        print(split, "random-score analysis:", json.dumps({k: a[k] for k in ("reads_scored", "reads_full_Pi_ge2", "P1")}),
              flush=True)
    print("DRY_RUN_OK (random scores; nothing written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu-stage", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--analyse", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default="")
    ap.add_argument("--thinker", default=P.V8)
    ap.add_argument("--splits", default="dev,train")
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first n windows with a read of |Pi| >= 2")
    ap.add_argument("--batch", type=int, default=4, help="sequences per forward (Engine.read-style left padding)")
    ap.add_argument("--max-gib", type=int, default=80)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--no-final-bar-wait", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.root:
        raise SystemExit("--root is required")
    if args.cpu_stage:
        run_cpu_stage(args)
    elif args.score:
        run_score(args)
    elif args.analyse:
        run_analyse(args)
    else:
        run_dry(args)


if __name__ == "__main__":
    main()
