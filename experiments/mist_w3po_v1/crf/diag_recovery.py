"""Recovery-probability diagnostic after the SerialCRF precheck (reports/mist_w3po_v1/serial_crf_precheck.zh-CN.md §12):
where do the overlapped short omissions sit, does v8 stop inside a multi-item read, and does it recover a skipped due
item at the next read? Same windows as crf/precheck_serial.py (train 500 / dev 300, seed 20261109), the corrected
kappa key (rho, source end, source start); dev is the primary evidence.

EVERY METRIC IS CONTENT-FREE (the lesson of P2): an item g is "opened" by its structural opening tokens only,
    p_open(g | prefix) = p(<spk_g> | prefix) * p(<t_start_g> | prefix, <spk_g>),
compared with the read-end token e_r (<wait>; <eos> at the final read, where <wait> is barred before the softmax, as
Engine.read). Text probabilities are never multiplied in (the text of items written BEFORE the probe position is part of
the conditioning prefix only).

D1  where omissions sit (CPU): for the dev ORDER_OMISSION overlapped (ov50) <= 1 s units of check A
    (diag_order_attr_v1/check_a_missed.jsonl), and separately for ALL overlapped <= 1 s misses, the share whose due read
    is the window's final read. Primary: the online window of check A (its "due_final"); also the reference window
    (sched_dev_v1; due read = rho of the unit's last chunk) and <= 0.5 s.
D2  same-read stopping (CPU, from the stored scores of the SerialCRF precheck, serial_precheck_v1/*_shard_*/reads.jsonl):
    per multi-item read and first item f in {label-first l, model-preferred f*}: right after f's tokens,
        p_open = sum over the remaining due items g that may come next of p_open(g | h_r, y_f),   p_e = p(e_r | h_r, y_f)
    f* (content-free) = argmax over the items that may come first of p_open(x | h_r) at the start of the read (ties ->
    label order). Medians of p_open and p_e, and the share of reads with p_e > p_open; final vs non-final reads.
    Groups: |Pi| >= 2 reads (all orders stored; primary), |Pi| = 1 and capped reads (label order only, f = l, the label
    second item only) reported apart; <= 1 s ov50 subgroup = reads with a <= 1 s ov50 unit among the remaining items.
D3  cross-read recovery (GPU; non-final due read r only; |Pi(B_r)| >= 2, |B_r| <= 4): a pair (f, m) = f an item that
    may come first in B_r, m an item of ANOTHER unit that may come next once f is written. Off-label state "at read r
    only f was written, then <wait>": h' = kappa(committed before r ∪ {f}) (all of B_r \\ {f} missing). At read r+1:
      R_first = p_open(m | h')                                  (m as the first item of read r+1; p(e_{r+1}) there too)
      R_end   = p_open(m | h', B'_{r+1}) vs p(e_{r+1} | h', B'_{r+1})
                B'_{r+1} = the label items of B_{r+1} in label order, minus the chunks of units with a missing chunk
                (they cannot precede their head; the count dropped is stored). B'_{r+1} empty -> R_end = R_first.
      control = the on-path p_open(m | h_r, y_f) at read r right after f, from the stored precheck scores (primary);
                also scored again in this run (fresh) as the alignment check.

PRE-REGISTERED READING (written and committed before any GPU run of this script; constants below):
  H_rec (cross-read recovery exposure bias), on the PRIMARY population = dev, all D3 pairs:
      SUPPORTED  iff median R_first < 0.3 AND median(control) - median(R_first) >= 0.3
      REJECTED   iff median R_first >= 0.6
      AMBIGUOUS  otherwise
    the same rule is also evaluated (secondary) on the pairs whose m is a <= 1 s ov50 unit, and on f = f* / f = l.
  H_final (omission = stopping early at the final read):
      SUPPORTED  iff D1 (primary: ORDER_OMISSION ov50 <= 1 s, online window) >= 50 % AND, on dev final reads with
                 |Pi| >= 2 and f = f*, median p_e > median p_open (D2); NOT SUPPORTED otherwise. (f = l reported too.)
  Note: the D1 inputs were already tabulated in sched_precheck_v8.zh-CN.md §8.1 (due_is_final_read) before this rule.
  SECONDARY reading, ADDED after the CPU stage (D1/D2) and BEFORE any GPU run of D3 (committed separately): D2 showed
  that the 80 ms start-time token often splits its mass (dev p(t_start | spk) median 0.36-0.55, q25 0.05-0.14), so
  products spk x t_start rarely reach 0.6 even for an item the model will open. The same H_rec rule (same thresholds)
  is therefore also evaluated with the speaker token alone (R_first_spk vs the speaker-only control); the primary
  verdict stays the speaker + start-time one above.

    python -m experiments.mist_w3po_v1.crf.diag_recovery --cpu-stage --root <new root>          # D1, D2, plan
    python -m experiments.mist_w3po_v1.crf.diag_recovery --score --root <root> --shard K/N --splits dev,train
    python -m experiments.mist_w3po_v1.crf.diag_recovery --score --root <work dir> --splits dev --limit 2   # smoke
    python -m experiments.mist_w3po_v1.crf.diag_recovery --analyse --root <root>                # D3 + decisions
    python -m experiments.mist_w3po_v1.crf.diag_recovery --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S

SERIAL = "eval_outputs/mist_w3po_v1/crf/serial_precheck_v1"
SHORT_S = 1.0
# pre-registered thresholds
REC_SUPPORT_MAX = 0.3
REC_GAP = 0.3
REC_REJECT_MIN = 0.6
FINAL_D1_MIN = 0.5


# ------------------------------------------------------------------------------------------------ decisions
def h_rec(r_first: list[float], control: list[float]) -> dict:
    a = float(np.median(r_first)) if len(r_first) else None
    c = float(np.median(control)) if len(control) else None
    if a is None or c is None:
        verdict = "NO_DATA"
    elif a < REC_SUPPORT_MAX and c - a >= REC_GAP:
        verdict = "SUPPORTED"
    elif a >= REC_REJECT_MIN:
        verdict = "REJECTED"
    else:
        verdict = "AMBIGUOUS"
    return {"n": len(r_first), "median_R_first": a, "median_control": c,
            "gap": None if a is None or c is None else c - a, "verdict": verdict}


def h_final(d1_share: float, p_e: list[float], p_open: list[float]) -> dict:
    me = float(np.median(p_e)) if len(p_e) else None
    mo = float(np.median(p_open)) if len(p_open) else None
    ok = me is not None and d1_share >= FINAL_D1_MIN and me > mo
    return {"D1": d1_share, "n_final_reads": len(p_e), "median_p_e": me, "median_p_open": mo,
            "verdict": "SUPPORTED" if ok else "NOT_SUPPORTED"}


# ------------------------------------------------------------------------------------------------ items
def openable(rec: dict, A) -> list[int]:
    """The items of A that may come next: no earlier chunk of the same unit in A (label order kept)."""
    A = list(A)
    return [c for c in A if not any(d < c and rec["chunks"][d]["utt"] == rec["chunks"][c]["utt"] for d in A)]


def opener_ids(rec: dict, c: int) -> list[int]:
    """The structural opening tokens of chunk c: its speaker token and its start-time token."""
    ch = rec["chunks"][c]
    ids = rec["S_ids"][ch["tok0"]: ch["tok0"] + 2]
    assert S.SPK0 <= ids[0] < S.NEW_END and S.NEW0 <= ids[1] < S.SPK0
    return ids


def d3_pairs(rec: dict, cap: int = S.CAP) -> list[dict]:
    rho = S.chunk_reads(rec)
    B = S.read_items(rec, rho)
    unit = {c: ch["utt"] for c, ch in enumerate(rec["chunks"])}
    ui = S.unit_info(rec)
    out = []
    for r in range(rec["n_read"] - 1):
        Br = B[r]
        if len(Br) < 2 or len(Br) > cap or S.n_orders(rec, Br) < 2:
            continue
        H = S.history(rec, rho, r)
        for f in openable(rec, Br):
            rest = [c for c in Br if c != f]
            miss_units = {unit[c] for c in rest}
            B1 = [c for c in B[r + 1] if unit[c] not in miss_units]
            for m in openable(rec, rest):
                if unit[m] == unit[f]:
                    continue
                out.append({"id": rec["id"], "r": r, "f": f, "m": m, "label_first": Br[0], "n_items": len(Br),
                            "hist_on": S.ids_of(rec, S.kappa(rec, rho, H)),
                            "hist_off": S.ids_of(rec, S.kappa(rec, rho, H + [f])),
                            "B1": B1, "B1_dropped": len(B[r + 1]) - len(B1), "r1_final": r + 1 == rec["n_read"] - 1,
                            "m_dur_s": ui[unit[m]]["dur_s"], "m_ov50": ui[unit[m]]["ov50"],
                            "m_short_ov": ui[unit[m]]["dur_s"] <= SHORT_S and ui[unit[m]]["ov50"]})
    return out


# ------------------------------------------------------------------------------------------------ D1 (CPU)
def d1(check_a: str = S.CHECK_A, ref: str = S.REF_DEV, sched: str = S.SCHEDS["dev"]) -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in open(check_a, encoding="utf-8")]
    rid_of = {}
    for l in open(ref, encoding="utf-8"):
        w = json.loads(l)
        rid_of[w["id"]] = [u["record_id"] for u in w["utts"]]
    where, recs = {}, {}
    for rec in S.load_all(sched):
        recs[rec["id"]] = rec
        for u in rec["units"]:
            where.setdefault(rid_of[rec["id"]][u["utt"]], (rec["id"], u["utt"]))
    units = []
    for o in rows:
        if not o["ov50"]:
            continue
        x = {k: o[k] for k in ("rid", "cls", "dur", "le05", "due_final", "window")}
        loc = where.get(o["rid"])
        if loc:
            rec = recs[loc[0]]
            rho = S.chunk_reads(rec)
            last = max(c for c, ch in enumerate(rec["chunks"]) if ch["utt"] == loc[1])
            x["ref_window"], x["ref_due_final"] = loc[0], rho[last] == rec["n_read"] - 1
        units.append(x)

    def grp(sel):
        g = [u for u in units if sel(u)]
        on = [u for u in g if u["due_final"] is not None]
        rf = [u for u in g if "ref_due_final" in u]
        return {"n": len(g), "online_with_window": len(on),
                "share_due_final_online": sum(u["due_final"] for u in on) / max(len(on), 1),
                "due_final_online": sum(u["due_final"] for u in on),
                "ref_found": len(rf), "share_due_final_ref": sum(u["ref_due_final"] for u in rf) / max(len(rf), 1)}
    rep = {"ORDER_OMISSION_ov50_le1": grp(lambda u: u["cls"] == "ORDER_OMISSION"),
           "ORDER_OMISSION_ov50_le05": grp(lambda u: u["cls"] == "ORDER_OMISSION" and u["le05"]),
           "all_ov50_le1_misses": grp(lambda u: True), "all_ov50_le05_misses": grp(lambda u: u["le05"]),
           "by_class": {c: grp(lambda u, c=c: u["cls"] == c) for c in sorted({u["cls"] for u in units})}}
    rep["D1_primary"] = rep["ORDER_OMISSION_ov50_le1"]["share_due_final_online"]
    return rep, units


# ------------------------------------------------------------------------------------------------ D2 (CPU)
def d2_read(row: dict) -> dict:
    """Content-free same-read stopping of one stored precheck read (serial_precheck_v1 reads.jsonl row)."""
    od = row["orders"]
    info = {it["chunk"]: it for it in row["items"]}
    open0 = defaultdict(list)
    for o in od:
        open0[o["pi"][0]].append(o["tok_lp"][0] + o["tok_lp"][1])
    pos = {c: i for i, c in enumerate(row["label"])}
    fstar = max(open0, key=lambda x: (float(np.mean(open0[x])), -pos[x]))
    lab = row["label"][0]

    def after(x):
        nx = info[x]["n_tok"]
        opens, ends = defaultdict(list), []
        for o in od:
            if o["pi"][0] != x or len(o["pi"]) < 2:
                continue
            opens[o["pi"][1]].append(o["tok_lp"][nx] + o["tok_lp"][nx + 1])
            ends.append(o["end_at_bounds"][1])
        p_open = sum(math.exp(float(np.mean(v))) for v in opens.values())
        p_e = math.exp(float(np.mean(ends)))
        rem_short = any(info[c]["dur_s"] <= SHORT_S and info[c]["ov50"] for c in row["label"] if c != x)
        return {"p_open": p_open, "p_e": p_e, "e_gt_open": p_e > p_open, "rem_short_ov": rem_short}
    kind = "capped" if row["capped"] else ("pi1" if len(od) < 2 else "pi2")
    return {"id": row["id"], "r": row["r"], "final": row["final"], "kind": kind, "n_items": len(row["label"]),
            "f_star": fstar, "label_first": lab, "f_star_is_label": fstar == lab,
            "after_label": after(lab), "after_fstar": after(fstar)}


def d2_summary(per: list[dict]) -> dict:
    def g(rows, key):
        xs = [r[key] for r in rows]
        return {"n": len(xs), "median_p_open": float(np.median([x["p_open"] for x in xs])) if xs else None,
                "median_p_e": float(np.median([x["p_e"] for x in xs])) if xs else None,
                "share_p_e_gt_p_open": float(np.mean([x["e_gt_open"] for x in xs])) if xs else None,
                "p_open": S.qstats([x["p_open"] for x in xs]), "p_e": S.qstats([x["p_e"] for x in xs])}
    out = {}
    for kind in ("pi2", "pi1", "capped"):
        rows = [r for r in per if r["kind"] == kind]
        for fin in ("final", "nonfinal"):
            rr = [r for r in rows if r["final"] == (fin == "final")]
            keys = ("after_label", "after_fstar") if kind == "pi2" else ("after_label",)
            for key in keys:
                f = "f=label" if key == "after_label" else "f=f*"
                out[f"{kind}|{fin}|{f}|all"] = g(rr, key)
                out[f"{kind}|{fin}|{f}|rem_short_ov"] = g([r for r in rr if r[key]["rem_short_ov"]], key)
    out["pi2_share_fstar_is_label"] = float(np.mean([r["f_star_is_label"] for r in per if r["kind"] == "pi2"]))
    return out


def stored_rows(split: str, root: str = SERIAL) -> list[dict]:
    return [json.loads(l) for d in sorted(Path(root).glob(f"{split}_shard_*of*"))
            for l in open(d / "reads.jsonl", encoding="utf-8")]


def stored_control(rows: list[dict], spk_only: bool = False) -> dict:
    """(id, r, f, m) -> on-path log p_open(m | h_r, y_f) from the stored precheck scores (spk_only: the speaker token
    alone)."""
    out = {}
    for row in rows:
        info = {it["chunk"]: it for it in row["items"]}
        acc = defaultdict(list)
        for o in row["orders"]:
            if len(o["pi"]) < 2:
                continue
            f, m = o["pi"][0], o["pi"][1]
            nf = info[f]["n_tok"]
            acc[(row["id"], row["r"], f, m)].append(o["tok_lp"][nf] + (0.0 if spk_only else o["tok_lp"][nf + 1]))
        out.update({k: float(np.mean(v)) for k, v in acc.items()})
    return out


# ------------------------------------------------------------------------------------------------ D3 scoring (GPU)
def pair_reqs(rec: dict, p: dict) -> list[tuple]:
    """(rec, read, history, forced) per pair: fresh on-path control, R_first, R_end (when B'_{r+1} is not empty)."""
    op = opener_ids(rec, p["m"])
    reqs = [(rec, p["r"], p["hist_on"], S.ids_of(rec, [p["f"]]) + op),
            (rec, p["r"] + 1, p["hist_off"], op)]
    if p["B1"]:
        reqs.append((rec, p["r"] + 1, p["hist_off"], S.ids_of(rec, p["B1"]) + op))
    return reqs


def pair_result(rec: dict, p: dict, res: list[tuple]) -> dict:
    nf = len(S.ids_of(rec, [p["f"]]))
    (tl0, ec0), (tl1, ec1) = res[0], res[1]
    out = {k: v for k, v in p.items() if k not in ("hist_on", "hist_off")}
    out["control_fresh_lp"], out["control_fresh_spk_lp"] = float(tl0[nf] + tl0[nf + 1]), float(tl0[nf])
    out["R_first_lp"], out["R_first_spk_lp"] = float(tl1[0] + tl1[1]), float(tl1[0])
    out["e_first_lp"] = float(ec1[0])
    if p["B1"]:
        tl2, ec2 = res[2]
        nb = len(S.ids_of(rec, p["B1"]))
        out["R_end_lp"], out["e_end_lp"] = float(tl2[nb] + tl2[nb + 1]), float(ec2[nb])
        out["R_end_spk_lp"] = float(tl2[nb])
    else:
        out["R_end_lp"], out["e_end_lp"] = out["R_first_lp"], out["e_first_lp"]
        out["R_end_spk_lp"] = out["R_first_spk_lp"]
    return out


def run_score(args) -> None:
    import torch

    from experiments.mist_w3po_v1.core.engine import Engine

    root = Path(args.root)
    sh = S.P.parse_shard(args.shard)
    jobs = []
    for split in [s for s in args.splits.split(",") if s]:
        recs = S.sample(split)
        if sh:
            recs = S.P.shard_select(recs, *sh)
        if args.limit:
            recs = [r for r in recs if d3_pairs(r)][: args.limit]
        name = f"{split}_shard_{sh[0]}of{sh[1]}" if sh else (f"{split}_limit{args.limit}" if args.limit else split)
        out = root / name
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        ex = S.check_exact_or_die(recs)                       # kappa exact, before any GPU use
        jobs.append((split, recs, out, ex))
    for split, recs, out, ex in jobs:
        out.mkdir(parents=True)
        (out / "windows.json").write_text(json.dumps({"split": split, "ids": [r["id"] for r in recs], "exactness": ex,
                                                      "pairs": sum(len(d3_pairs(r)) for r in recs),
                                                      "args": vars(args)}, indent=1))
    t0 = time.time()
    eng = Engine.from_merged(args.thinker, "zh2en", args.max_gib)
    load_s = time.time() - t0
    for split, recs, out, ex in jobs:
        t1, n_fw, n_pairs = time.time(), 0, 0
        with open(out / "pairs.jsonl", "x", encoding="utf-8") as fh:
            for rec in recs:
                ps = d3_pairs(rec)
                if not ps:
                    continue
                reqs, spans = [], []
                for p in ps:
                    q = pair_reqs(rec, p)
                    spans.append((len(reqs), len(reqs) + len(q)))
                    reqs += q
                res = S.score_reqs(eng, reqs, args.batch, True)
                n_fw += len(reqs)
                for p, (a, b) in zip(ps, spans):
                    fh.write(json.dumps(pair_result(rec, p, res[a:b])) + "\n")
                    n_pairs += 1
        peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
        st = {"windows": len(recs), "pairs": n_pairs, "forwards": n_fw, "scoring_s": time.time() - t1,
              "load_s": load_s, "peak_alloc_gib": peak}
        (out / "done.json").write_text(json.dumps(st, indent=1))
        print(f"RECOVERY_SCORED {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def d3_summary(pairs: list[dict], ctrl: dict, ctrl_spk: dict | None = None) -> dict:
    for p in pairs:
        k = (p["id"], p["r"], p["f"], p["m"])
        p["control_lp"] = ctrl.get(k)
        p["control_spk_lp"] = (ctrl_spk or {}).get(k)
    P = [p for p in pairs if p["control_lp"] is not None]

    def stats(g):
        e = lambda k: [math.exp(p[k]) for p in g]
        return {"pairs": len(g), "reads": len({(p["id"], p["r"]) for p in g}),
                "R_first": S.qstats(e("R_first_lp")), "R_first_spk_only": S.qstats(e("R_first_spk_lp")),
                "p_e_first": S.qstats(e("e_first_lp")), "R_end": S.qstats(e("R_end_lp")), "p_e_end": S.qstats(e("e_end_lp")),
                "share_p_e_end_gt_R_end": float(np.mean([p["e_end_lp"] > p["R_end_lp"] for p in g])) if g else None,
                "control": S.qstats(e("control_lp")), "control_fresh": S.qstats(e("control_fresh_lp")),
                "paired_control_minus_R_first": S.qstats([math.exp(p["control_lp"]) - math.exp(p["R_first_lp"])
                                                          for p in g]),
                "B1_empty": sum(1 for p in g if not p["B1"]), "B1_dropped_pairs": sum(1 for p in g if p["B1_dropped"]),
                "r1_final": sum(1 for p in g if p["r1_final"]),
                "H_rec": h_rec([math.exp(p["R_first_lp"]) for p in g], [math.exp(p["control_lp"]) for p in g]),
                "R_end_spk_only": S.qstats(e("R_end_spk_lp")),
                "control_spk_only": S.qstats([math.exp(p["control_spk_lp"]) for p in g if p["control_spk_lp"] is not None]),
                "H_rec_spk_only(secondary)": h_rec([math.exp(p["R_first_spk_lp"]) for p in g],
                                                   [math.exp(p["control_spk_lp"]) for p in g
                                                    if p["control_spk_lp"] is not None])}
    groups = {"all": P, "m_short_ov": [p for p in P if p["m_short_ov"]],
              "f_is_label_first": [p for p in P if p["f"] == p["label_first"]],
              "f_is_fstar": [p for p in P if p.get("f_is_fstar")],
              "m_short_ov&f_is_fstar": [p for p in P if p["m_short_ov"] and p.get("f_is_fstar")],
              "r1_final": [p for p in P if p["r1_final"]], "r1_nonfinal": [p for p in P if not p["r1_final"]]}
    out = {k: stats(v) for k, v in groups.items()}
    out["alignment_control_fresh_vs_stored_abs_lp"] = S.qstats([abs(p["control_fresh_lp"] - p["control_lp"]) for p in P])
    out["pairs_without_stored_control"] = len(pairs) - len(P)
    return out


def run_analyse(args) -> None:
    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cpu = json.loads((root / "cpu" / "cpu_stage.json").read_text())
    rep = {"D1": cpu["D1"], "D2": cpu["D2"]}
    for split in ("dev", "train"):
        dirs = sorted(root.glob(f"{split}_shard_*of*"))
        if not dirs:
            continue
        for d in dirs:
            if not (d / "done.json").exists():
                raise SystemExit(f"shard not finished: {d}")
        pairs = [json.loads(l) for d in dirs for l in open(d / "pairs.jsonl", encoding="utf-8")]
        ids = [i for d in dirs for i in json.loads((d / "windows.json").read_text())["ids"]]
        rows = stored_rows(split)
        fst = {(x["id"], x["r"]): x["f_star"] for x in (d2_read(r) for r in rows)}
        for p in pairs:
            p["f_is_fstar"] = fst.get((p["id"], p["r"])) == p["f"]
        rep[f"D3_{split}"] = d3_summary(pairs, stored_control(rows), stored_control(rows, spk_only=True))
        rep[f"D3_{split}"]["windows"] = len(ids)
        rep[f"D3_{split}"]["complete_sample"] = sorted(ids) == sorted(r["id"] for r in S.sample(split))
    rep["decision"] = {
        "H_rec_primary(dev, all pairs)": rep["D3_dev"]["all"]["H_rec"]["verdict"],
        "H_rec_secondary(dev, m <= 1 s ov50)": rep["D3_dev"]["m_short_ov"]["H_rec"]["verdict"],
        "H_rec_spk_only(dev, all pairs)": rep["D3_dev"]["all"]["H_rec_spk_only(secondary)"]["verdict"],
        "H_rec_spk_only(dev, m <= 1 s ov50)": rep["D3_dev"]["m_short_ov"]["H_rec_spk_only(secondary)"]["verdict"],
        "H_final": h_final(rep["D1"]["D1_primary"], cpu["D2_final_fstar_values"]["p_e"],
                           cpu["D2_final_fstar_values"]["p_open"])}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep["decision"], indent=1))
    print(f"RECOVERY_ANALYSED {out}", flush=True)


# ------------------------------------------------------------------------------------------------ CPU stage
def run_cpu_stage(args) -> None:
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {root}")
    rep = {}
    rep["D1"], d1_units = d1()
    rep["D2"], per = {}, {}
    for split in ("dev", "train"):
        per[split] = [d2_read(r) for r in stored_rows(split)]
        rep["D2"][split] = d2_summary(per[split])
    fin = [x for x in per["dev"] if x["kind"] == "pi2" and x["final"]]
    rep["D2_final_fstar_values"] = {"p_e": [x["after_fstar"]["p_e"] for x in fin],
                                    "p_open": [x["after_fstar"]["p_open"] for x in fin]}
    rep["H_final"] = h_final(rep["D1"]["D1_primary"], rep["D2_final_fstar_values"]["p_e"],
                             rep["D2_final_fstar_values"]["p_open"])
    plan = {}
    for split in ("dev", "train"):
        ps = [p for r in S.sample(split) for p in d3_pairs(r)]
        plan[split] = {"pairs": len(ps), "reads": len({(p["id"], p["r"]) for p in ps}),
                       "m_short_ov_pairs": sum(p["m_short_ov"] for p in ps),
                       "forwards": sum(3 if p["B1"] else 2 for p in ps)}
    rep["D3_plan"] = plan
    out = root / "cpu"
    out.mkdir(parents=True)
    (out / "cpu_stage.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "d1_units.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(u) + "\n" for u in d1_units)
    for split, xs in per.items():
        with open(out / f"d2_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in xs)
    print(json.dumps({k: rep[k] for k in ("H_final", "D3_plan")}, indent=1))
    print(f"RECOVERY_CPU_STAGE_DONE {out}", flush=True)


def run_dry(args) -> None:
    for split in ("dev", "train"):
        ps = [p for r in S.sample(split) for p in d3_pairs(r)]
        print(split, "pairs", len(ps), "m_short_ov", sum(p["m_short_ov"] for p in ps),
              "forwards", sum(3 if p["B1"] else 2 for p in ps), "B1 empty", sum(not p["B1"] for p in ps),
              "B1 dropped", sum(bool(p["B1_dropped"]) for p in ps), flush=True)
    print("DRY_RUN_OK (nothing written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu-stage", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--analyse", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default="")
    ap.add_argument("--thinker", default=S.P.V8)
    ap.add_argument("--splits", default="dev,train")
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-gib", type=int, default=80)
    args = ap.parse_args()
    if not args.dry_run and not args.root:
        raise SystemExit("--root is required")
    {"cpu_stage": run_cpu_stage, "score": run_score, "analyse": run_analyse, "dry_run": run_dry}[
        next(k for k in ("cpu_stage", "score", "analyse", "dry_run") if getattr(args, k))](args)


if __name__ == "__main__":
    main()
