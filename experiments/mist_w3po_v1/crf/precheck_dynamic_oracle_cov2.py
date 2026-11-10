"""CRF dynamic oracle precheck, coverage judge v2 (re-registration 2026-11-10 ~21:00 UTC, before any v2 result:
reports/mist_w3po_v1/crf_dynamic_oracle_precheck.zh-CN.md §5). The first precheck stopped at O1 (iv): the hand check
agreed with the coverage function on 0.865 < 0.95 of its decisions (§2.3). This module changes ONLY the coverage
function and what O1 (ii) / (iii) assert about it; everything else is precheck_dynamic_oracle.py, imported unchanged:
the same v8 trajectory (dev_v8_ctx0_guard_finalpost_s8), items, residual lattice, oracle, O2 population, budget,
scoring (v8), thresholds (O1 0.95, O2 D >= 0.05), tau = 0.5.

Coverage v2 (an utterance u is covered when some visible closed segment x satisfies one rule; x's text "matches" u by
  the content test M(x, u) = edge match of the normalised words (rule a of v1), or both are short responses of the same
  class (BACKCHANNEL / GREETING / THANKS), or the light-stemmed content words share >= half of the smaller set):
  (a') TEXT    x's span widened by 2 s overlaps u and (edge match or same short-response class), under any speaker
               label; NO explaining away by another utterance of x's mapped speaker (v1 lost correct short responses to
               it, §2.3). A short-class segment covers only the matching utterance whose time centre is closest to its
               own (one "Yeah." cannot cover several).
  (b') CREDIT  the scorer credits x to u's unit within 1 s of u (v1's rule b) AND M(x, u): a credited segment whose text
               shares nothing with u (garbage, a repeat of another sentence) no longer covers it (§2.3).
  (c') TIME    only for u longer than 1 s: x's own source span overlaps u by >= 0.3 of their union (IoU, instead of the
               scorer's best-overlap assignment) and M(x, u).
O1 (ii) now asserts v1's raw credit (b_raw) == recalled() for every scored dev utterance; (iii) asserts, on the 263
missed units, that b_raw never fires, rule c' never applies (<= 1 s), and the due window / due read equal S1's; the
S1 / S1' field reproduction of v1 is reported (agreement counts), not asserted, since rule a' differs from S1's text
rule by design. (iv) is unchanged except the sample seed (SEED + 2: a fresh sample). The O2 states are those of this
coverage (written to <root>/o1/states.jsonl), scored by precheck_dynamic_oracle --score --root <root> unchanged.

    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle_cov2 --o1 --root <new root> --procs 64
    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --review --root <root>
    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --score --root <root> --shard K/8     # GPU
    python -m experiments.mist_w3po_v1.crf.precheck_dynamic_oracle --analyse --root <root>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_dynamic_oracle as DO

IOU_C = 0.3
REVIEW_SEED = DO.SEED + 2
COVERAGE_TAG = "v2"          # cov3 (precheck_dynamic_oracle_cov3.py) sets "v3" together with its tables / seed
SHORT_CLASSES = {
    "BACKCHANNEL": {"yeah", "yes", "yep", "yup", "ya", "yea", "okay", "ok", "right", "sure", "mm", "mhm", "mmhmm",
                    "hmm", "uh", "huh", "alright", "all", "uhhuh", "true", "exactly", "indeed"},
    "GREETING": {"hello", "hi", "hey", "morning", "good"},
    "THANKS": {"thanks", "thank", "you"},
    "NEGATIVE": {"no", "nope", "not", "really"},
}
STOP = {"the", "a", "an", "to", "of", "and", "or", "is", "it", "that", "this", "i", "we", "you", "be", "in", "on",
        "for", "so", "do", "uh", "um", "um", "er", "like", "just", "s", "are", "was"}


def stem(w: str) -> str:
    w = re.sub(r"'s$", "", w)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def short_class(ws: list[str]) -> str | None:
    """the class of a short response (<= 4 words, all in one class), else None."""
    if not ws or len(ws) > 4:
        return None
    for c, vocab in SHORT_CLASSES.items():
        if all(w in vocab for w in ws):
            return c
    return None


def content_match(xw: list[str], r: list[str]) -> bool:
    from experiments.mist_w3po_v1.crf import diag_order_attr as D
    if D.edge_match(xw, r):
        return True
    cx, cr = short_class(xw), short_class(r)
    if cx is not None and cx == cr:
        return True
    X = {stem(w) for w in xw if w not in STOP}
    U = {stem(w) for w in r if w not in STOP}
    return bool(X and U) and len(X & U) >= 0.5 * min(len(X), len(U))


def meeting_tables_v2(st: dict, seglist: list[dict]) -> dict:
    """Per meeting: for every scored utterance the segment indices covering it by rules a' / b' / c' (keys a / b / c so
    the rest of precheck_dynamic_oracle works unchanged), plus v1's raw credit (b_raw) for O1 (ii)."""
    from experiments.mist_w3po_v1.crf import diag_order_attr as D

    sc, got, unit_of = st["sc"], st["got"], st["unit_of"]
    nr = {rid: D.norm(v["ref"]) for rid, v in sc.items()}
    nx = [D.norm(x["text"]) if x["text"] else [] for x in seglist]
    idx = {id(x): k for k, x in enumerate(seglist)}
    # short-class segments: the matching utterance closest in time
    closest = {}
    for k, x in enumerate(seglist):
        cx = short_class(nx[k])
        if cx is None:
            continue
        cands = [rid for rid, u in sc.items() if short_class(nr[rid]) == cx
                 and DO.ov(x["a"] - DO.COLLAR_S, x["b"] + DO.COLLAR_S, u["a"], u["b"]) > 0]
        if cands:
            xc = 0.5 * (x["a"] + x["b"])
            closest[k] = min(cands, key=lambda rid: (abs(0.5 * (sc[rid]["a"] + sc[rid]["b"]) - xc), rid))
    out = {}
    for rid, u in sc.items():
        r = nr[rid]
        ra, rb, rc, raw = [], [], [], []
        for x, t in got.get(unit_of[rid], []):
            if t.strip() and DO.ov(x["a"] - DO.RECALL_COLLAR, x["b"] + DO.RECALL_COLLAR, u["a"], u["b"]) > 0:
                k = idx[id(x)]
                raw.append(k)
                if content_match(nx[k], r):
                    rb.append(k)
        for k, x in enumerate(seglist):
            if not x["text"] or DO.ov(x["a"] - DO.COLLAR_S, x["b"] + DO.COLLAR_S, u["a"], u["b"]) <= 0:
                continue
            cx = short_class(nx[k])
            if cx is not None and cx == short_class(r):
                if closest.get(k) == rid:
                    ra.append(k)
            elif D.edge_match(nx[k], r):
                ra.append(k)
            if u["b"] - u["a"] > DO.LONG_S:
                inter = DO.ov(x["a"], x["b"], u["a"], u["b"])
                union = max(x["b"], u["b"]) - min(x["a"], u["a"])
                if union > 0 and inter / union >= IOU_C and content_match(nx[k], r):
                    rc.append(k)
        out[rid] = {"a": sorted(set(ra)), "b": sorted(set(rb)), "c": sorted(set(rc)), "b_raw": sorted(set(raw))}
    return out


TABLES = meeting_tables_v2   # the coverage-table function run_o1_v2 uses (cov3 swaps it)


def s1_check_v2(g: dict, by_w: dict) -> dict:
    """O1 (iii) v2: on the 263 missed units, assert b_raw never fires, c' never applies, the due window and the due read
    of the unit's last chunk equal S1's; report how often the v2 coverage agrees with S1's u_written_at_due."""
    rows = [json.loads(l) for l in open(DO.S1P_ROWS, encoding="utf-8")]
    st = Counter()
    for row in rows:
        m, rid = row["meeting"], row["rid"]
        st["rows"] += 1
        cov = g["cov"][m][rid]
        assert not cov["b_raw"] and not cov["b"], rid
        assert not g["cap"][m]["rec"][rid]
        assert not cov["c"], rid
        if row["window"] is None:
            assert g["due_of"][rid]["window"] is None
            st["no_window"] += 1
            continue
        w = row["window"]
        assert g["due_of"][rid]["window"] == w and rid in g["items_of"][w]
        ws = by_w[w]
        last = max((it for it in ws["items"] if it["rid"] == rid), key=lambda it: it["commit_ms"])
        assert last["rho"] == row["due_read"], (rid, last["rho"], row["due_read"])
        meta_m, seglist = g["meta"][m], g["segs"][m]
        A = set(cov["a"])
        at = [k for k in A if meta_m[k]["wid"] == w and meta_m[k]["write_ms"] == row["t_due_ms"] and seglist[k]["text"].strip()]
        st["agree_u_written_at_due"] += bool(at) == bool(row["u_written_at_due"])
        st["covered_v2_anywhere"] += bool(A)
        st["checked"] += 1
    st["asserted"] = True
    return dict(st)


def run_o1_v2(args) -> None:
    """precheck_dynamic_oracle.run_o1 with the v2 coverage; the steps, counts, budget rule and outputs are its own."""
    from multiprocessing import get_context

    root = Path(args.root)
    out = root / "o1"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    g = DO.load_dev_state(DO.DECODE)
    DO.tok()
    cov, rec_eq = {}, Counter()
    for m, st in g["cap"].items():
        cov[m] = TABLES(st, g["segs"][m])
        for rid in st["sc"]:                                  # O1 (ii) v2: v1's raw credit == recalled()
            ok = bool(cov[m][rid]["b_raw"]) == st["rec"][rid]
            rec_eq["utts"] += 1
            rec_eq["b_raw_eq_recalled"] += ok
            assert ok, (m, rid)
    g["cov"] = cov
    DO._G.update(g)
    print(f"[O1v2] scorer + tables {time.time() - t0:.0f}s", flush=True)
    wids = [w["id"] for w in g["W"]]
    with get_context("fork").Pool(args.procs) as pool:
        res = pool.map(DO._window_job, wids, chunksize=4)
    print(f"[O1v2] windows {time.time() - t0:.0f}s", flush=True)
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
    s1 = s1_check_v2(g, by_w)
    print(f"[O1v2] S1 check {json.dumps(s1)}", flush=True)
    pop = [(r["wid"], st) for r in res for st in r["states"] if st["pop"]]
    agg = Counter()
    for r in res:
        agg["windows"] += 1
        agg["reads"] += r["reads"]
        agg["items"] += r["n_items"]
        for st in r["states"]:
            agg["nonfinal_read_ends"] += not st["final"]
            agg["nonfinal_with_pending"] += (not st["final"]) and bool(st["pend"])
    rule_end = Counter()
    for r in res:
        last = r["states"][-1]
        for rid in g["items_of"].get(r["wid"], []):
            m = r["wid"].rsplit("_on_", 1)[0]
            rs = last["cov_rules"].get(rid, [])
            rule_end[f"{'+'.join(rs) or 'none'}|recalled={int(g['cap'][m]['rec'][rid])}"] += 1
    miss = {row["rid"] for row in map(json.loads, open(DO.S1P_ROWS, encoding="utf-8"))}
    n_fw = [st["n_fw"] for _, st in pop]
    for wid, st in pop:
        st["miss263"] = sorted(set(st["pend"]) & miss)
    keep = [(wid, st) for wid, st in pop if st["n_fw"] <= DO.FW_STATE_CAP]
    excluded = len(pop) - len(keep)
    order = list(range(len(keep)))
    random.Random(DO.SEED).shuffle(order)
    total = sum(st["n_fw"] for _, st in keep)
    if total <= DO.FW_BUDGET:
        uni = set(order)
    else:
        uni, acc = set(), 0
        for j in order:
            if acc + keep[j][1]["n_fw"] > DO.FW_BUDGET:
                break
            uni.add(j)
            acc += keep[j][1]["n_fw"]
    sel = [j for j in range(len(keep)) if j in uni or keep[j][1]["miss263"]]
    rv = random.Random(REVIEW_SEED).sample(range(len(pop)), min(DO.N_REVIEW, len(pop)))
    out.mkdir(parents=True)
    DO.write_review(out, g, by_w, [pop[j] for j in rv])
    with open(out / "states.jsonl", "x", encoding="utf-8") as fh:
        for j in sel:
            wid, st = keep[j]
            w = g["Wid"][wid]
            fh.write(json.dumps({"wid": wid, "i": st["i"], "uniform": j in uni, "miss263": st["miss263"],
                                 "pend": st["pend"], "n_fw": st["n_fw"], "rec": st["rec"], "block": st["block"],
                                 "audio": w["audio"], "post": w["post"]["lowlat"],
                                 "close_detect_ms": w["close_detect_ms"]}) + "\n")
    rep = {"coverage": COVERAGE_TAG, "decode": DO.DECODE, "recalled_eq_b_raw": dict(rec_eq), "s1_check_v2": s1,
           "counts": dict(agg), "coverage_rules_at_window_end": dict(rule_end),
           "population": {"states": len(pop), "excluded_over_cap": excluded,
                          "planned_forwards": {"sum": int(sum(n_fw)), "mean": float(np.mean(n_fw)) if n_fw else 0,
                                               "max": int(max(n_fw)) if n_fw else 0},
                          "budget": DO.FW_BUDGET, "uniform_selected": len(uni), "selected_total": len(sel),
                          "selected_forwards": int(sum(keep[j][1]["n_fw"] for j in sel)),
                          "states_with_missed_unit_pending": sum(bool(st["miss263"]) for _, st in pop)},
           "review_seed": REVIEW_SEED, "review_states": [f"{pop[j][0]}|{pop[j][1]['i']}" for j in rv],
           "elapsed_s": time.time() - t0}
    (out / "o1.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"O1V2_STAGE_DONE {out} states={len(pop)} selected={len(sel)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--o1", action="store_true")
    ap.add_argument("--root", required=True)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    if a.o1:
        run_o1_v2(a)


if __name__ == "__main__":
    main()
