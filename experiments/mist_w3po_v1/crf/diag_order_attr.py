"""Two read-only CPU checks after the opening-position rescoring (reports/mist_w3po_v1/sched_precheck_v8.zh-CN.md §7).

CHECK A  does ORDER cause actual omissions in v8's own online decoding?
  decode    the formal gate reference: eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0_guard_s8 (v8, ctx0, guarded protocol,
            8 shards; hyp.jsonl with per-segment write_ms and the read log), dev online windows
            eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl, references
            windows_validation_ref.jsonl (15 AMI dev meetings).
  matching  NOT re-implemented: reports/gstar_si_v8/overlap_score.score (online=True, the scorer of score_run.py) is run
            as is, and its per-meeting state (scored utterances sc, unit_of, the segment -> unit assignment got, the
            meeting speaker map mp, its own recalled() closure) is read from its frame with a call tracer when it first
            calls recalled() (the closure is evaluated there for every utterance of the meeting). A short utterance
            is MISSED iff the scorer's recalled(rid) is False. The short-recall
            counts are asserted equal to the decode's score_text.json.
  units     scored utterances with b - a <= 1 s (and <= 0.5 s), as the scorer's short recall. Isolated / overlapped as
            in the md: ov50 = the other speakers' scored utterances (union) cover >= 50 % of it
            (diag_short_units.ov50); the scorer's own split (>= 20 % of other annotated speech) is kept as ov20.
  due read  w = the online window with the largest time overlap with the utterance; due = the first read of w (its
            read log) with t >= (b - w.start) + 1.0 s (v8's label rule: commit = source end + 1.0 s lag, at the next
            1 s read), else w's last read. T_due = w.start + t_due (meeting clock). Commit times below are the raw read
            times (w.start + write_ms), not the scorer's detection-clamped ones.
  text match (pre-stated) norm(s) = the [a-z0-9'] words of s lower-cased; r = norm(reference of u) (non-empty).
            A hyp segment x MATCHES u when r == the first len(r) or the last len(r) words of norm(x.text) (the
            interjection is the whole segment or glued to one of its ends), x's source span widened by 2 s (the scorer's
            collar) overlaps u, and the match is not explained away: no other scored utterance of x's matched speaker
            (for (ii)) / of u's own speaker (for OWN) overlapping x's widened span contains r contiguously.
  classes, first that applies:
    (ii)  MISLABEL         a matching segment whose hyp slot is not mapped (scorer's mp) to u's speaker
    (iv)  OWN_UNCREDITED   a matching segment under u's own mapped slot (written, but not credited near u)
    (i)   ORDER_OMISSION   some other-speaker scored utterance v overlapping u in time has a segment credited to v's
                           unit by the scorer (got), whose source span is within 1 s (the recall collar) of u, committed
                           at or after T_due  (= the concurrent part of the other speaker was written instead, at or
                           after u's due read)
    (iii) SILENT           no non-empty segment of any speaker committed within T_due +- 2 s
    (iv)  OTHER            the rest
  sensitivity (not in the headline): MISLABEL_BC: r all in a fixed backchannel set BC and some segment of 1-3 words,
            all in BC, under another mapped speaker within the 2 s collar (translation of an interjection varies).
  reverse   overlapped (ov50) short utterances that were recalled: E_u = the segments credited to u's unit within 1 s of
            u (the recall evidence), E_o = segments credited to the units of the other-speaker utterances overlapping u,
            within 1 s of u; first = min by (raw commit, window start, position in the window's output). OTHER_FIRST
            when E_o's first precedes E_u's (split: at an earlier read / same read, earlier in the output).

CHECK B  what causes ATTRIBUTION for isolated short units?
  items     the rescoring items (eval_outputs/.../precheck_v8_{dev300,500}/openers_rescore/shard_*/items.jsonl) at the
            FINAL read with label ATTRIBUTION or ATTRIBUTION_NOPENDING (dev primary, train for reference); control:
            the GOLD items of the same groups.
  P         the window's lowlat posteriors (sched record "post", as Engine.audio_embeds loads them) as the prompt sees
            them at that read: frames >= (t_r - 1.04 s) / 80 ms zeroed. The unit's frames: [a / 80 ms, ceil(b / 80 ms)).
            Column of slot k = k - 1 (v8's <|spk k|>, meeting-level lowlat map).
  evidence  NONE when no frame of the unit has a slot >= 0.5 in the prompt P (split UNSEEN: the full P has one, i.e.
            the frames were zeroed by the look-ahead; SILENT: neither); else the argmax of the unit-mean prompt P:
            GOLD_COL (k - 1), K2_COL (k' - 1), OTHER_COL.
  window map data/mist_w3po_v1/build_windows_zh2en.window_slot_map (S0-5: Hungarian on frame overlap inside the window;
            members = the reference window's utterances, the whole window's P, session map = slot_map.json lowlat) ->
            the gold speaker's window slot; WIN_AGREE (= meeting slot k), WIN_K2 (= k'), WIN_OTHER; uncovered = the
            speaker's activity meets no active (> 0.5) frame of any slot in the window (the map falls back to k).
  cause (pre-stated partition)
    LABEL_NOISE       evidence K2_COL and window map WIN_K2: in this window the diarizer slot of the gold speaker IS k';
                      the meeting-level gold label disagrees with the local slot the model sees
    WRONG_EVIDENCE    evidence K2_COL and the window map not WIN_K2: the window keeps the speaker on k (or elsewhere),
                      only the unit's own frames point to k' -> the model follows wrong local diarizer evidence
    NO_EVIDENCE       evidence NONE (UNSEEN / SILENT reported separately)
    IGNORES_EVIDENCE  evidence GOLD_COL: the prompt P points to the gold column and the model still prefers k'
    OTHER_EVIDENCE    evidence OTHER_COL

POST-HOC (added after Check B; --posthoc, check_a_posthoc): on the same online decode, the visibility of the short
utterance's diarizer frames in the prompt at the read where its slot was decided (recalled vs MISLABEL), and the miss
rate by when the other speakers' concurrent part was first written relative to the due read.

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.diag_order_attr \
        --out eval_outputs/mist_w3po_v1/crf/diag_order_attr_v1
    ... --posthoc --out eval_outputs/mist_w3po_v1/crf/diag_order_attr_v1_posthoc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))

DECODE = "eval_outputs/mist_w3po_v1/decode/dev_v8_ctx0_guard_s8"
WINDOWS = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
REF_DEV = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"
REF_TRAIN = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_train_ref.jsonl"
SLOT_MAP = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/slot_map.json"
RESCORE = {"dev": "eval_outputs/mist_w3po_v1/crf/precheck_v8_dev300/openers_rescore",
           "train": "eval_outputs/mist_w3po_v1/crf/precheck_v8_500/openers_rescore"}
LAG_S = 1.0          # v8 label lag (build_ami_gstar_stream_rows_zh2en, utt_chunks: commit = source end + lag)
COLLAR_S = 2.0       # the scorer's segment-to-utterance collar (overlap_score.score collar=2.0)
RECALL_COLLAR = 1.0  # overlap_score.PAIR_COLLAR (recalled())
SILENT_S = 2.0
FRAME_MS, LOOKAHEAD_MS = 80, 1040
BC = {"yeah", "yes", "yep", "yup", "okay", "ok", "right", "sure", "alright", "mm", "mhm", "hmm", "uh", "huh", "um",
      "oh", "ah", "true", "exactly", "cool", "great", "good", "fine"}   # norm() splits "mm-hmm" / "uh-huh" at "-"
A_CLASSES = ["ORDER_OMISSION", "MISLABEL", "SILENT", "OWN_UNCREDITED", "OTHER"]


# ------------------------------------------------------------------------------------------------ helpers
def norm(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def edge_match(xw: list[str], r: list[str]) -> bool:
    n = len(r)
    return 0 < n <= len(xw) and (xw[:n] == r or xw[-n:] == r)


def contains(xw: list[str], r: list[str]) -> bool:
    n = len(r)
    return 0 < n <= len(xw) and any(xw[i: i + n] == r for i in range(len(xw) - n + 1))


def ov(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def due_read(reads_ms: list[int], w_start: float, b: float) -> int:
    """index of the first read with t >= (b - w_start) + LAG_S, else the last read."""
    need = (b - w_start + LAG_S) * 1000 - 1e-6
    for i, t in enumerate(reads_ms):
        if t >= need:
            return i
    return len(reads_ms) - 1


def ov50_of(u: dict, others: list[dict]) -> bool:
    from experiments.mist_w3po_v1.crf.diag_short_units import ov50

    units = [{"a_ms": round(v["a"] * 1000), "b_ms": round(v["b"] * 1000), "speaker_id": v["spk"]} for v in others]
    me = {"a_ms": round(u["a"] * 1000), "b_ms": round(u["b"] * 1000), "speaker_id": u["spk"]}
    return ov50(units + [me], me)


# ------------------------------------------------------------------------------------------------ check A
def run_scorer(W: list[dict], H: dict, RW: list[dict]) -> tuple[dict, dict, dict]:
    """overlap_score.score(online=True) as score_run.py calls it; returns (result, per-meeting captured state, segs)."""
    import overlap_score as O
    from meeting_score import universe

    U = universe("zh2en", {w["meeting"] for w in W})
    code = O.score.__code__
    rec_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == "recalled")
    cap, segs = {}, {}

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is rec_code:
            f = frame.f_back
            while f is not None and f.f_code is not code:
                f = f.f_back
            L = f.f_locals
            if L["m"] not in cap:
                # the closure's cells are rebound per meeting: evaluate it now (tracing is off inside the tracer)
                cap[L["m"]] = {k: L[k] for k in ("sc", "got", "unit_of", "units_", "mp", "us")}
                cap[L["m"]]["rec"] = {rid: bool(L["recalled"](rid)) for rid in L["sc"]}
                segs.update(L["segs"])
        return None

    sys.settrace(tracer)
    try:
        res = O.score("zh2en", W, H, RW, U, online=True)
    finally:
        sys.settrace(None)
    return res, cap, segs


def seg_meta(W: list[dict], H: dict, segs: dict) -> dict:
    """per meeting, parallel to the scorer's segs[m] (built in window order, then segment order)."""
    meta = defaultdict(list)
    for wi, w in enumerate(W):
        h = H[w["id"]]
        reads = [r["t"] for r in h["read_log"]]
        for si, s in enumerate(h["segments"]):
            meta[w["meeting"]].append({"wid": w["id"], "wi": wi, "si": si, "write_ms": s["write_ms"],
                                       "commit": w["start_s"] + s["write_ms"] / 1000.0,
                                       "final": s["write_ms"] >= reads[-1], "read": reads.index(s["write_ms"]),
                                       "w_start": w["start_s"]})
    for m, L in meta.items():
        assert len(L) == len(segs[m]), m
        for x, s in zip(L, segs[m]):
            assert abs(s["a"] - (W[x["wi"]]["start_s"] + H[x["wid"]]["segments"][x["si"]]["start_s"])) < 1e-9
    return meta


def due_of(u: dict, windows: list[dict], H: dict) -> dict:
    """the online window with the largest overlap with u and u's due read in it (module docstring)."""
    best, bw = 0.0, None
    for w in windows:
        o = ov(u["a"], u["b"], w["start_s"], w["start_s"] + w["dur_ms"] / 1000.0)
        if o > best:
            best, bw = o, w
    if bw is None:
        return {"window": None, "T_due": u["b"] + LAG_S, "due_final": None}
    reads = [x["t"] for x in H[bw["id"]]["read_log"]]
    i = due_read(reads, bw["start_s"], u["b"])
    return {"window": bw["id"], "T_due": bw["start_s"] + reads[i] / 1000.0, "due_final": i == len(reads) - 1,
            "n_reads": len(reads), "window_has_output": bool(H[bw["id"]]["segments"])}


def classify_missed(u: dict, rid: str, st: dict, seglist: list[dict], meta: list[dict], windows: list[dict],
                    H: dict) -> dict:
    """the pre-stated Check A classes for a missed short utterance u (scorer dict, meeting clock)."""
    sc, got, unit_of, mp = st["sc"], st["got"], st["unit_of"], st["mp"]
    r = norm(u["ref"])
    assert r, rid
    out = {"rid": rid, "spk": u["spk"], "a": u["a"], "b": u["b"], "dur": u["b"] - u["a"], "ref": u["ref"]}
    out.update(due_of(u, windows, H))
    T = out["T_due"]
    idx = {id(s): k for k, s in enumerate(seglist)}

    def explained(spk, x, exclude):
        if spk is None:
            return False
        for rid2, v in sc.items():
            if rid2 == exclude or v["spk"] != spk:
                continue
            if ov(x["a"] - COLLAR_S, x["b"] + COLLAR_S, v["a"], v["b"]) > 0 and contains(norm(v["ref"]), r):
                return True
        return False

    mis, own, mis_bc = [], [], []
    for k, x in enumerate(seglist):
        if not x["text"] or ov(x["a"] - COLLAR_S, x["b"] + COLLAR_S, u["a"], u["b"]) <= 0:
            continue
        xw = norm(x["text"])
        ms = mp.get(x["spk"])
        if ms != u["spk"]:
            if edge_match(xw, r) and not explained(ms, x, rid):
                mis.append(k)
            if r and all(t in BC for t in r) and 1 <= len(xw) <= 3 and all(t in BC for t in xw):
                mis_bc.append(k)
        elif edge_match(xw, r) and not explained(u["spk"], x, rid):
            own.append(k)
    # order: other-speaker utterances overlapping u, their credited segments within the recall collar of u
    order = []
    for rid2, v in sc.items():
        if v["spk"] == u["spk"] or ov(u["a"], u["b"], v["a"], v["b"]) <= 0:
            continue
        for x, t in got.get(unit_of[rid2], []):
            if not t.strip() or ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) <= 0:
                continue
            k = idx[id(x)]
            if meta[k]["commit"] >= T - 1e-6:
                order.append(k)
    order = sorted(set(order))
    near = [k for k, x in enumerate(seglist) if x["text"] and abs(meta[k]["commit"] - T) <= SILENT_S]
    if mis:
        cls = "MISLABEL"
    elif own:
        cls = "OWN_UNCREDITED"
    elif order:
        cls = "ORDER_OMISSION"
    elif not near:
        cls = "SILENT"
    else:
        cls = "OTHER"
    out["cls"] = cls
    out["mislabel_bc"] = bool(mis_bc)
    out["has_overlapping_other"] = any(v["spk"] != u["spk"] and ov(u["a"], u["b"], v["a"], v["b"]) > 0
                                       for v in sc.values())
    if order:
        k0 = min(order, key=lambda k: meta[k]["commit"])
        out["order_first_commit_minus_due"] = round(meta[k0]["commit"] - T, 3)
        out["order_same_read"] = any(abs(meta[k]["commit"] - T) < 1e-6 for k in order)
        out["order_spk"] = seglist[k0]["spk"]
        out["order_text"] = seglist[k0]["text"]
    if mis:
        out["mislabel_text"] = seglist[mis[0]]["text"]
        out["mislabel_k"] = mis[0]
        out["mislabel_slot"] = seglist[mis[0]]["spk"]
    if cls == "OTHER":
        conc_before = any(
            ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0 and t.strip()
            for rid2, v in sc.items() if v["spk"] != u["spk"] and ov(u["a"], u["b"], v["a"], v["b"]) > 0
            for x, t in got.get(unit_of[rid2], []))
        own_near = any(x["text"] and mp.get(x["spk"]) == u["spk"]
                       and ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0 for x in seglist)
        out["other_sub"] = ("OTHER_SPK_BEFORE_DUE" if conc_before else "OWN_SLOT_NEAR_NO_TEXT_MATCH" if own_near
                            else "REST")
    return out


def first_key(ks: list[int], meta: list[dict]):
    return min((meta[k]["commit"], meta[k]["w_start"], meta[k]["si"], k) for k in ks)


def reverse_one(u: dict, rid: str, st: dict, seglist: list[dict], meta: list[dict]) -> dict:
    sc, got, unit_of = st["sc"], st["got"], st["unit_of"]
    idx = {id(s): k for k, s in enumerate(seglist)}
    eu = [idx[id(x)] for x, t in got.get(unit_of[rid], [])
          if t.strip() and ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0]
    eo = []
    for rid2, v in sc.items():
        if v["spk"] == u["spk"] or ov(u["a"], u["b"], v["a"], v["b"]) <= 0:
            continue
        eo += [idx[id(x)] for x, t in got.get(unit_of[rid2], [])
               if t.strip() and ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0]
    assert eu, rid
    if not eo:
        return {"rid": rid, "rev": "NO_OTHER_WRITTEN"}
    fu, fo = first_key(eu, meta), first_key(eo, meta)
    if fo < fu:
        rev = "OTHER_FIRST_SAME_READ" if abs(fo[0] - fu[0]) < 1e-6 else "OTHER_FIRST_EARLIER_READ"
    else:
        rev = "SHORT_FIRST_SAME_READ" if abs(fo[0] - fu[0]) < 1e-6 else "SHORT_FIRST_EARLIER_READ"
    return {"rid": rid, "rev": rev, "dt": round(fo[0] - fu[0], 3)}


def check_a(args) -> dict:
    W = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(args.decode) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(args.ref_dev, encoding="utf-8")]
    W = [w for w in W if w["id"] in H]
    t0 = time.time()
    res, cap, segs = run_scorer(W, H, RW)
    print(f"scorer {time.time() - t0:.0f}s", flush=True)
    ref = json.load(open(Path(args.decode) / "score_text.json", encoding="utf-8"))
    for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "CR", "short_recall", "omitted_ov", "omitted_1spk"):
        assert res[k] == ref[k], (k, res[k], ref[k])
    meta = seg_meta(W, H, segs)
    wins_of = defaultdict(list)
    for w in W:
        wins_of[w["meeting"]].append(w)
    rows, rev_rows = [], []
    short_chk = defaultdict(lambda: [0, 0])
    for m in sorted(cap):
        st = cap[m]
        sc = st["sc"]
        from overlap_score import OV_SHARE, others_overlap

        for rid, u in sc.items():
            d = u["b"] - u["a"]
            if d > 1.0:
                continue
            got_it = st["rec"][rid]
            ov20 = others_overlap(u["a"], u["b"], u["spk"], st["us"]) / max(d, 1e-6) >= OV_SHARE
            band = "le0.5" if d <= 0.5 else "0.5-1"
            short_chk[f"{band}_{'ov' if ov20 else '1spk'}"][0] += got_it
            short_chk[f"{band}_{'ov' if ov20 else '1spk'}"][1] += 1
            o50 = ov50_of(u, [v for r2, v in sc.items() if r2 != rid and v["spk"] != u["spk"]])
            base = {"meeting": m, "rid": rid, "dur": round(d, 3), "le05": d <= 0.5, "ov50": o50, "ov20": ov20}
            if not got_it:
                rows.append({**base, **classify_missed(u, rid, st, segs[m], meta[m], wins_of[m], H)})
            else:
                rev_rows.append({**base, **reverse_one(u, rid, st, segs[m], meta[m])})
    for k, (a, n) in short_chk.items():
        assert ref["short_recall"][k]["n"] == n and abs(ref["short_recall"][k]["recall"] - round(a / n, 4)) < 1e-9, k
    groups = {"le1_iso": lambda x: not x["ov50"], "le1_ov50": lambda x: x["ov50"],
              "le05_iso": lambda x: x["le05"] and not x["ov50"], "le05_ov50": lambda x: x["le05"] and x["ov50"],
              "le1_all": lambda x: True,
              "le1_iso_ov20": lambda x: not x["ov20"], "le1_ov_ov20": lambda x: x["ov20"]}
    summ = {"decode": args.decode, "scorer_check": "score_text.json headline + short recall reproduced exactly",
            "short_utts": len(rows) + len(rev_rows), "missed": len(rows), "groups": {}}
    for g, f in groups.items():
        sub = [x for x in rows if f(x)]
        allg = [x for x in rows + rev_rows if f(x)]
        c = Counter(x["cls"] for x in sub)
        o = [x for x in sub if x["cls"] == "ORDER_OMISSION"]
        rv = [x for x in rev_rows if f(x)]
        rc = Counter(x["rev"] for x in rv)
        summ["groups"][g] = {
            "n_short": len(allg), "n_missed": len(sub), "miss_rate": round(len(sub) / max(len(allg), 1), 4),
            "counts": {k: c.get(k, 0) for k in A_CLASSES},
            "shares": {k: round(c.get(k, 0) / max(len(sub), 1), 4) for k in A_CLASSES},
            "other_sub": dict(Counter(x.get("other_sub") for x in sub if x["cls"] == "OTHER")),
            "mislabel_bc_extra": sum(1 for x in sub if x["mislabel_bc"] and x["cls"] != "MISLABEL"),
            "no_window": sum(1 for x in sub if x["window"] is None),
            "due_is_final_read": sum(1 for x in sub if x.get("due_final")),
            "has_overlapping_other": sum(1 for x in sub if x["has_overlapping_other"]),
            "order_same_read": sum(1 for x in o if x["order_same_read"]),
            "order_due_final": sum(1 for x in o if x.get("due_final")),
            "order_dt_median": float(np.median([x["order_first_commit_minus_due"] for x in o])) if o else None,
            "reverse": {"n": len(rv), "counts": dict(rc),
                        "other_first": sum(v for k, v in rc.items() if k.startswith("OTHER_FIRST")),
                        "other_written": sum(v for k, v in rc.items() if k != "NO_OTHER_WRITTEN")}}
    return {"summary": summ, "missed_rows": rows, "reverse_rows": rev_rows}


# ------------------------------------------------------------------------------------------------ check B
def evidence(P: np.ndarray, a_ms: int, b_ms: int, t_ms: int, k: int, k2: int) -> dict:
    """the unit's frames of the prompt P at read t (frames >= (t - 1.04 s) / 80 ms zeroed)."""
    F = len(P)
    f0 = min(int(a_ms // FRAME_MS), F - 1)
    f1 = min(max(int(np.ceil(b_ms / FRAME_MS)), f0 + 1), F)
    z = max(0, t_ms - LOOKAHEAD_MS) // FRAME_MS
    Pp = P.copy()
    Pp[z:] = 0.0
    up, uf = Pp[f0:f1], P[f0:f1]
    mp, mf = up.mean(0), uf.mean(0)

    def cat(x, m):
        if x.max() < 0.5:
            return "NONE"
        j = int(m.argmax())
        return "GOLD_COL" if j == k - 1 else "K2_COL" if j == k2 - 1 else "OTHER_COL"

    ev = cat(up, mp)
    if ev == "NONE":
        ev_sub = "UNSEEN" if uf.max() >= 0.5 else "SILENT"
    else:
        ev_sub = ev
    return {"frames": f1 - f0, "vis_frac": float(np.clip((z - f0) / (f1 - f0), 0, 1)), "ev": ev, "ev_sub": ev_sub,
            "ev_full": cat(uf, mf), "p_gold": float(mp[k - 1]), "p_k2": float(mp[k2 - 1]),
            "p_gold_full": float(mf[k - 1]), "p_k2_full": float(mf[k2 - 1])}


def window_map(win: dict, P: np.ndarray, sess: dict, spk: str) -> dict:
    from experiments.mist_w3po_v1.data.build_windows_zh2en import FRAME, window_slot_map

    members = [u for u in win["utts"]]
    wm = window_slot_map(members, P.astype(np.float32), sess)
    on = P > 0.5
    act = np.zeros(len(P), dtype=bool)
    for u in members:
        if u["speaker_id"] == spk:
            act[max(0, int(u["begin_s"] / FRAME)): int(np.ceil(u["end_s"] / FRAME))] = True
    covered = bool((act[:, None] & on).any())
    return {"win_slot": int(wm[spk]), "covered": covered, "n_spk_win": len({u["speaker_id"] for u in members})}


def cause_of(ev: str, win: str) -> str:
    if ev == "K2_COL":
        return "LABEL_NOISE" if win == "WIN_K2" else "WRONG_EVIDENCE"
    return {"NONE": "NO_EVIDENCE", "GOLD_COL": "IGNORES_EVIDENCE", "OTHER_COL": "OTHER_EVIDENCE"}[ev]


def check_b(args) -> dict:
    smap = json.loads(Path(SLOT_MAP).read_text(encoding="utf-8"))["lowlat"]
    out = {"rows": [], "summary": {}}
    for split, d in (("dev", args.rescore_dev), ("train", args.rescore_train)):
        items = [json.loads(l) for f in sorted(Path(d).glob("shard_*/items.jsonl")) for l in open(f, encoding="utf-8")]
        meta = json.loads(next(Path(d).glob("shard_*/meta.json")).read_text())
        items = [it for it in items if "final" in it["reads"]
                 and it["label"] in ("ATTRIBUTION", "ATTRIBUTION_NOPENDING", "GOLD")]
        want = {it["id"] for it in items}
        recs = {}
        for f in sorted(Path(meta["sched"]).glob("shard_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["id"] in want:
                    recs[r["id"]] = r
        refw = {}
        for l in open(args.ref_dev if split == "dev" else args.ref_train, encoding="utf-8"):
            w = json.loads(l)
            if w["id"] in want:
                refw[w["id"]] = w
        rows = []
        for it in items:
            rec, win = recs[it["id"]], refw[it["id"]]
            u = next(x for x in rec["units"] if x["utt"] == it["utt"])
            sess = smap[rec["meeting"]]
            assert sess[u["speaker_id"]] == it["slot"] == u["slot"]
            P = np.load(rec["post"]).astype(np.float32)
            k, k2 = it["slot"], it["k2"]
            e = evidence(P, u["a_ms"], u["b_ms"], rec["t_ms"][it["r"]], k, k2)
            wm = window_map(win, P, sess, u["speaker_id"])
            ws = wm["win_slot"]
            wcat = "WIN_AGREE" if ws == k else "WIN_K2" if ws == k2 else "WIN_OTHER"
            k2_spk = [s for s, kk in sess.items() if kk == k2]
            win_spks = {x["speaker_id"] for x in win["utts"]}
            rows.append({"split": split, "id": it["id"], "utt": it["utt"], "label": it["label"],
                         "attr": it["label"] != "GOLD", "dur_s": it["dur_s"], "ov50": it["ov50"], "slot": k, "k2": k2,
                         **e, "win_slot": ws, "win": wcat, "covered": wm["covered"], "n_spk_win": wm["n_spk_win"],
                         "k2_speaker_in_window": any(s in win_spks for s in k2_spk),
                         "win_dur_s": rec["dur_ms"] / 1000.0, "unit_end_to_win_end_s": (rec["dur_ms"] - u["b_ms"]) / 1000.0,
                         "cause": cause_of(e["ev"], wcat)})
        out["rows"] += rows
        groups = {"le1_iso": lambda x: not x["ov50"], "le05_iso": lambda x: x["dur_s"] <= 0.5 and not x["ov50"],
                  "le1_ov50": lambda x: x["ov50"], "le1_all": lambda x: True}
        for g, f in groups.items():
            for lab, sel in (("ATTR", lambda x: x["attr"]), ("GOLD_ctrl", lambda x: not x["attr"])):
                sub = [x for x in rows if f(x) and sel(x)]
                if not sub:
                    continue
                n = len(sub)

                def cnt(key):
                    c = Counter(x[key] for x in sub)
                    return {kk: [v, round(v / n, 4)] for kk, v in c.most_common()}

                def med(key):
                    return round(float(np.median([x[key] for x in sub])), 4)

                out["summary"][f"{split}|{g}|{lab}"] = {
                    "n": n, "cause": cnt("cause"), "ev_sub": cnt("ev_sub"), "ev_full": cnt("ev_full"), "win": cnt("win"),
                    "win_disagree": sum(x["win"] != "WIN_AGREE" for x in sub),
                    "uncovered": sum(not x["covered"] for x in sub),
                    "k2_speaker_in_window": sum(x["k2_speaker_in_window"] for x in sub),
                    "p_gold_median": med("p_gold"), "p_k2_median": med("p_k2"),
                    "p_gold_mean": round(float(np.mean([x["p_gold"] for x in sub])), 4),
                    "p_k2_mean": round(float(np.mean([x["p_k2"] for x in sub])), 4),
                    "p_gold_full_median": med("p_gold_full"), "p_k2_full_median": med("p_k2_full"),
                    "vis_frac_median": med("vis_frac"), "fully_unseen": sum(x["vis_frac"] == 0 for x in sub),
                    "win_dur_median": med("win_dur_s"), "unit_end_to_win_end_median": med("unit_end_to_win_end_s"),
                    "single_speaker_window": sum(x["n_spk_win"] == 1 for x in sub),
                    "cross_ev_win": dict(Counter(f"{x['ev']}|{x['win']}" for x in sub))}
    return out


# ------------------------------------------------------------------------------------------------ post-hoc (after B)
def other_relation(u: dict, rid: str, st: dict, seglist: list[dict], meta: list[dict], T: float) -> str:
    """when the other speakers' concurrent part (segments credited to the units of other-speaker utterances
    overlapping u, within 1 s of u) was first committed, relative to u's due read."""
    sc, got, unit_of = st["sc"], st["got"], st["unit_of"]
    idx = {id(s): k for k, s in enumerate(seglist)}
    cs = []
    for rid2, v in sc.items():
        if v["spk"] == u["spk"] or ov(u["a"], u["b"], v["a"], v["b"]) <= 0:
            continue
        cs += [meta[idx[id(x)]]["commit"] for x, t in got.get(unit_of[rid2], [])
               if t.strip() and ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0]
    if not cs:
        return "NONE"
    c = min(cs)
    return "BEFORE_DUE" if c < T - 1e-6 else "AT_DUE_READ" if c <= T + 1e-6 else "AFTER_DUE"


def aligned_other_slot(u: dict, st: dict, seglist: list[dict]) -> bool:
    """(post-hoc, descriptive) a short segment (<= 4 words) under a slot not mapped to u's speaker, whose source span
    lies >= 50 % within [a - 0.2 s, b + 0.2 s] of u and is not >= 50 % covered by a scored utterance of its own mapped
    speaker: something written at u's position under another speaker that the strict text match does not see."""
    sc, mp = st["sc"], st["mp"]
    for x in seglist:
        xw = norm(x["text"])
        if not xw or len(xw) > 4 or mp.get(x["spk"]) == u["spk"]:
            continue
        L = max(x["b"] - x["a"], 0.08)
        if ov(x["a"], x["b"], u["a"] - 0.2, u["b"] + 0.2) < 0.5 * L:
            continue
        own = mp.get(x["spk"])
        if own is not None and any(v["spk"] == own and ov(x["a"], x["b"], v["a"], v["b"]) >= 0.5 * L
                                   for v in sc.values()):
            continue
        return True
    return False


def check_a_posthoc(args) -> dict:
    """POST-HOC (added after Check B showed the unit frames masked at the final read): on the online decode,
    (1) the visibility of u's diarizer frames in the prompt at the read where u's slot was decided: the commit read of
        the first recall-evidence segment (recalled u) or of the first MISLABEL segment (missed u); prompt P = the
        online window's lowlat P with frames >= (t - 1.04 s) / 80 ms zeroed (Engine.audio_embeds); expected slot =
        the hyp slot the scorer maps to u's speaker, written slot = the segment's slot; evidence() as in Check B;
    (2) the miss rate of short utterances by when the other speakers' concurrent part was first written relative to
        u's due read (other_relation)."""
    W = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(args.decode) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(args.ref_dev, encoding="utf-8")]
    W = [w for w in W if w["id"] in H]
    Wid = {w["id"]: w for w in W}
    res, cap, segs = run_scorer(W, H, RW)
    ref = json.load(open(Path(args.decode) / "score_text.json", encoding="utf-8"))
    assert res["short_recall"] == ref["short_recall"]
    meta = seg_meta(W, H, segs)
    wins_of = defaultdict(list)
    for w in W:
        wins_of[w["meeting"]].append(w)
    rows = []
    for m in sorted(cap):
        st = cap[m]
        sc, got, unit_of = st["sc"], st["got"], st["unit_of"]
        inv = {v: k for k, v in st["mp"].items()}
        idx = {id(s): k for k, s in enumerate(segs[m])}
        for rid, u in sc.items():
            d = u["b"] - u["a"]
            if d > 1.0:
                continue
            o50 = ov50_of(u, [v for r2, v in sc.items() if r2 != rid and v["spk"] != u["spk"]])
            due = due_of(u, wins_of[m], H)
            row = {"meeting": m, "rid": rid, "dur": round(d, 3), "le05": d <= 0.5, "ov50": o50,
                   "missed": not st["rec"][rid], "due_final": due["due_final"],
                   "other_rel": other_relation(u, rid, st, segs[m], meta[m], due["T_due"])}
            k = None
            if st["rec"][rid]:
                eu = [idx[id(x)] for x, t in got.get(unit_of[rid], [])
                      if t.strip() and ov(x["a"] - RECALL_COLLAR, x["b"] + RECALL_COLLAR, u["a"], u["b"]) > 0]
                k = first_key(eu, meta[m])[3]
                row["cls"] = "RECALLED"
            else:
                c = classify_missed(u, rid, st, segs[m], meta[m], wins_of[m], H)
                row["cls"] = c["cls"]
                row["mislabel_bc"] = c["mislabel_bc"]
                row["aligned_other_slot"] = aligned_other_slot(u, st, segs[m])
                k = c.get("mislabel_k")
            exp = inv.get(u["spk"])
            if k is not None and exp is not None and isinstance(segs[m][k]["spk"], int):
                x, mt = segs[m][k], meta[m][k]
                w = Wid[mt["wid"]]
                P = np.load(w["post"]["lowlat"]).astype(np.float32)
                a_ms = max(0, round((u["a"] - w["start_s"]) * 1000))
                b_ms = max(a_ms + 1, round((u["b"] - w["start_s"]) * 1000))
                if a_ms < len(P) * FRAME_MS:
                    e = evidence(P, a_ms, b_ms, mt["write_ms"], exp, x["spk"] if x["spk"] != exp else 0)
                    row.update({"written_slot": x["spk"], "expected_slot": exp, "decision_final": mt["final"],
                                "win_dur_s": w["dur_ms"] / 1000.0, **e})
            rows.append(row)
    groups = {"le1_iso": lambda x: not x["ov50"], "le05_iso": lambda x: x["le05"] and not x["ov50"],
              "le1_ov50": lambda x: x["ov50"], "le05_ov50": lambda x: x["le05"] and x["ov50"]}
    summ = {"note": "post-hoc, after Check B", "visibility": {}, "miss_by_other_rel": {}, "hidden_mislabel": {}}
    for g, f in groups.items():
        for c in ("RECALLED", "MISLABEL"):
            sub = [x for x in rows if f(x) and x["cls"] == c and "ev_sub" in x]
            if not sub:
                continue
            n = len(sub)
            summ["visibility"][f"{g}|{c}"] = {
                "n": n, "ev_sub": {kk: [v, round(v / n, 4)] for kk, v in Counter(x["ev_sub"] for x in sub).most_common()},
                "ev_full": {kk: [v, round(v / n, 4)] for kk, v in Counter(x["ev_full"] for x in sub).most_common()},
                "vis_frac_median": round(float(np.median([x["vis_frac"] for x in sub])), 4),
                "fully_unseen": sum(x["vis_frac"] == 0 for x in sub),
                "decision_at_final_read": sum(bool(x["decision_final"]) for x in sub),
                "win_dur_median": round(float(np.median([x["win_dur_s"] for x in sub])), 3)}
        sub = [x for x in rows if f(x)]
        summ["hidden_mislabel"][g] = {
            c: {"n": sum(1 for x in sub if x["cls"] == c),
                "aligned_other_slot": sum(1 for x in sub if x["cls"] == c and x["aligned_other_slot"]),
                "mislabel_bc": sum(1 for x in sub if x["cls"] == c and x["mislabel_bc"])}
            for c in A_CLASSES}
        summ["miss_by_other_rel"][g] = {
            rel: {"n": len(s2), "missed": sum(x["missed"] for x in s2),
                  "miss_rate": round(sum(x["missed"] for x in s2) / max(len(s2), 1), 4),
                  "order_omission": sum(x["cls"] == "ORDER_OMISSION" for x in s2),
                  "mislabel": sum(x["cls"] == "MISLABEL" for x in s2)}
            for rel in ("BEFORE_DUE", "AT_DUE_READ", "AFTER_DUE", "NONE")
            for s2 in [[x for x in sub if x["other_rel"] == rel]]}
    return {"summary": summ, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode", default=DECODE)
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--ref-dev", default=REF_DEV)
    ap.add_argument("--ref-train", default=REF_TRAIN)
    ap.add_argument("--rescore-dev", default=RESCORE["dev"])
    ap.add_argument("--rescore-train", default=RESCORE["train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", choices=["", "A", "B"])
    ap.add_argument("--posthoc", action="store_true", help="only the post-hoc online-decode analysis (after B)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res = {}
    if args.posthoc:
        ph = check_a_posthoc(args)
        print(json.dumps(ph["summary"], indent=1), flush=True)
        out.mkdir(parents=True)
        (out / "posthoc_summary.json").write_text(json.dumps(ph["summary"], indent=1), encoding="utf-8")
        with open(out / "posthoc_rows.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in ph["rows"])
        print(f"DIAG_ORDER_ATTR_POSTHOC_DONE {out}", flush=True)
        return
    if args.only in ("", "B"):
        res["B"] = check_b(args)
        print(json.dumps(res["B"]["summary"], indent=1), flush=True)
    if args.only in ("", "A"):
        res["A"] = check_a(args)
        print(json.dumps(res["A"]["summary"], indent=1), flush=True)
    out.mkdir(parents=True)
    if "A" in res:
        (out / "check_a_summary.json").write_text(json.dumps(res["A"]["summary"], indent=1), encoding="utf-8")
        with open(out / "check_a_missed.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in res["A"]["missed_rows"])
        with open(out / "check_a_reverse.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in res["A"]["reverse_rows"])
    if "B" in res:
        (out / "check_b_summary.json").write_text(json.dumps(res["B"]["summary"], indent=1), encoding="utf-8")
        with open(out / "check_b_rows.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in res["B"]["rows"])
    print(f"DIAG_ORDER_ATTR_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
