"""Metrics-first headroom analysis of a dev decode (CPU only; reports/mist_w3po_v1/metrics_first_si_analysis.zh-CN.md).

Re-implements the unit assignment of reports/gstar_si_v8/overlap_score.py (score(), online=True, split, collar 2 s,
merge 1 s) so that the per-unit (hyp, ref) pairs are visible, checks that the re-implementation reproduces the
scorer's tcpBLEU all / ov / 1spk exactly, then measures how far each error category can move the headline metrics:

  oracles (each alone, applied to the scorer's own assignment; upper bounds -- a filled utterance gets its reference)
    fill_short_{iso,ov,all}   every non-recalled scored utterance <= 1 s (iso / ov by the scorer's 20% rule) gets its
                              reference text inserted into its unit at its time position
    fill_long                 the same for non-recalled utterances > 1 s
    fill_all                  every non-recalled utterance
    drop_ins_{ov,1spk,all}    insertions removed
    drop_dup                  in units holding a duplicate pair (chrF >= 50), the later segment of the pair removed
    spk_oracle                every segment relabelled to the speaker of the scored utterance within +-2 s whose
                              reference is closest to its text (chrF; see relabel_oracle), speaker mapping = identity
    spk_oracle_short          the same, only for segments whose best utterance is <= 1 s
    trans_oracle              every unit that holds text gets its reference (content ceiling at the current recall)
    under_trans_lt0.5         units whose text has fewer than half the reference's words get their reference
    fill_all_drop_ins         fill_all and drop_ins_all together
  breakdown                   reference words by utterance duration band x region x recalled; hyp words of insertions
  latency (text)              per unit: StartOffset = first commit - unit start, EndOffset = last commit - unit end,
                              LAAL over the unit's words (delay = commit - unit start, |X| = unit length,
                              |Y| = max(|hyp|, |ref|), tau = first word committed at or after the unit end)
  speech                      --speech-dirs: talker results (StartOffset / EndOffset per window x speaker stream,
                              output overlap between streams); --asr-hyps: the speech tcpBLEU re-scored, plus the
                              text hyp merged per window x speaker exactly as asr_hyp.py merges the ASR (granularity
                              artifact of the speech scorer, without any TTS / ASR error)

Writes <out> (json; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.metric_headroom --name c0fp --run <decode dir> \
        [--speech-dirs <talker dir> ...] [--asr-hyps <hyp_asr.jsonl> ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
import overlap_score as O  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402

WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
REF_WIN = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"
DR = "zh2en"
SEP = " "


def build_segs(W, H, online=True):
    segs = defaultdict(list)
    for w in W:
        for s in H[w["id"]]["segments"]:
            c = None
            if "write_ms" in s:
                c = s["write_ms"]
                if online:
                    c = max(c, w["open_detect_ms"])
                    if s["write_ms"] >= w["dur_ms"]:
                        c = max(c, w["close_detect_ms"])
                c = w["start_s"] + c / 1000
            segs[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                       "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08),
                                       "text": s["text"].strip(), "commit": c, "win": w["id"]})
    return segs


def scored_utts(W, RW):
    meetings = {w["meeting"] for w in W}
    sc = {}
    for w in RW:
        if w["meeting"] not in meetings:
            continue
        for u in w["utts"]:
            if not u["hesitation"] and u.get("ref", u["zh"]).strip():
                sc[u["record_id"]] = {"meeting": w["meeting"], "spk": u["speaker_id"], "a": w["start_s"] + u["begin_s"],
                                      "b": w["start_s"] + u["end_s"], "ref": u.get("ref", u["zh"]).strip()}
    return sc


def assign(m, msegs, sc_all, us, identity=False, collar=2.0, merge=1.0, split=True):
    """overlap_score.score() for one meeting, returning the assignment instead of the corpus statistics."""
    sc = {rid: v for rid, v in sc_all.items() if v["meeting"] == m}
    hyp_t, ref_t = defaultdict(str), defaultdict(str)
    for s in msegs:
        hyp_t[s["spk"]] += SEP + s["text"]
    for v in sorted(sc.values(), key=lambda v: v["a"]):
        ref_t[v["spk"]] += SEP + v["ref"]
    if identity:
        mp = {k: k for k in hyp_t if k in ref_t}
    else:
        mp = match({k: v for k, v in hyp_t.items() if k != "unk"}, dict(ref_t))
    unit_of, units_ = {}, {}
    for spk in {v["spk"] for v in sc.values()}:
        cur = None
        for rid, v in sorted(((r, v) for r, v in sc.items() if v["spk"] == spk), key=lambda kv: kv[1]["a"]):
            if cur is not None and v["a"] - units_[cur]["b"] < merge:
                units_[cur]["b"] = max(units_[cur]["b"], v["b"])
                units_[cur]["ref"] += SEP + v["ref"]
                units_[cur]["utts"].append(rid)
            else:
                cur = rid
                units_[cur] = {"spk": spk, "a": v["a"], "b": v["b"], "ref": v["ref"], "utts": [rid]}
            unit_of[rid] = cur
    for rid, v in units_.items():
        d = max(v["b"] - v["a"], 1e-6)
        v["share"] = O.others_overlap(v["a"], v["b"], v["spk"], us) / d
        v["reg"] = "ov" if v["share"] >= O.OV_SHARE else "1spk"
    got = defaultdict(list)
    ins = []
    for s in sorted(msegs, key=lambda s: s["a"]):
        if not s["text"]:
            continue
        a, b = s["a"] - collar, s["b"] + collar
        best, bu = 0.0, None
        for u in us:
            ov = min(b, u[3]) - max(a, u[2])
            if ov > best:
                best, bu = ov, u
        if bu is None:
            ins.append(s)
            continue
        if bu[0] not in sc:
            continue
        spk = mp.get(s["spk"])
        target = unit_of[bu[0]] if sc[bu[0]]["spk"] == spk else None
        if target is None and spk is not None:
            best2 = 0.0
            for rid, v in units_.items():
                if v["spk"] != spk:
                    continue
                ov = min(b, v["b"]) - max(a, v["a"])
                if ov > best2:
                    best2, target = ov, rid
        if target is None:
            ins.append(s)
            continue
        if split and spk is not None:
            L = max(s["b"] - s["a"], 1e-6)
            cand = [(rid, O.overlap(s["a"], s["b"], v["a"], v["b"])) for rid, v in units_.items() if v["spk"] == spk]
            cand = sorted([(r, o) for r, o in cand if o / L >= O.OV_SHARE], key=lambda ro: units_[ro[0]]["a"])
            if len(cand) >= 2:
                pc = O.pieces(s["text"], True)
                tot = sum(o for _, o in cand)
                i0 = 0
                for j, (r, o) in enumerate(cand):
                    i1 = len(pc) if j == len(cand) - 1 else i0 + int(round(len(pc) * o / tot))
                    if i1 > i0:
                        got[r].append((s, SEP.join(pc[i0:i1])))
                    i0 = i1
                continue
        got[target].append((s, s["text"]))
    for s in ins:
        s["reg"] = "ov" if O.multi_talk(s["a"], s["b"], us) else "1spk"

    def recalled(rid):
        v = sc[rid]
        return any(O.overlap(x["a"] - O.PAIR_COLLAR, x["b"] + O.PAIR_COLLAR, v["a"], v["b"]) > 0 and t.strip()
                   for x, t in got.get(unit_of[rid], []))
    rec = {rid: recalled(rid) for rid in sc}
    for rid, v in sc.items():
        d = v["b"] - v["a"]
        v["dur"] = d
        v["uov"] = O.others_overlap(v["a"], v["b"], v["spk"], us) / max(d, 1e-6) >= O.OV_SHARE
    return {"sc": sc, "units": units_, "unit_of": unit_of, "got": got, "ins": ins, "rec": rec, "mp": mp}


def bleu_stats(pairs):
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize="13a")
    if not pairs:
        return None
    st = bleu._extract_corpus_statistics([x for x, _ in pairs], [[y for _, y in pairs]])
    return np.sum(np.array(st, dtype=np.float64), 0)


def bleu_from(st):
    from sacrebleu.metrics import BLEU
    b = BLEU(tokenize="13a")
    r = b._compute_score_from_stats(list(st))
    return {"bleu": round(r.score, 2), "bp": round(r.bp, 4), "sys_len": int(r.sys_len), "ref_len": int(r.ref_len),
            "prec": [round(p, 2) for p in r.precisions]}


def pairs_of(A, fill=lambda rid, v: False, drop_ins=lambda s: False, drop_dup=False, trans=False, under=0.0):
    """unit and insertion pairs per region after the oracle edits."""
    from sacrebleu.metrics import CHRF
    ch = CHRF()
    P = {"ov": [], "1spk": [], "all": []}
    extra = defaultdict(list)
    for rid, v in A["sc"].items():
        if not A["rec"][rid] and fill(rid, v):
            extra[A["unit_of"][rid]].append(({"a": v["a"]}, v["ref"]))
    for uid, u in A["units"].items():
        ss = sorted(A["got"].get(uid, []), key=lambda x: x[0]["a"])
        if drop_dup and len(ss) >= 2:
            keep = []
            for x in ss:
                if any(ch.sentence_score(x[1], [y[1]]).score >= 50 for y in keep):
                    continue
                keep.append(x)
            ss = keep
        ss = sorted(ss + extra.get(uid, []), key=lambda x: x[0]["a"])
        hyp = SEP.join(t for _, t in ss).strip()
        if trans and hyp:
            hyp = u["ref"]
        if under and hyp and len(hyp.split()) < under * len(u["ref"].split()):
            hyp = u["ref"]
        P[u["reg"]].append((hyp, u["ref"]))
        P["all"].append((hyp, u["ref"]))
    for s in A["ins"]:
        if drop_ins(s):
            continue
        P[s["reg"]].append((s["text"], ""))
        P["all"].append((s["text"], ""))
    return P


def corpus(As, **kw):
    acc = {r: None for r in ("all", "ov", "1spk")}
    for A in As.values():
        P = pairs_of(A, **kw)
        for r in acc:
            st = bleu_stats(P[r])
            if st is not None:
                acc[r] = st if acc[r] is None else acc[r] + st
    return {r: bleu_from(acc[r]) for r in acc}


def relabel_oracle(msegs, us, sc, only_short=False, collar=2.0):
    """oracle speaker of a segment: among the scored utterances within the collar, the one whose reference is closest
    to the segment text (chrF; ties and text-free cases by time overlap); without any, the most-overlapping
    annotated utterance. Content-aware, so that a short interjection inside another speaker's long turn is not
    relabelled to the long turn just because of the time overlap."""
    from sacrebleu.metrics import CHRF
    ch = CHRF()
    out = []
    for s in msegs:
        cands = [u for u in us if u[0] in sc and min(s["b"] + collar, u[3]) - max(s["a"] - collar, u[2]) > 0]
        bu = None
        if cands and s["text"]:
            bu = max(cands, key=lambda u: (ch.sentence_score(s["text"], [sc[u[0]]["ref"]]).score,
                                           min(s["b"], u[3]) - max(s["a"], u[2])))
        if bu is None:
            best = 0.0
            for u in us:
                ov = min(s["b"] + collar, u[3]) - max(s["a"] - collar, u[2])
                if ov > best:
                    best, bu = ov, u
        t = dict(s)
        if bu is not None and (not only_short or bu[3] - bu[2] <= 1.0):
            t["spk"] = bu[1]
        elif only_short:
            t["spk"] = ("hyp", s["spk"])      # mapped later by the meeting's own chrF matching
        out.append(t)
    return out


def assign_all(W, H, sc, U, mode="base"):
    segs = build_segs(W, H)
    As = {}
    for m in sorted({w["meeting"] for w in W}):
        us = U[m]
        ms = segs[m]
        if mode == "spk_oracle":
            As[m] = assign(m, relabel_oracle(ms, us, sc), sc, us, identity=True)
        elif mode == "spk_oracle_short":
            # base mapping for the untouched segments, identity for the relabelled ones
            base = assign(m, ms, sc, us)
            rl = relabel_oracle(ms, us, sc, only_short=True)
            for t in rl:
                if isinstance(t["spk"], tuple):
                    t["spk"] = base["mp"].get(t["spk"][1], ("unmapped", t["spk"][1]))
            As[m] = assign(m, rl, sc, us, identity=True)
        else:
            As[m] = assign(m, ms, sc, us)
    return As


def breakdown(As):
    bands = [("le0.5", 0, 0.5), ("0.5-1", 0.5, 1.0), ("1-2", 1.0, 2.0), ("2-5", 2.0, 5.0), ("gt5", 5.0, 1e9)]
    out = defaultdict(lambda: {"n": 0, "words": 0, "n_miss": 0, "words_miss": 0})
    tot = 0
    for A in As.values():
        for rid, v in A["sc"].items():
            nw = len(v["ref"].split())
            tot += nw
            band = next(b for b, lo, hi in bands if lo < v["dur"] <= hi or (lo == 0 and v["dur"] <= hi))
            for key in (f"{band}_{'ov' if v['uov'] else 'iso'}", f"{band}_any"):
                out[key]["n"] += 1
                out[key]["words"] += nw
                if not A["rec"][rid]:
                    out[key]["n_miss"] += 1
                    out[key]["words_miss"] += nw
    ins_w = {"ov": 0, "1spk": 0}
    for A in As.values():
        for s in A["ins"]:
            ins_w[s["reg"]] += len(s["text"].split())
    res = {k: dict(v, word_share=round(v["words"] / tot, 4), miss_rate=round(v["n_miss"] / max(v["n"], 1), 4),
                   missed_word_share=round(v["words_miss"] / tot, 4)) for k, v in sorted(out.items())}
    return {"ref_words": tot, "bands": res, "insertion_words": ins_w}


def latency(As):
    R = {"ov": defaultdict(list), "1spk": defaultdict(list)}
    for A in As.values():
        for uid, u in A["units"].items():
            ss = sorted(A["got"].get(uid, []), key=lambda x: x[0]["a"])
            cs = [x[0]["commit"] for x in ss if x[0]["commit"] is not None]
            if not cs:
                continue
            r = R[u["reg"]]
            r["start"].append(min(cs) - u["a"])
            r["end"].append(max(cs) - u["b"])
            X = max(u["b"] - u["a"], 1e-3)
            d = []
            for x, t in ss:
                if x["commit"] is None:
                    continue
                d += [x["commit"] - u["a"]] * len(t.split())
            if not d:
                continue
            d.sort()
            Y = max(len(d), len(u["ref"].split()))
            tau = next((i for i, di in enumerate(d) if di >= X), len(d) - 1)
            r["laal"].append(float(np.mean([d[i] - i * X / Y for i in range(tau + 1)])))
    out = {}
    for reg in R:
        for k, v in R[reg].items():
            a = np.array(v)
            out[f"{k}_{reg}"] = {"mean": round(float(a.mean()), 3), "p50": round(float(np.median(a)), 3),
                                 "p90": round(float(np.percentile(a, 90)), 3), "n": int(len(a))}
    for k in ("start", "end", "laal"):
        a = np.array(R["ov"][k] + R["1spk"][k])
        out[f"{k}_all"] = {"mean": round(float(a.mean()), 3), "p50": round(float(np.median(a)), 3),
                           "p90": round(float(np.percentile(a, 90)), 3), "n": int(len(a))}
    return out


def merged_text_hyp(H):
    """the text hyp merged per window x speaker the way asr_hyp.py merges the ASR transcripts."""
    out = {}
    for wid, h in H.items():
        by = defaultdict(list)
        for s in h["segments"]:
            by[s["spk"]].append(s)
        segs = []
        for k, ss in by.items():
            t = " ".join(s["text"].strip() for s in ss if s["text"].strip())
            if t:
                segs.append({"spk": k, "start_s": min(s["start_s"] for s in ss), "end_s": max(s["end_s"] for s in ss),
                             "text": t, "write_ms": max(s.get("write_ms", 0) for s in ss)})
        out[wid] = {"id": wid, "segments": segs}
    return out


def speech_stats(talker_dir, W):
    wins = {w["id"]: w for w in W}
    rows = [json.loads(l) for l in open(Path(talker_dir) / "results.jsonl", encoding="utf-8")]
    so, eo, ratio = [], [], []
    per_win = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        so.append(r["start_offset_ms"] / 1000)
        eo.append(r["end_offset_ms"] / 1000)
        bl = r["blocks"]
        if isinstance(bl, str):
            import ast
            bl = ast.literal_eval(bl)
        wid = r["id"].split("__S")[0]
        iv = [(b["start_ms"] / 1000, b["start_ms"] / 1000 + b["dur_ms"] / 1000) for b in bl if b.get("dur_ms")]
        per_win[wid].append(iv)
        if wid in wins:
            ratio.append(sum(b - a for a, b in iv) / max(wins[wid]["dur_ms"] / 1000, 1e-3))
    # output overlap: time during which >= 2 streams of the same window speak / total speech time
    tot = ov = 0.0
    # cross-window backlog: speech of a window still playing when the next window of the meeting starts speaking
    ends = {}
    for wid, streams in per_win.items():
        ev = []
        for k, iv in enumerate(streams):
            for a, b in iv:
                ev += [(a, 1), (b, -1)]
                tot += b - a
        ev.sort()
        act, last = 0, None
        for t, d in ev:
            if last is not None and act >= 2:
                ov += t - last
            act += d
            last = t
        if wid in wins and streams and any(streams):
            w = wins[wid]
            ends[wid] = (w["meeting"], w["start_s"], w["start_s"] + max(b for iv in streams for _, b in iv),
                         w["start_s"] + min(a for iv in streams for a, _ in iv))
    bylm = defaultdict(list)
    for wid, (m, s0, e, st) in ends.items():
        bylm[m].append((s0, st, e))
    back, nb = [], 0
    for m, L in bylm.items():
        L.sort()
        for (s0, st, e), (s1, st1, e1) in zip(L, L[1:]):
            nb += 1
            back.append(max(0.0, e - st1))
    so, eo, back = np.array(so), np.array(eo), np.array(back)

    def q(a):
        return {"mean": round(float(a.mean()), 3), "p50": round(float(np.median(a)), 3),
                "p90": round(float(np.percentile(a, 90)), 3), "n": int(len(a))}
    return {"start_offset_s": q(so), "end_offset_s": q(eo), "speech_to_window_ratio": q(np.array(ratio)),
            "output_overlap_share": round(ov / max(tot, 1e-6), 4), "speech_s": round(tot, 1),
            "backlog_into_next_window_s": q(back), "backlog_share_gt0": round(float((back > 0).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--run", required=True, help="decode dir holding hyp.jsonl and score_text.json")
    ap.add_argument("--speech-dirs", nargs="*", default=[])
    ap.add_argument("--asr-hyps", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from meeting_score import universe
    run = Path(args.run)
    W = [json.loads(l) for l in open(REPO / WIN, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(run / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(REPO / REF_WIN, encoding="utf-8")]
    U = universe(DR, {w["meeting"] for w in W})
    sc = scored_utts(W, RW)
    ref = json.load(open(run / "score_text.json", encoding="utf-8"))
    res = {"name": args.name, "run": str(run)}

    As = assign_all(W, H, sc, U)
    base = corpus(As)
    res["base"] = base
    res["check"] = {r: (base[r]["bleu"], ref[f"tcpBLEU_{r}"]) for r in ("all", "ov", "1spk")}
    if any(abs(a - b) > 1e-6 for a, b in res["check"].values()):
        raise SystemExit(f"re-implementation does not reproduce the scorer: {res['check']}")
    res["breakdown"] = breakdown(As)
    res["latency_text"] = latency(As)
    short = lambda v: v["dur"] <= 1.0  # noqa: E731
    O_ = {
        "fill_short_iso": dict(fill=lambda r, v: short(v) and not v["uov"]),
        "fill_short_ov": dict(fill=lambda r, v: short(v) and v["uov"]),
        "fill_short_all": dict(fill=lambda r, v: short(v)),
        "fill_long": dict(fill=lambda r, v: not short(v)),
        "fill_all": dict(fill=lambda r, v: True),
        "drop_ins_ov": dict(drop_ins=lambda s: s["reg"] == "ov"),
        "drop_ins_1spk": dict(drop_ins=lambda s: s["reg"] == "1spk"),
        "drop_ins_all": dict(drop_ins=lambda s: True),
        "drop_dup": dict(drop_dup=True),
        "trans_oracle": dict(trans=True),
        "under_trans_lt0.5": dict(under=0.5),
        "fill_all_drop_ins": dict(fill=lambda r, v: True, drop_ins=lambda s: True),
    }
    res["oracle"] = {k: corpus(As, **kw) for k, kw in O_.items()}
    for mode in ("spk_oracle", "spk_oracle_short"):
        Am = assign_all(W, H, sc, U, mode=mode)
        res["oracle"][mode] = corpus(Am)
        res["oracle"][mode + "_recall_short"] = {
            "le1_iso": round(np.mean([A["rec"][r] for A in Am.values() for r, v in A["sc"].items()
                                      if v["dur"] <= 1 and not v["uov"]]), 4),
            "le1_ov": round(np.mean([A["rec"][r] for A in Am.values() for r, v in A["sc"].items()
                                     if v["dur"] <= 1 and v["uov"]]), 4)}
    res["oracle"]["spk_oracle_fill_all_drop_ins"] = corpus(assign_all(W, H, sc, U, mode="spk_oracle"),
                                                           fill=lambda r, v: True, drop_ins=lambda s: True)
    res["recall_short_base"] = {
        "le1_iso": round(np.mean([A["rec"][r] for A in As.values() for r, v in A["sc"].items() if v["dur"] <= 1 and not v["uov"]]), 4),
        "le1_ov": round(np.mean([A["rec"][r] for A in As.values() for r, v in A["sc"].items() if v["dur"] <= 1 and v["uov"]]), 4)}
    # speech-scorer granularity: the text hyp merged per window x speaker (no TTS, no ASR)
    Hm = merged_text_hyp(H)
    res["text_merged_per_stream"] = corpus(assign_all(W, Hm, sc, U))
    res["speech"] = []
    for ah in args.asr_hyps:
        Ha = {h["id"]: h for h in map(json.loads, open(ah, encoding="utf-8"))}
        Aa = assign_all(W, Ha, sc, U)
        sa = Path(ah).parent / "score_asr.json"
        chk = json.load(open(sa, encoding="utf-8")) if sa.exists() else {}
        res["speech"].append({"asr_hyp": ah, "check_scorer": {r: chk.get(f"tcpBLEU_{r}") for r in ("all", "ov", "1spk")},
                              "base": corpus(Aa), "spk_oracle": corpus(assign_all(W, Ha, sc, U, "spk_oracle")),
                              "breakdown_missed": {k: v["missed_word_share"] for k, v in breakdown(Aa)["bands"].items()
                                                   if k.endswith("_any")}})
    res["speech_timing"] = [dict(dir=d, **speech_stats(d, W)) for d in args.speech_dirs]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"name": args.name, "check": res["check"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
