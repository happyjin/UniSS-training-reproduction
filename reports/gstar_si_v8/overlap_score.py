"""Paper v8 (prereg_v8 §2-3): overlap-aware whole-meeting scores. Extends reports/gstar_si_v6/meeting_score.py (tcpBLEU:
the same scored utterances, reference units, chrF slot matching and segment assignment), adding

  region      a reference unit is "ov" when >= 20% of its time overlaps annotated speech of another speaker (hesitations
              included), else "1spk"; an insertion is "ov" when, within its span, two or more speakers are talking
  split       (default; --no-split gives v7's rule) a segment overlapping two or more units of its matched speaker by
              >= 20% of its length each is split between them in proportion to the time overlap (characters for
              Chinese, words for English)
  tcpBLEU-ov / tcpBLEU-1spk, omitted / inserted per region
  CR          concurrent-speaker recall: pairs of scored utterances of two speakers overlapping >= 0.5 s; a pair is
              recalled when each utterance has text assigned to its unit from a segment meeting it within 1 s
  collapse    pairs where one side is not recalled but a segment of the other side's matched slot covers it
  duplicate   units holding two segments with chrF >= 50 to each other
  short       recall of scored utterances <= 0.5 s / 0.5-1 s, with or without overlap (>= 20% of the utterance)
  lag         text completion lag (commit of a unit's last segment - unit end) per region; pre-end commit share:
              characters / words committed before the end of the unit they are assigned to
--relabel: causal post-hoc relabelling of a speaker-blind run (prereg_v8 §3): each segment gets the slot with the
highest mean diarizer posterior over its source span, using only frames up to commit - 1.04 s; with no frame there,
the most active slot over [a - 2 s, commit - 1.04 s]; else an unknown slot (scored as an unmatched output speaker).
Writes <out> (json, refuses to overwrite) with corpus scores and per-meeting statistics for the permutation tests.

    python reports/gstar_si_v8/overlap_score.py --dir en2zh --windows <decoded windows> --hyp <hyp.jsonl> \
        --ref-windows <windows_test_ref.jsonl> [--online] [--relabel] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402
from meeting_score import universe  # noqa: E402

OV_SHARE = 0.2
PAIR_MIN = 0.5
PAIR_COLLAR = 1.0
LOOKAHEAD = 1.04
FRAME = 0.08


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def others_overlap(a, b, spk, uts):
    """time of [a, b] during which a speaker other than spk talks (union of their annotated utterances)."""
    iv = sorted((max(a, u[2]), min(b, u[3])) for u in uts if u[1] != spk and u[3] > a and u[2] < b)
    tot, cur0, cur1 = 0.0, None, None
    for x0, x1 in iv:
        if cur1 is None or x0 > cur1:
            if cur1 is not None:
                tot += cur1 - cur0
            cur0, cur1 = x0, x1
        else:
            cur1 = max(cur1, x1)
    if cur1 is not None:
        tot += cur1 - cur0
    return tot


def multi_talk(a, b, uts):
    """whether two or more speakers talk at some moment of [a, b]."""
    ev = []
    for u in uts:
        x0, x1 = max(a, u[2]), min(b, u[3])
        if x1 > x0:
            ev += [(x0, 1, u[1]), (x1, -1, u[1])]
    act = defaultdict(int)
    for _, d, s in sorted(ev, key=lambda e: (e[0], e[1])):
        act[s] += d
        if sum(1 for v in act.values() if v > 0) >= 2:
            return True
    return False


def pieces(text, eng):
    return text.split() if eng else list(text)


def relabel_causal(s, post, win_start, commit):
    """slot (1-based) of a speaker-blind segment from the causal posterior of its window (prereg_v8 §3)."""
    if post is None or commit is None:
        return 0
    lim = int((commit - LOOKAHEAD - win_start) / FRAME)   # rows available at the commit
    lim = min(lim, len(post))
    if lim <= 0:
        return 0
    r0, r1 = int((s["a"] - win_start) / FRAME), int(np.ceil((s["b"] - win_start) / FRAME))
    r1 = min(r1, lim)
    if r1 > r0:
        m = post[max(r0, 0):r1].astype(np.float32).mean(0)
        if m.max() > 0:
            return int(m.argmax()) + 1
    q0 = max(0, int((s["a"] - 2.0 - win_start) / FRAME))
    if lim > q0:
        m = post[q0:lim].astype(np.float32).mean(0)
        if m.max() > 0:
            return int(m.argmax()) + 1
    return 0


def chrf2(x, y):
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(x, [y]).score


def score(dr, W, H, ref_windows, U, online=True, relabel=False, split=True, collar=2.0, merge=1.0):
    from sacrebleu.metrics import BLEU
    eng = dr != "en2zh"
    sep = " " if eng else ""
    bleu = BLEU(tokenize="13a" if eng else "zh")
    meetings = {w["meeting"] for w in W}
    scored = {}
    for w in ref_windows:
        if w["meeting"] not in meetings:
            continue
        for u in w["utts"]:
            if not u["hesitation"] and u.get("ref", u["zh"]).strip():
                scored[u["record_id"]] = {"meeting": w["meeting"], "spk": u["speaker_id"], "a": w["start_s"] + u["begin_s"],
                                          "b": w["start_s"] + u["end_s"], "ref": u.get("ref", u["zh"]).strip()}
    segs = defaultdict(list)
    for w in W:
        h = H[w["id"]]
        post = None
        if relabel:
            pp = (w.get("post") or {}).get("lowlat")
            post = np.load(pp) if pp and Path(pp).exists() else None
        for s in h["segments"]:
            c = None
            if "write_ms" in s:
                c = s["write_ms"]
                if online:
                    c = max(c, w["open_detect_ms"])
                    if s["write_ms"] >= w["dur_ms"]:
                        c = max(c, w["close_detect_ms"])
                c = w["start_s"] + c / 1000
            seg = {"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                   "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08), "text": s["text"].strip(), "commit": c}
            if relabel:
                k = relabel_causal(seg, post, w["start_s"], c)
                seg["spk"] = k if k else "unk"
            segs[w["meeting"]].append(seg)
    R = {"ov": [], "1spk": [], "all": []}
    PM = {}
    cnt = defaultdict(int)
    lag = {"ov": [], "1spk": []}
    pre_end = [0, 0]
    short = defaultdict(lambda: [0, 0])
    for m in sorted(meetings):
        us = U[m]
        sc = {rid: v for rid, v in scored.items() if v["meeting"] == m}
        hyp_t, ref_t = defaultdict(str), defaultdict(str)
        for s in segs[m]:
            hyp_t[s["spk"]] += sep + s["text"]
        for v in sorted(sc.values(), key=lambda v: v["a"]):
            ref_t[v["spk"]] += sep + v["ref"]
        mp = match({k: v for k, v in hyp_t.items() if k != "unk"}, dict(ref_t))
        unit_of, units_ = {}, {}
        for spk in {v["spk"] for v in sc.values()}:
            cur = None
            for rid, v in sorted(((r, v) for r, v in sc.items() if v["spk"] == spk), key=lambda kv: kv[1]["a"]):
                if cur is not None and v["a"] - units_[cur]["b"] < merge:
                    units_[cur]["b"] = max(units_[cur]["b"], v["b"])
                    units_[cur]["ref"] += sep + v["ref"]
                    units_[cur]["utts"].append(rid)
                else:
                    cur = rid
                    units_[cur] = {"spk": spk, "a": v["a"], "b": v["b"], "ref": v["ref"], "utts": [rid]}
                unit_of[rid] = cur
        for rid, v in units_.items():
            d = max(v["b"] - v["a"], 1e-6)
            v["reg"] = "ov" if others_overlap(v["a"], v["b"], v["spk"], us) / d >= OV_SHARE else "1spk"
        got = defaultdict(list)   # unit -> [(segment, text piece)]
        ins = []
        for s in sorted(segs[m], key=lambda s: s["a"]):
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
                cnt["dropped_dont_care"] += 1
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
                cand = [(rid, overlap(s["a"], s["b"], v["a"], v["b"])) for rid, v in units_.items() if v["spk"] == spk]
                cand = sorted([(r, o) for r, o in cand if o / L >= OV_SHARE], key=lambda ro: units_[ro[0]]["a"])
                if len(cand) >= 2:
                    pc = pieces(s["text"], eng)
                    tot = sum(o for _, o in cand)
                    i0 = 0
                    for j, (r, o) in enumerate(cand):
                        i1 = len(pc) if j == len(cand) - 1 else i0 + int(round(len(pc) * o / tot))
                        if i1 > i0:
                            got[r].append((s, sep.join(pc[i0:i1])))
                        i0 = i1
                    cnt["split_segments"] += 1
                    continue
            got[target].append((s, s["text"]))
        pm = {"ov": [], "1spk": [], "all": []}
        for rid, v in sorted(units_.items(), key=lambda kv: kv[1]["a"]):
            ss = sorted(got.get(rid, []), key=lambda x: x[0]["a"])
            hyp = sep.join(t for _, t in ss).strip()
            pm[v["reg"]].append((hyp, v["ref"]))
            pm["all"].append((hyp, v["ref"]))
            cnt[f"units_{v['reg']}"] += 1
            cnt[f"omitted_{v['reg']}"] += not ss
            cs = [x[0]["commit"] for x in ss if x[0]["commit"] is not None]
            if cs:
                lag[v["reg"]].append(max(cs) - v["b"])
            for x, t in ss:
                n = len(pieces(t, eng))
                pre_end[1] += n
                if x["commit"] is not None and x["commit"] < v["b"]:
                    pre_end[0] += n
            if len(ss) >= 2:
                cnt["units_multi_seg"] += 1
                if any(chrf2(ss[i][1], ss[j][1]) >= 50 for i in range(len(ss)) for j in range(i + 1, len(ss))):
                    cnt["duplicate_units"] += 1
        for s in ins:
            reg = "ov" if multi_talk(s["a"], s["b"], us) else "1spk"
            pm[reg].append((s["text"], ""))
            pm["all"].append((s["text"], ""))
            cnt[f"inserted_{reg}"] += 1

        # recall of scored utterances: text assigned to its unit from a segment within PAIR_COLLAR of the utterance
        def recalled(rid):
            v = sc[rid]
            return any(overlap(x["a"] - PAIR_COLLAR, x["b"] + PAIR_COLLAR, v["a"], v["b"]) > 0 and t.strip()
                       for x, t in got.get(unit_of[rid], []))
        rec = {rid: recalled(rid) for rid in sc}
        inv = {}
        for k, v in mp.items():
            inv[v] = k
        rids = sorted(sc, key=lambda r: sc[r]["a"])
        pairs = rec_pairs = coll = 0
        for i, r1 in enumerate(rids):
            for r2 in rids[i + 1:]:
                if sc[r2]["a"] >= sc[r1]["b"]:
                    break
                if sc[r1]["spk"] == sc[r2]["spk"]:
                    continue
                if overlap(sc[r1]["a"], sc[r1]["b"], sc[r2]["a"], sc[r2]["b"]) < PAIR_MIN:
                    continue
                pairs += 1
                if rec[r1] and rec[r2]:
                    rec_pairs += 1
                    continue
                for miss, oth in ((r1, r2), (r2, r1)):
                    if not rec[miss]:
                        k = inv.get(sc[oth]["spk"])
                        if any(x["spk"] == k and overlap(x["a"], x["b"], sc[miss]["a"], sc[miss]["b"]) > 0 for x in segs[m]):
                            coll += 1
                            break
        cnt["pairs"] += pairs
        cnt["pairs_recalled"] += rec_pairs
        cnt["pairs_collapsed"] += coll
        for rid, v in sc.items():
            d = v["b"] - v["a"]
            if d > 1.0:
                continue
            band = "le0.5" if d <= 0.5 else "0.5-1"
            o = "ov" if others_overlap(v["a"], v["b"], v["spk"], us) / max(d, 1e-6) >= OV_SHARE else "1spk"
            short[f"{band}_{o}"][0] += rec[rid]
            short[f"{band}_{o}"][1] += 1
        rec_m = {}
        for reg in ("ov", "1spk", "all"):
            R[reg] += pm[reg]
            st = bleu._extract_corpus_statistics([x for x, _ in pm[reg]], [[y for _, y in pm[reg]]]) if pm[reg] else None
            rec_m[f"{reg}_stats"] = np.sum(np.array(st, dtype=np.float64), 0).tolist() if st else None
        rec_m["pairs"], rec_m["pairs_recalled"] = pairs, rec_pairs
        mlag = {r: [x for x in lag[r]] for r in lag}
        PM[m] = rec_m
        PM[m]["lag_units"] = {r: len(mlag[r]) for r in mlag}
    res = {"dir": dr, "relabel": relabel, "split": split, "collar_s": collar, "meetings": len(meetings)}
    for reg in ("ov", "1spk", "all"):
        res[f"tcpBLEU_{reg}"] = round(bleu.corpus_score([x for x, _ in R[reg]], [[y for _, y in R[reg]]]).score, 2) if R[reg] else None
    for reg in ("ov", "1spk"):
        res[f"units_{reg}"] = cnt[f"units_{reg}"]
        res[f"omitted_{reg}"] = round(cnt[f"omitted_{reg}"] / max(cnt[f"units_{reg}"], 1), 4)
        res[f"inserted_{reg}"] = cnt[f"inserted_{reg}"]
        a = np.array(lag[reg]) if lag[reg] else None
        res[f"lag_{reg}"] = {"p50": round(float(np.median(a)), 3), "mean": round(float(a.mean()), 3),
                             "p90": round(float(np.percentile(a, 90)), 3), "n": int(len(a))} if a is not None else None
    res["CR"] = round(cnt["pairs_recalled"] / max(cnt["pairs"], 1), 4)
    res["pairs"] = cnt["pairs"]
    res["collapse"] = round(cnt["pairs_collapsed"] / max(cnt["pairs"], 1), 4)
    res["duplicate_units"] = cnt["duplicate_units"]
    res["duplicate_rate"] = round(cnt["duplicate_units"] / max(cnt["units_ov"] + cnt["units_1spk"], 1), 4)
    res["short_recall"] = {k: {"recall": round(v[0] / max(v[1], 1), 4), "n": v[1]} for k, v in sorted(short.items())}
    res["pre_end_share"] = round(pre_end[0] / max(pre_end[1], 1), 4)
    res["split_segments"] = cnt["split_segments"]
    res["dropped_dont_care_segments"] = cnt["dropped_dont_care"]
    res["per_meeting"] = PM
    res["lag_values"] = {r: [round(x, 3) for x in lag[r]] for r in lag}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, choices=["en2zh", "zh2en", "ali"])
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref-windows", required=True)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--collar", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    H = {h["id"]: h for h in map(json.loads, open(args.hyp, encoding="utf-8"))}
    W = [w for w in map(json.loads, open(args.windows, encoding="utf-8")) if w["id"] in H]
    RW = list(map(json.loads, open(args.ref_windows, encoding="utf-8")))
    U = universe(args.dir, {w["meeting"] for w in W})
    res = score(args.dir, W, H, RW, U, online=args.online, relabel=args.relabel, split=not args.no_split,
                collar=args.collar)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_meeting", "lag_values")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
