"""v11 P1-4 (reports/mist_w3po_v1/LOG.zh-CN.md 2026-11-15 09:35 Prague, definitions fixed before computing):
extended overlap-aware scoring of EXISTING outputs, on top of reports/gstar_si_v8/overlap_score.py (same windows,
references, slot matching, collar, splitting and segment assignment; the default call reproduces its tcpBLEU exactly).

  dedup     inside each reference unit, pieces are visited by time overlap with the unit (largest first) and a piece is
            dropped when its chrF to an already kept piece (max of both directions) is >= 50, the scorer's duplicate
            threshold; kept pieces are joined in time order
  blind     speaker-agnostic tcpBLEU (the BLEU analogue of meeteval's tcORC: speaker labels ignored): every output
            segment and every reference utterance get one common speaker, units are the reference utterances of all
            speakers less than 1 s apart, then the scorer's assignment and splitting; a unit is "ov" when any of its
            utterances is overlapped (true speakers). Speaker-attributed minus speaker-agnostic = cost of attribution.
            (A first variant that sent each segment to the unit it overlaps most in time scored below the attributed
            score on AMI test, so time overlap is not an attribution oracle; it was dropped before any table.)
  short     content recall of short utterances (<= 1.5 s), overlapped (>= 20% by others) or not: the text assigned to
            the utterance's unit from segments within 1 s of it (CR's rule); token recall against the utterance's
            reference (English: lower-case words; Chinese: characters; punctuation dropped); recalled when >= 0.5.
            Secondary: reference <= 3 words (English) / <= 5 characters (Chinese), overlapped.
  units     with --units: every scored unit as {meeting, unit, reg, src, ref, hyp, hyp_dedup} for COMET

    python -m experiments.mist_w3po_v1.evaluation.v11_metrics run --out <dir>      (all systems of SYSTEMS, CPU pool)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))

import overlap_score as O  # noqa: E402
from meeting_score import universe  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402

SHORT_S = 1.5
CONTENT_MIN = 0.5
DEC = "eval_outputs/mist_w3po_v1/decode"
RT_B = "joint_v8h6k_20261113T073658Z_s06000"
RT_W = "joint_v2p_20261113T120415Z_s03500_wbzm1em1"
DD = "eval_outputs/mist_w3po_v1/data"
WIN = {"devz": f"{DD}/windows_zh2en_c2g06_v1/windows_dev_online.jsonl",
       "deve": f"{DD}/windows_en2zh_c2g06_v1/windows_dev_online.jsonl",
       "ami": f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
       "ali": f"{DD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
       "en": f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl"}
REF = {"devz": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl",
       "deve": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl",
       "ami": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "ali": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl",
       "en": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl"}
DR = {"devz": "zh2en", "deve": "en2zh", "ami": "zh2en", "ali": "ali", "en": "en2zh"}
TXT = {"devz": "jdev_bz", "deve": "jdev_be", "ami": "jtest_bami", "ali": "jtest_bali", "en": "jtest_ben"}
OFF = "eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/scores_ov2"


def systems() -> list[dict]:
    """(key, set, dr, hyp, windows, ref windows, online, relabel, reference score file) of every row analysed."""
    out = []
    for s in ("devz", "deve", "ami", "ali", "en"):
        b = f"{TXT[s]}_{RT_B}"
        m = f"{TXT[s].replace('_b', '_m')}_{RT_W}"
        rows = [("b", b, "hyp.jsonl", "score_text.json"), ("m", m, "hyp.jsonl", "score_text.json")]
        for sysk, base, sp in (("b", b, "__t1c"), ("e2", b, "__rlplem"), ("m", m, "__rlplem")):
            for seed in ("", "_tts2"):
                rows.append((f"{sysk}:asr{seed or '_tts1'}", base + sp + seed, "hyp_asr.jsonl", "score_asr.json"))
        for k, run, hf, sf in rows:
            kind = "asr" if "asr" in k else "text"
            out.append({"key": f"{s}:{k if kind == 'asr' else k + ':text'}", "set": s, "dr": DR[s],
                        "hyp": f"{DEC}/{run}/{hf}", "windows": WIN[s], "ref": REF[s], "online": True,
                        "relabel": False, "check": f"{DEC}/{run}/{sf}"})
    for s, dr in (("en2zh", "en2zh"), ("zh2en", "zh2en"), ("ali", "ali")):
        for k in ("C-ft2", "O2-nosd-relabel", "O2"):
            f = REPO / OFF / f"{s}_{k}.json"
            if not f.exists():
                continue
            r = json.load(open(f))
            out.append({"key": f"off_{s}:{k}", "set": f"off_{s}", "dr": dr, "hyp": r["hyp"], "windows": r["windows"],
                        "ref": {"en2zh": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl",
                                "zh2en": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
                                "ali": "data/alimeeting_si_v1/windows_test_ref.jsonl"}[s],
                        "online": False, "relabel": r["relabel"] in (True, "True"), "check": str(f)})
    return out


def toks(t: str, eng: bool) -> list[str]:
    return re.findall(r"[a-z0-9']+", t.lower()) if eng else [c for c in t if re.match(r"[一-鿿0-9a-zA-Z]", c)]


def tok_recall(hyp: str, ref: str, eng: bool) -> float | None:
    r = toks(ref, eng)
    if not r:
        return None
    from collections import Counter
    h, c = Counter(toks(hyp, eng)), Counter(r)
    return sum(min(n, h[t]) for t, n in c.items()) / len(r)


def score_ext(dr, W, H, ref_windows, U, online=True, relabel=False, blind=False, dedup=False, collect=False,
              split=True, collar=2.0, merge=1.0, post_key="lowlat"):
    """overlap_score.score restricted to tcpBLEU all/ov/1spk (+ per-meeting BLEU statistics), with the options above."""
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
                src = (u["zh"] if dr != "en2zh" else u.get("en", "")).strip()
                scored[u["record_id"]] = {"meeting": w["meeting"], "spk": u["speaker_id"], "a": w["start_s"] + u["begin_s"],
                                          "b": w["start_s"] + u["end_s"], "ref": u.get("ref", u["zh"]).strip(),
                                          "src": src.lower() if dr == "en2zh" else src}
    segs = defaultdict(list)
    for w in W:
        h = H[w["id"]]
        post = None
        if relabel:
            pp = (w.get("post") or {}).get(post_key)
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
                k = O.relabel_causal(seg, post, w["start_s"], c)
                seg["spk"] = k if k else "unk"
            if blind:
                seg["spk"] = "X"
            segs[w["meeting"]].append(seg)
    if blind:
        for v in scored.values():
            v["tspk"], v["spk"] = v["spk"], "X"
    R = defaultdict(list)
    PM, UNITS = {}, []
    cnt = defaultdict(int)
    short = defaultdict(lambda: [0, 0, 0.0])   # band -> [recalled with content, n, sum token recall]
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
                    units_[cur]["src"] += (" " if dr == "en2zh" else "") + v["src"]
                    units_[cur]["utts"].append(rid)
                else:
                    cur = rid
                    units_[cur] = {"spk": spk, "a": v["a"], "b": v["b"], "ref": v["ref"], "src": v["src"], "utts": [rid]}
                unit_of[rid] = cur
        def is_ov(a_, b_, spk_):
            return O.others_overlap(a_, b_, spk_, us) / max(b_ - a_, 1e-6) >= O.OV_SHARE
        for rid, v in units_.items():
            if blind:
                v["reg"] = "ov" if any(is_ov(sc[r]["a"], sc[r]["b"], sc[r]["tspk"]) for r in v["utts"]) else "1spk"
            else:
                v["reg"] = "ov" if is_ov(v["a"], v["b"], v["spk"]) else "1spk"
        got = defaultdict(list)
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
                    pc = O.pieces(s["text"], eng)
                    tot = sum(o for _, o in cand)
                    i0 = 0
                    for j, (r, o) in enumerate(cand):
                        i1 = len(pc) if j == len(cand) - 1 else i0 + int(round(len(pc) * o / tot))
                        if i1 > i0:
                            got[r].append((s, sep.join(pc[i0:i1])))
                        i0 = i1
                    continue
            got[target].append((s, s["text"]))
        pm = defaultdict(list)
        for rid, v in sorted(units_.items(), key=lambda kv: kv[1]["a"]):
            ss = sorted(got.get(rid, []), key=lambda x: x[0]["a"])
            hyp = sep.join(t for _, t in ss).strip()
            hyp_d = hyp
            if len(ss) >= 2:
                order = sorted(range(len(ss)), key=lambda i: (-O.overlap(ss[i][0]["a"], ss[i][0]["b"], v["a"], v["b"]),
                                                              ss[i][0]["a"]))
                kept = []
                for i in order:
                    t = ss[i][1]
                    if any(max(O.chrf2(t, ss[k][1]), O.chrf2(ss[k][1], t)) >= 50 for k in kept):
                        cnt["dedup_dropped"] += 1
                        continue
                    kept.append(i)
                hyp_d = sep.join(ss[i][1] for i in sorted(kept)).strip()
            use = hyp_d if dedup else hyp
            pm[v["reg"]].append((use, v["ref"]))
            pm["all"].append((use, v["ref"]))
            cnt["units"] += 1
            if collect:
                UNITS.append({"meeting": m, "unit": rid, "reg": v["reg"], "spk": v["spk"], "src": v["src"], "ref": v["ref"],
                              "hyp": hyp, "hyp_dedup": hyp_d})
        for s in ins:
            reg = "ov" if O.multi_talk(s["a"], s["b"], us) else "1spk"
            pm[reg].append((s["text"], ""))
            pm["all"].append((s["text"], ""))
        pmc = defaultdict(lambda: [0, 0, 0.0])
        for rid, v in sc.items():
            d = v["b"] - v["a"]
            if d > SHORT_S:
                continue
            ovs = is_ov(v["a"], v["b"], v.get("tspk", v["spk"]))
            txt = sep.join(t for x, t in got.get(unit_of[rid], [])
                           if O.overlap(x["a"] - O.PAIR_COLLAR, x["b"] + O.PAIR_COLLAR, v["a"], v["b"]) > 0 and t.strip())
            tr = tok_recall(txt, v["ref"], eng)
            if tr is None:
                continue
            bands = ["le1.5_ov" if ovs else "le1.5_1spk"]
            if ovs and len(toks(v["ref"], eng)) <= (3 if eng else 5):
                bands.append("tiny_ov")
            for bd in bands:
                for D_ in (short[bd], pmc[bd]):
                    D_[0] += tr >= CONTENT_MIN
                    D_[1] += 1
                    D_[2] += tr
        rec_m = {"short": {k: list(v) for k, v in pmc.items()}}
        for reg in ("all", "ov", "1spk"):
            R[reg] += pm[reg]
            st = bleu._extract_corpus_statistics([x for x, _ in pm[reg]], [[y for _, y in pm[reg]]]) if pm[reg] else None
            rec_m[f"{reg}_stats"] = np.sum(np.array(st, dtype=np.float64), 0).tolist() if st else None
        PM[m] = rec_m
    res = {"dir": dr, "blind": blind, "dedup": dedup, "relabel": relabel, "online": online}
    for reg in ("all", "ov", "1spk"):
        res[f"tcpBLEU_{reg}"] = round(bleu.corpus_score([x for x, _ in R[reg]], [[y for _, y in R[reg]]]).score, 2) if R[reg] else None
    res["dedup_dropped_pieces"] = cnt["dedup_dropped"]
    res["units"] = cnt["units"]
    res["short_content"] = {k: {"recall": round(v[0] / max(v[1], 1), 4), "mean_tok_recall": round(v[2] / max(v[1], 1), 4),
                                "n": v[1]} for k, v in sorted(short.items())}
    res["per_meeting"] = PM
    if collect:
        res["unit_rows"] = UNITS
    return res


def one(job: dict, out: str) -> str:
    od = Path(out)
    f = od / f"{job['key'].replace(':', '__')}.json"
    if f.exists():
        return f"{job['key']}: exists, kept"
    if not Path(job["hyp"]).exists():
        return f"{job['key']}: MISSING {job['hyp']}"
    H = {h["id"]: h for h in map(json.loads, open(job["hyp"], encoding="utf-8"))}
    W = [w for w in map(json.loads, open(job["windows"], encoding="utf-8")) if w["id"] in H]
    RW = list(map(json.loads, open(job["ref"], encoding="utf-8")))
    U = universe(job["dr"], {w["meeting"] for w in W})
    kw = dict(online=job["online"], relabel=job["relabel"])
    res = {"job": job}
    std = score_ext(job["dr"], W, H, RW, U, collect=True, **kw)
    res["std"] = std
    res["dedup"] = score_ext(job["dr"], W, H, RW, U, dedup=True, **kw)
    res["blind"] = score_ext(job["dr"], W, H, RW, U, blind=True, **kw)
    ref = json.load(open(job["check"])) if Path(job["check"]).exists() else {}
    res["check"] = {k: (ref.get(k), std[k]) for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk")}
    res["check_ok"] = all(a is None or abs(float(a) - float(b)) < 0.006 for a, b in res["check"].values())
    od.mkdir(parents=True, exist_ok=True)
    with open(f, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False)
    return (f"{job['key']}: all {std['tcpBLEU_all']} dedup {res['dedup']['tcpBLEU_all']} blind {res['blind']['tcpBLEU_all']} "
            f"check_ok {res['check_ok']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "list"])
    ap.add_argument("--out", default="eval_outputs/mist_w3po_v1/v11_metrics/scores")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    J = [j for j in systems() if not a.only or re.search(a.only, j["key"])]
    if a.cmd == "list":
        for j in J:
            print(j["key"], j["hyp"], Path(j["hyp"]).exists())
        return
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, J, [a.out] * len(J)):
            print(r, flush=True)


if __name__ == "__main__":
    main()
