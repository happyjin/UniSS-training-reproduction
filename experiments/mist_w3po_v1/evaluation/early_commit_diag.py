"""Why do early-writing Thinkers lose text tcpBLEU? CPU diagnosis of dev5 decodes against A' on the same windows
(reports/mist_w3po_v1/t45_quality_diag_v1.zh-CN.md; also the extra report of thinker_rc5_prereg_v1 §4).

Uses metric_headroom's re-implementation of overlap_score.score() (online, split, collar 2 s, merge 1 s), which
reproduces the scorer's tcpBLEU all / ov / 1spk exactly (checked here against each run's score_text.json).

Per candidate run (vs the reference run, A' by default):
  components     BP, sys / ref length, p1-p4 (all / ov / 1spk); unmatched tokens (clipped per unit, + insertions)
  swaps          tcpBLEU change when the candidate's windows of a group are replaced by the reference's
                 (groups: window duration < 2, 2-4, 4-6, 6-10, >= 10 s; candidate output with 1 / >= 2 speakers)
  joins          2/3-gram precision inside a written segment vs across two segments of one unit (after a sentence end,
                 same read, later read), words per segment, non-final segments ending on a function word
  early          units by the share of their words written before the window's final read: candidate vs reference
                 BLEU on the same units
  edits          inference-only edits re-scored on the candidate and the reference: exact boundary repetitions of >= 2
                 words dropped, a text-only loop guard, trailing commas dropped, same-read segments merged
  oracle         every non-recalled unit filled with its reference (metric_headroom fill_all)

    python -m experiments.mist_w3po_v1.evaluation.early_commit_diag --run <decode dir> [--run ...] \
        [--ref-run eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule] [--win <dev5 windows>] [--light] --out <new json>
--light: components, swaps (>= 10 s and >= 2 speakers only) and joins (the RC5 extra report).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.evaluation import metric_headroom as MH  # noqa: E402

WIN5 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
REF_RUN = "eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule"
FW = set(("the a an of to at in on for with and or but that this these those is are was were be been my your our their "
          "his her its i you we they he she it so if as by from than then which who what when where how not do does did "
          "have has had will would can could should just very").split())
SENT_END = tuple(".?!")


def tok13a():
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
    return Tokenizer13a()


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


class Ctx:
    def __init__(self, win: str):
        from meeting_score import universe
        self.W = [json.loads(l) for l in open(win, encoding="utf-8")]
        self.WD = {w["id"]: w for w in self.W}
        self.RW = [json.loads(l) for l in open(MH.REF_WIN, encoding="utf-8")]
        self.U = universe("zh2en", {w["meeting"] for w in self.W})
        self.sc = MH.scored_utts(self.W, self.RW)
        self.bym = collections.defaultdict(list)
        for w in self.W:
            self.bym[w["meeting"]].append(w)

    def assign(self, H: dict) -> dict:
        segs = MH.build_segs(self.W, H)
        return {m: MH.assign(m, segs[m], self.sc, self.U[m]) for m in sorted(segs)}

    def pairs(self, As: dict) -> dict:
        P = {"ov": [], "1spk": [], "all": []}
        for A in As.values():
            for rid, v in A["units"].items():
                ss = sorted(A["got"].get(rid, []), key=lambda x: x[0]["a"])
                hyp = " ".join(t for _, t in ss).strip()
                P[v["reg"]].append((hyp, v["ref"]))
                P["all"].append((hyp, v["ref"]))
            for s in A["ins"]:
                P[s["reg"]].append((s["text"], ""))
                P["all"].append((s["text"], ""))
        return P

    def bleu(self, H: dict) -> dict:
        return {r: MH.bleu_from(MH.bleu_stats(p))["bleu"] for r, p in self.pairs(self.assign(H)).items()}


def load_hyp(run: str) -> dict:
    return {r["id"]: r for r in map(json.loads, open(Path(run) / "hyp.jsonl", encoding="utf-8"))}


def edit(H: dict, f) -> dict:
    return {i: dict(h, segments=f(i, [dict(s) for s in h["segments"]])) for i, h in H.items()}


def components(cx: Ctx, H: dict, tok) -> dict:
    As = cx.assign(H)
    P = cx.pairs(As)
    out = {r: MH.bleu_from(MH.bleu_stats(p)) for r, p in P.items()}
    um = 0
    for hyp, ref in P["all"]:
        hc, rc = collections.Counter(tok(hyp).split()), collections.Counter(tok(ref).split())
        um += sum(max(0, n - rc.get(t, 0)) for t, n in hc.items())
    out["unmatched_tokens"] = um
    segs = [s for h in H.values() for s in h["segments"] if s["text"].strip()]
    out["segments"] = len(segs)
    out["words_per_segment"] = round(float(np.mean([len(s["text"].split()) for s in segs])), 3)
    return out


def swap_gain(cx: Ctx, H: dict, R: dict, ids: list, base: dict) -> dict:
    X = dict(H)
    for i in ids:
        X[i] = R[i]
    b = cx.bleu(X)
    return {r: round(b[r] - base[r], 2) for r in ("all", "ov", "1spk")} | {"n_windows": len(ids)}


def nspk(h: dict) -> int:
    return len({s["spk"] for s in h["segments"] if s["text"].strip()})


def swaps(cx: Ctx, H: dict, R: dict, light: bool) -> dict:
    base = cx.bleu(H)
    ids = [w["id"] for w in cx.W]
    g = {}
    if not light:
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10)):
            g[f"dur_{lo}-{hi}s"] = [i for i in ids if lo <= cx.WD[i]["dur_ms"] / 1000 < hi]
        g["out_1spk"] = [i for i in ids if nspk(H[i]) <= 1]
    g["dur_ge10s"] = [i for i in ids if cx.WD[i]["dur_ms"] >= 10000]
    g["out_ge2spk"] = [i for i in ids if nspk(H[i]) >= 2]
    return {"base": base, "swap_in_reference": {k: swap_gain(cx, H, R, v, base) for k, v in g.items()}}


def raw_writes(cx: Ctx, H: dict) -> dict:
    out = {}
    for i, h in H.items():
        w = cx.WD[i]
        for j, s in enumerate(h["segments"]):
            out[(i, round(w["start_s"] + s["start_s"], 4), s["text"].strip())] = (s["write_ms"], j)
    return out


def joins(cx: Ctx, H: dict, tok) -> dict:
    As = cx.assign(H)
    RAW = raw_writes(cx, H)
    st = collections.defaultdict(lambda: [0, 0])
    for A in As.values():
        for rid, lst in A["got"].items():
            ss = sorted(lst, key=lambda x: x[0]["a"])
            toks, pid, info = [], [], {}
            for j, (x, t) in enumerate(ss):
                tt = tok(t).split()
                if not tt:
                    continue
                toks += tt
                pid += [j] * len(tt)
                info[j] = (x["win"], RAW.get((x["win"], round(x["a"], 4), x["text"]), (None, None))[0],
                           t.strip().endswith(SENT_END))
            ref = tok(A["units"][rid]["ref"]).split()
            for n in (2, 3):
                rc = collections.Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
                for i in range(len(toks) - n + 1):
                    a, b = pid[i], pid[i + n - 1]
                    if a == b:
                        c = "within"
                    elif info[a][2]:
                        c = "cross_after_sentence_end"
                    elif info[a][0] != info[b][0] or info[a][1] is None:
                        c = "cross_other"
                    elif info[a][1] == info[b][1]:
                        c = "cross_same_read"
                    else:
                        c = "cross_later_read"
                    g = tuple(toks[i:i + n])
                    st[c][1] += 1
                    if rc.get(g, 0) > 0:
                        rc[g] -= 1
                        st[c][0] += 1
    nonfinal = fw = 0
    for h in H.values():
        for s in h["segments"]:
            t = s["text"].strip()
            if t and not t.endswith(SENT_END):
                nonfinal += 1
                fw += norm(t.split()[-1]) in FW
    out = {c: {"precision": round(v[0] / max(v[1], 1), 4), "n": v[1]} for c, v in sorted(st.items())}
    cross = [v for c, v in st.items() if c.startswith("cross") and c != "cross_after_sentence_end"]
    out["cross_mid_sentence"] = {"precision": round(sum(v[0] for v in cross) / max(sum(v[1] for v in cross), 1), 4),
                                 "n": sum(v[1] for v in cross)}
    out["nonfinal_segments"] = nonfinal
    out["nonfinal_end_function_word"] = round(fw / max(nonfinal, 1), 4)
    return out


def early_units(cx: Ctx, H: dict, R: dict) -> dict:
    Ac, Ar = cx.assign(H), cx.assign(R)
    RAW = raw_writes(cx, H)

    def rows(As, raw):
        out = {}
        for m, A in As.items():
            for rid, v in A["units"].items():
                ss = sorted(A["got"].get(rid, []), key=lambda x: x[0]["a"])
                n = sum(len(t.split()) for _, t in ss)
                e = 0
                if raw is not None:
                    e = sum(len(t.split()) for x, t in ss
                            if raw.get((x["win"], round(x["a"], 4), x["text"]), (1e12, 0))[0] < cx.WD[x["win"]]["dur_ms"])
                out[(m, rid)] = {"hyp": " ".join(t for _, t in ss).strip(), "ref": v["ref"], "reg": v["reg"], "n": n, "e": e}
        return out
    C, Rr = rows(Ac, RAW), rows(Ar, None)
    G = collections.defaultdict(list)
    for k, r in C.items():
        g = "omitted" if r["n"] == 0 else "all_at_final_read" if r["e"] == 0 else "all_early" if r["e"] == r["n"] else "mixed"
        G[f"{g}|{r['reg']}"].append(k)
    out = {}
    for g, ks in sorted(G.items()):
        bc = MH.bleu_from(MH.bleu_stats([(C[k]["hyp"], C[k]["ref"]) for k in ks]))
        br = MH.bleu_from(MH.bleu_stats([(Rr[k]["hyp"], Rr[k]["ref"]) for k in ks if k in Rr]))
        out[g] = {"units": len(ks), "cand": bc["bleu"], "cand_len": bc["sys_len"], "ref_run": br["bleu"],
                  "ref_run_len": br["sys_len"], "ref_len": bc["ref_len"]}
    return out


def drop_boundary_rep(i, ss, kmin=2):
    by = collections.defaultdict(list)
    for j, s in enumerate(ss):
        by[s["spk"]].append(j)
    out = [dict(s) for s in ss]
    for js in by.values():
        prev = []
        for j in sorted(js, key=lambda j: (ss[j].get("write_ms", 0), j)):
            ws = out[j]["text"].split()
            nw = [norm(x) for x in ws]
            k = next((kk for kk in range(min(len(prev), len(nw)), 0, -1) if prev[-kk:] == nw[:kk] and all(nw[:kk])), 0)
            if k >= kmin:
                out[j]["text"] = " ".join(ws[k:])
            prev += [norm(x) for x in out[j]["text"].split()]
    return out


def text_loop_guard(i, ss, k=3, maxp=6):
    out, hist = [], collections.defaultdict(list)
    for s in ss:
        ws = [norm(x) for x in s["text"].split()]
        seq = hist[s["spk"]] + ws
        loop = any(len(seq) >= k * p and all(seq[len(seq) - (j + 1) * p:len(seq) - j * p] == seq[len(seq) - p:]
                                             for j in range(1, k)) for p in range(1, maxp + 1))
        if loop and ws:
            continue
        hist[s["spk"]] += ws
        out.append(s)
    return out


def drop_trailing_comma(i, ss):
    return [dict(s, text=re.sub(r",\s*$", "", s["text"].strip())) for s in ss]


def merge_same_read(i, ss):
    out = []
    for s in ss:
        if out and out[-1]["spk"] == s["spk"] and out[-1].get("write_ms") == s.get("write_ms"):
            o = out[-1]
            o["text"] = (o["text"].strip() + " " + s["text"].strip()).strip()
            o["end_s"] = max(o["end_s"], s["end_s"])
        else:
            out.append(dict(s))
    return out


EDITS = {"drop_boundary_repetition_ge2": drop_boundary_rep, "text_loop_guard": text_loop_guard,
         "drop_trailing_comma": drop_trailing_comma, "merge_same_read_segments": merge_same_read}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ref-run", default=REF_RUN)
    ap.add_argument("--win", default=WIN5)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cx, tok = Ctx(a.win), tok13a()
    R = load_hyp(a.ref_run)
    res = {"ref_run": a.ref_run, "win": a.win, "light": a.light, "runs": {}}
    for run in [a.ref_run] + a.run:
        H = load_hyp(run)
        sc = json.loads((Path(run) / "score_text.json").read_text())
        r = {"components": components(cx, H, tok), "joins": joins(cx, H, tok)}
        r["check_scorer"] = {g: (r["components"][g]["bleu"], sc[f"tcpBLEU_{g}"]) for g in ("all", "ov", "1spk")}
        assert all(abs(x - y) < 0.006 for x, y in r["check_scorer"].values()), r["check_scorer"]
        if run != a.ref_run:
            r["swaps"] = swaps(cx, H, R, a.light)
            if not a.light:
                r["early_units"] = early_units(cx, H, R)
        if not a.light:
            r["edits"] = {k: cx.bleu(edit(H, f)) for k, f in EDITS.items()}
            As = MH.assign_all(cx.W, H, cx.sc, cx.U)
            r["oracle_fill_all"] = {g: v["bleu"] for g, v in MH.corpus(As, fill=lambda rid, v: True).items()}
        res["runs"][run] = r
        print(run, json.dumps({k: r["components"][k]["bleu"] for k in ("all", "ov", "1spk")}), flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"EARLY_COMMIT_DIAG_DONE {out}")


if __name__ == "__main__":
    main()
