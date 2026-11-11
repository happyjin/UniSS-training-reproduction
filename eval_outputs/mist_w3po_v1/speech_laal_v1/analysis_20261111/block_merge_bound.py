"""Block-merge upper bound (CPU, read-only, descriptive; reports/mist_w3po_v1/block_merge_bound_v1.zh-CN.md).

Replays the speech timeline of an existing dev talker run with the talker blocks of each window x speaker stream
regrouped, using the SAME playback model as speech_latency_lit (online availability = max(write, open_detect), a
final-read block >= close_detect; per-stream FIFO queue start = max(avail, previous end); words spread evenly over
their block's playback; one-channel FIFO variant) and the MEASURED per-block durations (dur_ms) of that talker run.

A regrouped block ("group") of members j = 1..k:
  release   the raw commit time at which the group goes to the talker (window-relative ms; = write_ms of the member
            that closes it, or the window's final read if the rule holds it to the end)
  duration  member j keeps its measured duration d_j; every member after the first loses the per-block overhead c
            (one block onset / offset fewer), floored at FLOOR_S_PER_WORD * words:
                d'_1 = d_1,  d'_j = max(d_j - c, FLOOR * n_j)  (j >= 2),   D = sum_j d'_j
            c = 0 is pure regrouping (no overhead saved); c = the run's OLS intercept of dur ~ words + 1 is the
            "measured overhead" variant (optimistic: it is confounded with the talker's slower short blocks); c = half
            of it in between. Members play back to back inside the group, words evenly inside a member,
            so c = 0 with every group of size 1 is exactly the original timeline (checked: --check reproduces the
            stored speech_latency_lit json).
Rules (per stream, blocks in order):
  none        identity (reproduction check)
  sameread    consecutive blocks with the same write_ms merged (no hold)
  m<k>        read-level hold: blocks of one read merged first, then a read group shorter than k words is held and
              joined with the next read groups until the held text reaches k words, a final-read block arrives, or
              the stream ends (then released at the window's final read, write = dur_ms)
  lit<k>      the literal per-block version of m<k> (no same-read pre-merge)
  oracle      per stream, the contiguous partition of the blocks (release = last member's write, no hold beyond
              that) minimising the stream's sum over words of the time the word has been spoken (exact Pareto DP over
              (queue end, cost)); an upper bound for regrouping under this proxy, evaluated on the real metrics.
Text: the segment's commit becomes its group's release (as if the Thinker held the write); the Thinker-side text
LAAL is the identity row. Speech quality cannot be measured on CPU (no re-synthesis): durations, the talker's
backlog-driven speed and its prosody are kept from the original blocks.

    python block_merge_bound.py --out <dir> [--procs N]      (writes <dir>/results.json, refuses to overwrite)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.evaluation import metric_headroom as M  # noqa: E402
from experiments.mist_w3po_v1.evaluation.speech_latency_lit import laal, q, speech_word_delays  # noqa: E402

FLOOR_S_PER_WORD = 0.15
WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl"
SP = "eval_outputs/mist_w3po_v1/speech"
RUNS = {
    "Aprime_s1": ("eval_outputs/mist_w3po_v1/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06",
                  f"{SP}/dev_crf_v2p_s3000_finalpost_s8_c2g06/talker",
                  "eval_outputs/mist_w3po_v1/talker_crf/eval/aprime/lat_aprime.json"),
    "Aprime_s2": ("eval_outputs/mist_w3po_v1/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06",
                  f"{SP}/dev_crf_v2p_s3000_finalpost_s8_c2g06_tts2/talker",
                  "eval_outputs/mist_w3po_v1/talker_crf/eval/aprime/lat_aprime_tts2.json"),
    "T2_s1": ("eval_outputs/mist_w3po_v1/decode/dev_t2_latent_t0_s3000_w05",
              f"{SP}/dev_t2_latent_t0_s3000_w05/talker",
              "eval_outputs/mist_w3po_v1/crf/t_eval/dev_t2_latent_t0_s3000_w05/lat_dev_t2_latent_t0_s3000_w05.json"),
    "T2_s2": ("eval_outputs/mist_w3po_v1/decode/dev_t2_latent_t0_s3000_w05_tts2",
              f"{SP}/dev_t2_latent_t0_s3000_w05_tts2/talker",
              "eval_outputs/mist_w3po_v1/crf/t_eval/dev_t2_latent_t0_s3000_w05/lat_dev_t2_latent_t0_s3000_w05_tts2.json"),
}
RULES = ["none", "sameread", "m3", "m4", "m5", "m6", "lit3", "lit4", "lit5", "lit6", "oracle"]


def online(w: dict, wm: float) -> float:
    c = max(wm, w["open_detect_ms"])
    if wm >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return c


def nwords(s: str) -> int:
    return len(s.split())


def fit_overhead(rows: dict) -> tuple:
    """OLS dur_s = a * words + c over the non-empty blocks of a talker run -> (a, c)."""
    X, Y = [], []
    for r in rows.values():
        for b in r["blocks"]:
            n = nwords(b["text"])
            if n:
                X.append(n)
                Y.append(b["dur_ms"] / 1000)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    a, c = np.linalg.lstsq(np.vstack([X, np.ones_like(X)]).T, Y, rcond=None)[0]
    return float(a), float(c)


# ---------------------------------------------------------------- grouping rules (pure functions, unit-tested)
def group_hold(writes: list, words: list, dur_ms: int, m: int, read_level: bool) -> list:
    """-> [(members, release_ms)]. writes / words: per block of one stream, in stream order. m = 0: no hold (with
    read_level: same-read merge only). A held group closes when it reaches m words, when a final-read block
    (write >= dur_ms) joins it, or at the stream's end (released at the final read, dur_ms)."""
    units = []   # (members, write) after the optional same-read pre-merge
    for i, wr in enumerate(writes):
        if read_level and units and units[-1][1] == wr:
            units[-1][0].append(i)
        else:
            units.append(([i], wr))
    if m <= 0:
        return [(mem, wr) for mem, wr in units]
    out, cur, cw = [], [], 0
    for mem, wr in units:
        cur = cur + mem
        cw += sum(words[i] for i in mem)
        if cw >= m or wr >= dur_ms:
            out.append((cur, wr))
            cur, cw = [], 0
    if cur:
        out.append((cur, max(dur_ms, writes[cur[-1]])))
    return out


def member_durs(d: list, n: list, c: float) -> list:
    """d'_1 = d_1, d'_j = max(d_j - c, FLOOR * n_j) for j >= 2 (seconds)."""
    return [d[0]] + [max(dj - c, FLOOR_S_PER_WORD * nj) for dj, nj in zip(d[1:], n[1:])]


def group_oracle(writes: list, words: list, durs: list, avail_of, c: float) -> list:
    """Exact Pareto DP over contiguous partitions. A group [i..j] is released at writes[j] (blocks in stream order;
    release = max write of its members), becomes available at avail_of(release) (s), starts at max(avail, queue
    end E) and lasts sum(member_durs). Cost = sum over its words of the time the word has been spoken (member j's
    words evenly over its own playback). Minimises total cost; ties -> fewer merges. -> [(members, release_ms)]."""
    n = len(writes)
    front = {0: [(-1e18, 0.0, 0, None)]}   # (E, cost, merges, back)
    for i in range(n):
        for E, cost, mg, back in front.get(i, []):
            for j in range(i, n):
                mem = list(range(i, j + 1))
                rel = max(writes[k] for k in mem)
                dd = member_durs([durs[k] for k in mem], [words[k] for k in mem], c)
                t = max(avail_of(rel), E)
                cc = cost
                for k, dk in zip(mem, dd):
                    nk = words[k]
                    cc += nk * t + dk * (nk + 1) / 2 if nk else 0.0
                    t += dk
                front.setdefault(j + 1, []).append((t, cc, mg + (j - i), (i, E, cost, mg, back, mem, rel)))
        if i + 1 in front:   # prune to the Pareto set of (E, cost, merges)
            L = sorted(front[i + 1], key=lambda x: (x[0], x[1], x[2]))
            keep, best = [], (float("inf"), float("inf"))
            for e in L:
                if e[1] < best[0] - 1e-9 or (abs(e[1] - best[0]) <= 1e-9 and e[2] < best[1]):
                    keep.append(e)
                    best = (min(best[0], e[1]), e[2] if e[1] <= best[0] + 1e-9 else best[1])
            front[i + 1] = keep
    if n == 0:
        return []
    fin = min(front[n], key=lambda x: (x[1], x[2]))
    out = []
    back = fin[3]
    while back is not None:
        _i, _E, _cost, _mg, prev, mem, rel = back
        out.append((mem, rel))
        back = prev
    return out[::-1]


# ---------------------------------------------------------------- data (loaded once per run, shared by fork)
D: dict = {}


def load_run(name: str) -> None:
    run, talker, _ = RUNS[name]
    W = [json.loads(l) for l in open(REPO / WIN, encoding="utf-8")]
    wins = {w["id"]: w for w in W}
    H = {h["id"]: h for h in map(json.loads, open(REPO / run / "hyp.jsonl", encoding="utf-8"))}
    rows = {}
    for l in open(REPO / talker / "results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r.get("error"):
            rows[r["id"]] = r
    key = (run, WIN)
    if key not in D.setdefault("_assign", {}):
        from meeting_score import universe
        RW = [json.loads(l) for l in open(REPO / M.REF_WIN, encoding="utf-8")]
        U = universe(M.DR, {w["meeting"] for w in W})
        sc = M.scored_utts(W, RW)
        segs = defaultdict(list)
        for w in W:
            h = H.get(w["id"])
            if h is None:
                continue
            idx = defaultdict(int)
            for s in h["segments"]:
                if not s["text"].strip():
                    continue
                k = f"{w['id']}__S{s['spk']}"
                i = idx[s["spk"]]
                idx[s["spk"]] += 1
                segs[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                           "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08),
                                           "text": s["text"].strip(), "key": (k, i), "wid": w["id"],
                                           "seg_write": s["write_ms"]})
        A = {m: M.assign(m, segs[m], sc, U[m]) for m in sorted(segs)}
        D["_assign"][key] = (segs, A)
    D[name] = {"wins": wins, "rows": rows, "segs": D["_assign"][key][0], "A": D["_assign"][key][1],
               "fit": fit_overhead(rows)}


def replay(name: str, rule: str, c: float) -> dict:
    d = D[name]
    wins, rows = d["wins"], d["rows"]
    blk, groups, stats = {}, [], defaultdict(float)
    for k, r in rows.items():
        w = wins[k.split("__S")[0]]
        t0 = w["start_s"]
        bl = r["blocks"]
        if not bl:
            continue
        wr = [b["write_ms"] for b in bl]
        nw = [nwords(b["text"]) for b in bl]
        du = [b["dur_ms"] / 1000 for b in bl]
        if rule == "none":
            G = [([i], wr[i]) for i in range(len(bl))]
        elif rule == "sameread":
            G = group_hold(wr, nw, w["dur_ms"], 0, True)
        elif rule.startswith("m"):
            G = group_hold(wr, nw, w["dur_ms"], int(rule[1:]), True)
        elif rule.startswith("lit"):
            G = group_hold(wr, nw, w["dur_ms"], int(rule[3:]), False)
        elif rule == "oracle":
            G = group_oracle(wr, nw, du, lambda rel, w=w: online(w, rel) / 1000, c)
        else:
            raise ValueError(rule)
        assert sorted(i for mem, _ in G for i in mem) == list(range(len(bl)))
        prev = None
        for mem, rel in G:
            av = online(w, rel) / 1000
            st = av if prev is None else max(av, prev)
            dd = member_durs([du[i] for i in mem], [nw[i] for i in mem], c)
            stats["floor_hits"] += sum(1 for i, x in zip(mem[1:], dd[1:]) if x == FLOOR_S_PER_WORD * nw[i] and du[i] - c < x)
            gid = len(groups)
            groups.append({"meeting": w["meeting"], "stream": k, "avail": t0 + av, "dur": sum(dd), "first": mem[0],
                           "words": sum(nw[i] for i in mem), "members": mem})
            t = st
            for i, x in zip(mem, dd):
                blk[(k, i)] = {"rel_ms": rel, "write": t0 + rel / 1000, "orig_write": t0 + wr[i] / 1000, "avail": t0 + av,
                               "on_start": t0 + t, "on_end": t0 + t + x, "g_start": t0 + st, "g_dur": sum(dd),
                               "final_read": rel >= w["dur_ms"], "text": bl[i]["text"], "gid": gid}
                t += x
            prev = t
            groups[gid]["on_start"], groups[gid]["on_end"] = t0 + st, t0 + t
    # one output channel per meeting (FIFO over groups; a stream's own groups stay in order)
    bym = defaultdict(list)
    for g in groups:
        bym[g["meeting"]].append(g)
    for m, L in bym.items():
        prev = -1e9
        for g in sorted(L, key=lambda g: (g["avail"], g["stream"], g["first"])):
            g["ch_start"] = max(g["avail"], prev)
            prev = g["ch_start"] + g["dur"]
    for (k, i), b in blk.items():
        g = groups[b["gid"]]
        b["ch_start"] = g["ch_start"] + (b["on_start"] - g["on_start"])
        b["ch_end"] = g["ch_start"] + (b["on_end"] - g["on_start"])
    # stream level (online)
    st_start, st_end = [], []
    for k, r in rows.items():
        if not r["blocks"]:
            continue
        w = wins[k.split("__S")[0]]
        B = [blk[(k, i)] for i in range(len(r["blocks"]))]
        st_start.append(B[0]["on_start"] - w["start_s"])
        st_end.append(B[-1]["on_end"] - w["start_s"] - w["dur_ms"] / 1000)
    # unit level
    U_ = defaultdict(lambda: defaultdict(list))
    parts = defaultdict(list)
    miss = 0
    for m, ss in d["segs"].items():
        for x in ss:
            b = blk.get(x["key"])
            if b is None or b["text"].strip() != x["text"]:
                miss += 1
                b = None
            w = wins[x["wid"]]
            # identity: the segment's own write (= speech_latency_lit); merged: its group's release (never earlier)
            rel = x["seg_write"] if (b is None or rule == "none") else max(b["rel_ms"], x["seg_write"])
            x["_blk"] = b
            x["_commit"] = w["start_s"] + online(w, rel) / 1000
            x["_raw"] = w["start_s"] + rel / 1000
            x["_orig_commit"] = w["start_s"] + online(w, x["seg_write"]) / 1000
    for m in sorted(d["A"]):
        A = d["A"][m]
        for uid, u in A["units"].items():
            got = A["got"].get(uid, [])
            ss = [x for x, _ in got]
            if not ss:
                continue
            reg = u["reg"]
            bs = [x["_blk"] for x in ss if x["_blk"] is not None]
            if not bs:
                continue
            U_[reg]["speech_end_online"].append(max(b["on_end"] for b in bs) - u["b"])
            U_[reg]["speech_start_online"].append(min(b["on_start"] for b in bs) - u["a"])
            pw = [({"blk": x["_blk"], "text": x["text"]}, t) for x, t in sorted(got, key=lambda xt: xt[0]["a"])]
            td = [x["_commit"] - u["a"] for x, t in sorted(got, key=lambda xt: xt[0]["a"]) for _ in t.split()]
            tdo = [x["_orig_commit"] - u["a"] for x, t in sorted(got, key=lambda xt: xt[0]["a"]) for _ in t.split()]
            if td:
                U_[reg]["text_laal_online"].append(laal(td, u))
                U_[reg]["text_laal_thinker"].append(laal(tdo, u))
            sd = speech_word_delays(pw, u["a"])
            if sd:
                U_[reg]["speech_laal_online"].append(laal(sd, u))
            sd1 = speech_word_delays(pw, u["a"], "ch_start", "ch_end")
            if sd1:
                U_[reg]["speech_laal_online_1ch"].append(laal(sd1, u))
            last = max(bs, key=lambda b: b["on_end"])
            parts["model_lag_orig"].append(last["orig_write"] - u["b"])
            parts["hold"].append(last["write"] - last["orig_write"])
            parts["model_lag"].append(last["write"] - u["b"])
            parts["detector_wait"].append(last["avail"] - last["write"])
            parts["queue_wait"].append(last["g_start"] - last["avail"])
            parts["play_to_last_word"].append(last["on_end"] - last["g_start"])
            parts["last_block_dur"].append(last["g_dur"])
            parts["final_read_last"].append(float(last["final_read"]))
    gw = [g["words"] for g in groups if g["words"] > 0]
    res = {"run": name, "rule": rule, "c": round(c, 4), "segment_block_mismatch": miss,
           "stream_online": {"start": q(st_start), "end": q(st_end)},
           "unit": {key: {"all": q(U_["ov"][key] + U_["1spk"][key]), "ov": q(U_["ov"][key]), "1spk": q(U_["1spk"][key])}
                    for key in sorted({k for dd in U_.values() for k in dd})},
           "parts": {k: q(v) for k, v in parts.items()},
           "talker_blocks": {"n": len(groups), "words_mean": round(float(np.mean(gw)), 3),
                             "share_1w": round(float(np.mean(np.asarray(gw) == 1)), 4),
                             "speech_s": round(sum(g["dur"] for g in groups), 1),
                             "floor_hits": int(stats["floor_hits"])}}
    return res


def _job(args):
    name, rule, c_kind = args
    a, c_ols = D[name]["fit"]
    c = {"c0": 0.0, "chalf": 0.5 * c_ols, "cols": c_ols}[c_kind]
    r = replay(name, rule, c)
    r["c_kind"] = c_kind
    return r


def check_against(stored: dict, res: dict) -> dict:
    """identity replay vs the stored speech_latency_lit json: every compared mean must agree exactly (3 dp)."""
    out = {}
    for key in ["speech_laal_online", "text_laal_online", "speech_end_online", "speech_start_online",
                "speech_laal_online_1ch"]:
        for reg in ["all", "ov", "1spk"]:
            if key in stored["unit"]:
                out[f"{key}.{reg}"] = (stored["unit"][key][reg]["mean"], res["unit"][key][reg]["mean"],
                                       stored["unit"][key][reg]["n"], res["unit"][key][reg]["n"])
    for p in ["model_lag", "detector_wait", "queue_wait", "last_block_dur", "final_read_last"]:
        out[f"parts.{p}"] = (stored["unit_end_parts_online"][p]["mean"], res["parts"][p]["mean"],
                             stored["unit_end_parts_online"][p]["n"], res["parts"][p]["n"])
    for e in ["start", "end"]:
        out[f"stream.{e}"] = (stored["stream"]["online"][e]["mean"], res["stream_online"][e]["mean"],
                              stored["stream"]["online"][e]["n"], res["stream_online"][e]["n"])
    out["mismatch"] = (stored["segment_block_mismatch"], res["segment_block_mismatch"], 0, 0)
    ok = all(abs(v[0] - v[1]) < 5e-4 and v[2] == v[3] for v in out.values())
    return {"ok": ok, "pairs": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--runs", default=",".join(RUNS))
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out / 'results.json'}")
    names = args.runs.split(",")
    for n in names:
        load_run(n)
    fits = {n: D[n]["fit"] for n in names}
    jobs = [(n, "none", "c0") for n in names]
    jobs += [(n, r, ck) for n in names for r in RULES if r != "none" for ck in ("c0", "chalf", "cols")]
    with mp.get_context("fork").Pool(min(args.procs, len(jobs))) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    checks = {}
    for n in names:
        ident = next(r for r in res if r["run"] == n and r["rule"] == "none")
        checks[n] = check_against(json.load(open(REPO / RUNS[n][2], encoding="utf-8")), ident)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "x", encoding="utf-8") as fh:
        json.dump({"fits_ols_a_c": fits, "floor_s_per_word": FLOOR_S_PER_WORD, "checks": checks, "results": res}, fh,
                  indent=1, ensure_ascii=False)
    print(json.dumps({n: checks[n]["ok"] for n in names}))
    for n in names:
        if not checks[n]["ok"]:
            print(n, json.dumps({k: v for k, v in checks[n]["pairs"].items() if abs(v[0] - v[1]) >= 5e-4 or v[2] != v[3]}))
    hdr = "%-10s %-9s %-5s %7s %7s %7s %7s %7s %7s %7s %6s" % ("run", "rule", "c", "spLAAL", "ov", "1spk", "EndOff",
                                                                "txtLAAL", "queue", "1chLAAL", "w/blk")
    print(hdr)
    for r in res:
        u = r["unit"]
        print("%-10s %-9s %-5s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %6.2f" % (
            r["run"], r["rule"], r["c_kind"], u["speech_laal_online"]["all"]["mean"], u["speech_laal_online"]["ov"]["mean"],
            u["speech_laal_online"]["1spk"]["mean"], u["speech_end_online"]["all"]["mean"],
            u["text_laal_online"]["all"]["mean"], r["parts"]["queue_wait"]["mean"],
            u["speech_laal_online_1ch"]["all"]["mean"], r["talker_blocks"]["words_mean"]))


if __name__ == "__main__":
    main()
