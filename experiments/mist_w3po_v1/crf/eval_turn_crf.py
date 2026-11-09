"""TurnCRF CPU stage, dev evaluation (pre-registered in reports/mist_w3po_v1/turn_crf_cpu_stage.zh-CN.md §4).

Systems, each offline (whole window) and streaming:
  raw    P >= 0.5 per slot (per frame, so streaming == offline)
  med5   P >= 0.5 then a 5-frame median filter per slot (scipy.ndimage.median_filter, mode "nearest"); streaming:
         applied to the prefix at each read
  chain  linear-chain CRF, slot marginal pi_t^k >= 0.5
  semi   semi-CRF, slot marginal pi_t^k >= 0.5
  (secondary, not in the stop rule: chain_vit / semi_vit = offline Viterbi)
Streaming: reads at 1 s, 2 s, ... (window-relative) and at the window end; frame t is labelled from the prefix
posterior of the first read whose prefix contains it (prefix length floor(read_ms / 80) frames; the window-end read
uses the whole window, closed). CRF prefixes are right-censored (open last segment, no end score).

Metrics against the full reference slot sets R (pooled over windows): turn IoU-F1 (turn = maximal run of one slot;
one-to-one match, same slot, IoU >= 0.5), recall of reference turns <= 1 s, frame miss / false-alarm / confusion rates
(normalised by reference slot-frames), overlap-frame F1 (frames with >= 2 active slots). Meeting-level paired
bootstrap (1000 resamples, seed 0) of CRF minus the better baseline of the same mode.

Writes --out (refuses an existing --out): metrics.json, per_meeting.json, table.md.

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.eval_turn_crf --models eval_outputs/mist_w3po_v1/crf/turn_v1 \
        --out eval_outputs/mist_w3po_v1/crf/turn_v1_dev_eval --workers 32
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter

from experiments.mist_w3po_v1.crf.turn_crf import build_model

FRAME_MS = 80
SHORT_FRAMES = 12            # <= 1 s: 12 frames * 80 ms = 0.96 s (13 frames = 1.04 s)
IOU = 0.5
SYSTEMS = ["raw", "med5", "chain", "semi"]
MODES = ["offline", "streaming"]
SECONDARY = ["chain_vit", "semi_vit"]
STAT_KEYS = ["hyp_turns", "ref_turns", "matched", "ref_short", "matched_short", "ref_slot_frames", "miss", "fa", "conf",
             "ov_tp", "ov_hyp", "ov_ref", "frames"]

_G: dict = {}


def read_prefixes(dur_ms: int, T: int) -> list[int]:
    taus = [min(T, (n * 1000) // FRAME_MS) for n in range(1, (dur_ms - 1) // 1000 + 1)]
    taus = sorted({t for t in taus if 1 <= t < T})
    return taus + [T]


def med5(B: np.ndarray) -> np.ndarray:
    return median_filter(B.astype(np.uint8), size=(5, 1), mode="nearest").astype(bool)


def stream(fn, T: int, taus: list[int]) -> np.ndarray:
    out = np.zeros((T, 8), dtype=bool)
    prev = 0
    for tau in taus:
        out[prev:tau] = fn(tau)[prev:tau]
        prev = tau
    return out


def runs(x: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def window_stats(H: np.ndarray, R: np.ndarray) -> dict:
    s = dict.fromkeys(STAT_KEYS, 0)
    for k in range(R.shape[1]):
        rr, hh = runs(R[:, k]), runs(H[:, k])
        s["ref_turns"] += len(rr)
        s["hyp_turns"] += len(hh)
        short = [(b - a) <= SHORT_FRAMES for a, b in rr]
        s["ref_short"] += sum(short)
        pairs = []
        for i, (a, b) in enumerate(rr):
            for j, (c, d) in enumerate(hh):
                inter = min(b, d) - max(a, c)
                if inter > 0:
                    iou = inter / (max(b, d) - min(a, c))
                    if iou >= IOU:
                        pairs.append((-iou, i, j))
        used_r, used_h = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_r or j in used_h:
                continue
            used_r.add(i)
            used_h.add(j)
            s["matched"] += 1
            s["matched_short"] += int(short[i])
    nR, nH = R.sum(1), H.sum(1)
    nC = (R & H).sum(1)
    s["ref_slot_frames"] = int(nR.sum())
    s["miss"] = int(np.maximum(0, nR - nH).sum())
    s["fa"] = int(np.maximum(0, nH - nR).sum())
    s["conf"] = int((np.minimum(nR, nH) - nC).sum())
    s["ov_tp"] = int(((nR >= 2) & (nH >= 2)).sum())
    s["ov_hyp"] = int((nH >= 2).sum())
    s["ov_ref"] = int((nR >= 2).sum())
    s["frames"] = len(R)
    return {k: int(v) for k, v in s.items()}


def metrics(s: dict) -> dict:
    p = s["matched"] / max(1, s["hyp_turns"])
    r = s["matched"] / max(1, s["ref_turns"])
    op = s["ov_tp"] / max(1, s["ov_hyp"])
    orc = s["ov_tp"] / max(1, s["ov_ref"])
    n = max(1, s["ref_slot_frames"])
    return {"turn_f1": 2 * p * r / max(1e-12, p + r), "turn_p": p, "turn_r": r,
            "short_recall": s["matched_short"] / max(1, s["ref_short"]),
            "miss": s["miss"] / n, "fa": s["fa"] / n, "conf": s["conf"] / n,
            "err": (s["miss"] + s["fa"] + s["conf"]) / n,
            "ov_f1": 2 * op * orc / max(1e-12, op + orc), "ov_p": op, "ov_r": orc}


def _init(models_dir: str) -> None:
    torch.set_num_threads(1)
    for kind in ("semi", "chain"):
        ck = torch.load(Path(models_dir) / kind / "params.pt", weights_only=False)
        m = build_model(ck["kind"], **ck["model_kw"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        _G[kind] = m


def _crf_hyp(m, P: np.ndarray, tau: int, T: int) -> np.ndarray:
    Pt = torch.from_numpy(P[:tau].astype(np.float64))
    _, mu = m.marginals(Pt, censored=tau < T)
    return (m.slot_marginals(mu) >= 0.5).numpy()


def _one(args) -> tuple:
    i, P, R, dur_ms = args
    T = len(P)
    taus = read_prefixes(dur_ms, T)
    B = P.astype(np.float32) >= 0.5
    hyp = {("raw", "offline"): B, ("raw", "streaming"): stream(lambda tau: B[:tau], T, taus),
           ("med5", "offline"): med5(B), ("med5", "streaming"): stream(lambda tau: med5(B[:tau]), T, taus)}
    for kind in ("chain", "semi"):
        m = _G[kind]
        hyp[(kind, "offline")] = _crf_hyp(m, P, T, T)
        hyp[(kind, "streaming")] = stream(lambda tau: _crf_hyp(m, P, tau, T), T, taus)
        yv, _ = m.viterbi(torch.from_numpy(P.astype(np.float64)))
        hyp[(kind + "_vit", "offline")] = m.space.M[yv].numpy() > 0.5
    return i, {f"{s}|{md}": window_stats(H, R) for (s, md), H in hyp.items()}, len(taus)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="eval_outputs/mist_w3po_v1/crf/turn_refs_v1/dev.npz")
    ap.add_argument("--windows", default="eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl")
    ap.add_argument("--models", default="eval_outputs/mist_w3po_v1/crf/turn_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    d = np.load(args.data)
    ids, meet, offs, P, R = d["ids"], d["meeting"], d["offs"], d["P"], d["R"]
    dur = {r["id"]: r["dur_ms"] for r in map(json.loads, open(args.windows, encoding="utf-8"))}
    N = len(ids) if not args.limit else min(args.limit, len(ids))
    jobs = [(i, P[offs[i]:offs[i + 1]], R[offs[i]:offs[i + 1]], dur[str(ids[i])]) for i in range(N)]
    ctx = mp.get_context("fork")
    per_win = [None] * N
    n_reads = 0
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.models,)) as pool:
        for i, st, nr in pool.imap_unordered(_one, jobs, chunksize=4):
            per_win[i] = st
            n_reads += nr
    keys = list(per_win[0].keys())
    meetings = sorted(set(str(m) for m in meet[:N]))
    per_meet = {k: {m: dict.fromkeys(STAT_KEYS, 0) for m in meetings} for k in keys}
    for i in range(N):
        for k in keys:
            for s, v in per_win[i][k].items():
                per_meet[k][str(meet[i])][s] += v
    tot = {k: {s: sum(per_meet[k][m][s] for m in meetings) for s in STAT_KEYS} for k in keys}
    res = {k: metrics(tot[k]) for k in keys}
    # meeting-level paired bootstrap: CRF minus the better baseline (point estimate) of the same mode
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(meetings), size=(args.boot, len(meetings)))
    boot = {}
    for md in MODES:
        for metric in ("turn_f1", "short_recall"):
            best = max(("raw", "med5"), key=lambda s: res[f"{s}|{md}"][metric])
            for crf in ("chain", "semi"):
                diffs = []
                for row in draws:
                    agg = {}
                    for k in (f"{crf}|{md}", f"{best}|{md}"):
                        agg[k] = metrics({s: sum(per_meet[k][meetings[j]][s] for j in row) for s in STAT_KEYS})[metric]
                    diffs.append(agg[f"{crf}|{md}"] - agg[f"{best}|{md}"])
                diffs = np.array(diffs)
                boot[f"{crf}-{best}|{md}|{metric}"] = {
                    "diff": res[f"{crf}|{md}"][metric] - res[f"{best}|{md}"][metric],
                    "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                    "p_le0": float((diffs <= 0).mean()), "baseline": best}
    sm = "streaming"
    verdict = {f"semi>{b}|{sm}|{m}": res[f"semi|{sm}"][m] > res[f"{b}|{sm}"][m]
               for b in ("raw", "med5") for m in ("turn_f1", "short_recall")}
    verdict["pass"] = all(verdict.values())
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"metrics": res, "totals": tot, "bootstrap": boot,
                                                  "stop_rule": verdict, "n_windows": N, "meetings": meetings,
                                                  "n_reads": n_reads, "args": vars(args)}, indent=1),
                                      encoding="utf-8")
    (out / "per_meeting.json").write_text(json.dumps(per_meet), encoding="utf-8")
    lines = ["| system | mode | turn F1 | turn P | turn R | <=1s recall | miss | FA | conf | err | ov F1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for md in MODES:
        for s in SYSTEMS + (SECONDARY if md == "offline" else []):
            k = f"{s}|{md}"
            if k not in res:
                continue
            r = res[k]
            lines.append(f"| {s} | {md} | {r['turn_f1']:.4f} | {r['turn_p']:.4f} | {r['turn_r']:.4f} | "
                         f"{r['short_recall']:.4f} | {r['miss']:.4f} | {r['fa']:.4f} | {r['conf']:.4f} | "
                         f"{r['err']:.4f} | {r['ov_f1']:.4f} |")
    lines += ["", "| comparison | mode | metric | diff | 95% CI | P(diff<=0) |", "|---|---|---|---|---|---|"]
    for k, v in boot.items():
        c, md, m = k.split("|")
        lines.append(f"| {c} | {md} | {m} | {v['diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                     f"{v['p_le0']:.3f} |")
    lines += ["", f"stop rule (streaming semi > raw and med5 on turn F1 and <=1s recall): {json.dumps(verdict)}"]
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
