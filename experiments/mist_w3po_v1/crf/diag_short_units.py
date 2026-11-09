"""Scheme 1' precheck, short-unit diagnostic (CPU, read-only on scored dirs of crf/precheck_sched.py).

Question: when v8 misses short units, is it a TIMING / decision problem (it can translate the unit but would not start
it at that read) or a CONTENT problem (it cannot translate it even with the whole window's audio)?

Units are the reference utterances of a window (as the precheck's C1). Per unit, from the saved v8 log-probs
(tok_lp[r, j] = log p(S_j | x<=t_r, S^{<j}), end_lp[r, b] = log p(e_r | x<=t_r, S^{<b}); last read: <wait> barred):
  dur_s, words; ov50 = the other speakers' units (union) cover >= 50 % of it; ov_any = ov_s > 0 (the precheck's C2)
  (a) p_content_final   exp(mean log-prob of the unit's TEXT tokens at the final read): full audio, gold prefix
      lp_unit_final     mean log-prob of all its tokens (speaker, times, text) at the final read
  (b) p_open_final      prob of the unit's opening token in S (its first chunk's speaker token) at the final read
      p_open_max        max of that prob over the reads where some feasible schedule can open it (the trainer's
                        reachable set A, W cap included); r_open_max_rel = that read - the heuristic opening read
      p_end_at_open_final  prob of <eos> at the opening position, final read (ending instead of starting it)
  (c) P_early_{lam0,lamstar}_{any,chunk}  posterior prob that the unit is committed (its last token) before the final
      read, at lam = 0 and lam*, both boundary modes; only windows with >= 2 reads count
  (d) p_content_heur, p_open_heur, p_end_at_open_heur: (a) and (b) at the heuristic (v8 label) read of each token
Groups: <= 1 s vs > 1 s, <= 0.5 s, ov50 vs isolated, and their crosses; median, quartiles, mean, share < 0.3 / < 0.5.

--extras (meant for the 500 train windows) adds, for three pending training decisions:
  (i)  lambda_spread: per-window lam* by vectorised bisection (target = the window's own Lat(heur); "any" mode, W),
       and per-batch lam* on simulated trainer steps (train_crf_sft.plan_steps, ~64 read rows; target = the global
       L_ref of sched summary.json, as the trainer, and the step's own mean Lat(heur), as --lat-target-batch)
  (ii) state_anticipation: over the reachable pairs A (train_crf_sft.set_reach), the share with r BEFORE the
       heuristic commit read of token j, and mean p / mean -log p there vs at / after it, by token kind

CAVEAT: on train windows v8 was trained on these very targets, so the content ability (a) is inflated; dev windows
(sched_dev_v1, AMI dev meetings) are the cleaner read.

    python -m experiments.mist_w3po_v1.crf.diag_short_units --scored <dir>[,<dir>...] --report <merged report.json> \
        --out <new dir> [--extras]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.mist_w3po_v1.crf import precheck_sched as P
from experiments.mist_w3po_v1.crf import schedule_lattice as sl

K_SPK, K_TA, K_TEXT, K_TB = 0, 1, 2, 3
KIND = {K_SPK: "speaker", K_TA: "start_time", K_TEXT: "text", K_TB: "end_time"}
PROB_METRICS = ["p_content_final", "p_open_final", "p_open_max", "p_end_at_open_final", "p_content_heur",
                "p_open_heur", "p_end_at_open_heur", "P_early_lam0_any", "P_early_lamstar_any",
                "P_early_lam0_chunk", "P_early_lamstar_chunk"]
OTHER_METRICS = ["dur_s", "words", "lp_unit_final", "r_open_max_rel"]


def load(scored: str):
    sched, ids, lp_of = P.load_scored([d for d in scored.split(",") if d])
    want, allr = set(ids), {}
    for f in sorted(Path(sched).glob("shard_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in want:
                allr[r["id"]] = r
    recs = [allr[i] for i in ids]
    lps = {}
    for r in recs:
        z = np.load(lp_of[r["id"]] / f"{r['id']}.npz")
        lps[r["id"]] = (z["tok_lp"], z["end_lp"])
    return sched, recs, lps


def lattices(recs, lps, mode: str, W: int, bs: int) -> list:
    wins = []
    for rec in recs:
        tok_lp, end_lp = lps[rec["id"]]
        end_lp = end_lp.copy()
        if mode == "chunk":                                   # as precheck_sched.analyse
            allowed = np.zeros(rec["n_tok"] + 1, bool)
            allowed[[0] + rec["seg_bounds"]] = True
            end_lp[:, ~allowed] = -np.inf
        wins.append({"tok_lp": tok_lp, "end_lp": end_lp, "lo": rec["lo"], "hi": rec["hi"], "lag": P.lag_matrix(rec)})
    return [sl.pack(wins[i: i + bs], W=W, device="cpu") for i in range(0, len(wins), bs)]


def w_tok_all(lats, lam) -> list[np.ndarray]:
    out = []
    for lat in lats:
        mg = sl.marginals(lat, lam)
        for i in range(lat.B):
            out.append(mg["w_tok"][i].numpy() if bool(mg["feasible"][i]) else None)
    return out


def reach_all(recs, W: int) -> list[np.ndarray]:
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    wins = [T.make_window(i, r) for i, r in enumerate(recs)]
    for i in range(0, len(wins), 64):
        T.set_reach(wins[i: i + 64], W)
    return [w.reach for w in wins]


def heur_reads(rec) -> np.ndarray:
    return P.commit_reads(np.asarray(rec["beta_heur"]), rec["n_tok"])


def ov50(units: list[dict], u: dict) -> bool:
    a, b = u["a_ms"], u["b_ms"]
    iv = sorted((max(a, v["a_ms"]), min(b, v["b_ms"])) for v in units
                if v["speaker_id"] != u["speaker_id"] and min(b, v["b_ms"]) > max(a, v["a_ms"]))
    cov, cur_a, cur_b = 0, None, None
    for x, y in iv:                                            # union length
        if cur_b is None or x > cur_b:
            if cur_b is not None:
                cov += cur_b - cur_a
            cur_a, cur_b = x, y
        else:
            cur_b = max(cur_b, y)
    if cur_b is not None:
        cov += cur_b - cur_a
    return cov >= 0.5 * max(b - a, 1)


def unit_rows(recs, lps, reach, wt) -> list[dict]:
    rows = []
    for k, rec in enumerate(recs):
        tok_lp, end_lp = lps[rec["id"]]
        R, N = rec["n_read"], rec["n_tok"]
        unit_of, kind = np.asarray(rec["unit"]), np.asarray(rec["kind"])
        rh = heur_reads(rec)
        units = rec["units"]
        for u in units:
            J = np.nonzero(unit_of == u["utt"])[0]
            if not len(J):
                continue
            jt = J[kind[J] == K_TEXT]
            j0, jL = int(J.min()), int(J.max())
            rr = np.nonzero(reach[k][:, j0])[0] if reach[k] is not None else np.arange(R)
            rr = rr if len(rr) else np.array([R - 1])
            best = int(rr[np.argmax(tok_lp[rr, j0])])
            row = {"id": rec["id"], "utt": u["utt"], "dur_s": (u["b_ms"] - u["a_ms"]) / 1000.0,
                   "words": u["words"], "ov50": ov50(units, u), "ov_any": u["ov_s"] > 0, "R": R,
                   "n_tok": len(J), "n_text": len(jt),
                   "heur_open_read": int(rh[j0]), "heur_commit_read": int(rh[jL]),
                   "heur_early": bool(rh[jL] < R - 1),
                   "p_content_final": float(np.exp(tok_lp[R - 1, jt].mean())),
                   "lp_unit_final": float(tok_lp[R - 1, J].mean()),
                   "p_open_final": float(np.exp(tok_lp[R - 1, j0])),
                   "p_open_max": float(np.exp(tok_lp[best, j0])), "r_open_max_rel": best - int(rh[j0]),
                   "p_end_at_open_final": float(np.exp(end_lp[R - 1, j0])),
                   "p_content_heur": float(np.exp(tok_lp[rh[jt], jt].mean())),
                   "p_open_heur": float(np.exp(tok_lp[rh[j0], j0])),
                   "p_end_at_open_heur": float(np.exp(end_lp[rh[j0], j0]))}
            for key, w in wt.items():
                if R > 1 and w[k] is not None:
                    row[f"P_early_{key}"] = float(np.clip(1.0 - w[k][R - 1, jL], 0.0, 1.0))
            rows.append(row)
    return rows


def stats(x) -> dict | None:
    x = np.asarray([v for v in x if v is not None], np.float64)
    if not len(x):
        return None
    return {"n": int(len(x)), "median": float(np.median(x)), "q1": float(np.percentile(x, 25)),
            "q3": float(np.percentile(x, 75)), "mean": float(x.mean()),
            "share_lt_0.3": float((x < 0.3).mean()), "share_lt_0.5": float((x < 0.5).mean())}


GROUPS = {
    "all": lambda r: True,
    "le1s": lambda r: r["dur_s"] <= 1.0, "gt1s": lambda r: r["dur_s"] > 1.0, "le0.5s": lambda r: r["dur_s"] <= 0.5,
    "ov50": lambda r: r["ov50"], "iso": lambda r: not r["ov50"],
    "le1s_ov50": lambda r: r["dur_s"] <= 1.0 and r["ov50"], "le1s_iso": lambda r: r["dur_s"] <= 1.0 and not r["ov50"],
    "gt1s_ov50": lambda r: r["dur_s"] > 1.0 and r["ov50"], "gt1s_iso": lambda r: r["dur_s"] > 1.0 and not r["ov50"],
}


def group_table(rows: list[dict]) -> dict:
    out = {}
    for g, f in GROUPS.items():
        sub = [r for r in rows if f(r)]
        d = {"units": len(sub), "share_ov_any": float(np.mean([r["ov_any"] for r in sub])) if sub else None,
             "share_heur_early": float(np.mean([r["heur_early"] for r in sub])) if sub else None}
        for m in PROB_METRICS + OTHER_METRICS:
            d[m] = stats([r.get(m) for r in sub])
        early = [r for r in sub if r["heur_early"]]
        for m in ("P_early_lam0_any", "P_early_lamstar_any", "P_early_lam0_chunk", "P_early_lamstar_chunk"):
            d[m + "|heur_early"] = stats([r.get(m) for r in early])
        out[g] = d
    return out


# ------------------------------------------------------------------------------------------------ extras
def per_window_lam(lat, target: torch.Tensor, big: float = 1e4, iters: int = 60) -> dict:
    """Vectorised bisection: lam_w with E_q[Lat]_w = target_w for every window of one packed batch."""
    def e(lam):
        return sl.marginals(lat, lam)["exp_lat"]

    lo = torch.full((lat.B,), -big, dtype=sl.DTYPE)
    hi = torch.full((lat.B,), big, dtype=sl.DTYPE)
    e_lo, e_hi = e(lo), e(hi)                                  # E[Lat] is non-increasing in lam (~ latest / earliest)
    eps = 1e-6
    degenerate = (e_lo - e_hi) < eps                           # Lat is the same for every feasible schedule
    at_latest = ~degenerate & ((e_lo - target).abs() <= eps)   # the target is the latest schedule: lam* = -inf
    at_earliest = ~degenerate & ((e_hi - target).abs() <= eps)  # ... the earliest: lam* = +inf
    bracketed = (e_lo > target + eps) & (e_hi < target - eps) & ~degenerate
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = e(mid) - target
        lo = torch.where(f > 0, mid, lo)
        hi = torch.where(f > 0, hi, mid)
    lam = 0.5 * (lo + hi)
    return {"lam": lam.numpy(), "bracketed": bracketed.numpy(), "degenerate": degenerate.numpy(),
            "at_latest": at_latest.numpy(), "at_earliest": at_earliest.numpy(),
            "below": ((e_lo < target - eps) & ~degenerate).numpy(), "above": ((e_hi > target + eps) & ~degenerate).numpy(),
            "resid": (e(lam) - target).abs().numpy()}


def qstats(x) -> dict | None:
    x = np.asarray(x, np.float64)
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "p10": float(np.percentile(x, 10)),
            "q1": float(np.percentile(x, 25)), "median": float(np.median(x)), "q3": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max()),
            "share_neg": float((x < 0).mean()), "share_abs_gt_10": float((np.abs(x) > 10).mean())}


def lambda_spread(recs, lps, sched: str, W: int, bs: int, passes: float) -> dict:
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    lats = lattices(recs, lps, "any", W, bs)
    keys = ("lam", "bracketed", "degenerate", "at_latest", "at_earliest", "below", "above", "resid")
    acc = {key: [] for key in keys}
    k = 0
    for lat in lats:
        tgt = torch.tensor([recs[k + i]["lat_heur"] for i in range(lat.B)], dtype=sl.DTYPE)
        k += lat.B
        o = per_window_lam(lat, tgt)
        for key in keys:
            acc[key] += o[key].tolist()
    a = {key: np.asarray(v) for key, v in acc.items()}
    br = a["bracketed"].astype(bool)
    lam = a["lam"]
    multi = np.asarray([r["n_read"] > 1 for r in recs])
    # all windows on one scale: -inf / +inf for the boundary classes (rank statistics stay defined)
    lam_all = np.where(a["at_latest"].astype(bool), -np.inf, np.where(a["at_earliest"].astype(bool), np.inf, lam))
    keep = br | a["at_latest"].astype(bool) | a["at_earliest"].astype(bool)
    pct = (lambda q: float(np.percentile(lam_all[keep], q, method="nearest"))) if keep.any() else (lambda q: None)
    per_window = {"windows": len(recs), "single_read_windows": int((~multi).sum()),
                  "interior_bracketed": int(br.sum()),
                  "degenerate_same_lat_every_schedule": int(a["degenerate"].sum()),
                  "target_is_latest_schedule_lam_minus_inf": int(a["at_latest"].sum()),
                  "target_is_earliest_schedule_lam_plus_inf": int(a["at_earliest"].sum()),
                  "target_unreachable_other": int((a["below"] | a["above"]).sum()),
                  "max_resid_bracketed": float(np.max(a["resid"][br])) if br.any() else None,
                  "lam_star_interior": qstats(lam[br]),
                  "lam_star_incl_boundary_percentiles": {f"p{q}": pct(q) for q in (10, 25, 50, 75, 90)}}
    # simulated trainer steps on this sample
    L_ref = json.loads((Path(sched) / "summary.json").read_text())["L_ref_mean_lat_heur_windows"]
    Rs = [r["n_read"] for r in recs]
    n_steps = int(passes * sum(Rs) / 64)
    steps = T.plan_steps(Rs, 0, n_steps, 64)
    one = {r["id"]: lat_one for r, lat_one in zip(recs, _singles(recs, lps, W))}
    out = {"global": [], "batch": []}
    br_n = {"global": 0, "batch": 0}
    lat_h, rows = [], []
    for st in steps:
        sub = [recs[i] for i in st]
        lat = sl.pack([one[r["id"]] for r in sub], W=W, device="cpu")
        lat_h.append(float(np.mean([r["lat_heur"] for r in sub])))
        rows.append(sum(r["n_read"] for r in sub))
        for key, tgt in (("global", L_ref), ("batch", lat_h[-1])):
            sol = sl.solve_lambda([lat], tgt)
            if sol["bracketed"]:
                out[key].append(sol["lam"])
                br_n[key] += 1
    return {"per_window_target_own_lat_heur": per_window,
            "per_step": {"steps": len(steps), "rows_per_step": qstats(rows), "step_mean_lat_heur": qstats(lat_h),
                         "target_global_L_ref": L_ref,
                         "lam_target_global": qstats(out["global"]), "bracketed_global": br_n["global"],
                         "lam_target_batch": qstats(out["batch"]), "bracketed_batch": br_n["batch"]}}


def _singles(recs, lps, W):
    return [{"tok_lp": lps[r["id"]][0], "end_lp": lps[r["id"]][1], "lo": r["lo"], "hi": r["hi"],
             "lag": P.lag_matrix(r)} for r in recs]


def state_anticipation(recs, lps, reach) -> dict:
    acc = defaultdict(lambda: defaultdict(list))
    for k, rec in enumerate(recs):
        if reach[k] is None:
            continue
        tok_lp = lps[rec["id"]][0]
        rh = heur_reads(rec)
        R = rec["n_read"]
        rs, js = np.nonzero(reach[k])
        kinds = np.asarray(rec["kind"])[js]
        lp = tok_lp[rs, js]
        rel = np.where(rs < rh[js], "before", np.where(rs == rh[js], "at", "after"))
        final = rs == R - 1
        for grp in ("before", "at", "after"):
            m = rel == grp
            for kd in ("all",) + tuple(KIND.values()):
                mk = m if kd == "all" else m & (kinds == [k_ for k_, v in KIND.items() if v == kd][0])
                acc[kd][grp].append(lp[mk])
                acc[kd][grp + "_nonfinal"].append(lp[mk & ~final])
    out = {}
    for kd, d in acc.items():
        tot = sum(np.concatenate(d[g]).size for g in ("before", "at", "after"))
        tot_nll = sum(-np.concatenate(d[g]).sum() for g in ("before", "at", "after"))
        o = {"pairs": int(tot)}
        for g, v in d.items():
            x = np.concatenate(v)
            o[g] = {"pairs": int(x.size), "share_of_A": float(x.size / max(tot, 1)),
                    "mean_p": float(np.exp(x).mean()) if x.size else None,
                    "median_p": float(np.median(np.exp(x))) if x.size else None,
                    "mean_nll": float(-x.mean()) if x.size else None,
                    "share_of_L_state": float(-x.sum() / tot_nll) if x.size and tot_nll > 0 else None}
        out[kd] = o
    return out


# ------------------------------------------------------------------------------------------------ main
def fmt_table(groups: dict, metrics: list[str]) -> str:
    gs = ["le1s", "gt1s", "le0.5s", "ov50", "iso", "le1s_ov50", "le1s_iso", "gt1s_ov50", "gt1s_iso"]
    lines = ["| metric | " + " | ".join(f"{g} (n={groups[g]['units']})" for g in gs) + " |",
             "|---|" + "---|" * len(gs)]
    for m in metrics:
        cells = []
        for g in gs:
            s = groups[g].get(m)
            cells.append("—" if not s else f"{s['median']:.2f} [{s['q1']:.2f},{s['q3']:.2f}] "
                                           f"<.3:{s['share_lt_0.3']:.0%} <.5:{s['share_lt_0.5']:.0%}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True, help="scored dir(s) of precheck_sched (comma-separated shards)")
    ap.add_argument("--report", default="", help="a precheck report.json for lam* per mode (else: bisection here, "
                                                  "L_ref = this sample's mean Lat(heur))")
    ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--lattice-batch", type=int, default=64)
    ap.add_argument("--extras", action="store_true", help="(i) lambda spread, (ii) state-loss anticipation")
    ap.add_argument("--passes", type=float, default=3.0, help="extras: simulated trainer passes over the sample")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    sched, recs, lps = load(args.scored)
    print(f"{len(recs)} windows from {sched}", flush=True)
    lam_star, lam_src = {}, args.report or "bisection on this sample (L_ref = its mean Lat(heur))"
    if args.report:
        rep = json.loads(Path(args.report).read_text())
        lam_star = {m: rep[m]["lam_star"] for m in ("any", "chunk")}
    reach = reach_all(recs, args.W)
    wt = {}
    for mode in ("any", "chunk"):
        lats = lattices(recs, lps, mode, args.W, args.lattice_batch)
        if mode not in lam_star:
            lam_star[mode] = sl.solve_lambda(lats, float(np.mean([r["lat_heur"] for r in recs])))["lam"]
        wt[f"lam0_{mode}"] = w_tok_all(lats, 0.0)
        wt[f"lamstar_{mode}"] = w_tok_all(lats, lam_star[mode])
        print(f"{mode}: marginals done {time.time() - t0:.0f}s (lam* {lam_star[mode]:.4f})", flush=True)
    rows = unit_rows(recs, lps, reach, wt)
    groups = group_table(rows)
    res = {"meta": {"scored": args.scored, "sched": sched, "windows": len(recs), "units": len(rows),
                    "lam_star": lam_star, "lam_source": lam_src, "W": args.W,
                    "caveat": "train windows: v8 was trained on these targets, content ability is inflated"
                    if "sched_v1" in sched else "dev windows (AMI validation meetings): not trained on"},
           "groups": groups}
    if args.extras:
        res["lambda_spread"] = lambda_spread(recs, lps, sched, args.W, args.lattice_batch, args.passes)
        print(f"lambda spread done {time.time() - t0:.0f}s", flush=True)
        res["state_anticipation"] = state_anticipation(recs, lps, reach)
    res["meta"]["elapsed_s"] = time.time() - t0
    out.mkdir(parents=True)
    with open(out / "units.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    (out / "diag.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(fmt_table(groups, PROB_METRICS + ["dur_s", "words"]))
    if args.extras:
        print(json.dumps({k: res[k] for k in ("lambda_spread", "state_anticipation")}, indent=1))
    print(f"DIAG_SHORT_UNITS_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
