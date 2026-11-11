"""T4: off-schedule state augmentation (DART-style noise injection) for the T2 latent write-time CRF, and the 1.0 s
sub-grid training data (pre-registration reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T4).

Noise injection (Laskey et al., CoRL 2017, "DART"; in the spirit of DAgger, Ross et al., AISTATS 2011): the training
schedule is perturbed so that the history falls behind the labels, and at every such off-schedule state the target is
the expert's action, "write every overdue item now". This removes the training shortcut "an item missing from the
history means it is not due yet" (T1's dev5 stall diagnosis, LOG 2026-11-11 14:25 UTC).

Per (optimiser step s, window w) of the trainer's plan (one setting, registered before any result):
  draw      rng = numpy.random.default_rng([20261111, s, w.idx]); the window is perturbed iff rng.random() < 0.5;
            then per item (chunk index order) hit_i = rng.random(n) < 0.3 and k_i = rng.integers(1, 4, n) (both arrays
            drawn for every perturbed window, so the stream does not depend on the outcome); delta_i = k_i if hit_i.
  path      rho~_i = min(R - 1, e_i + delta_i), then the running max along each speaker's kap order (prefix closure);
            e_i = the chunk's label read (S.chunk_reads == e_read, T2's lower bound).
  history   H_r = {i : rho~_i <= r - 1}  (kap order; a written item sits at its label position, as in T2)
  overdue   O_r = {i not in H_r : e_i <= r - 1}; one extra row at every read r with O_r non-empty (final read included)
  target    W*_r = {i not in H_r : e_i <= r} in kap order (v2''s canon order), then e_r (<wait>; <eos> at the final read)
  weight    the label-path row's: h_t / D per target token, 1 / D at the end token; D = T2's D (augmentation rows are not
            added to it), so T2's rows and their gradients are unchanged and the extra rows add on the same scale.
  passes    pass 2 only (gradient); pass 1, the posterior and lam are T2's.
The draws are written once to a manifest (dart.jsonl, CPU, multiprocessing over steps); the trainer reads the manifest
(--dart), checks that its (step, window) entries are the draws of its own plan, and builds the rows from it.

1.0 s variant (--build-g1000): the 14,874 T0-labelled train windows, reads every 1.0 s plus the window end
(read_times(dur, 1000), asserted a subset of the 0.5 s grid), labels recomputed exactly from T0's stored per-read
agreement counts a_j(r) on the sub-grid: e = earliest_stable(a[sub-grid rows], u), u = S_prop's read on the 1.0 s grid,
a_j(R) = 0 -> u_j, then the running max per speaker (t0_agree_labels' definition, only the grid differs). The records are
written by build_piece_sched's own _job (bounds free) and checked by its own --finalize.

    python -m experiments.mist_w3po_v1.crf.dart_aug --build-g1000 --t0-root eval_outputs/mist_w3po_v1/crf/t0_agree_train \\
        --out <new sched dir> [--workers 32]
    python -m experiments.mist_w3po_v1.crf.dart_aug --manifest --sched <sched dir> --out <new dir> [--workers 48]
    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_dart_aug
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

DART_SEED = 20261111
FRAC, P_ITEM, KMAX = 0.5, 0.3, 3
PLAN_SEED, PLAN_STEPS, PLAN_ROWS = 20261109, 3000, 64
GRID_G1000 = 1000


# ------------------------------------------------------------------------------------------------ draws and rows
def draw_delays(n_items: int, step: int, widx: int, seed: int = DART_SEED, frac: float = FRAC, p: float = P_ITEM,
                kmax: int = KMAX) -> list[int] | None:
    """None (window not perturbed) or the per-item delay in reads (0 = on schedule)."""
    rng = np.random.default_rng([seed, step, widx])
    if rng.random() >= frac:
        return None
    hit = rng.random(n_items) < p
    k = rng.integers(1, kmax + 1, n_items)
    return [int(k[i]) if hit[i] else 0 for i in range(n_items)]


def speaker_orders(rec: dict, e: list[int]) -> dict:
    order = V.kap(rec, e, range(len(rec["chunks"])))
    spk = sorted({c["slot"] for c in rec["chunks"]})
    return {k: [c for c in order if rec["chunks"][c]["slot"] == k] for k in spk}


def perturbed_reads(rec: dict, delay: list[int]) -> list[int]:
    """rho~ per chunk: min(R - 1, e + delay), running max along each speaker's kap order."""
    e = S.chunk_reads(rec)
    R = rec["n_read"]
    assert len(delay) == len(e)
    out = [0] * len(e)
    for it in speaker_orders(rec, e).values():
        best = -1
        for c in it:
            best = max(best, min(R - 1, e[c] + int(delay[c])))
            out[c] = best
    return out


def aug_requests(rec: dict, delay: list[int], mode: str = "t2") -> list[dict]:
    """The extra rows of one perturbed window: per read r with an overdue item, the history H_r and the target W*_r
    (kap order), their token ids, the overdue lags and whether H_r is a state of T2's lattice at read r - 1."""
    from experiments.mist_w3po_v1.crf import latent_write_crf as LW

    e = S.chunk_reads(rec)
    R, n = rec["n_read"], len(rec["chunks"])
    rt = perturbed_reads(rec, delay)
    _, h = LW.bounds(rec, mode)
    out = []
    for r in range(R):
        H = [i for i in range(n) if rt[i] <= r - 1]
        Hs = set(H)
        O = [i for i in range(n) if i not in Hs and e[i] <= r - 1]
        if not O:
            continue
        Wst = [i for i in range(n) if i not in Hs and e[i] <= r]
        hist, items = V.kap(rec, e, H), V.kap(rec, e, Wst)
        out.append({"r": r, "hist": hist, "items": items, "hist_ids": S.ids_of(rec, hist),
                    "forced_ids": S.ids_of(rec, items), "lags": [r - e[i] for i in O],
                    "in_t2": all(i in Hs for i in range(n) if h[i] <= r - 1)})
    return out


def aug_rows(wins: list, step_draws: dict, D: float) -> list:
    """T.Row objects (coefficients set) of the step's windows that the manifest perturbs; step_draws {w.idx: delay}."""
    from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    rows = []
    for w in wins:
        dl = step_draws.get(w.idx)
        if dl is None:
            continue
        for q in aug_requests(w.rec, dl):
            seq = list(q["hist_ids"]) + list(q["forced_ids"])
            h0, m = len(q["hist_ids"]), len(q["forced_ids"])
            rw = T.Row(w, q["r"], len(seq), np.arange(h0, h0 + m + 1), seq=seq, end=w.end_ids[q["r"]],
                       key=("dart", w.idx, q["r"]), aug=True)
            ct, ce = np.zeros(m + 1), np.zeros(m + 1)
            ct[:m] = V2T.token_weights(w, q["forced_ids"]) / D
            ce[m] = 1.0 / D
            rw.c_tok, rw.c_end, rw.c_state = ct, ce, np.zeros(m + 1)
            rw.c_lab_tok, rw.c_lab_end = np.zeros(m + 1), np.zeros(m + 1)
            rows.append(rw)
    return rows


# ------------------------------------------------------------------------------------------------ D1 (learner roll-in)
def check_rho(rec: dict, rho: list[int]) -> None:
    """a learner's write reads (crf/d1_rollin.py): 0 <= rho <= R - 1, the final read writes what is left, monotone
    along every speaker's kappa order (rho may be below e: written ahead of the label)."""
    e = S.chunk_reads(rec)
    R = rec["n_read"]
    assert len(rho) == len(e) and all(0 <= x <= R - 1 for x in rho), rec["id"]
    for it in speaker_orders(rec, e).values():
        assert all(rho[a] <= rho[b] for a, b in zip(it, it[1:])), rec["id"]


def aug_requests_rho(rec: dict, rho: list[int], mode: str = "t2", keep: list[int] | None = None) -> list[dict]:
    """D1 (reports/mist_w3po_v1/thinker_dagger_d1_prereg_v1.zh-CN.md §1.3): aug_requests with the learner's own write
    reads rho in place of the perturbed reads (rho < e allowed). Per read r with an overdue item (e_i <= r - 1, not in
    H_r = {i : rho_i <= r - 1}): history kappa(H_r), target W*_r = {i not in H_r : e_i <= r} in kappa order. keep: only
    these reads (the manifest's 25 % cap); None = all."""
    from experiments.mist_w3po_v1.crf import latent_write_crf as LW

    check_rho(rec, rho)
    e = S.chunk_reads(rec)
    R, n = rec["n_read"], len(rec["chunks"])
    _, h = LW.bounds(rec, mode)
    ks = None if keep is None else set(keep)
    out = []
    for r in range(R):
        if ks is not None and r not in ks:
            continue
        H = [i for i in range(n) if rho[i] <= r - 1]
        Hs = set(H)
        O = [i for i in range(n) if i not in Hs and e[i] <= r - 1]
        if not O:
            continue
        Wst = [i for i in range(n) if i not in Hs and e[i] <= r]
        hist, items = V.kap(rec, e, H), V.kap(rec, e, Wst)
        out.append({"r": r, "hist": hist, "items": items, "hist_ids": S.ids_of(rec, hist),
                    "forced_ids": S.ids_of(rec, items), "lags": [r - e[i] for i in O],
                    "ahead": sum(1 for i in H if e[i] > r - 1),
                    "in_t2": all(i in Hs for i in range(n) if h[i] <= r - 1)})
    return out


def aug_rows_rho(wins: list, step_d1: dict, D: float) -> list:
    """T.Row objects of the D1 rows of one step; step_d1 {w.idx: (rho, keep)} (crf/d1_rollin.load_manifest). The rows
    are built and weighted exactly as aug_rows' (key ("d1", idx, r), aug=True)."""
    from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    rows = []
    for w in wins:
        x = step_d1.get(w.idx)
        if x is None:
            continue
        rho, keep = x
        for q in aug_requests_rho(w.rec, rho, keep=keep):
            seq = list(q["hist_ids"]) + list(q["forced_ids"])
            h0, m = len(q["hist_ids"]), len(q["forced_ids"])
            rw = T.Row(w, q["r"], len(seq), np.arange(h0, h0 + m + 1), seq=seq, end=w.end_ids[q["r"]],
                       key=("d1", w.idx, q["r"]), aug=True)
            ct, ce = np.zeros(m + 1), np.zeros(m + 1)
            ct[:m] = V2T.token_weights(w, q["forced_ids"]) / D
            ce[m] = 1.0 / D
            rw.c_tok, rw.c_end, rw.c_state = ct, ce, np.zeros(m + 1)
            rw.c_lab_tok, rw.c_lab_end = np.zeros(m + 1), np.zeros(m + 1)
            rows.append(rw)
    return rows


# ------------------------------------------------------------------------------------------------ manifest
def _manifest_job(args):
    steps, mode = args          # steps: [(s, [(idx, rec), ...])], lattice mode (only for the in-lattice statistic)
    out, st = [], Counter()
    lags = Counter()
    for s, items in steps:
        n_rows = n_tok = 0
        for idx, rec in items:
            dl = draw_delays(len(rec["chunks"]), s, idx)
            if dl is None:
                continue
            qs = aug_requests(rec, dl, mode)
            nt = sum(len(q["hist_ids"]) + len(q["forced_ids"]) + 1 for q in qs)
            out.append({"step": s, "idx": idx, "id": rec["id"], "delay": dl, "rows": [q["r"] for q in qs],
                        "tok": nt, "off_t2": sum(not q["in_t2"] for q in qs),
                        "final_rows": sum(q["r"] == rec["n_read"] - 1 for q in qs)})
            st["perturbed"] += 1
            st["delayed_items"] += sum(x > 0 for x in dl)
            st["items"] += len(dl)
            st["rows"] += len(qs)
            st["off_t2"] += sum(not q["in_t2"] for q in qs)
            for q in qs:
                lags[max(q["lags"])] += 1
            n_rows += len(qs)
            n_tok += nt
        st["windows"] += len(items)
        out.append({"step": s, "summary": True, "rows": n_rows, "tok": n_tok})
    return out, dict(st), dict(lags)


def build_manifest(a) -> None:
    import multiprocessing as mp

    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    wins = T.load_windows(a.sched, True)
    plan = T.plan_steps([w.R for w in wins], a.plan_seed, a.steps, a.rows_per_step)
    jobs, chunk = [], max(1, len(plan) // (a.workers * 4))
    for k in range(0, len(plan), chunk):
        jobs.append(([(s, [(i, wins[i].rec) for i in plan[s]]) for s in range(k, min(len(plan), k + chunk))], a.lw_mode))
    out.mkdir(parents=True)
    st, lags, per_step = Counter(), Counter(), {}
    with open(out / "dart.jsonl", "x", encoding="utf-8") as fh, mp.get_context("spawn").Pool(a.workers) as pool:
        for res, s1, l1 in pool.imap(_manifest_job, jobs):
            st.update(s1)
            lags.update({int(k): v for k, v in l1.items()})
            for r in res:
                if r.get("summary"):
                    per_step[r["step"]] = (r["rows"], r["tok"])
                else:
                    fh.write(json.dumps(r) + "\n")
    rows = np.array([per_step[s][0] for s in range(len(plan))])
    tok = np.array([per_step[s][1] for s in range(len(plan))])
    normal = np.array([sum(wins[i].R for i in s) for s in plan])
    summ = {"sched": a.sched, "plan": {"seed": a.plan_seed, "steps": a.steps, "rows_per_step": a.rows_per_step},
            "dart": {"seed": DART_SEED, "frac": FRAC, "p_item": P_ITEM, "kmax": KMAX}, "lw_mode": a.lw_mode,
            "windows_in_plan": len(wins), "stats": dict(st),
            "aug_rows_per_step": {"mean": float(rows.mean()), "p50": float(np.median(rows)),
                                  "p95": float(np.percentile(rows, 95)), "max": int(rows.max())},
            "aug_tokens_per_step": {"mean": float(tok.mean()), "p95": float(np.percentile(tok, 95)),
                                    "max": int(tok.max())},
            "read_rows_per_step_mean": float(normal.mean()),
            "aug_rows_over_read_rows": float(rows.sum() / normal.sum()),
            "off_t2_share": st["off_t2"] / max(1, st["rows"]),
            "max_overdue_lag_reads": {int(k): int(v) for k, v in sorted(lags.items())},
            "elapsed_s": round(time.time() - t0, 1)}
    with open(out / "summary.json", "x", encoding="utf-8") as fh:
        fh.write(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    print(f"DART_MANIFEST_DONE {out}", flush=True)


def manifest_mode(path: str) -> str:
    """the lattice mode the manifest's in-lattice statistic was computed for (manifests before --lw-mode: t2)."""
    return json.loads((Path(path) / "summary.json").read_text()).get("lw_mode", "t2")


def load_manifest(path: str, wins: list, plan: list[list[int]], sched: str, seed: int, steps: int,
                  rows_per_step: int) -> dict:
    """{step: {w.idx: delay}} from <path>/dart.jsonl; checks the manifest's sched / plan arguments, that every entry is a
    window of that step of the trainer's own plan with the same id, and that the draws of every plan (step, window)
    reproduce the manifest exactly (perturbed or not)."""
    root = Path(path)
    summ = json.loads((root / "summary.json").read_text())
    want = {"seed": seed, "steps": steps, "rows_per_step": rows_per_step}
    if summ["plan"] != want or Path(summ["sched"]).resolve() != Path(sched).resolve():
        raise SystemExit(f"--dart manifest made for {summ['sched']} {summ['plan']}, trainer has {sched} {want}")
    out: dict = {}
    for l in open(root / "dart.jsonl", encoding="utf-8"):
        r = json.loads(l)
        s, i = r["step"], r["idx"]
        if s >= len(plan) or i not in plan[s] or wins[i].rec["id"] != r["id"]:
            raise SystemExit(f"--dart manifest entry not in the plan: {r['step']} {r['id']}")
        out.setdefault(s, {})[i] = r["delay"]
    for s, ids in enumerate(plan):
        for i in ids:
            dl = draw_delays(len(wins[i].rec["chunks"]), s, i)
            if dl != out.get(s, {}).get(i):
                raise SystemExit(f"--dart manifest differs from the draw of step {s} window {wins[i].rec['id']}")
    return out


# ------------------------------------------------------------------------------------------------ 1.0 s sub-grid data
def subgrid_labels(tg: dict, a: np.ndarray, grid: int = GRID_G1000) -> tuple[dict, dict]:
    """(target on the sub-grid, labels dict as t0_agree_labels.run_labels writes) of one T0 train target tg and its
    per-read agreement counts a [R0, J] on the 0.5 s grid."""
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.crf import t0_agree_labels as T0

    t0 = list(tg["t_ms"])
    t1 = B.read_times(tg["dur_ms"], grid)
    pos = {t: k for k, t in enumerate(t0)}
    assert all(t in pos for t in t1) and t1[-1] == t0[-1], tg["id"]
    sub = [pos[t] for t in t1]
    assert a.shape == (len(t0), len(tg["pieces"])), tg["id"]
    tg1 = dict(tg, t_ms=t1)
    u = T0.prop_reads(tg1)
    e_raw, zero = T0.earliest_stable(a[sub], u)
    e = B.cummax_by_speaker(e_raw, [p["slot"] for p in tg["pieces"]])
    return tg1, {"id": tg["id"], "t_ms": t1, "u": u, "e_raw": e_raw.tolist(), "e": e,
                 "zero": zero.astype(int).tolist()}


def build_g1000(a) -> None:
    import multiprocessing as mp

    from experiments.mist_w3po_v1.crf import build_piece_sched as BPS
    from experiments.mist_w3po_v1.crf import t0_agree_labels as T0

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    root = Path(a.t0_root)
    scores, _ = T0.load_scores(root)
    tgs, labs, st = {}, {}, Counter()
    for l in open(root / "cpu" / "targets.jsonl", encoding="utf-8"):
        tg = json.loads(l)
        tg1, lb = subgrid_labels(tg, np.asarray(scores[tg["id"]]["a"], np.int64), a.grid)
        tgs[tg["id"]], labs[tg["id"]] = tg1, lb
        R = len(lb["t_ms"])
        st["windows"] += 1
        st["pieces"] += len(lb["e"])
        st["reads"] += R
        st["zero"] += sum(lb["zero"])
        st["final"] += sum(x == R - 1 for x in lb["e"])
        st["gt_u_plus1"] += sum(x > u + 1 for x, u in zip(lb["e"], lb["u"]))
    src = Path(T0.SRC_TRAIN)
    smap = json.loads((src / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    W = [w for w in map(json.loads, open(src / "windows_train_ref.jsonl", encoding="utf-8")) if w["id"] in tgs]
    W.sort(key=lambda w: w["id"])
    assert len(W) == len(tgs)
    out.mkdir(parents=True)
    with open(out / "labels_g1000.jsonl", "x", encoding="utf-8") as fh:
        for i in sorted(labs):
            fh.write(json.dumps(labs[i]) + "\n")
    shards = [W[i: i + a.shard_size] for i in range(0, len(W), a.shard_size)]
    jobs = [(k, sh, {w["id"]: tgs[w["id"]] for w in sh}, {w["id"]: labs[w["id"]] for w in sh},
             {m: smap[m] for m in {w["meeting"] for w in sh}}, "t0", str(out), "free") for k, sh in enumerate(shards)]
    bad = []
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        for i, (s, c) in enumerate(pool.imap_unordered(BPS._job, jobs)):
            bad += [k for k, v in c.items() if not v["ok"]]
            print(f"shard {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    (out / "labels_g1000_summary.json").write_text(json.dumps({**st, "grid_ms": a.grid, "check_bad": bad[:20],
                                                               "n_bad": len(bad)}, indent=1), encoding="utf-8")
    print(json.dumps({**st, "n_bad": len(bad)}), flush=True)
    if bad:
        raise SystemExit("G1000_CHECK_FAILED")
    print(f"G1000_SHARDS_DONE {out} (next: build_piece_sched --finalize)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-g1000", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--t0-root", default="eval_outputs/mist_w3po_v1/crf/t0_agree_train")
    ap.add_argument("--sched", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=GRID_G1000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--plan-seed", type=int, default=PLAN_SEED)
    ap.add_argument("--steps", type=int, default=PLAN_STEPS)
    ap.add_argument("--rows-per-step", type=int, default=PLAN_ROWS)
    ap.add_argument("--lw-mode", default="t2", choices=["t2", "t5", "v2p"],
                    help="lattice mode of the in-lattice statistic (v2p: VE1, thinker_v2p_evidence_ve1_prereg_v1)")
    a = ap.parse_args()
    if a.build_g1000:
        return build_g1000(a)
    if a.manifest:
        return build_manifest(a)
    raise SystemExit("--build-g1000 or --manifest")


if __name__ == "__main__":
    main()
