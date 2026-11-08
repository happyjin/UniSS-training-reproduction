"""CPU pre-check of the EA-GRPO event term before launch (2026-11-08 amendment, item 4). Groups = existing AMI dev
decodes of the same online windows by 8 systems (v8 and RL checkpoints, guarded protocol; a diagnostic like
diag_reward_noise, nothing is trained or tuned on it). Per group: STAMP credits (rl/stamp.credit, as in training) and
advantage.eagrpo_advantages; E(t) = A(t) - z(cpBLEU) is the event part of each token's advantage.
  (a) corr(E_when, commit earliness) and corr(E_when, segments per rollout), group-centred and pooled: the miss credit
      must not reward committing earlier or emitting more segments (W3PO's drift); want <= ~0;
  (b) sign agreement: per (unit, rollout) with e_q != 0, sign of the mean E over the tokens of that rollout's segments
      of the unit vs sign of the unit quality change SQ - mean_g SQ;
  (c) mean |E| / mean |z(cpBLEU)| over tokens (want <= 1);
for --ev-norm center (clip 1, the launch setting) and std (the first version) side by side.

    python -m experiments.mist_w3po_v1.rl.precheck_eagrpo --out reports/mist_w3po_v1/precheck_eagrpo.json [--n 400]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

D = "eval_outputs/mist_w3po_v1/decode"
SYSTEMS = ["dev_v8_ctx0_guard_s8", "dev_v8_ctx0_guard_s4", "dev_v8_ctx2_guard_s8", "dev_pilot50_dia_w128_ctx0_guard_s8",
           "dev_pilot100_dia_w128_ctx0_guard_s8", "dev_pilot50_w3dia_w128_ctx0_guard_s8",
           "dev_pilot125_diahard_ctx0_guard_s8", "dev_pilot75_w3dia_w128_ctx0_guard_s8"]


def one_group(job):
    from experiments.mist_w3po_v1.rl import advantage as A
    from experiments.mist_w3po_v1.rl import stamp as S
    w, utts, trajs = job
    units = S.units_of(utts)
    dur = w["dur_ms"] / 1000
    post = np.load(w["post"]["lowlat"]).astype(np.float32)
    creds = [S.credit(t.segs, units, dur, t.wait_reads, [rd.t_ms / 1000 for rd in t.reads], post=post) for t in trajs]
    zq = A.z(np.array([S.seq_cpbleu(t.segs, units) for t in trajs]))
    out = {}
    for norm in ("center", "std"):
        adv = A.eagrpo_advantages(trajs, creds, units, dur, ev_norm=norm, ev_clip=1.0 if norm == "center" else 0.0)
        ew, segs, early, absE, absz = [], [], [], 0.0, 0.0
        agree = [0, 0]
        for g, tr in enumerate(trajs):
            Ew = [x[4] - zq[g] for row in adv["tok"][g] for x in row if x[0]]
            for row in adv["tok"][g]:
                for x in row:
                    absE += abs(x[4] - zq[g])
                    absz += abs(zq[g])
            ew.append(float(np.mean(Ew)) if Ew else 0.0)
            segs.append(len(tr.segs))
            early.append(-float(np.mean([s["t"] - s["b"] for s in tr.segs])) if tr.segs else np.nan)
        sq = np.array([[c.SQ[u] for u in range(len(units))] for c in creds]) if units else np.zeros((len(trajs), 0))
        for u in range(len(units)):
            dq = sq[:, u] - sq[:, u].mean()
            for g, tr in enumerate(trajs):
                if abs(dq[g]) < 1e-9:
                    continue
                vals = [x[4] - zq[g] for r, row in enumerate(adv["tok"][g]) for p, x in enumerate(row)
                        if tr.tok_seg[r][p] >= 0 and creds[g].unit_of[tr.tok_seg[r][p]] == u]
                if vals and abs(np.mean(vals)) > 1e-12:
                    agree[0] += np.sign(np.mean(vals)) == np.sign(dq[g])
                    agree[1] += 1
        out[norm] = {"ew": ew, "segs": segs, "early": early, "absE": absE, "absz": absz, "agree": agree}
    return out


def pooled_corr(groups, key_x, key_y):
    xs, ys = [], []
    for gr in groups:
        x, y = np.array(gr[key_x], dtype=float), np.array(gr[key_y], dtype=float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 2:
            continue
        xs += list(x[ok] - x[ok].mean())
        ys += list(y[ok] - y[ok].mean())
    xs, ys = np.array(xs), np.array(ys)
    r = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else 0.0
    return {"r": r, "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    from experiments.mist_w3po_v1.core.engine import Prompter
    from experiments.mist_w3po_v1.evaluation.diag_reward_noise import WIN, ref_utts, window_utts
    from experiments.mist_w3po_v1.rl.test_pipeline_cpu import traj_from
    tok = Prompter("zh2en").tok
    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    by = ref_utts()
    H = [{r["id"]: r for r in map(json.loads, open(f"{D}/{s}/hyp.jsonl", encoding="utf-8"))} for s in SYSTEMS]
    cand = [w for w in W if window_utts(w, by) and all(h.get(w["id"], {}).get("read_log") for h in H)]
    pick = [cand[round(i * (len(cand) - 1) / (a.n - 1))] for i in range(min(a.n, len(cand)))]
    jobs = []
    for w in pick:
        trajs = [traj_from(h[w["id"]], tok, w) for h in H]
        jobs.append((w, window_utts(w, by), trajs))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_group, jobs, chunksize=4))
    rep = {"systems": SYSTEMS, "groups": len(res), "windows_with_units": len(cand)}
    for norm in ("center", "std"):
        gs = [r[norm] for r in res]
        ag = np.array([g["agree"] for g in gs]).sum(0)
        rep[norm] = {"corr_Ewhen_earliness": pooled_corr(gs, "ew", "early"), "corr_Ewhen_segs": pooled_corr(gs, "ew", "segs"),
                     "sign_agreement": float(ag[0] / max(ag[1], 1)), "sign_n": int(ag[1]),
                     "E_over_z": float(sum(g["absE"] for g in gs) / max(sum(g["absz"] for g in gs), 1e-9))}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
