"""Per-token advantages for one group (window) of trajectories (plan §3.3 W3, §6.1 baselines, §6.2 ablations).

Token categories (core/engine.py): when = positions between segments of a non-final read; who = speaker tokens;
what = the rest. A token may be when and who at once (a speaker token opening a block).

modes
  w3po      C1-C4: what <- LOO quality credit with a per-unit group baseline; who <- RBC; when <- FLAD latency reward +
            miss credit, reward-to-go, normalised per read index across the group (Hibiki-Zero's checkpoints)
  noC1      STAMP replaced by the window's sequence-level cpBLEU (what and who: z(cpBLEU); when: FLAD on cpBLEU)
  noC2      one advantage z(R + LAT_W r_lat / L_MAX) on every token (no routing, no projection)
  noC3      FLAD replaced by HPO's absolute gate (no anchors)
  noC4      RBC replaced by a group-mean baseline on speaker correctness (1[pi*(k) = speaker of the matched unit])
  seqpo     (a) SeqPO: z(q) - z(max(m, L)) on every token, q = window quality
  hibiki    (b) Hibiki-Zero: per read r_r = (1 - a) prefixBLEU_r + a BLEU_final, z per read, reward-to-go
  hpo       (c) HPO: 0.5 z(mean unit quality) + 0.5 z(-gated lag) on every token
  dia       (d) DiaScriber: z(cpBLEU) on every token
Returns per trajectory, per read: list of (when?, who?, a_when, a_who, a_what, seg index) per token, plus group stats.
"""
from __future__ import annotations

import numpy as np

from experiments.mist_w3po_v1.core.engine import NEW_END, SPK0, seg_state, step_state
from experiments.mist_w3po_v1.rl import stamp as S

FLOOR = 0.02
LAT_W = 0.3
HIBIKI_ALPHA = 0.4
SEQPO_M = 1.0
W3_MODES = ("w3po", "noC1", "noC3", "noC4")


def categories(before: list[int], ids: list[int], final: bool) -> list[tuple[bool, bool]]:
    st = seg_state(before)
    out = []
    for i in ids:
        out.append(((st == "out") and not final, SPK0 <= i < NEW_END))
        st = step_state(st, i)
    return out


def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / max(sd, FLOOR) if sd > 1e-9 else np.zeros_like(x)


def group_advantages(mode: str, trajs: list, credits: list, units: list, dur: float, anchor_ql: list) -> dict:
    """trajs/credits: the group members (policy rollouts, then anchors when they are in the group).
    anchor_ql: [(Q_a, L_a)] of the anchors for FLAD. Returns {"tok": per member per read [(when, who, aw, ah, at)],
    "stats": {...}}."""
    n = len(trajs)
    R = np.array([c.R for c in credits])
    Lat = np.array([c.lat for c in credits])
    stats = {"R": R.tolist(), "lat": Lat.tolist()}
    cats = [[categories(rd.before, rd.ids, rd.final) for rd in tr.reads] for tr in trajs]

    def fill(fn):
        """fn(member, read, pos, when, who, seg) -> (aw, ah, at)"""
        return [[[(c[0], c[1], *fn(m, r, p, c[0], c[1], tr.tok_seg[r][p])) for p, c in enumerate(cats[m][r])]
                 for r in range(len(tr.reads))] for m, tr in enumerate(trajs)]

    if mode in W3_MODES:
        if mode == "noC1":
            Q = np.array([S.seq_cpbleu(tr.segs, units) for tr in trajs])
        else:
            Q = R
        if mode == "noC3":
            rlat = np.array([S.hpo_gate_reward(q, l) for q, l in zip(Q, Lat)])
        else:
            rlat = np.array([S.flad_reward(q, l, anchor_ql) for q, l in zip(Q, Lat)])
        stats["r_lat"] = rlat.tolist()
        # when: reward-to-go of r_lat / L_MAX + miss credit, normalised per read index
        n_reads = max(len(tr.reads) for tr in trajs)
        ret = np.full((n, n_reads), np.nan)
        for m, (tr, c) in enumerate(zip(trajs, credits)):
            for r in range(len(tr.reads)):
                ret[m, r] = rlat[m] / S.L_MAX + sum(v for rr, v in c.miss.items() if rr >= r)
        aw = np.zeros_like(ret)
        acr_when = True
        for r in range(n_reads):
            col = ret[:, r]
            ok = ~np.isnan(col)
            if ok.sum() >= 2 and col[ok].std() > 1e-9:
                acr_when = False
                aw[ok, r] = (col[ok] - col[ok].mean()) / max(col[ok].std(), FLOOR)
        # what
        if mode == "noC1":
            at_traj = z(Q)
            seg_what = [[at_traj[m]] * len(tr.segs) for m, tr in enumerate(trajs)]
            acr_what = bool(np.all(at_traj == 0))
        else:
            n_u = max(len(units), 1)
            sq_mean = np.array([np.mean([c.SQ[i] for c in credits]) for i in range(len(units))]) if units else np.zeros(0)
            xs = []
            for tr, c in zip(trajs, credits):
                lens = {}
                for j, s in enumerate(tr.segs):
                    u = c.unit_of[j]
                    if u >= 0:
                        lens[u] = lens.get(u, 0) + max(len(s["text"]), 1)
                x = []
                for j, s in enumerate(tr.segs):
                    u = c.unit_of[j]
                    beta = sq_mean[u] / n_u * max(len(s["text"]), 1) / lens[u] if u >= 0 else 0.0
                    x.append(c.delta_q[j] - beta)
                xs.append(x)
            flat = np.array([v for x in xs for v in x])
            sd = max(flat.std(), FLOOR) if len(flat) > 1 else 1.0
            seg_what = [[v / sd for v in x] for x in xs]
            acr_what = bool(len(flat) == 0 or np.all(np.abs(flat) < 1e-9))
            at_traj = z(np.array([c.RQ for c in credits]))
        # who
        if mode == "noC4":
            xs = []
            for tr, c in zip(trajs, credits):
                pi = S.speaker_map(tr.segs, units)
                xs.append([float(c.unit_of[j] >= 0 and pi.get(s["spk"]) == units[c.unit_of[j]]["spk"])
                           for j, s in enumerate(tr.segs)])
        elif mode == "noC1":
            xs = [[float(Q[m])] * len(tr.segs) for m, tr in enumerate(trajs)]
        else:
            xs = [list(c.who) for c in credits]
        flat = np.array([v for x in xs for v in x])
        if len(flat) > 1 and flat.std() > 1e-9:
            mu, sd = flat.mean(), max(flat.std(), FLOOR)
            seg_who = [[(v - mu) / sd for v in x] for x in xs]
            acr_who = False
        else:
            seg_who = [[0.0] * len(x) for x in xs]
            acr_who = True
        stats.update(acr_when=acr_when, acr_who=acr_who, acr_what=acr_what)

        def fn(m, r, p, w, h, sj):
            a_w = float(aw[m, r]) if w else 0.0
            a_h = float(seg_who[m][sj]) if (h and sj >= 0) else 0.0
            a_t = 0.0
            if not h:
                a_t = float(seg_what[m][sj]) if sj >= 0 else float(at_traj[m])
            return a_w, a_h, a_t
        return {"tok": fill(fn), "stats": stats}

    # single-advantage modes: the same value on every token of a read (stored in the "what" slot)
    if mode == "noC2":
        rlat = np.array([S.flad_reward(q, l, anchor_ql) for q, l in zip(R, Lat)])
        A = z(R + LAT_W * rlat / S.L_MAX)
        per_read = [[A[m]] * len(tr.reads) for m, tr in enumerate(trajs)]
    elif mode == "seqpo":
        q = np.array([S.window_quality(tr.segs, units) for tr in trajs])
        A = z(q) - z(np.maximum(SEQPO_M, Lat))
        per_read = [[A[m]] * len(tr.reads) for m, tr in enumerate(trajs)]
    elif mode == "dia":
        A = z(np.array([S.seq_cpbleu(tr.segs, units) for tr in trajs]))
        per_read = [[A[m]] * len(tr.reads) for m, tr in enumerate(trajs)]
    elif mode == "hpo":
        qs, ls = [], []
        for tr, c in zip(trajs, credits):
            qu = c.SQ if units else [0.0]
            qs.append(float(np.mean(qu)))
            lags = []
            for i, u in enumerate(units):
                ts = [tr.segs[j]["t"] for j in range(len(tr.segs)) if c.unit_of[j] == i]
                lags.append(max(ts) - u["b"] if ts and c.SQ[i] >= 0.3 else S.L_MAX)
            ls.append(float(np.mean(lags)) if lags else S.L_MAX)
        A = 0.5 * z(np.array(qs)) + 0.5 * z(-np.array(ls))
        per_read = [[A[m]] * len(tr.reads) for m, tr in enumerate(trajs)]
    elif mode == "hibiki":
        n_reads = max(len(tr.reads) for tr in trajs)
        rr = np.full((n, n_reads), np.nan)
        for m, tr in enumerate(trajs):
            fin = S.prefix_bleu(tr.segs, units, len(tr.reads), 1e9)
            for r, rd in enumerate(tr.reads):
                rr[m, r] = (1 - HIBIKI_ALPHA) * S.prefix_bleu(tr.segs, units, r, rd.t_ms / 1000) + HIBIKI_ALPHA * fin
        zz = np.zeros_like(rr)
        for r in range(n_reads):
            ok = ~np.isnan(rr[:, r])
            if ok.sum() >= 2:
                zz[ok, r] = z(rr[ok, r])
        per_read = [[float(np.nansum(zz[m, r:len(tr.reads)])) for r in range(len(tr.reads))] for m, tr in enumerate(trajs)]
        A = np.array([v[0] for v in per_read])
    else:
        raise ValueError(mode)
    stats.update(acr_when=None, acr_who=None, acr_what=bool(np.all(np.abs(np.array([v for x in per_read for v in x])) < 1e-9)))
    return {"tok": fill(lambda m, r, p, w, h, sj: (0.0, 0.0, float(per_read[m][r]))), "stats": stats}
