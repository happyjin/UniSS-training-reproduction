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
  w3dia     (2026-11-07) W3 routing / projected ratios with DiaScriber's quality reward and C4: what and when <- z(cpBLEU)
            of the trajectory (no latency term, no FLAD), who <- RBC (STAMP speaker counterfactual, as w3po)
  w3dia2    (2026-11-08) w3dia with the who channel rebuilt: who <- z(cpBLEU) of the trajectory (what dia gives speaker
            tokens) + who_sc x group-normalised per-segment speaker correctness (noC4's 1[pi*(spk) = speaker of the
            matched unit], reference labels) + who_rbc x group-normalised RBC; what / when as w3dia
  eagrpo    (2026-11-08) Event-Anchored GRPO: dia's z(cpBLEU) on every token + lam_ev x E(t), E = per-event group-relative
            credit. Every reference unit u of the window is an event shared by the G rollouts; per unit and rollout
            q = SQ_u, c = speaker correctness of u's segments (rollouts covering u), m = 1[covered]; each is normalised
            across the group per event (skipped when constant), so a 0.4 s interjection weighs as much as a long
            sentence. Routing: what tokens of u's segments <- e(q); who tokens <- e(c); when tokens <- e(q) (a segment's
            opening token; the <wait> of a read committing a segment of u) and, for a missed u, the <wait> of reads in
            [u.end, u.end + 2 s] <- e(m); unmatched / duplicate segments' what and when tokens <- -z(count). No latency
            term. Same token-level loss as dia (not a W3 mode), so lam_ev = 0 is dia exactly.
            ev_norm (2026-11-08 amendment): "center" (default) e = x - mean_g(x), x in [0, 1] (no division by the
            group std, which blew tiny differences up to unit scale in W3PO), bad-segment term = -(count/|U| - mean),
            and E(t) clipped to [-ev_clip, ev_clip] (default 1); "std" = the first version (std division, no clip).
            ev_clip_scaled (off by default): the scaled event term lam_ev x E is clipped to +-sqrt(G - 1), the largest
            possible |z(cpBLEU)| of a group of G, so no token's event term exceeds the backbone's range.
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
W3_MODES = ("w3po", "noC1", "noC3", "noC4", "w3dia", "w3dia2")


def categories(before: list[int], ids: list[int], final: bool) -> list[tuple[bool, bool]]:
    st = seg_state(before)
    out = []
    for i in ids:
        out.append(((st == "out") and not final, SPK0 <= i < NEW_END))
        st = step_state(st, i)
    return out


def admit_anchors(policy_R: list[float], anchor_R: list[float], rule: str = "minpol") -> list[bool]:
    """Which FLAD anchors join the group as LUFFY guidance (2026-11-07 pilot audit: the Lag1 anchor sometimes falls
    into a fragment loop ("the, | the, | ...") whose spurious penalties give R down to -32; inside the group it
    inflates every normalising std and shrinks the policy members' advantages, while teaching nothing).
      none     every anchor joins (the pilot's steps 1-13)
      minpol   an anchor joins only if its R is at least the lowest policy R of the group
    Anchors always stay FLAD latency references (a degenerate anchor never dominates, so it is inert there)."""
    if rule == "none" or not policy_R:
        return [True] * len(anchor_R)
    if rule == "minpol":
        lo = min(policy_R)
        return [a >= lo for a in anchor_R]
    raise ValueError(rule)


def rep_score(segs: list, n: int = 4) -> float:
    """Repetition score of a trajectory (2026-11-08, MMDuet2-style r_rep / r_pfx): the share of the word n-grams of its
    committed text (segments in commit order) that repeat an earlier n-gram of the same trajectory. Catches loops
    ("with the with the ...") and re-translation of already committed content; 0 for a clean output."""
    words = " ".join(s["text"] for s in segs).lower().split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


def dup_segments(segs: list, n: int = 4) -> set:
    """eagrpo: segments (commit order) that repeat earlier committed text of the same rollout: >= half of their word
    n-grams seen before, or (fewer than n words) the same normalised text as an earlier segment."""
    out, seen, texts = set(), set(), set()
    for j, s in enumerate(segs):
        words = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s["text"]).split()
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if grams:
            if sum(g in seen for g in grams) * 2 >= len(grams):
                out.add(j)
        elif words and " ".join(words) in texts:
            out.add(j)
        seen.update(grams)
        texts.add(" ".join(words))
    return out


def event_norm(x: list, ok: list | None = None, center: bool = False) -> list:
    """eagrpo: (x - mean) / max(std, FLOOR) over the rollouts with ok (all by default); 0 elsewhere, and everywhere
    when fewer than two such rollouts or the values are constant. center: x - mean only (no std division)."""
    ok = [True] * len(x) if ok is None else ok
    v = np.array([a for a, k in zip(x, ok) if k], dtype=float)
    if len(v) < 2 or v.std() <= 1e-9:
        return [0.0] * len(x)
    mu, sd = v.mean(), (1.0 if center else max(v.std(), FLOOR))
    return [float((a - mu) / sd) if k else 0.0 for a, k in zip(x, ok)]


EV_MISS_S = 2.0


def eagrpo_advantages(trajs: list, credits: list, units: list, dur: float, q_penalty: list | None = None,
                      lam_ev: float = 1.0, ev_norm: str = "center", ev_clip: float = 1.0,
                      ev_clip_scaled: bool = False) -> dict:
    """--mode eagrpo (see the module docstring). Returns {"tok", "stats"} like group_advantages; the advantage of every
    token is in the "what" slot (as dia), the when / who flags are kept for the record."""
    n, U = len(trajs), len(units)
    R = np.array([c.R for c in credits])
    q = np.array([S.seq_cpbleu(tr.segs, units) for tr in trajs])
    if q_penalty is not None:
        q = q - np.asarray(q_penalty, dtype=float)
    zq = z(q)
    stats = {"R": R.tolist(), "lat": [c.lat for c in credits]}
    cats = [[categories(rd.before, rd.ids, rd.final) for rd in tr.reads] for tr in trajs]
    segs_of = [[[j for j, uu in enumerate(c.unit_of) if uu == u] for u in range(U)] for c in credits]
    bad = [dup_segments(tr.segs) | {j for j, uu in enumerate(c.unit_of) if uu < 0} for tr, c in zip(trajs, credits)]
    cov = [[1.0 if any(j not in bad[g] for j in segs_of[g][u]) else 0.0 for u in range(U)] for g in range(n)]
    assert ev_norm in ("center", "std"), ev_norm
    ctr = ev_norm == "center"
    pis = [S.speaker_map(tr.segs, units) for tr in trajs]
    e_q, e_c, e_m = [], [], []
    for u in range(U):
        e_q.append(event_norm([float(credits[g].SQ[u]) for g in range(n)], center=ctr))
        e_m.append(event_norm([cov[g][u] for g in range(n)], center=ctr))
        corr = []
        for g, tr in enumerate(trajs):
            js = [j for j in segs_of[g][u] if j not in bad[g]]
            corr.append(float(np.mean([pis[g].get(tr.segs[j]["spk"]) == units[u]["spk"] for j in js])) if js else 0.0)
        e_c.append(event_norm(corr, [cov[g][u] > 0 for g in range(n)], center=ctr))
    if ctr:
        e_bad = [-v for v in event_norm([len(b) / max(U, 1) for b in bad], center=True)]
    else:
        e_bad = [-v for v in event_norm([float(len(b)) for b in bad])]
    tok, mag = [], {"when": [0.0, 0], "who": [0.0, 0], "what": [0.0, 0]}
    tot = [0.0, 0.0, 0]   # sum |E(t)| (after the clip), sum |z(cpBLEU)|, tokens
    bound = float(np.sqrt(max(n - 1, 1)))   # --ev-clip-scaled: max |z| of a group of n
    n_hit = 0
    for g, tr in enumerate(trajs):
        c = credits[g]
        wait_ev = {}   # read -> event values routed to its <wait> / end-of-read when token
        for u in range(U):
            for j in segs_of[g][u]:
                if j not in bad[g] and e_q[u][g]:
                    wait_ev.setdefault(tr.segs[j]["r"], []).append(e_q[u][g])
            if not cov[g][u] and e_m[u][g]:
                b = units[u]["b"]
                rs = [r for r, rd in enumerate(tr.reads) if not rd.final and b <= rd.t_ms / 1000 <= b + EV_MISS_S]
                if not rs:
                    rs = [r for r, rd in enumerate(tr.reads) if not rd.final and rd.t_ms / 1000 >= b]
                for r in rs:
                    wait_ev.setdefault(r, []).append(e_m[u][g])
        per_read = []
        for r, rd in enumerate(tr.reads):
            row = []
            for p, (wh, who) in enumerate(cats[g][r]):
                sj = tr.tok_seg[r][p]
                E_w = E_h = E_t = 0.0
                if sj >= 0:
                    if sj in bad[g]:
                        E_w = e_bad[g] if wh else 0.0
                        E_t = e_bad[g] if not (wh or who) else 0.0
                    else:
                        u = c.unit_of[sj]
                        E_w = e_q[u][g] if wh else 0.0
                        E_h = e_c[u][g] if who else 0.0
                        E_t = e_q[u][g] if not (wh or who) else 0.0
                elif wh and wait_ev.get(r):
                    E_w = float(np.mean(wait_ev[r]))
                for k, (on, v) in (("when", (wh, E_w)), ("who", (who, E_h)), ("what", (not (wh or who), E_t))):
                    if on:
                        mag[k][0] += abs(v)
                        mag[k][1] += 1
                E = E_w + E_h + E_t
                if ctr and ev_clip > 0:
                    E = min(max(E, -ev_clip), ev_clip)
                tot[0] += abs(E)
                tot[1] += abs(float(zq[g]))
                tot[2] += 1
                if ev_clip_scaled:
                    t = lam_ev * E
                    n_hit += abs(t) > bound
                    row.append((wh, who, 0.0, 0.0, float(zq[g]) + min(max(t, -bound), bound)))
                else:
                    row.append((wh, who, 0.0, 0.0, float(zq[g]) + lam_ev * E))
            per_read.append(row)
        tok.append(per_read)
    short = [u for u in range(U) if units[u]["b"] - units[u]["a"] <= 1.0]
    stats.update(acr_when=None, acr_who=None, acr_what=bool(np.all(zq == 0)) and not any(
        v for e in (e_q, e_c, e_m) for x in e for v in x),
        ev={"units": U, "units_nz": sum(1 for u in range(U) if any(e_q[u]) or any(e_c[u]) or any(e_m[u])),
            "abs_when": mag["when"][0], "n_when": mag["when"][1], "abs_who": mag["who"][0], "n_who": mag["who"][1],
            "abs_what": mag["what"][0], "n_what": mag["what"][1], "short": len(short) * n,
            "short_cov": sum(cov[g][u] for g in range(n) for u in short),
            "abs_E": tot[0], "abs_z": tot[1], "n_tok": tot[2], "n_scaled_clip": n_hit})
    return {"tok": tok, "stats": stats}


def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / max(sd, FLOOR) if sd > 1e-9 else np.zeros_like(x)


def group_advantages(mode: str, trajs: list, credits: list, units: list, dur: float, anchor_ql: list,
                     q_penalty: list | None = None, who_sc: float = 0.5, who_rbc: float = 0.0, lam_ev: float = 1.0,
                     ev_norm: str = "center", ev_clip: float = 1.0, ev_clip_scaled: bool = False) -> dict:
    """trajs/credits: the group members (policy rollouts, then anchors when they are in the group).
    anchor_ql: [(Q_a, L_a)] of the anchors for FLAD. Returns {"tok": per member per read [(when, who, aw, ah, at)],
    "stats": {...}}. who_sc / who_rbc: weights of the w3dia2 who channel (ignored by every other mode); lam_ev: weight
    of the eagrpo event term (ignored by every other mode)."""
    if mode == "eagrpo":
        return eagrpo_advantages(trajs, credits, units, dur, q_penalty=q_penalty, lam_ev=lam_ev, ev_norm=ev_norm,
                                 ev_clip=ev_clip, ev_clip_scaled=ev_clip_scaled)
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
        if mode in ("noC1", "w3dia", "w3dia2"):
            Q = np.array([S.seq_cpbleu(tr.segs, units) for tr in trajs])
            if q_penalty is not None:     # --rep-penalty (default None: unchanged)
                Q = Q - np.asarray(q_penalty, dtype=float)
        else:
            Q = R
        if mode == "noC3":
            rlat = np.array([S.hpo_gate_reward(q, l) for q, l in zip(Q, Lat)])
        elif mode in ("w3dia", "w3dia2"):
            rlat = np.zeros(n)     # no latency reward: when tokens carry the quality advantage only (below)
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
        if mode in ("w3dia", "w3dia2"):     # terminal quality reward, same at every read (reward-to-go of a terminal reward)
            zq = z(Q)
            for m, tr in enumerate(trajs):
                aw[m, : len(tr.reads)] = zq[m]
            acr_when = bool(np.all(zq == 0))
        for r in (range(n_reads) if mode not in ("w3dia", "w3dia2") else []):
            col = ret[:, r]
            ok = ~np.isnan(col)
            if ok.sum() >= 2 and col[ok].std() > 1e-9:
                acr_when = False
                aw[ok, r] = (col[ok] - col[ok].mean()) / max(col[ok].std(), FLOOR)
        # what
        if mode in ("noC1", "w3dia", "w3dia2"):
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
        if mode in ("noC4", "w3dia2"):
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
        if mode == "w3dia2":     # seg_who is now the normalised speaker correctness (noC4)
            rbc = [[0.0] * len(tr.segs) for tr in trajs]
            if who_rbc:
                xr = [list(c.who) for c in credits]
                fr = np.array([v for x in xr for v in x])
                if len(fr) > 1 and fr.std() > 1e-9:
                    rbc = [[(v - fr.mean()) / max(fr.std(), FLOOR) for v in x] for x in xr]
            seg_who = [[float(at_traj[m]) + who_sc * seg_who[m][j] + who_rbc * rbc[m][j] for j in range(len(tr.segs))]
                       for m, tr in enumerate(trajs)]
            acr_who = bool(all(abs(v) < 1e-12 for x in seg_who for v in x))
        stats.update(acr_when=acr_when, acr_who=acr_who, acr_what=acr_what)

        def fn(m, r, p, w, h, sj):
            a_w = float(aw[m, r]) if w else 0.0
            a_h = float(seg_who[m][sj]) if (h and sj >= 0) else 0.0
            if mode == "w3dia2" and h and sj < 0:     # speaker token outside a segment: the trajectory's z(cpBLEU)
                a_h = float(at_traj[m])
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
        q = np.array([S.seq_cpbleu(tr.segs, units) for tr in trajs])
        if q_penalty is not None:     # --rep-penalty (default None: unchanged)
            q = q - np.asarray(q_penalty, dtype=float)
        A = z(q)
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


# --- ECF-GRPO (2026-11-09, --ecf-branches K with --mode dia; K = 0: never called, dia unchanged) -----------------
ECF_SHORT_S = 1.0


def ecf_specs(trajs: list, credits: list, units: list, rng, cap: int, short_s: float = ECF_SHORT_S) -> list:
    """Counterfactual branch points of one group: for every member, the short reference units (<= short_s) it never
    covers (STAMP unit_of) whose first read at or after the unit's end is non-final and ends with <wait> (the member
    heard the whole unit and kept waiting). One (read, unit) per member, at most cap members; rng: random.Random.
    Returns [(member, read, unit)]."""
    from experiments.mist_w3po_v1.core.engine import WAIT_ID
    per = []
    for m, (tr, c) in enumerate(zip(trajs, credits)):
        cand = []
        for i, u in enumerate(units):
            if u["b"] - u["a"] > short_s or any(x == i for x in c.unit_of):
                continue
            r = next((r for r, rd in enumerate(tr.reads) if rd.t_ms / 1000 >= u["b"]), None)
            if r is not None and not tr.reads[r].final and tr.reads[r].ids and tr.reads[r].ids[-1] == WAIT_ID:
                cand.append((r, i))
        if cand:
            r, i = cand[rng.randrange(len(cand))]
            per.append((m, r, i))
    if len(per) > cap:
        per = [per[k] for k in sorted(rng.sample(range(len(per)), cap))]
    return per


def ecf_apply(tok: list, q, specs: list, q_branch: list, lam: float = 1.0) -> dict:
    """Adds lam * clip((q_parent - q_branch) / sigma_q, +-sqrt(G - 1)) to the advantage (the "what" slot, as dia) of
    the parent's terminal <wait> token at the branch read; sigma_q = max(std_g q, FLOOR) (the backbone's z scale).
    tok: group_advantages(...)["tok"] (rows are replaced, not mutated). Returns stats."""
    q = np.asarray(q, dtype=float)
    sig = max(float(q.std()), FLOOR)
    lim = float(np.sqrt(max(len(q) - 1, 1)))
    st = {"n": 0, "better": 0, "dq": 0.0, "abs_extra": 0.0}
    for (m, r, _), qb in zip(specs, q_branch):
        ex = lam * float(np.clip((q[m] - qb) / sig, -lim, lim))
        row = list(tok[m][r])
        wh, who, aw, ah, at = row[-1]
        row[-1] = (wh, who, aw, ah, at + ex)
        tok[m][r] = row
        st["n"] += 1
        st["better"] += int(qb > q[m])
        st["dq"] += float(qb - q[m])
        st["abs_extra"] += abs(ex)
    return st


# --- HiCA-GRPO (2026-11-09, --hica-ms D with --mode dia; D = 0: never called, dia unchanged) -------------------
def hica_weights(adv, delta, apply, clip: float = 1.0):
    """Hindsight credit allocation (RLSD-style, arXiv 2604.03128, with a hindsight-privileged self-teacher).
    adv: per-token advantage (torch tensor), delta: log pi_ref(y | audio <= t + D) - log pi_ref(y | audio <= t) per
    token, apply: bool mask of the re-weighted tokens (content and who; never pure when / anchors).
    Returns (weights, clipped mask): w = exp(clip(sign(adv) * delta, +-clip)) where apply, else 1. The sign of the
    update stays the sign of adv; only its per-token magnitude moves."""
    x = adv.sign() * delta
    clipped = apply & (x.abs() > clip)
    w = x.clamp(-clip, clip).exp()
    w = w.where(apply, w.new_ones(()))
    return w, clipped
