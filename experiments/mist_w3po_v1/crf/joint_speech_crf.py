"""T3: the joint CRF of the Thinker's write schedule and the talker's playback ("scheme 3";
reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T3, aprime_plem_crf_formal_and_latency.zh-CN.md §1.3).

Factor graph (one training window, reads r = 0..R-1, one output stream per speaker k)
  Thinker   T2's chain (crf/latent_write_crf.py, mode "t2", imported unchanged): set state A_r, order o_r, edge term
            E_r(A' -> A) = logsumexp_o tau psi^T_{r,o} (tau = 0.5).
  talker    for every speaker k whose block B_{k,r} (its pieces in W_r, kap order) is non-empty: a rate class
            v_{k,r} in V = PL-EM's candidate offsets (0 = the A' queue-rule speed, +2 = two steps faster).
  playback  per speaker the queue end E_k on the absolute 80 ms grid (PL-EM's Delta), backlog b_k(r) = max(0, E_k - t_r):
              b'_k = max(0, b_k(r-1) - (t_r - t_{r-1}))       the queue left when read r writes
              E_k <- nearest_Delta(t_r + b'_k + D(B_{k,r}, v))  only when a block is added (no rounding on empty reads,
                                                               so no drift), b capped at 8 s; availability a = t_r (A3)
  factors   log pi(v_{k,r} | bin(b'_k))  (A2; 0.5 s bins, 8 bins)
            - lam * l_{k,r},  l = sum_{i in B} W_i (t_r - src_end_i) / 1000 + W_B b'_k / 1000 + (W_B + 1) / 2 * D / 1000
            (seconds x words: each word's "spoken" time minus its piece's source end; the words of a block spread evenly
            over its playback, as speech_latency_lit / PL-EM's l_n; the first term is T2's text term)
  posterior q_lam(z, v) ∝ exp(sum_r [E-terms of the Thinker edge and order] + sum_k [log pi - lam l]).

Exact inference: the augmented chain over (A_r, b-vector) is a linear chain with deterministic transitions given
(A_{r-1} -> A_r, v-vector); its forward-backward is the exact sum-product of q_lam. States are the reachable ones only
(hashed per read); the edge arrays (src, dst, Thinker edge, lam-free potential, l) are built once per window, so every
lam of the bisection re-runs only the vectorised sum-product. A5: a window whose augmented states exceed K_CAP at a read
is rebuilt on a coarser grid (2 Delta, 4 Delta, ...; still exact on that grid; counted).
Thinker edge-order marginals = the augmented edge marginals summed over (b, v), times the order conditional
exp(tau psi_o - E_order) (T2's), -> cend -> serial_v2p_train.set_coefs (the M-step is T2's / v2''s).

lam: per optimiser step, sum_w E_{q_lam,w}[l] = sum_w E_{q_T1,w}[l] where q_T1 is the same joint model restricted to T1's
Thinker sub-lattice (states with c_k >= #{e_i <= r - 1}), lam = 0; 0 when not binding; T2's bracket / bisection.

Talker model (A1, A2; --fit-talker, CPU, train only): PL-EM's E step on the train candidates of
speech/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands (talker_rows.lattice: the per-stream playback lattice,
lam = the PL-EM budget's): per block the conditional q(c | b_n = the A' path's state); q(v | b_n) = its mass per offset class;
D(B, v) = a_v n(B) + c_v fitted (least squares) to the class's q-weighted candidate duration, n = the text feature
(syllables = si_infer.spoken_units, words or characters) with the highest train R^2; pi(v | bin) = the bin's mean of
q(v | b_n) with one pseudo-count block per class (add-one).

    python -m experiments.mist_w3po_v1.crf.joint_speech_crf --fit-talker --out <new json>
    python -m experiments.mist_w3po_v1.crf.joint_speech_crf --checks --talker-model <json> --out <new json>
    python -m experiments.mist_w3po_v1.crf.joint_speech_crf --sizes --talker-model <json> --sched <dir> --out <new json>
    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_joint_speech_crf
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V

TAU = 0.5
DELTA_MS, CAP_MS = 80, 8000
BIN_MS, N_BIN = 500, 8
K_CAP = 4096
OFFS = (0, 2)
CANDS = "eval_outputs/mist_w3po_v1/speech/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands/cands_*.jsonl"
CAND_WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl"
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")       # si_infer.WORD
FEATURES = ("syllables", "words", "chars")


# ------------------------------------------------------------------------------------------------ talker model
def text_feature(text: str, kind: str) -> float:
    if kind == "syllables":   # si_infer.spoken_units(text, "eng")
        return float(sum(2 * len(w) if w.isdigit() else max(1, len(re.findall(r"[aeiouy]+", w.lower())))
                         for w in WORD.findall(text)))
    if kind == "words":
        return float(len(text.split()))
    if kind == "chars":
        return float(len(re.sub(r"\s+", "", text)))
    raise ValueError(kind)


class TalkerModel:
    """A1 / A2: D(B, v) = max(0, a_v n(B) + c_v) ms, log pi(v | bin(b')) over the rate classes."""

    def __init__(self, d: dict):
        self.d = d
        self.feature = d["feature"]
        self.offs = tuple(d["offs"])
        self.a = np.asarray([d["dur"][str(o)][0] for o in self.offs], np.float64)
        self.c = np.asarray([d["dur"][str(o)][1] for o in self.offs], np.float64)
        self.logpi = np.log(np.asarray(d["pi"], np.float64))          # [N_BIN, |V|]
        assert self.logpi.shape == (N_BIN, len(self.offs))

    @classmethod
    def load(cls, path: str) -> "TalkerModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def null(cls) -> "TalkerModel":
        """one class, D = 0, pi = 1: no talker latent, no playback (the T2-equivalence test)."""
        return cls({"feature": "words", "offs": [0], "dur": {"0": [0.0, 0.0]}, "pi": [[1.0]] * N_BIN})

    def n(self, text: str) -> float:
        return text_feature(text, self.feature)

    def dur(self, n: float) -> np.ndarray:
        """[|V|] durations (ms) of a block with feature n."""
        return np.maximum(0.0, self.a * n + self.c)

    def logpi_of(self, b_ms: np.ndarray) -> np.ndarray:
        """[m, |V|] log pi(v | bin(b'))."""
        return self.logpi[np.minimum(N_BIN - 1, (np.maximum(0, b_ms) // BIN_MS).astype(np.int64))]


def fit_talker(args) -> None:
    """A1 / A2 from PL-EM's E step on the train candidates (talker_rows.lattice, unchanged)."""
    from experiments.mist_w3po_v1.talker import talker_rows as TR

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    items = TR.load_items(args.cands)
    W = TR.windows(args.cand_win)
    streams, lam, st = TR.lattice(items, W)
    rows = []                                   # per block: text, q(v | b), class-weighted durations, A' backlog
    for it, s in zip(items, streams):
        _, _, _, beta = s.forward_backward(lam)
        states = s.path_states([b["chosen"] for b in it["blocks"]])
        for r, b in enumerate(it["blocks"]):
            q = s.conditional(r, states[r], lam, beta)
            qv, dv = [], []
            for o in OFFS:
                js = [j for j, c in enumerate(b["cands"]) if c["o"] == o]
                m = float(sum(q[j] for j in js))
                d = [len(b["cands"][j]["codes"]) * 20.0 for j in js]
                qv.append(m)
                dv.append(float(sum(q[j] * x for j, x in zip(js, d)) / m) if m > 1e-12 else float(np.mean(d)))
            rows.append({"text": b["text"], "qv": qv, "dv": dv, "b": states[r] * s.delta})
    qv = np.asarray([x["qv"] for x in rows])
    qv = qv / qv.sum(1, keepdims=True)
    dv = np.asarray([x["dv"] for x in rows])
    fits = {}
    for f in FEATURES:
        n = np.asarray([text_feature(x["text"], f) for x in rows])
        X = np.stack([n, np.ones_like(n)], 1)
        co, r2 = {}, []
        for k, o in enumerate(OFFS):
            sol = np.linalg.lstsq(X, dv[:, k], rcond=None)[0]
            pred = X @ sol
            r2.append(1.0 - float(((dv[:, k] - pred) ** 2).sum() / ((dv[:, k] - dv[:, k].mean()) ** 2).sum()))
            co[str(o)] = [float(sol[0]), float(sol[1])]
        fits[f] = {"dur": co, "r2": r2, "r2_mean": float(np.mean(r2))}
    best = max(FEATURES, key=lambda f: fits[f]["r2_mean"])
    bins = np.minimum(N_BIN - 1, np.asarray([x["b"] for x in rows]) // BIN_MS).astype(int)
    pi, cnt = [], []
    for k in range(N_BIN):
        sel = bins == k
        pi.append(((qv[sel].sum(0) + 1.0) / (sel.sum() + len(OFFS))).tolist())
        cnt.append(int(sel.sum()))
    res = {"feature": best, "offs": list(OFFS), "dur": fits[best]["dur"], "pi": pi, "bin_blocks": cnt,
           "fits": fits, "plem": st, "plem_lambda": lam, "blocks": len(rows), "delta_ms": DELTA_MS, "cap_ms": CAP_MS,
           "bin_ms": BIN_MS, "cands": args.cands, "cand_win": args.cand_win, "cpu_s": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    np.save(str(out) + ".blocks.npy", np.concatenate([qv, dv, bins[:, None]], 1))
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=1), flush=True)
    print(json.dumps(fits, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ augmented chain
def next_backlog(t: int, bp, d: float, delta: int = DELTA_MS, cap: int = CAP_MS):
    """the backlog after a block of d ms is queued at read time t with queue left bp: the queue end t + bp + d on the
    absolute delta grid (nearest), minus t, clipped to [0, cap]."""
    return np.clip(np.rint((t + np.asarray(bp) + d) / delta).astype(np.int64) * delta - t, 0, cap)


def block_info(rec: dict, W: tuple, r: int, spk: list, tm: TalkerModel) -> list:
    """per speaker index: None (no block) or (W_B words, D [|V|] ms, text term sum W_i (t_r - src_end_i) / 1000)."""
    ch = rec["chunks"]
    t = rec["t_ms"][r]
    out = [None] * len(spk)
    pos = {k: x for x, k in enumerate(spk)}
    by = {}
    for c in W:
        by.setdefault(pos[ch[c]["slot"]], []).append(c)
    for x, cs in by.items():
        words = sum(LW.n_words(ch[c]) for c in cs)
        text = " ".join(ch[c]["text"] for c in cs)
        tt = sum(LW.n_words(ch[c]) * (t - ch[c]["src_end_ms"]) / 1000.0 for c in cs)
        out[x] = (words, tm.dur(tm.n(text)), tt)
    return out


def _encode(j: np.ndarray, B: np.ndarray, base: int) -> np.ndarray | list:
    if (base ** B.shape[1]) * (int(j.max()) + 1) < 2 ** 62:
        key = j.astype(np.int64)
        for x in range(B.shape[1]):
            key = key * base + B[:, x].astype(np.int64)
        return key
    return [tuple([int(a)] + [int(v) for v in row]) for a, row in zip(j, B)]


def augment(rec: dict, pl: dict, Eo: list, tm: TalkerModel, delta: int = DELTA_MS, cap: int = CAP_MS,
            kcap: int = K_CAP, zero_play: bool = False) -> dict | None:
    """The augmented chain of one window: per read the states (Thinker state j, backlog vector) and the edges
    (src, dst, flat Thinker edge i * n_cur + j, lam-free potential E_order + sum log pi, l). None when a read exceeds kcap.
    zero_play: l = the text term only (the playback terms set to 0)."""
    lat = pl["lat"]
    spk, R, t_ms = lat["spk"], lat["R"], rec["t_ms"]
    K = len(spk)
    nV = len(tm.offs)
    pj = np.zeros(1, np.int64)
    pB = np.zeros((1, K), np.int64)
    reads = []
    for r in range(R):
        dt = t_ms[r] - (t_ms[r - 1] if r > 0 else 0)
        Er, edge = Eo[r], pl["edge"][r]
        ncur = Er.shape[1]
        Bp_all = np.maximum(0, pB - dt)
        src, keys, jj, BB, te, pot, ll = [], [], [], [], [], [], []
        for i in range(Er.shape[0]):
            sel = np.nonzero(pj == i)[0]
            if not len(sel):
                continue
            Bp = Bp_all[sel]
            for j in np.nonzero(edge[i])[0]:
                info = block_info(rec, pl["W"][(r, i)][j], r, spk, tm)
                wk = [x for x in range(K) if info[x] is not None]
                base_lat = sum(info[x][2] for x in wk)
                for combo in itertools.product(range(nV), repeat=len(wk)):
                    nB = Bp.copy()
                    lp = np.zeros(len(sel))
                    l_ = np.full(len(sel), base_lat)
                    for x, v in zip(wk, combo):
                        words, D, _ = info[x]
                        d = float(D[v])
                        if d > 0.0:          # D = 0 (the null model): no playback, the queue is unchanged
                            nB[:, x] = next_backlog(t_ms[r], Bp[:, x], d, delta, cap)
                        lp += tm.logpi_of(Bp[:, x])[:, v]
                        if not zero_play:
                            l_ += words * Bp[:, x] / 1000.0 + (words + 1) / 2.0 * d / 1000.0
                    src.append(sel)
                    jj.append(np.full(len(sel), j, np.int64))
                    BB.append(nB)
                    te.append(np.full(len(sel), i * ncur + j, np.int64))
                    pot.append(Er[i, j] + lp)
                    ll.append(l_)
        src, jj, BB = np.concatenate(src), np.concatenate(jj), np.concatenate(BB)
        te, pot, ll = np.concatenate(te), np.concatenate(pot), np.concatenate(ll)
        key = _encode(jj, BB, cap + 1)
        if isinstance(key, list):
            uniq = {}
            dst = np.asarray([uniq.setdefault(k_, len(uniq)) for k_ in key], np.int64)
            first = np.zeros(len(uniq), np.int64)
            first[dst[::-1]] = np.arange(len(dst))[::-1]
        else:
            _, first, dst = np.unique(key, return_index=True, return_inverse=True)
        n = len(first)
        if n > kcap:
            return None
        pj, pB = jj[first], BB[first]
        reads.append({"src": src, "dst": dst, "te": te, "pot": pot, "l": ll, "n": n, "j": pj.copy(),
                      "ncur": ncur, "nprev": Er.shape[0]})
    for rd in reads:
        rd["by_dst"] = _segments(rd["dst"], rd["n"])
    for r, rd in enumerate(reads):
        rd["by_src"] = _segments(rd["src"], reads[r - 1]["n"] if r > 0 else 1)
    return {"reads": reads, "delta": delta, "states": [rd["n"] for rd in reads],
            "edges": int(sum(len(rd["src"]) for rd in reads))}


def augment_capped(rec, pl, Eo, tm, zero_play=False, kcap=K_CAP) -> dict:
    """A5: the 80 ms grid, or the first of 160, 320, ... ms that keeps every read <= kcap states."""
    d = DELTA_MS
    while True:
        a = augment(rec, pl, Eo, tm, delta=d, kcap=kcap, zero_play=zero_play)
        if a is not None:
            a["coarsened"] = d != DELTA_MS
            return a
        d *= 2


def _segments(ids: np.ndarray, n: int) -> tuple:
    perm = np.argsort(ids, kind="stable")
    s = ids[perm]
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1] if len(s) else np.zeros(0, np.int64)
    return perm, starts, s[starts] if len(s) else s, n


def _seg_lse(v: np.ndarray, seg: tuple) -> np.ndarray:
    perm, starts, uid, n = seg
    out = np.full(n, -np.inf)
    if not len(v):
        return out
    vs = v[perm]
    m = np.maximum.reduceat(vs, starts)
    mm = np.where(np.isfinite(m), m, 0.0)
    cnt = np.diff(np.r_[starts, len(vs)])
    with np.errstate(divide="ignore"):
        out[uid] = np.log(np.add.reduceat(np.exp(vs - np.repeat(mm, cnt)), starts)) + mm
    return out


def forward_backward(aug: dict, lam: float, mask: list | None = None) -> dict:
    """Sum-product on the augmented chain; mask[r]: bool per edge (T1's sub-lattice) or None."""
    reads = aug["reads"]
    R = len(reads)
    pots = []
    for r, rd in enumerate(reads):
        p = rd["pot"] if lam == 0.0 else rd["pot"] - lam * rd["l"]
        if mask is not None:
            p = np.where(mask[r], p, -np.inf)
        pots.append(p)
    alpha, a = [], np.zeros(1)
    for rd, p in zip(reads, pots):
        a = _seg_lse(a[rd["src"]] + p, rd["by_dst"])
        alpha.append(a)
    logZ = float(V.lse(alpha[-1]))
    beta = [None] * R
    b = np.zeros(reads[-1]["n"])
    beta[R - 1] = b
    for r in range(R - 1, 0, -1):
        rd = reads[r]
        b = _seg_lse(pots[r] + b[rd["dst"]], rd["by_src"])
        beta[r - 1] = b
    q, el = [], 0.0
    for r, (rd, p) in enumerate(zip(reads, pots)):
        ap = alpha[r - 1] if r > 0 else np.zeros(1)
        with np.errstate(invalid="ignore"):
            qe = np.exp(ap[rd["src"]] + p + beta[r][rd["dst"]] - logZ)
        qe = np.nan_to_num(qe)
        q.append(qe)
        el += float((qe * rd["l"]).sum())
    return {"logZ": logZ, "alpha": alpha, "beta": beta, "q": q, "Eq_l": el}


def t1_mask(aug: dict, pl: dict) -> list:
    out = []
    for rd, (pm, cm) in zip(aug["reads"], pl["t1"]):
        i, j = rd["te"] // rd["ncur"], rd["te"] % rd["ncur"]
        out.append(pm[i] & cm[j])
    return out


def window_parts(w, pl: dict, res: dict, tm: TalkerModel, tau: float = TAU, zero_play: bool = False) -> dict:
    """The lam-free parts of one window: T2's psi / order terms, the augmented chain, T1's target E_{q_T1}[l]."""
    reads = LW.assemble(pl, LW.full_res(w, pl, res))
    Eo = LW.edge_terms(reads, tau)
    aug = augment_capped(w.rec, pl, Eo, tm, zero_play=zero_play)
    m1 = t1_mask(aug, pl)
    tgt = forward_backward(aug, 0.0, m1)["Eq_l"]
    sw = float(sum(LW.n_words(c) for c in w.rec["chunks"]))
    return {"reads": reads, "Eo": Eo, "aug": aug, "target": tgt, "sum_words": sw, "edge": pl["edge"]}


def solve_lambda(parts: list[dict]) -> tuple[float, dict]:
    """T2's step-level dual (LW constants) on the joint chain's E_q[l]."""
    tgt = sum(p["target"] for p in parts)

    def f(lam):
        return sum(forward_backward(p["aug"], lam)["Eq_l"] for p in parts)
    f0 = f(0.0)
    info = {"target": tgt, "Eq_l_lam0": f0, "binding": f0 > tgt, "bracketed": True, "capped": False, "evals": 1}
    if f0 <= tgt:
        return 0.0, info
    lo, hi = 0.0, LW.LAM_START
    while True:
        fh = f(hi)
        info["evals"] += 1
        if fh <= tgt:
            break
        lo = hi
        if hi >= LW.LAM_MAX:
            info.update(bracketed=False, capped=True)
            return LW.LAM_MAX, info
        hi = min(2.0 * hi, LW.LAM_MAX)
    for _ in range(LW.BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if f(mid) > tgt:
            lo = mid
        else:
            hi = mid
        info["evals"] += 1
    return hi, info


def edge_blocks(rec: dict, pl: dict, r: int, nprev: int, ncur: int) -> np.ndarray:
    """[nprev, ncur] number of non-empty (read, speaker) blocks the Thinker edge i -> j of read r writes (0 off the
    edges): the speakers of W = pl["W"][(r, i)][j]. J1's block statistic (= the talker blocks under read-level input)."""
    ch = rec["chunks"]
    out = np.zeros((nprev, ncur))
    edge = pl["edge"][r]
    for i in range(nprev):
        for j in np.nonzero(edge[i])[0]:
            out[i, j] = len({ch[c]["slot"] for c in pl["W"][(r, i)][j]})
    return out


def posterior_from_parts(w, pl: dict, parts: dict, tau: float, lam: float) -> dict:
    """Thinker edge marginals from the joint chain, then LW.posterior_from_parts' cend construction."""
    reads, Eo, edge, aug = parts["reads"], parts["Eo"], parts["edge"], parts["aug"]
    fb = forward_backward(aug, lam)
    cend = [np.zeros(len(q["forced_ids"]) + 1) for q in pl["reqs"]]
    exp_nf, n_real, worst, exp_defer = 0.0, 0, 0.0, 0.0
    exp_blocks = 0.0           # J1 log field: E_q[number of non-empty (read, speaker) blocks] (no effect on the posterior)
    lat, e = pl["lat"], pl["lat"]["rho"]
    by_e = [[x for x in range(len(e)) if e[x] == r] for r in range(lat["R"])]
    for r, (rd, pr, ar) in enumerate(zip(reads, pl["reads"], aug["reads"])):
        C, Nf = rd["C"], rd["Nf"]
        qe = np.bincount(ar["te"], weights=fb["q"][r], minlength=ar["nprev"] * ar["ncur"]).reshape(ar["nprev"],
                                                                                                  ar["ncur"])
        with np.errstate(invalid="ignore", over="ignore"):
            qC = qe * np.where(edge[r], np.exp(tau * C - Eo[r]), 0.0)
        if Nf is not None and not np.isnan(Nf).all():
            real = ~np.isnan(Nf)
            with np.errstate(invalid="ignore", over="ignore"):
                qN = np.where(real, qe * np.exp(tau * np.where(real, Nf, 0.0) - Eo[r]), 0.0)
            n_real += 1
            exp_nf += float(qN.sum())
        else:
            qN = np.zeros_like(qC)
        worst = max(worst, abs(float(qC.sum() + qN.sum()) - 1.0))
        exp_blocks += float((qe * edge_blocks(w.rec, pl, r, ar["nprev"], ar["ncur"])).sum())
        ms = np.exp(fb["alpha"][r] + fb["beta"][r] - fb["logZ"])
        mj = np.bincount(ar["j"], weights=ms, minlength=ar["ncur"])
        for j, A in enumerate(lat["sets"][r]):
            exp_defer += float(mj[j]) * sum(1 for x in by_e[r] if x not in A)
        for cx in pr["ctx"]:
            i = cx["i"]
            for j, (k, b) in cx["cells"]["new"].items():
                cend[k][b] += qC[i, j]
            for j, (k, b, _bn, _s, _nl) in cx["nf"].items():
                cend[k][b] += qN[i, j]
    if worst > 1e-6:
        raise RuntimeError(f"T3 posterior of {w.rec['id']}: per-read mass off by {worst:.2e}")
    return {"cend": cend, "logZ": fb["logZ"], "exp_nf": exp_nf, "n_real": n_real, "exp_defer": exp_defer,
            "exp_blocks": exp_blocks, "Eq_l": fb["Eq_l"], "target": parts["target"], "sum_words": parts["sum_words"], "lam": lam,
            "aug_states_max": int(max(aug["states"])), "aug_edges": aug["edges"], "coarsened": aug["coarsened"]}


def posterior(w, pl: dict, res: dict, tm: TalkerModel, tau: float = TAU, lam: float = 0.0,
              zero_play: bool = False) -> dict:
    return posterior_from_parts(w, pl, window_parts(w, pl, res, tm, tau, zero_play), tau, lam)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-talker", action="store_true")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--sizes", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--talker-model", default="")
    ap.add_argument("--sched", default="eval_outputs/mist_w3po_v1/crf/sched_piece_t0_v1")
    ap.add_argument("--cands", default=CANDS)
    ap.add_argument("--cand-win", default=CAND_WIN)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    if args.fit_talker:
        fit_talker(args)
    if args.checks:
        from experiments.mist_w3po_v1.crf import joint_speech_checks as JC

        JC.run_checks(args)
    if args.sizes:
        from experiments.mist_w3po_v1.crf import joint_speech_checks as JC

        JC.run_sizes(args)


if __name__ == "__main__":
    main()
