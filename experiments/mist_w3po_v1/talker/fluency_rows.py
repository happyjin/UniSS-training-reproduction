"""Fluency-constrained PL-EM rows (paper v11 P2, reports/mist_w3po_v1/v11_p2_fluency_pa_prereg.zh-CN.md). CPU only.

PL-EM (talker_rows --mode plem) projects the playback CRF onto one expectation constraint, the corpus latency budget
E_q[sum_n l_n] / sum W <= b, with one dual lam. The PA Talker trained on it speaks ~45 % faster than the base Talker
and pauses about twice as often. Here a second expectation constraint bounds the speaking rate from above, written as
a lower bound on the expected speech time per word,

    E_q[sum_n d_{c_n}] / sum_n W_n >= s_min        (seconds of speech per word / character),

so the KL projection of the lam=0 CRF onto {latency <= b, time per word >= s_min} is again a linear-chain CRF with

    psi_n(b, c) = tau*lpbar_c + beta*f_c - lam * l_n(b, c) + mu * d_c / 1000,      lam, mu >= 0,

and (lam, mu) maximise the concave dual. Solved by exact dual coordinate ascent: for fixed mu, lam = the smallest
lam >= 0 meeting the latency budget (E_q[sum l] falls monotonically in lam); for fixed lam, mu = the smallest mu >= 0
meeting the time-per-word bound (E_q[sum d] rises monotonically in mu); repeat until both hold. Forward-backward stays
exact over the backlog (playback_lattice.Stream; d_c only enters the node potential).

Everything else is talker_rows' plem construction verbatim (q(c | b_n = the rule path's state), rows with q >= 0.01,
the speed soft label from q, own = backlog format, ref = native format), saved with mode "plem", variant "fluency",
so pretrain_talker_crf --crf-mode plem reads the file unchanged.

    --stats   train-set statistics that fix s_min before any training (prereg): time per word of the base-speed
              candidates, the rule path, the lam=0 CRF, the PL-EM posterior (lam*, mu=0). No file written but --stats-out.
    --out     rows with s_min = --spw-min (or --rate-frac f: s_min = 1 / (r0 + f * (r_pa - r0)), r = 1 / time per word,
              r0 the base-speed candidates, r_pa the PL-EM posterior)

    python -m experiments.mist_w3po_v1.talker.fluency_rows --cands '<dir>/cands_*.jsonl' --win <windows> \\
        [--lang cmn --units char] (--stats --stats-out <json> | --rate-frac 0.5 --out <rows.pt>)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.talker import playback_lattice as PL
from experiments.mist_w3po_v1.talker import talker_rows as TR


class FStream(PL.Stream):
    """PL.Stream with the fluency dual: psi gains + mu * d_c (seconds)."""

    mu = 0.0

    def psi(self, n: int, lam: float) -> np.ndarray:
        return self.feat[n][None, :] - lam * self.lat[n] + self.mu * self.dur[n][None, :] / 1000.0


def build(items: list[dict], W: dict) -> list[FStream]:
    out = []
    for it in items:
        s = TR.stream_of(it, W[it["id"].split("__S")[0]])
        s.__class__ = FStream
        out.append(s)
    return out


def corpus(streams: list[FStream], lam: float, mu: float) -> tuple[float, float]:
    """(E_q[sum l] / sum W, E_q[sum d] / sum W) at (lam, mu), seconds."""
    el = ed = 0.0
    words = 0
    for st in streams:
        st.mu = mu
        _, marg, e, _ = st.forward_backward(lam)
        el += e
        ed += sum(float(m.sum(0) @ st.dur[n]) for n, m in enumerate(marg)) / 1000.0
        words += sum(st.words)
    return el / words, ed / words


def bisect(f, target: float, increasing: bool, hi: float = 50.0, iters: int = 40) -> float:
    """smallest x in [0, hi] with f(x) meeting the target (f(x) <= target if not increasing, >= if increasing)."""
    ok = (lambda v: v >= target) if increasing else (lambda v: v <= target)
    if ok(f(0.0)):
        return 0.0
    if not ok(f(hi)):
        return hi
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if ok(f(mid)):
            hi = mid
        else:
            lo = mid
    return hi


def solve(streams: list[FStream], budget: float, spw_min: float, rounds: int = 50, tol_l: float = 2e-3,
          tol_d: float = 2e-4) -> dict:
    lam = mu = 0.0
    hist = []
    for k in range(rounds):
        lam = bisect(lambda x: corpus(streams, x, mu)[0], budget, increasing=False)
        mu = bisect(lambda x: corpus(streams, lam, x)[1], spw_min, increasing=True)
        el, ed = corpus(streams, lam, mu)
        hist.append((k, lam, mu, round(el, 4), round(ed, 4)))
        print(json.dumps({"round": k, "lam": lam, "mu": mu, "latency_s": el, "spw_s": ed}), flush=True)
        if el <= budget + tol_l and ed >= spw_min - tol_d:
            break
    feasible = el <= budget + tol_l and ed >= spw_min - tol_d and lam < 50.0 and mu < 50.0
    return {"lambda": lam, "mu": mu, "eq_latency_s": round(el, 4), "eq_spw_s": round(ed, 4), "rounds": len(hist),
            "feasible": feasible, "history": hist}


def stats(items: list[dict], W: dict) -> dict:
    streams = build(items, W)
    words = sum(sum(st.words) for st in streams)
    b0_d = b0_w = 0.0
    for it in items:
        for b in it["blocks"]:
            ds = [len(c["codes"]) * 0.02 for c in b["cands"] if c["v"] == b["speed_base"]]
            if ds:
                b0_d += float(np.mean(ds))
                b0_w += max(1, TR.n_units(b["text"]))
    rule_d = sum(len(b["cands"][b["chosen"]]["codes"]) * 0.02 for it in items for b in it["blocks"]) / words
    a_lat = sum(st.path_latency([b["chosen"] for b in it["blocks"]]) for st, it in zip(streams, items)) / words
    lam, _ = PL.solve_lambda(streams, a_lat - TR.BUDGET_DROP_S)
    l0, d0 = corpus(streams, 0.0, 0.0)
    lp, dp = corpus(streams, lam, 0.0)
    return {"streams": len(streams), "words": words, "spw_base_speed_s": round(b0_d / b0_w, 4),
            "spw_rule_s": round(rule_d, 4), "spw_lam0_s": round(d0, 4), "spw_plem_s": round(dp, 4),
            "aprule_latency_s": round(a_lat, 4), "budget_s": round(a_lat - TR.BUDGET_DROP_S, 4), "plem_lambda": lam,
            "latency_lam0_s": round(l0, 4), "latency_plem_s": round(lp, 4),
            "rate_base": round(b0_w / b0_d, 4), "rate_plem": round(1 / dp, 4), "rate_rule": round(1 / rule_d, 4)}


def rows(items: list[dict], W: dict, spw_min: float) -> tuple[list[dict], dict]:
    streams = build(items, W)
    words = sum(sum(st.words) for st in streams)
    a_lat = sum(st.path_latency([b["chosen"] for b in it["blocks"]]) for st, it in zip(streams, items)) / words
    budget = a_lat - TR.BUDGET_DROP_S
    sol = solve(streams, budget, spw_min)
    lam, mu = sol["lambda"], sol["mu"]
    blocks, kl_u, ent = [], [], []
    for it, st in zip(items, streams):
        st.mu = mu
        _, marg, _, beta = st.forward_backward(lam)
        states = st.path_states([b["chosen"] for b in it["blocks"]])
        for r in range(len(it["blocks"])):
            k, h = TR._kl_ent(marg[r].sum(0))
            kl_u.append(k)
            ent.append(h)
            bk = TR.block_rows(it, r, st.conditional(r, states[r], lam, beta))
            if bk is not None:
                blocks.append(bk)
    out = {"aprule_latency_s": round(a_lat, 4), "budget_s": round(budget, 4), "spw_min_s": round(spw_min, 4),
           **{k: v for k, v in sol.items() if k != "history"}, "dual_history": sol["history"],
           "streams": len(streams), "words": words, "blocks": len(blocks),
           "rows": sum(len(x["cands"]) for x in blocks),
           "p3_kl_uniform_mean": round(float(np.mean(kl_u)), 4) if kl_u else None,
           "posterior_entropy_mean": round(float(np.mean(ent)), 4) if ent else None}
    return blocks, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--win", required=True)
    ap.add_argument("--lang", choices=("eng", "cmn"), default="eng")
    ap.add_argument("--units", choices=("word", "char"), default="word")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--stats-out")
    ap.add_argument("--spw-min", type=float)
    ap.add_argument("--rate-frac", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    TR.LANG, TR.UNITS = (TR.LANG_ENG if a.lang == "eng" else TR.LANG_CMN), a.units
    items, W = TR.load_items(a.cands), TR.windows(a.win)
    if a.stats:
        s = stats(items, W)
        s.update({"cands": a.cands, "lang": a.lang, "units": a.units})
        print(json.dumps(s))
        if a.stats_out:
            p = Path(a.stats_out)
            if p.exists():
                raise SystemExit(f"拒绝覆盖已有输出: {p}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(s, indent=1) + "\n")
        return
    out = Path(a.out)
    if out.exists() or Path(str(out) + ".json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if (a.spw_min is None) == (a.rate_frac is None):
        raise SystemExit("exactly one of --spw-min / --rate-frac")
    spw_min = a.spw_min
    extra = {}
    if spw_min is None:
        s = stats(items, W)
        r0, rp = s["rate_base"], s["rate_plem"]
        spw_min = 1.0 / (r0 + a.rate_frac * (rp - r0))
        extra = {"rate_frac": a.rate_frac, "rate_base": r0, "rate_plem": rp, "rate_cap": round(1 / spw_min, 4)}
    blocks, st = rows(items, W, spw_min)
    st.update(extra)
    st.update({} if (a.lang, a.units) == ("eng", "word") else {"lang": a.lang, "units": a.units})
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mode": "plem", "variant": "fluency", "blocks": blocks, "stats": st, "cands": a.cands}, out)
    Path(str(out) + ".json").write_text(json.dumps({"mode": "plem", "variant": "fluency", "cands": a.cands, **st},
                                                   indent=1) + "\n")
    print(json.dumps({k: v for k, v in st.items() if k != "dual_history"}))


if __name__ == "__main__":
    main()
