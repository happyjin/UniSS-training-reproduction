"""Bayes belief tracking for the commit decision (docs/si_bayes_belief_plan.zh-CN.md, section 2).

Each tentative unit of a read's greedy draft carries a belief that it survives the future -- that the frozen
Thinker, the same committed prefix before it and the whole sentence heard, would say the same up to it:

  Z_u = 1[ draft[:u+1] == oracle[:u+1] ],   oracle = Thinker(x_1:T, committed)

A unit is born the first read it appears (or when anything before it changed); its log-odds start at the prior
and every later read that repeats it adds a log likelihood ratio:

  l_born = g(x_born) + b_talk          g: this read's features alone (no cross-read signal -- that is evidence)
  l_s    = l_{s-1} + h(o_s)            o_s = (x_s, x_s - x_{s-1}, age, log(1+age)): what the new read says of it
  P(Z_u = 1 | o) = sigmoid(l_s)

g and h are trained together (BCE of sigmoid(l_s) against Z at every read), so h is a log likelihood ratio, not a
score. b_talk is a talk-level prior offset, N(0, sigma_b^2), updated after each sentence by an assumed-density
(Laplace) step with the units that sentence has since revealed. The decision commits the leading units with
P >= tau; a commit shorter than --min units that does not end a clause waits (the talker's fragment cost).

  prep    dumps from si_thinker_stream --policy head --oracle --dump  ->  one npz of unit trajectories
  check   held-out AUC / calibration: prior alone, the full belief, and LocalAgreement's signal
  (run_belief is the inference loop si_thinker_stream --policy belief calls)

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_belief prep --dumps 'data/si_belief_v1/shard_*/dumps/*.jsonl' --out rows.npz
    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_belief check --belief belief.pt --rows rows.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_X = 15
N_O = 2 * N_X + 2
PUNCT_END = tuple(",.;:!?，。；：！？、…")
MAX_UNITS, MAX_STEPS = 16, 8


def unit_features(tok: dict, gu: list[str], heard_s: float, n_committed: int, zh: bool) -> np.ndarray:
    """One row per unit of the draft: its tokens' least / mean log-probability, least top-5 margin, largest
    entropy, whether all are the argmax; its place; how much is heard and committed; the draft's running
    least / mean log-probability up to it; whether it is last, the direction, whether it ends a clause."""
    n = len(gu)
    lp, top, uo = tok["lp"], tok["top"], tok["unit_of"]
    per = [[] for _ in range(n)]
    for i, u in enumerate(uo):
        if 0 <= u < n:
            per[u].append(i)
    out = np.zeros((n, N_X), dtype=np.float32)
    last, cmin, csum = None, 0.0, 0.0
    for j in range(n):
        idx = per[j] or ([last] if last is not None else [])
        if per[j]:
            last = per[j][-1]
        if idx:
            l = [lp[i] for i in idx]
            mg = [(top[i][0] - top[i][1]) if len(top[i]) > 1 else 5.0 for i in idx]
            en = []
            for i in idx:
                p = np.exp(np.asarray(top[i]) - max(top[i]))
                p /= p.sum()
                en.append(float(-(p * np.log(p + 1e-9)).sum()))
            t1 = all(abs(lp[i] - top[i][0]) < 1e-4 for i in idx)
        else:
            l, mg, en, t1 = [-5.0], [0.0], [1.6], False
        cmin = min(l) if j == 0 else min(cmin, min(l))
        csum += float(np.mean(l))
        out[j] = [min(l), float(np.mean(l)), min(mg), max(en), float(t1), j, j / max(n, 1), n, heard_s,
                  n_committed / (heard_s + 1e-3), cmin, csum / (j + 1), float(j == n - 1), float(zh),
                  float(gu[j].endswith(PUNCT_END))]
    return out


def observation(x: np.ndarray, x_prev: np.ndarray, age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([x, x - x_prev, age, np.log1p(age)], axis=-1).astype(np.float32)


def mlp(d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))


class Belief(nn.Module):
    """g (prior log-odds from one read) and h (log likelihood ratio of a repeat), each on normalised inputs."""

    def __init__(self, mx=None, sx=None, mo=None, so=None):
        super().__init__()
        for name, v, d in (("mx", mx, N_X), ("sx", sx, N_X), ("mo", mo, N_O), ("so", so, N_O)):
            default = torch.zeros(d) if name.startswith("m") else torch.ones(d)
            self.register_buffer(name, default if v is None else torch.as_tensor(v, dtype=torch.float32))
        self.g, self.h = mlp(N_X), mlp(N_O)

    def prior(self, x):
        return self.g((x - self.mx) / self.sx).squeeze(-1)

    def update(self, o):
        return self.h((o - self.mo) / self.so).squeeze(-1)

    def trajectory_logits(self, x0, obs, m):
        """x0 [B, N_X] at birth, obs [B, S, N_O] (step 0 unused), m [B, S] -> log-odds at every step [B, S]."""
        inc = self.update(obs) * m
        inc = torch.cat([torch.zeros_like(inc[:, :1]), inc[:, 1:]], dim=1)
        return self.prior(x0).unsqueeze(1) + torch.cumsum(inc, dim=1)


def load_belief(path: str) -> Belief:
    ck = torch.load(path, map_location="cpu")
    b = Belief()
    b.load_state_dict(ck["state"])
    return b.eval()


TERMINAL = tuple(".!?。！？")


def safe_prefix(draft: list[str], oracle: list[str], max_gap: int = 3) -> int:
    """How many leading draft units the whole-sentence translation would also say, in order: each a unit of the
    oracle at most max_gap units past the last one matched (insertions allowed, rewording of what is said not);
    words compared without their punctuation, except that a draft unit closing a sentence must close one there."""
    norm = lambda u: "".join(c for c in u.lower() if c.isalnum())
    k, pos = 0, -1
    for u in draft:
        nu, found = norm(u), -1
        for q in range(pos + 1, min(len(oracle), pos + 2 + max_gap)):
            o = oracle[q]
            if (nu == norm(o) if nu else u == o) and (not u.endswith(TERMINAL) or o.endswith(TERMINAL)):
                found = q
                break
        if found < 0:
            break
        k, pos = k + 1, found
    return k


def lcp(a: list, b: list) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


# ---------------------------------------------------------------------------------------------- prep

def cmd_prep(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    xs, zs, hyp_of, step_of = [], [], [], []
    starts, lens, items = [], [], []
    item_names = []
    n_files = 0
    for f in sorted(glob.glob(args.dumps)):
        n_files += 1
        zh = "_en2zh_" in Path(f).name
        by = defaultdict(list)
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if "oracle_units" in r:
                by[r["id"]].append(r)
        for iid, recs in by.items():
            recs.sort(key=lambda r: r["t"])
            item_idx = len(item_names)
            item_names.append(f"{'en2zh' if zh else 'zh2en'}:{iid}")
            prev = None   # (x, hyp ids, keep) of the read before
            for r in recs:
                gu = r["draft_units"][:MAX_UNITS]
                if not gu or not r["lp"]:
                    prev = None
                    continue
                x = unit_features(r, gu, r["heard_s"], r["n_committed"], zh)
                stable = (safe_prefix(r["draft_units"], r["oracle_units"], 6 if zh else 4) if args.label == "subseq" else
                          lcp(r["draft_units"], r["oracle_units"]))
                agree = r["prev_agree"] if prev is not None else 0
                hyp_ids = []
                for j in range(len(gu)):
                    z = float(j < stable)
                    jp = (prev[2] + j) if prev is not None else -1
                    h = prev[1][jp] if (prev is not None and j < agree and 0 <= jp < len(prev[1])) else -1
                    if h >= 0 and lens[h] < MAX_STEPS:
                        lens[h] += 1
                    else:
                        h = len(starts)
                        starts.append(len(xs))
                        lens.append(1)
                        items.append(item_idx)
                    xs.append(x[j])
                    zs.append(z)
                    hyp_of.append(h)
                    step_of.append(lens[h] - 1)
                    hyp_ids.append(h)
                prev = (x, hyp_ids, r["policy_keep"])
    if not xs:
        raise SystemExit(f"no oracle records under {args.dumps}")
    # steps were appended in read order, interleaving hypotheses: regroup so each one's steps are contiguous
    hyp_of = np.asarray(hyp_of, dtype=np.int64)
    step_of = np.asarray(step_of, dtype=np.int64)
    order = np.lexsort((step_of, hyp_of))
    X = np.stack(xs)[order]
    Z = np.asarray(zs, dtype=np.uint8)[order]
    lens = np.asarray(lens, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    items = np.asarray(items, dtype=np.int64)
    rng = np.random.default_rng(1)
    val_items = rng.random(len(item_names)) < args.val_frac
    val = val_items[items]
    # normalisation from training steps (x) and training repeats (observations)
    tr_steps = np.repeat(~val, lens)
    mx, sx = X[tr_steps].mean(0), X[tr_steps].std(0) + 1e-4
    rep = np.flatnonzero(np.concatenate([np.arange(L) for L in lens]) > 0)
    rep = rep[tr_steps[rep]]
    age = np.concatenate([np.arange(L) for L in lens])[rep]
    O = observation(X[rep], X[rep - 1], age)
    mo, so = O.mean(0), O.std(0) + 1e-4
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=X, z=Z, start=starts, len=lens, item=items, val=val, mx=mx, sx=sx, mo=mo, so=so,
             item_names=np.asarray(item_names))
    print(json.dumps({"files": n_files, "items": len(item_names), "hypotheses": int(len(lens)), "steps": int(len(X)),
                      "repeats": int((lens - 1).sum()), "stable_rate": round(float(Z.mean()), 4),
                      "val_hyp": int(val.sum()), "len_hist": np.bincount(lens).tolist(), "out": str(out)}), flush=True)


def batch_of(data: dict, hyps: np.ndarray) -> dict:
    """Padded tensors for these hypotheses: birth features, observations, labels, mask."""
    S = int(data["len"][hyps].max())
    B = len(hyps)
    x0 = np.zeros((B, N_X), np.float32)
    obs = np.zeros((B, S, N_O), np.float32)
    z = np.zeros((B, S), np.float32)
    m = np.zeros((B, S), np.float32)
    X, Z = data["x"], data["z"]
    for b, h in enumerate(hyps):
        s0, L = int(data["start"][h]), int(data["len"][h])
        xh = np.asarray(X[s0: s0 + L], dtype=np.float32)
        x0[b] = xh[0]
        if L > 1:
            obs[b, 1:L] = observation(xh[1:], xh[:-1], np.arange(1, L))
        z[b, :L] = Z[s0: s0 + L]
        m[b, :L] = 1
    return {"x0": torch.from_numpy(x0), "obs": torch.from_numpy(obs), "z": torch.from_numpy(z), "m": torch.from_numpy(m)}


# --------------------------------------------------------------------------------------------- check

def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        k = (p >= a) & (p < b if b < 1 else p <= b)
        if k.any():
            e += k.mean() * abs(y[k].mean() - p[k].mean())
    return float(e)


def cmd_check(args) -> None:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import auc_score

    belief = load_belief(args.belief)
    data = dict(np.load(args.rows))
    hyps = np.flatnonzero(data["val"])
    ys, pb, pp, la = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(hyps), 4096):
            bt = batch_of(data, hyps[s: s + 4096])
            l = belief.trajectory_logits(bt["x0"], bt["obs"], bt["m"])
            # the prior alone at every step: each read's own features, as if nothing were remembered
            X = bt["obs"][..., :N_X].clone()
            X[:, 0] = bt["x0"]
            lp = belief.prior(X)
            k = bt["m"] > 0
            ys.append(bt["z"][k].numpy())
            pb.append(torch.sigmoid(l)[k].numpy())
            pp.append(torch.sigmoid(lp)[k].numpy())
            step = torch.arange(bt["m"].shape[1]).expand_as(bt["m"])
            la.append((step[k] > 0).float().numpy())   # LocalAgreement: repeated by a later read
    y, pb, pp, la = map(np.concatenate, (ys, pb, pp, la))
    res = {"val_steps": int(len(y)), "stable_rate": round(float(y.mean()), 4),
           "auc_prior": round(auc_score(y, pp), 4), "auc_belief": round(auc_score(y, pb), 4),
           "auc_la_signal": round(auc_score(y, la), 4),
           "ece_prior": round(ece(y, pp), 4), "ece_belief": round(ece(y, pb), 4)}
    for name, k in (("born", la == 0), ("repeated", la == 1)):
        if k.any():
            res[f"auc_belief_{name}"] = round(auc_score(y[k], pb[k]), 4)
            res[f"stable_rate_{name}"] = round(float(y[k].mean()), 4)
    print(json.dumps(res), flush=True)


# ----------------------------------------------------------------------------------------- inference

def talk_of(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]


def run_belief(llm, params_lp, items, head, zh, window, out, tag, belief: Belief, tau: float, min_units: int = 1,
               sigma_b: float | None = None, look_units: int = 0):
    """The belief policy on RealSI: each talk's sentences in order (a wave is the k-th sentence of every talk), so
    the talk's prior offset can learn from the sentences before; within a wave every sentence reads in step."""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import draft_tokens, join, units

    talks = defaultdict(list)
    for it in items:
        talks[talk_of(it["id"])].append(it)
    bias = {t: 0.0 for t in talks}
    prec = {t: (1.0 / sigma_b ** 2 if sigma_b else None) for t in talks}
    t0 = time.time()
    waves = max(len(v) for v in talks.values())
    for w in range(waves):
        wave = [v[w] for v in talks.values() if len(v) > w]
        for it in wave:
            it.update(committed=[], blocks=[], drafts=[], done=False, b_prev=None, seen=[])
        r = 0
        while any(not it["done"] for it in wave):
            live = [it for it in wave if not it["done"]]
            prompts = []
            for it in live:
                it["t"] = min((r + 1) * window, it["dur_ms"])
                prompts.append({"prompt": head + join(it["committed"], zh),
                                "multi_modal_data": {"audio": (it["audio"][: it["t"] * 16], 16000)}})
            results = llm.generate(prompts, params_lp, use_tqdm=False)
            for it, res in zip(live, results):
                cont = res.outputs[0].text
                sep = "" if zh or not it["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
                draft = units(join(it["committed"], zh) + sep + cont, zh)
                it["drafts"].append({"t": it["t"], "draft": join(draft, zh)})
                n = len(it["committed"])
                if it["t"] >= it["dur_ms"]:
                    new, it["done"] = draft[n:], True
                else:
                    gu = units(cont, zh)
                    keep = 0
                    if not (gu and res.outputs[0].logprobs):
                        it["b_prev"] = None
                    else:
                        tok = draft_tokens(res.outputs[0], cont, zh)
                        g = gu[:MAX_UNITS]
                        x = unit_features(tok, g, it["t"] / 1000, n, zh)
                        prev = it["b_prev"]   # (units, x, log-odds, age) of what the read before left uncommitted
                        agree = lcp(g, prev[0]) if prev is not None else 0
                        with torch.no_grad():
                            l = belief.prior(torch.from_numpy(x)).numpy()
                            age = np.zeros(len(g), dtype=np.float32)
                            if agree:
                                age[:agree] = prev[3][:agree] + 1
                                o = observation(x[:agree], prev[1][:agree], age[:agree])
                                l[:agree] = prev[2][:agree] + belief.update(torch.from_numpy(o)).numpy()
                        p = 1.0 / (1.0 + np.exp(-(l + bias[talk_of(it["id"])])))
                        while keep < len(g) and p[keep] >= tau:
                            keep += 1
                        if 0 < keep < min_units and not g[keep - 1].endswith(PUNCT_END):
                            keep = 0
                        it["seen"].append((list(it["committed"]), g, l))
                        it["b_prev"] = (g[keep:], x[keep:], l[keep:], age[keep:])
                        it["look"] = join(g[keep: keep + look_units], zh) if look_units else ""
                    new = draft[n: n + keep]
                if new or it["done"]:
                    it["blocks"].append({"write_ms": it["t"], "text": ("" if zh or not n else " ") + join(new, zh),
                                         "final": it["done"],
                                         **({"look": it.get("look", "")} if look_units and not it["done"] else {})})
                    it["committed"] += new
            r += 1
        if sigma_b:   # each finished sentence reveals which of its tentative units the future kept
            # judged against the whole sentence translated afresh -- the policy's own translation would confirm
            # every unit it committed, and the offset would only ever rise
            fresh = llm.generate([{"prompt": head, "multi_modal_data": {"audio": (it["audio"], 16000)}} for it in wave],
                                 params_lp, use_tqdm=False)
            gap = 6 if zh else 4
            for it, fr in zip(wave, fresh):
                t = talk_of(it["id"])
                ref = units(fr.outputs[0].text, zh)
                grad, hess = 0.0, 0.0
                for said, g, l in it["seen"]:
                    n = len(said)
                    if safe_prefix(said, ref, gap) < n:   # what was already said strays from it: no evidence here
                        continue
                    k = safe_prefix(said + g, ref, gap) - n
                    z = (np.arange(len(g)) < k).astype(np.float64)
                    q = 1.0 / (1.0 + np.exp(-(l + bias[t])))
                    grad += float((z - q).sum())
                    hess += float((q * (1 - q)).sum())
                prec[t] += hess
                bias[t] += grad / prec[t]
        print(f"{tag} wave {w + 1}/{waves}: {len(wave)} sentences, {time.time() - t0:.0f}s", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 "drafts": it["drafts"], "talk_bias": round(bias[talk_of(it["id"])], 4)},
                                ensure_ascii=False) + "\n")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}; talk biases "
          f"{ {t: round(b, 3) for t, b in bias.items()} }", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--dumps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--label", choices=("prefix", "subseq"), default="prefix",
                   help="Z: the draft's prefix is the oracle's (prefix), or an in-order subsequence of it with small "
                        "insertions and no early sentence end (subseq: rewording that keeps what was said is safe)")
    c = sub.add_parser("check")
    c.add_argument("--belief", required=True)
    c.add_argument("--rows", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
