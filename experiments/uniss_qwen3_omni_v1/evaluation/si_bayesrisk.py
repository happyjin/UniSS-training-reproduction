"""Test-time Bayes-risk commitment (docs/m2_test_time_bayes_model.zh-CN.md).

At every read the Thinker writes its greedy draft with the wait marker barred; the raw log-probabilities (vLLM's
default logprobs_mode) still carry the marker's probability at every position, which is the ABC hazard h(u) of
stopping after u units. Each unit gets a statistic vector T (FEATS); the survival model
    P(S >= u+1 | S >= u, T, theta) = sigmoid(theta . T)
with theta ~ N(m, P) (prior mean: the ABC logit with weight 1, so the model alone is ABC) gives the posterior
predictive P(S >= a) by Monte Carlo over theta. The action minimises the posterior expected loss
    R(a) = C_err (1 - P(S>=a)) + C_lat D sum_{u>=a} P(S>=u) + C_sil l_sil(a) + C_frag [short block, idle]
           + C_seam [dangling end] + inf [repeat loop] + inf [a inside one of the draft's tokens],
with C_err = 1, C_lat = (1 - sigmoid(b)) / window (b: the bias, as in ABC), D = max(window - p, D_min) for the p
seconds of speech still to play, and l_sil the listener-silence cost phi(g) = (g-g0)+ + lam (g-g0)+^2 per silence.
With C_sil = C_frag = C_seam = 0 and P = 0 it is ABC + SDR (Prop. 1 of the doc).
--br-loss unit instead charges every wrong unit heard and every stable unit withheld,
    R(a) = C_err E[(a - S)+] + C_lat D E[(min(S, K) - a)+],   C_lat = (1 - sigmoid(b)) / sigmoid(b) / window,
whose argmin commits the longest prefix with P(S >= a) > 1 / (1 + C_lat D): under the event loss (one revision
charged once) the global argmin commits long drafts past a doubtful unit.

v2 (docs/m2_bayesrisk_v2_model.zh-CN.md):
  --br-hyg-min m   hard action set: a block is admissible only if v3's hygiene (si_fluency_rules.hygiene) would commit
                   it unchanged (no dangling end, >= m content units while the Talker is idle unless it ends a clause);
                   this is C_frag -> infinity (Prop. 2 of the v1 model)
  --br-hyg-mode    argmin: the Bayes action over the admissible blocks; trim: over the admissible blocks no longer than
                   the unconstrained Bayes action (the rule may only delay speech, never advance it, as v3's hygiene
                   trims the self-wait prefix; argmin can jump past the confident prefix to the next admissible end)
  --br-next json   posterior of the next-commit hazard P(J = j | J >= j, x) = sigmoid(eta . x_j + c_n) (fitted by
                   data_clean/fit_next_commit_hazard.py), x known at the read (NEXT_BASE) plus the step j
  --br-abrupt lam  the expected abrupt-pause cost lam * 1[source speaking] * 1[mid-phrase seam] * P(G >= g0), G the
                   silence after the block, G = J window - playback - block duration, J from the hazard's posterior
                   predictive (Monte Carlo over eta and the sentence intercept c_n); lam is the Lagrange multiplier of
                   the abrupt-pause constraint, set by dual ascent on dev
  --br-dump also writes, for every non-final committing read, the hazard row {x, J} (type "next").

--br-dump writes every non-final read's statistics and, once its sentence ends, its stable length S (the whole-
sentence continuation from the read's committed prefix, no reference): the data the posterior is fitted on
(data_clean/fit_bayesrisk_posterior.py). --br-online updates a talk's posterior after each of its sentences
(Laplace, assumed-density filtering); the talk's sentences then run one after another.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np

FEATS = ("bias", "abc", "forced", "content", "agree", "pause", "pos", "start")
CAP = 15.0
NEXT_BASE = ("one", "logit_next", "draft_end", "withheld", "seam_mid", "seam_clause", "speaking", "wbias", "lam", "pos", "play")
NEXT_J = 6        # steps 1..NEXT_J; the last one absorbs J >= NEXT_J
MID = ("P", "P-fw", "W")


def next_x(base: np.ndarray, j: int) -> np.ndarray:
    """Hazard covariates at step j: base (known at the read) and one-hot j = 2..NEXT_J (j = 1 is the intercept)."""
    e = np.zeros(NEXT_J - 1)
    if j >= 2:
        e[min(j, NEXT_J) - 2] = 1.0
    return np.concatenate([base, e])


def next_base(ge: np.ndarray, a: int, K: int, seam: str | None, speaking: float, bias: float, lam: float, n: int,
              play: float) -> np.ndarray:
    if a < K and ge[a] > 0:
        r = float(np.clip(ge[a + 1] / ge[a], 1e-6, 1 - 1e-6))
        logit_next, end = float(np.log(r) - np.log1p(-r)), 0.0
    else:
        logit_next, end = 0.0, 1.0
    return np.array([1.0, logit_next, end, np.log1p(K - a), float(seam in MID), float(seam in ("C", "S")), speaking,
                     bias, lam, np.log1p(n), play])


def load_next(path: str) -> dict | None:
    if not path:
        return None
    j = json.load(open(path, encoding="utf-8"))
    assert tuple(j["base"]) == NEXT_BASE and j["n_j"] == NEXT_J, (j["base"], j["n_j"])
    return {"mu": np.asarray(j["mu"], float), "Sigma": np.asarray(j["Sigma"], float), "tau_c": float(j.get("tau_c", 0.0))}


def hazard_pmf(base: np.ndarray, nm: dict, rng, M: int) -> np.ndarray:
    """Posterior predictive P(J = j), j = 1..NEXT_J (the last absorbs J >= NEXT_J), by Monte Carlo over eta and c_n."""
    X = np.stack([next_x(base, j) for j in range(1, NEXT_J + 1)])
    L = np.linalg.cholesky(nm["Sigma"] + 1e-9 * np.eye(len(nm["mu"])))
    eta = nm["mu"][None] + rng.standard_normal((M, len(nm["mu"]))) @ L.T
    c = nm["tau_c"] * rng.standard_normal((M, 1))
    h = 1.0 / (1.0 + np.exp(-np.clip(eta @ X.T + c, -30, 30)))       # [M, NEXT_J]
    h[:, -1] = 1.0
    surv = np.concatenate([np.ones((M, 1)), np.cumprod(1 - h[:, :-1], axis=1)], axis=1)
    return (h * surv).mean(0)


def draft_seams(committed: list[str], gu: list[str], zh: bool) -> list[str | None]:
    """Seam type (si_fluency_stats.seam_types) after committing gu[:a], a = 0..K (None at a = 0); after the whole
    draft the next unit is unknown: S/C from its punctuation, otherwise P."""
    from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import seam_types
    K = len(gu)
    out: list[str | None] = [None] * (K + 1)
    if K == 0:
        return out
    if zh:
        blocks = ([{"text": "".join(committed)}] if committed else []) + [{"text": u} for u in gu]
    else:
        lead = bool(committed)
        blocks = ([{"text": " ".join(committed)}] if committed else []) + [
            {"text": (" " if (lead or k) else "") + u} for k, u in enumerate(gu)]
    text = "".join(b["text"] for b in blocks)
    types = seam_types(blocks, "cmn" if zh else "eng", text) if len(blocks) > 1 else []
    off = 1 if committed else 0
    for a in range(1, K):
        out[a] = types[a - 1 + off] if a - 1 + off < len(types) else "P"
    last = gu[-1]
    out[K] = "S" if last.endswith(tuple(".?!。？！")) else "C" if last.endswith(tuple(",;:，；：、")) else "P"
    return out


def prior_model() -> dict:
    m = np.zeros(len(FEATS))
    m[FEATS.index("abc")] = 1.0
    m[FEATS.index("forced")] = CAP   # where the model cannot stop, ABC never stops: survival ~1
    return {"mu": m, "P": np.zeros((len(FEATS), len(FEATS))), "psi": np.zeros(len(FEATS)), "like_w": 1.0}


def load_model(path: str, direction: str) -> dict:
    if not path:
        return prior_model()
    j = json.load(open(path, encoding="utf-8"))
    assert tuple(j["feats"]) == FEATS, j["feats"]
    d = j["dirs"][direction]
    return {"mu": np.asarray(d["mu"], float), "P": np.asarray(d["Sigma"], float),
            "psi": np.asarray(d.get("psi", [0.0] * len(FEATS)), float), "like_w": float(d.get("like_w", 1.0))}


def unit_stats(out, cont: str, zh: bool, wait_id: int, eos_id: int) -> tuple[list[float], list[float]]:
    """Per unit of cont: the wait marker's log-probability where the unit starts (-inf inside a token that began
    with an earlier unit: the model cannot stop there) and the chosen token's share of the non-wait mass."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S" if zh else r"\S+", cont)]
    lw = [float("-inf")] * len(spans)
    lc = [0.0] * len(spans)
    seen, pos = set(), 0
    for tid, alts in zip(out.token_ids, out.logprobs or []):
        if tid == eos_id:
            break
        ch = alts.get(tid)
        text = ch.decoded_token if ch is not None and ch.decoded_token is not None else ""
        lead = len(text) - len(text.lstrip())
        start = pos + lead
        u = next((k for k, (a, b) in enumerate(spans) if a <= start < b), None)
        if u is not None and u not in seen and spans[u][0] == start:
            seen.add(u)
            w = alts.get(wait_id)
            lwu = float(w.logprob) if w is not None else min(float(a.logprob) for a in alts.values())
            lw[u] = lwu
            lch = float(ch.logprob) if ch is not None else -20.0
            lc[u] = lch - float(np.log1p(-min(np.exp(lwu), 1 - 1e-9)))
        pos += len(text)
    return lw, lc


def features(lw, lc, gu, prev_abs, n, pause, start) -> np.ndarray:
    X = np.zeros((len(gu), len(FEATS)))
    for u in range(len(gu)):
        forced = 0.0
        if np.isfinite(lw[u]):
            h = min(np.exp(lw[u]), 1 - 1e-9)
            abc = float(np.log1p(-h) - lw[u])
        else:   # inside a token that began with an earlier unit: the model cannot stop here
            abc, forced = 0.0, 1.0
        agree = 0.0
        if prev_abs is not None and n + u < len(prev_abs):
            agree = 1.0 if prev_abs[n + u] == gu[u] else -1.0
        X[u] = (1.0, float(np.clip(abc, -10, CAP)), forced, float(np.clip(lc[u], -10, 0)), agree, pause, np.log1p(u), start)
    return X


def has_loop(us: list[str]) -> bool:
    """The sequence ends in three copies of a 2-6 unit phrase or four of one unit."""
    for k in range(1, 7):
        reps = 4 if k == 1 else 3
        if len(us) >= reps * k and all(us[-k * (i + 1): len(us) - k * i] == us[-k:] for i in range(reps)):
            return True
    return False


def loop_free(committed: list[str], gu: list[str]) -> int:
    """The longest prefix of gu that does not complete a repeat loop."""
    for a in range(1, len(gu) + 1):
        if has_loop(committed + gu[:a]):
            return max(0, a - 1)
    return len(gu)


def p_survive(X: np.ndarray, mu: np.ndarray, P: np.ndarray, rng, M: int) -> np.ndarray:
    """Posterior predictive P(S >= a), a = 0..K, by Monte Carlo over theta (exact when P = 0)."""
    K = len(X)
    if K == 0:
        return np.ones(1)
    if not P.any():
        th = mu[None, :]
    else:
        L = np.linalg.cholesky(P + 1e-9 * np.eye(len(mu)))
        th = mu[None, :] + rng.standard_normal((M, len(mu))) @ L.T
    rho = 1.0 / (1.0 + np.exp(-np.clip(th @ X.T, -30, 30)))   # [M, K]
    ge = np.concatenate([np.ones((len(th), 1)), np.cumprod(rho, axis=1)], axis=1)
    return ge.mean(0)


def laplace_update(mu, P, rows, y, w=1.0, iters=25):
    """Gaussian prior N(mu, P) x Bernoulli-logit likelihood -> Laplace approximation of the posterior."""
    if not len(y):
        return mu, P
    Pinv = np.linalg.inv(P + 1e-9 * np.eye(len(mu)))

    def neg_log_post(t):
        z = rows @ t
        return 0.5 * (t - mu) @ Pinv @ (t - mu) + w * float(np.sum(np.logaddexp(0, z) - y * z))

    th = mu.copy()
    f = neg_log_post(th)
    for _ in range(iters):   # Newton with backtracking: the ABC logit's cap separates some rows
        s = 1.0 / (1.0 + np.exp(-np.clip(rows @ th, -30, 30)))
        g = Pinv @ (th - mu) - w * rows.T @ (y - s)
        H = Pinv + w * (rows * (s * (1 - s))[:, None]).T @ rows
        step = np.linalg.solve(H, g)
        lr = 1.0
        while lr > 1e-4 and neg_log_post(th - lr * step) > f:
            lr /= 2
        th = th - lr * step
        f_new = neg_log_post(th)
        if abs(f - f_new) < 1e-8 * max(1.0, abs(f)):
            f = f_new
            break
        f = f_new
    s = 1.0 / (1.0 + np.exp(-np.clip(rows @ th, -30, 30)))
    H = Pinv + w * (rows * (s * (1 - s))[:, None]).T @ rows
    return th, np.linalg.inv(H)


def label_rows(X: np.ndarray, S: int):
    """The chain's labels: units before S survived (1); unit S, when the draft goes past it, did not (0)."""
    K = len(X)
    if K == 0:
        return X[:0], np.zeros(0)
    n = min(S + 1, K)
    y = np.ones(n)
    if S < K:
        y[S] = 0.0
    return X[:n], y


def phi(g: float, g0: float, lam: float) -> float:
    e = max(0.0, g - g0)
    return e + lam * e * e


def run_bayesrisk(llm, items, head, zh, window, out, tag, bias, cfg, model, units, join, wait_id, eos_id,
                  clean=False, dump=None, online=False):
    from vllm import SamplingParams

    from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules import clean as clean_text, content_units, hygiene, trim_dangling

    draft_p = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[eos_id], logit_bias={wait_id: -100.0},
                             logprobs=20)
    full_p = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[eos_id], logit_bias={wait_id: -100.0})
    rng = np.random.default_rng(cfg.get("seed", 0))
    W = window / 1000.0
    unit_loss = cfg.get("loss", "event") == "unit"
    tau0 = 1.0 / (1.0 + np.exp(-bias))
    if unit_loss:   # commit iff P(S >= a+1) > 1 / (1 + C_lat D): tau0 at D = window
        c_lat = (1.0 - tau0) / tau0 / W
        d_min = (1.0 - cfg["tau_max"]) / cfg["tau_max"] / c_lat
    else:
        c_lat = (1.0 - tau0) / W
        d_min = (1.0 - cfg["tau_max"]) / c_lat
    talk_post = {}
    nm = cfg.get("next_model")
    lam = cfg.get("abrupt", 0.0)
    for it in items:
        it.update(committed=[], blocks=[], drafts=[], done=False, busy=0.0, spoken="", prev_abs=None, reads=[], r=0,
                  waiting=False)
    nxt = {}
    if online:   # a talk's sentences in turn, so a sentence's labels reach only the later ones
        last = {}
        for it in items:
            talk = it["id"].rsplit("_", 1)[0]
            it["waiting"] = talk in last
            if talk in last:
                nxt[last[talk]["id"]] = it
            last[talk] = it
    rec_out = open(dump, "x", encoding="utf-8") if dump else None
    trace = {}
    t0, rnd = time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"] and not it["waiting"]]
        prompts = []
        for it in live:
            it["t"] = min((it["r"] + 1) * window, it["dur_ms"])
            prompts.append({"prompt": head + join(it["committed"], zh),
                            "multi_modal_data": {"audio": (it["audio"][: it["t"] * 16], 16000)}})
        res = llm.generate(prompts, draft_p, use_tqdm=False)
        for it, r in zip(live, res):
            o = r.outputs[0]
            cont = o.text
            sep = "" if zh or not it["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
            draft = units(join(it["committed"], zh) + sep + cont, zh)
            it["drafts"].append({"t": it["t"], "draft": join(draft, zh)})
            n = len(it["committed"])
            gu = draft[n:]
            final = it["t"] >= it["dur_ms"]
            if final:
                a = loop_free(it["committed"], gu)
                it["done"] = True
            else:
                lw, lc = unit_stats(o, cont, zh, wait_id, eos_id)
                off = len(lw) - len(gu)   # a leading punctuation unit merged into the committed last word
                lw, lc = (lw[off:], lc[off:]) if off >= 0 else (lw + [float("-inf")] * -off, lc + [0.0] * -off)
                db = it["frame_db"]
                hi = min(it["t"] // 20, len(db))
                pause = float(hi >= 10 and (db[hi - 10: hi] < db.max() - 25).all())
                X = features(lw, lc, gu, it["prev_abs"], n, pause, float(n == 0))
                talk = it["id"].rsplit("_", 1)[0]
                mu, P = talk_post.get(talk, (model["mu"], model["P"] + np.diag(model["psi"])))
                ge = p_survive(X, mu, P, rng, cfg["mc"])
                K = len(gu)
                p = max(0.0, it["busy"] - it["t"]) / 1000.0
                speed = 1.0 + (cfg["speed_step"] * min(cfg["shift_max"], cfg["shift_per_s"] * p) if cfg["playback"] else 0.0)
                D = max(W - p, d_min)
                g = max(0.0, it["t"] - it["busy"]) / 1000.0 if p == 0 else 0.0
                if unit_loss:   # E[(a - S)+] and E[(min(S, K) - a)+]
                    err_c = np.concatenate([[0.0], np.cumsum(1.0 - ge[1:K + 1])])
                    tail = np.concatenate([np.cumsum(ge[1:K + 1][::-1])[::-1], [0.0]])
                else:   # 1[S < a] and sum_{u >= a} P(S >= u)
                    err_c = 1.0 - ge[:K + 1]
                    tail = np.concatenate([np.cumsum(ge[:K][::-1])[::-1], [0.0]])
                best, a = None, 0
                fi = FEATS.index("forced")
                speaking = 1.0 - pause
                a_free = K + 1
                if cfg.get("hyg_min") and cfg.get("hyg_mode") == "trim":   # the unconstrained unit-loss action
                    a_free = int(np.argmin(err_c + c_lat * D * tail))
                v2 = cfg.get("hyg_min", 0) or lam > 0 or rec_out is not None
                seams = draft_seams(it["committed"], gu, zh) if v2 else [None] * (K + 1)
                idle = it["busy"] <= it["t"]
                for cand in range(K + 1):
                    if 0 < cand < K and X[cand, fi] > 0:   # the action set: token boundaries only -- a prompt that ends
                        continue                          # inside the draft's token is out of the Thinker's distribution
                    blk = gu[:cand]
                    if cand and has_loop(it["committed"] + blk) and not has_loop(it["committed"]):
                        continue
                    if cand > a_free:
                        break
                    if cand and cfg.get("hyg_min") and hygiene(it["committed"], blk, zh, idle, cfg["hyg_min"]) != blk:
                        continue   # v2 hard action set: v3's hygiene would not commit this block as it is
                    ncu = content_units(blk, zh)
                    risk = err_c[cand] + c_lat * D * tail[cand]
                    if lam > 0 and cand and ncu and speaking and seams[cand] in MID:
                        pmf = hazard_pmf(next_base(ge, cand, K, seams[cand], speaking, bias, lam, n, p), nm, rng, cfg["mc"])
                        dur = len(blk) * cfg["rate"] / speed
                        gaps = np.arange(1, NEXT_J + 1) * W - p - dur
                        risk += lam * float(pmf[gaps >= cfg["g0"]].sum())
                    if cfg["c_sil"]:
                        dur = len(blk) * cfg["rate"] / speed
                        if ncu == 0:
                            risk += cfg["c_sil"] * (phi(g + max(0.0, W - p), cfg["g0"], cfg["lam"]) - phi(g, cfg["g0"], cfg["lam"]))
                        else:
                            risk += cfg["c_sil"] * phi(max(0.0, W - p - dur), cfg["g0"], cfg["lam"])
                    if cfg["c_frag"] and 1 <= ncu < cfg["frag_min"] and p == 0 and not blk[-1].endswith(
                            tuple(",.;:!?，。；：！？、…")):
                        risk += cfg["c_frag"]
                    if cfg["c_seam"] and ncu and len(trim_dangling(it["committed"], blk, zh)) < cand:
                        risk += cfg["c_seam"]
                    if best is None or risk < best - 1e-12:
                        best, a = risk, cand
                it["reads"].append((join(it["committed"], zh), gu, X))
                if rec_out is not None or online:
                    it.setdefault("rd", []).append({"t": it["t"], "n": n, "K": K, "a": a, "gu": gu,
                                                    "X": np.round(X, 4).tolist(), "ge_a": float(ge[a]),
                                                    "nx": np.round(next_base(ge, a, K, seams[a], speaking, bias, lam, n, p),
                                                                   4).tolist() if v2 and a else None})
                it["prev_abs"] = draft
            new = gu[:a]
            if final:
                it["final_new"] = bool(new)
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                said = len(new)
                if clean:
                    raw, text = text, clean_text(it["spoken"], text, zh)
                    said = len(units(text, zh))
                    it["spoken"] += text
                    if content_units(units(text, zh), zh) or it["done"]:
                        it["blocks"].append({"write_ms": it["t"], "text": text, "raw": raw, "final": it["done"]})
                else:
                    it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"]})
                it["committed"] += new
                if said:
                    p = max(0.0, it["busy"] - it["t"]) / 1000.0
                    speed = 1.0 + (cfg["speed_step"] * min(cfg["shift_max"], cfg["shift_per_s"] * p) if cfg["playback"] else 0.0)
                    it["busy"] = max(it["busy"], it["t"]) + said * cfg["rate"] / speed * 1000
            it["r"] += 1
        fin = [it for it in live if it["done"] and it["reads"]]
        if fin and (rec_out is not None or online):   # label the finished sentences' reads: S from the whole sentence
            todo = [(it, rd) for it in fin for rd in it["reads"]]
            full = [{"prompt": head + rd[0], "multi_modal_data": {"audio": (it["audio"], 16000)}} for it, rd in todo]
            outs = llm.generate(full, full_p, use_tqdm=False)
            per = {}
            for (it, (_, gu, X)), o in zip(todo, outs):
                ou, L = units(o.outputs[0].text, zh), 0
                while L < min(len(gu), len(ou)) and gu[L] == ou[L]:
                    L += 1
                per.setdefault(it["id"], []).append((X, L))
            for it in fin:
                labs = per.get(it["id"], [])
                if rec_out is not None:   # hazard rows: J = reads until the next read that commits (the final read does)
                    rds = it.get("rd", [])
                    commit_at = [k for k, rd in enumerate(rds) if rd["a"] > 0] + [len(rds)]
                    for k, rd in enumerate(rds):
                        if rd["a"] > 0 and rd.get("nx") is not None:
                            nxt_k = next(c for c in commit_at if c > k)
                            rec_out.write(json.dumps({"type": "next", "id": it["id"], "t": rd["t"], "x": rd["nx"],
                                                      "J": nxt_k - k, "cens": int(nxt_k == len(rds) and not it["final_new"])}) + "\n")
                if rec_out is not None:
                    for rd, (_, L) in zip(it.get("rd", []), labs):
                        rec_out.write(json.dumps({"id": it["id"], **rd, "S": L}, ensure_ascii=False) + "\n")
                if online and labs:
                    talk = it["id"].rsplit("_", 1)[0]
                    mu, P = talk_post.get(talk, (model["mu"], model["P"] + np.diag(model["psi"])))
                    rows = [label_rows(X, L) for X, L in labs]
                    R = np.concatenate([r_[0] for r_ in rows])
                    y = np.concatenate([r_[1] for r_ in rows])
                    mu, P = laplace_update(mu, P, R, y, cfg["like_w"] or model["like_w"])
                    talk_post[talk] = (mu, P)
                    trace.setdefault(talk, []).append({"id": it["id"], "mu": np.round(mu, 3).tolist(),
                                                       "sd": np.round(np.sqrt(np.diag(P)), 3).tolist()})
        for it in [it for it in live if it["done"]]:
            it["reads"] = []
            it.pop("rd", None)
            if it["id"] in nxt:
                nxt[it["id"]]["waiting"] = False
        rnd += 1
        print(f"{tag} round {rnd}: {len(live)} sentences, {time.time() - t0:.0f}s", flush=True)
    if rec_out is not None:
        rec_out.close()
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 **({"spoken": it["spoken"]} if clean else {}), "drafts": it["drafts"]},
                                ensure_ascii=False) + "\n")
    if online:
        out.with_suffix(".posterior.json").write_text(json.dumps(trace), encoding="utf-8")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)


def _selftest() -> None:
    assert has_loop("Then go to the . Then go to the . Then go to the .".split())
    assert has_loop(["Or."] * 4) and not has_loop(["Or."] * 3)
    assert not has_loop("a b c d e f".split())
    assert loop_free([], ["x", "y", "x", "y", "x", "y", "z"]) == 5
    X = np.array([[1, 2.0, 0, 0, 0, 0, 0, 0], [1, -1.0, 0, 0, 0, 0, 0, 0]])
    m = prior_model()
    ge = p_survive(X, m["mu"], m["P"], np.random.default_rng(0), 8)
    assert p_survive(np.array([[1, 0, 1.0, 0, 0, 0, 0, 0]]), m["mu"], m["P"], None, 1)[1] > 0.999
    s = 1 / (1 + np.exp(-np.array([2.0, -1.0])))
    assert np.allclose(ge, [1, s[0], s[0] * s[1]])
    R, y = label_rows(X, 1)
    assert y.tolist() == [1.0, 0.0] and len(R) == 2
    R, y = label_rows(X, 2)
    assert y.tolist() == [1.0, 1.0]
    mu, P = laplace_update(m["mu"], np.eye(8), np.tile(X, (50, 1)), np.tile([1.0, 0.0], 50))
    assert P.shape == (8, 8) and np.all(np.linalg.eigvalsh(P) > 0)
    # Prop. 1: with no extra costs and P = 0, the risk's first difference has the threshold's sign
    rho = np.array([0.9, 0.5]); ge = np.concatenate([[1], np.cumprod(rho)]); c, Dk = 0.4, 1.0
    tail = np.concatenate([np.cumsum(ge[:2][::-1])[::-1], [0]])
    R_ = [(1 - ge[a]) + c * Dk * tail[a] for a in range(3)]
    for a in range(2):
        assert np.sign(R_[a + 1] - R_[a]) == np.sign((1 - rho[a]) - c * Dk)
    # unit loss: the argmin is the longest prefix whose marginal survival passes 1 / (1 + C_lat D)
    rho = np.array([0.95, 0.6, 0.99, 0.99]); ge = np.concatenate([[1], np.cumprod(rho)]); cD = 0.5
    err_c = np.concatenate([[0.0], np.cumsum(1.0 - ge[1:])]); tail = np.concatenate([np.cumsum(ge[1:][::-1])[::-1], [0.0]])
    a_star = int(np.argmin(err_c + cD * tail))
    assert a_star == max([a for a in range(5) if all(ge[u + 1] > 1 / (1 + cD) for u in range(a))])
    # v2: seam types per candidate, hazard pmf sums to one, the hard action set is v3's hygiene
    sm = draft_seams(["I", "think"], ["that", "the", "city,", "is", "big."], False)
    assert sm[0] is None and sm[3] == "C" and sm[5] == "S", sm
    nm = {"mu": np.zeros(len(NEXT_BASE) + NEXT_J - 1), "Sigma": 0.01 * np.eye(len(NEXT_BASE) + NEXT_J - 1), "tau_c": 0.5}
    pmf = hazard_pmf(next_base(np.array([1, 0.9, 0.5]), 1, 2, "P", 1.0, 0.0, 1.0, 2, 0.3), nm, np.random.default_rng(0), 64)
    assert abs(pmf.sum() - 1) < 1e-9 and np.all(pmf >= 0) and len(pmf) == NEXT_J
    from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules import hygiene
    assert hygiene([], ["the"], False, True, 4) == []                    # dangling function word: inadmissible
    assert hygiene(["I", "said"], ["it", "is", "fine,"], False, True, 4) == ["it", "is", "fine,"]   # a clause end
    print("si_bayesrisk selftest ok")


if __name__ == "__main__":
    _selftest()
