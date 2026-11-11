"""Tests of J1 (reports/mist_w3po_v1/thinker_joint_crf_prereg_v1.zh-CN.md §5 PC1), CPU only:

  (a) brute force in lattice mode t5 (and t2): every rho in [e, h] (non-decreasing per speaker) x per-read order x per
      (read, speaker) block rate class, playback simulated per path without the lattice: log Z, per-sequence weights,
      E_q[l], E_q[blocks] (the new edge_blocks statistic) and the T1-sub-lattice target (rho <= min(e + 1, h)) equal
      the augmented chain's to 1e-9
  (b) the null talker with the playback terms zeroed = latent_write's posterior in mode t5 (log Z, cend, E[Lat], T1
      target) to 1e-10
  (c) the label lattice (one path): E_q[blocks] = the number of (read, speaker) groups of the label path
  (d) the trainer step: joint_speech_train.run_step's default call == (mode "t2", dart None) == (mode "t2", dart {})
      bit for bit (rows, coefficients, loss, gradients); with DART draws the T3 rows are unchanged, the extra rows are
      latent_write_train's DART rows field for field, loss = T3 loss + loss_aug; mode t5 + DART finite with the J1 log
      fields; --prefetch 1 == 0
  (e) joint_speech_proxy.stream_blocks gives the talker prep's per-stream block order: on A''s dev decode the actual
      talker results hold exactly these texts in this order (>= 99.5 % of the streams)
  (f) j1_decide's registered rules on constructed inputs (every branch), stall / fragment flags
  (g) j1_precheck.load_plan = the trainer's load_windows + plan_steps (both candidate data sets, first 100 steps)

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_j1
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch

if not torch.cuda.is_available():    # CPU-only run: transformers' model modules probe the accelerator at import
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import j1_decide as JD
from experiments.mist_w3po_v1.crf import j1_precheck as JPC
from experiments.mist_w3po_v1.crf import joint_speech_crf as JS
from experiments.mist_w3po_v1.crf import joint_speech_proxy as JP
from experiments.mist_w3po_v1.crf import joint_speech_train as JST
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import latent_write_train as LWT
from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V
from experiments.mist_w3po_v1.crf import serial_v2p_train as V2T
from experiments.mist_w3po_v1.crf import test_joint_speech_crf as TJ
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import test_serial_v2p_train as TV
from experiments.mist_w3po_v1.crf import test_train_crf_sft as TT
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)


# ---------------------------------------------------------------- brute force (T3's, with the lattice mode and blocks)
def brute(rec: dict, tm: JS.TalkerModel, tau: float, lam: float, mode: str, delta: int = JS.DELTA_MS,
          rho_sets=None) -> dict:
    e, h = LW.bounds(rec, mode)
    R, n = rec["n_read"], len(rec["chunks"])
    ch, t_ms = rec["chunks"], rec["t_ms"]
    spk = sorted({c["slot"] for c in ch})
    by = defaultdict(list)
    for c in V.kap(rec, e, range(n)):
        by[ch[c]["slot"]].append(c)
    pairs = [(x, y) for it in by.values() for x, y in zip(it, it[1:])]
    keys, scores, lats, nbs = [], [], [], []
    ranges = [range(e[c], h[c] + 1) for c in range(n)] if rho_sets is None else rho_sets
    for rho in itertools.product(*ranges):
        if any(rho[x] > rho[y] for x, y in pairs):
            continue
        per_read, blocks = [], []
        for r in range(R):
            hist = V.kap(rec, e, [c for c in range(n) if rho[c] <= r - 1])
            Wr = V.kap(rec, e, [c for c in range(n) if rho[c] == r])
            opts = [Wr]
            carried = [c for c in Wr if e[c] <= r - 1]
            if carried:
                ps = {ch[c]["slot"] for c in carried}
                N = [c for c in Wr if e[c] == r and ch[c]["slot"] not in ps]
                if N:
                    opts.append(N + V.kap(rec, e, [c for c in Wr if c not in set(N)]))
            per_read.append([(r, tuple(S.ids_of(rec, hist)), tuple(S.ids_of(rec, o))) for o in opts])
            blocks.append([[c for c in Wr if ch[c]["slot"] == k] for k in spk])
        nb = [(r, x) for r in range(R) for x in range(len(spk)) if blocks[r][x]]
        for rates in itertools.product(range(len(tm.offs)), repeat=len(nb)):
            rate = dict(zip(nb, rates))
            b = [0] * len(spk)
            lp, l_ = 0.0, 0.0
            for r in range(R):
                dt = t_ms[r] - (t_ms[r - 1] if r > 0 else 0)
                bp = [max(0, x - dt) for x in b]
                b = list(bp)
                for x in range(len(spk)):
                    cs = blocks[r][x]
                    if not cs:
                        continue
                    words = sum(LW.n_words(ch[c]) for c in cs)
                    d = float(tm.dur(tm.n(" ".join(ch[c]["text"] for c in cs)))[rate[(r, x)]])
                    l_ += sum(LW.n_words(ch[c]) * (t_ms[r] - ch[c]["src_end_ms"]) / 1000.0 for c in cs)
                    l_ += words * bp[x] / 1000.0 + (words + 1) / 2.0 * d / 1000.0
                    lp += float(tm.logpi[min(JS.N_BIN - 1, bp[x] // JS.BIN_MS), rate[(r, x)]])
                    if d > 0:
                        end = t_ms[r] + bp[x] + d
                        b[x] = int(min(JS.CAP_MS, max(0, int(np.rint(end / delta)) * delta - t_ms[r])))
            for combo in itertools.product(*per_read):
                sc = 0.0
                for (r, hi, wi) in combo:
                    tl, ec = V.dummy_scores(r, r == R - 1, list(hi), list(wi), "new")
                    sc += float(tl.sum() + ec[-1])
                keys.append(combo)
                scores.append(tau * sc + lp - lam * l_)
                lats.append(l_)
                nbs.append(len(nb))
    scores = np.asarray(scores)
    logZ = V.lse(scores)
    q = np.exp(scores - logZ)
    Wb = defaultdict(float)
    for combo, qq in zip(keys, q):
        for k in combo:
            Wb[k] += qq
    return {"logZ": logZ, "W": Wb, "Eq_l": float((q * np.asarray(lats)).sum()),
            "E_blocks": float((q * np.asarray(nbs)).sum()), "n_paths": len(keys)}


def chain_vs_brute(rec, tm, mode, taus=(0.5, 1.0), lams=(0.0, 0.3, 1.5)) -> tuple[float, int]:
    pl = LW.plan_window(rec, LW.lattice(rec, mode), mode=mode)
    res = TL.all_scored(pl)
    w = TL.W_(0, rec)
    worst, n = 0.0, 0
    e, h = LW.bounds(rec, mode)
    for tau in taus:
        parts = JS.window_parts(w, pl, res, tm, tau)
        delta = parts["aug"]["delta"]
        for lam in lams:
            post = JS.posterior_from_parts(w, pl, parts, tau, lam)
            bf = brute(rec, tm, tau, lam, mode, delta=delta)
            d = [abs(post["logZ"] - bf["logZ"]), abs(post["Eq_l"] - bf["Eq_l"]), abs(post["exp_blocks"] - bf["E_blocks"])]
            Wt = TL.lattice_weights(pl, post)
            ks = set(Wt) | set(bf["W"])
            d.append(max(abs(Wt.get(k, 0.0) - bf["W"].get(k, 0.0)) for k in ks))
            worst = max(worst, max(d))
            n = bf["n_paths"]
        bf1 = brute(rec, tm, tau, 0.0, mode, delta=delta, rho_sets=[range(x, min(x + 1, y) + 1) for x, y in zip(e, h)])
        worst = max(worst, abs(parts["target"] - bf1["Eq_l"]))
    return worst, n


def records(rng, n: int, mode: str, max_paths: int = 150, need_free: bool = True) -> list[dict]:
    out = []
    while len(out) < n:
        rec = TL.synthetic(rng, int(rng.integers(2, 6)), int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        for c in rec["chunks"]:
            c["text"] = " ".join(["word"] * c["n_words"])
        e, h = LW.bounds(rec, mode)
        if need_free and all(x == y for x, y in zip(e, h)):
            continue
        pl = LW.plan_window(rec, LW.lattice(rec, mode), mode=mode)
        npath = math.exp(V.forward_backward([np.where(x, 0.0, -np.inf) for x in pl["edge"]], 1.0)["logZ"])
        if npath > max_paths:
            continue
        out.append(rec)
    return out


def test_brute_modes():
    tm = TJ.toy_talker()
    rng = np.random.default_rng(5)
    out = {}
    for mode in ("t5", "t2"):
        worst, paths, multi = 0.0, [], 0
        for rec in records(rng, 25 if mode == "t5" else 12, mode):
            wv, n = chain_vs_brute(rec, tm, mode)
            worst = max(worst, wv)
            paths.append(n)
            multi += len({c["slot"] for c in rec["chunks"]}) > 1
        assert worst < 1e-9 and multi >= 4, (mode, worst, multi)
        out[mode] = (worst, max(paths), multi)
    print(f"  (a) brute force (rho x order x rate) incl. E_q[blocks] and the T1 target: t5 max |d| {out['t5'][0]:.1e} "
          f"(up to {out['t5'][1]} paths, {out['t5'][2]} multi-speaker), t2 {out['t2'][0]:.1e}")


def test_null_t5():
    rng = np.random.default_rng(9)
    worst, recs = 0.0, records(rng, 25, "t5", max_paths=3000)
    for rec in recs:
        pl = LW.plan_window(rec, LW.lattice(rec, "t5"), mode="t5")
        res = TL.all_scored(pl)
        w = TL.W_(0, rec)
        for lam in (0.0, 0.4):
            p2 = LW.posterior(w, pl, res, 0.5, lam)
            p3 = JS.posterior(w, pl, res, JS.TalkerModel.null(), 0.5, lam, zero_play=True)
            d = [abs(p2["logZ"] - p3["logZ"]), abs(p2["Eq_lat"] - p3["Eq_l"]), abs(p2["exp_nf"] - p3["exp_nf"])]
            d += [float(np.max(np.abs(a - b))) if len(a) else 0.0 for a, b in zip(p2["cend"], p3["cend"])]
            worst = max(worst, max(d))
        worst = max(worst, abs(LW.window_parts(w, pl, res, 0.5)["target"]
                               - JS.window_parts(w, pl, res, JS.TalkerModel.null(), 0.5, zero_play=True)["target"]))
    assert worst < 1e-10, worst
    print(f"  (b) t5: null talker + zero playback == latent_write (log Z, cend, E[Lat], T1 target) on {len(recs)} "
          f"windows: max |d| {worst:.1e}")


def test_label_blocks():
    rng = np.random.default_rng(13)
    tm = TJ.toy_talker()
    for rec in records(rng, 20, "label", max_paths=10 ** 9, need_free=False):
        pl = LW.plan_window(rec, LW.lattice(rec, "label"))
        res = TL.all_scored(pl)
        post = JS.posterior(TL.W_(0, rec), pl, res, tm, 0.5, 0.7)
        e = S.chunk_reads(rec)
        want = len({(r, c["slot"]) for r, c in zip(e, rec["chunks"])})
        assert abs(post["exp_blocks"] - want) < 1e-9, (post["exp_blocks"], want)
    print("  (c) one-path lattice: E_q[blocks] = the label path's (read, speaker) groups on 20 windows")


# ---------------------------------------------------------------- (d) trainer step
def _run(m, wins, D, pf=False, **kw):
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True, "talker_model": TJ.toy_talker()}
    TT.zero(m)
    v = JST.run_step(m, wins, D, ctx, 3, JS.TAU, 0, 1, prefetch=pf, **kw)
    ctx["prefetched_batches"] = v["batches"]
    lg = T.grad_pass(m, v["rows"], ctx, 3)
    return v, lg, TT.grads(m)


def _same_rows(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.key == y.key and x.r == y.r and list(x.seq or []) == list(y.seq or [])
        assert np.array_equal(x.c_tok, y.c_tok) and np.array_equal(x.c_end, y.c_end)


def test_run_step():
    base = TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TT.tiny_model(seed=37)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    v0, l0, g0 = _run(m, wins, D)
    for kw in ({"mode": "t2"}, {"mode": "t2", "dart": None}, {"mode": "t2", "dart": {}}):
        v, lg, g = _run(m, wins, D, **kw)
        _same_rows(v0["rows"], v["rows"])
        assert lg["loss_main"] == l0["loss_main"], kw
        for k in g0:
            assert torch.equal(g0[k], g[k]), (kw, k)
    assert "exp_blocks" not in v0["stats"]
    draws = {w.idx: [2] * len(w.rec["chunks"]) for w in wins}
    v1, l1, g1 = _run(m, wins, D, mode="t2", dart=draws)
    v1p, l1p, g1p = _run(m, wins, D, pf=True, mode="t2", dart=draws)
    assert l1["loss_main"] == l1p["loss_main"] and all(torch.equal(g1[k], g1p[k]) for k in g1)
    n0 = len(v0["rows"])
    _same_rows(v0["rows"], v1["rows"][:n0])
    aug = v1["rows"][n0:]
    assert len(aug) == v1["stats"]["dart_rows"] > 0 and all(rw.aug for rw in aug)
    ctx = {**TV.vctx(perturb=0.2, budget=1500), "final_full_post": True}
    TT.zero(m)
    vl = LWT.run_step(m, wins, D, ctx, 3, LW.TAU, 0, 1, mode="t2", dart=draws)
    _same_rows([rw for rw in vl["rows"] if rw.aug], aug)
    tol = 1e-5 * max(1.0, abs(l1["loss_main"]))
    assert abs(l1["loss_main"] - (l0["loss_main"] + l1["loss_aug"])) < tol
    st = v1["stats"]
    assert {"exp_blocks", "lw_mode", "dart_rows", "dart_windows", "lam", "Eq_speech_lat_T3"} <= set(st)
    v5, l5, _ = _run(m, wins, D, mode="t5", dart=draws)
    assert math.isfinite(l5["loss_main"]) and v5["stats"]["lw_mode"] == "t5" and v5["stats"]["exp_blocks"] > 0
    print(f"  (d) step: default == (t2, None) == (t2, {{}}) bit for bit; DART rows = latent_write_train's "
          f"({len(aug)} rows), T3 rows unchanged, loss {l1['loss_main']:.6f} = {l0['loss_main']:.6f} + aug "
          f"{l1['loss_aug']:.6f}; prefetch 1 == 0; t5 + DART loss {l5['loss_main']:.4f}, E[blocks] "
          f"{v5['stats']['exp_blocks']:.2f}")


# ---------------------------------------------------------------- (e) proxy block order
def test_proxy_order():
    run = "eval_outputs/mist_w3po_v1/decode/dev_crf_v2p_s3000_finalpost_s8_c2g06"
    talker = "eval_outputs/mist_w3po_v1/speech/dev_crf_v2p_s3000_finalpost_s8_c2g06/talker"
    wins = {w["id"]: w for w in map(json.loads, open(JPC.WIN, encoding="utf-8"))}
    H = {h["id"]: h for h in map(json.loads, open(f"{run}/hyp.jsonl", encoding="utf-8"))}
    n = ok = 0
    for l in open(f"{talker}/results.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("error"):
            continue
        wid, k = r["id"].split("__S")
        sb = JP.stream_blocks(H[wid], wins[wid]).get(int(k), [])
        n += 1
        ok += [b["text"] for b in sb] == [b["text"] for b in r["blocks"]] and \
            [b["write_ms"] for b in sb] == [b["write_ms"] for b in r["blocks"]]
    assert ok >= 0.995 * n, (ok, n)
    print(f"  (e) proxy blocks = the actual talker's blocks (text, write_ms, order) in {ok}/{n} A' dev streams")


# ---------------------------------------------------------------- (f) decision rules
def test_decide():
    def inf(ok=True, go=True, ta=28.0, tl=3.5, px=4.0):
        return {"ok": ok, "GO": go, "text_all": ta, "text_laal": tl, "proxy": px,
                "stall": {"stall": False}, "frag": {"frag_or_quality": False}}
    assert JD.j1_base({"t4": inf(go=True), "t5": inf(go=False)})["base"] == "t4"
    assert JD.j1_base({"t4": inf(px=4.0), "t5": inf(px=3.9)})["base"] == "t5"
    assert JD.j1_base({"t4": inf(px=4.0, ta=28.5), "t5": inf(px=3.97, ta=28.0)})["base"] == "t4"
    assert JD.j1_base({"t4": inf(go=False, ta=28.0), "t5": inf(go=False, ta=27.5)})["base"] == "t4"
    assert JD.j1_base({"t4": inf(go=False, ta=28.0), "t5": inf(go=False, ta=27.95)})["base"] == "t5"
    assert JD.j1_base({"t4": inf(ok=False, go=False), "t5": inf(go=False)})["base"] == "t5"
    assert JD.j1_base({"t4": inf(ok=False), "t5": inf(ok=False)})["base"] is None
    assert JD.d1_base({"t4": inf(ta=28.3), "t5": inf(ta=28.0)})["base"] == "t4"
    assert JD.d1_base({"t4": inf(ta=28.0, tl=3.6), "t5": inf(ta=28.05, tl=3.4)})["base"] == "t5"
    assert JD.d1_base({"t4": inf(ok=False), "t5": inf(ta=20)})["base"] == "t5"
    st_ = inf()
    st_["stall"] = {"stall": True}
    assert JD.schedule({"t4": st_, "t5": inf()}) == {"any_stall": True, "any_frag_or_quality": False, "any_GO": True,
                                                       "run_D1": True, "run_J1": True, "first": "D1"}
    s2 = JD.schedule({"t4": inf(go=False), "t5": inf(go=False)})
    assert s2["run_D1"] and s2["first"] == "J1"
    s3 = JD.schedule({"t4": inf(), "t5": inf(go=False)})
    assert not s3["run_D1"] and s3["first"] == "J1"
    m = {"by_dur": {">=6s": {"A": 3, "B": 4, "C": 3}}, "B_mean_silent_reads": 4.0}
    a = {"by_dur": {">=6s": {"A": 1, "B": 4, "C": 5}}, "B_mean_silent_reads": 2.0}
    f = JD.stall_flags({"final_share": 0.45}, m, a)
    assert f["S2"] and f["S3"] and not f["S1"] and f["stall"]
    f = JD.stall_flags({"final_share": 0.50}, a, a)
    assert f["S1"] and not f["S2"] and not f["S3"]
    f = JD.stall_flags({"final_share": 0.40}, {**m, "B_mean_silent_reads": 3.9, "by_dur": {">=6s": {"A": 2, "C": 8}}}, a)
    assert not f["stall"]
    txt = 'T1 (0.5 s reads) {"windows": 3, "B_mean_silent_reads": 1.5}\nA\' (1.0 s reads) {"windows": 3}\n'
    p1, p2 = JD.parse_stall(txt)
    assert p1["B_mean_silent_reads"] == 1.5 and p2 == {"windows": 3}
    assert JD.frag_or_quality({"go_text": True, "talker_blocks": {"segment": {"mean": 3.9}}})["frag_or_quality"]
    assert not JD.frag_or_quality({"go_text": True, "talker_blocks": {"segment": {"mean": 4.2}}})["frag_or_quality"]
    assert JD.frag_or_quality({"go_text": False, "talker_blocks": {"segment": {"mean": 6}}})["frag_or_quality"]
    print("  (f) decision rules: J1 base (one GO / both GO by proxy, tie by text / neither GO by text, tie T5 / "
          "failed trainings), D1 base, schedule, S1-S3, fragment flag, stall parsing")


# ---------------------------------------------------------------- (g) plan
def test_plan():
    for name, (sched, _) in JPC.SCHEDS.items():
        recs, plan = JPC.load_plan(sched, steps=100)
        wins = T.load_windows(sched)
        p2 = T.plan_steps([w.R for w in wins], 20261109, 100, 64)
        assert plan == p2 and [r["id"] for r in recs] == [w.rec["id"] for w in wins], name
    print("  (g) j1_precheck.load_plan == the trainer's windows and plan (T4 and T5 data, 100 steps)")


def main():
    t0 = time.time()
    tests = [test_decide, test_label_blocks, test_null_t5, test_brute_modes, test_run_step, test_proxy_order, test_plan]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
