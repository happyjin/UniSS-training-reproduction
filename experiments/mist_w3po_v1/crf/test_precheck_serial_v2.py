"""CPU tests of crf/precheck_serial_v2.py: the set-state lattice with one-read slack, the canonical write order, the
forward cover, S3 (a) kappa on the label path == the S prefix, S3 (b) forward log Z / path masses / marginals ==
brute-force enumeration (synthetic and real windows, dummy log-probs, with and without forced-read skipping), the
final-read P protocol, the engine-path scoring with a fake model, the S1 classification and the analysis on dummy
scores. pytest is not installed:

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_serial_v2
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S
from experiments.mist_w3po_v1.crf import precheck_serial_v2 as V


def multi_windows(split: str, n: int, max_def: int = V.BRUTE_MAX_DEFERRABLE) -> list[dict]:
    """sample windows with a non-final label read of >= 2 items of >= 2 speakers and few deferrable items."""
    out = []
    for rec in S.sample(split):
        lat = V.lattice(rec)
        d = sum(1 for r in lat["rho"] if r < lat["R"] - 1)
        if d <= max_def and any(len({rec["chunks"][c]["slot"] for c in lat["B"][r]}) >= 2 for r in range(lat["R"] - 1)):
            out.append(rec)
        if len(out) >= n:
            break
    return out


def test_states_and_covers():
    rng = np.random.default_rng(1)
    recs = [V.synthetic_rec(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)), int(rng.integers(1, 4)))
            for _ in range(200)] + S.sample("dev")[:150]
    n_nonprefix = 0
    for rec in recs:
        lat = V.lattice(rec)
        for r, st in enumerate(lat["states"]):
            B = lat["B"][r]
            assert st[0] == tuple(B) and len(st) == len(set(st))
            if r == lat["R"] - 1:
                assert st == [tuple(B)]
                continue
            by = defaultdict(list)
            for c in B:
                by[rec["chunks"][c]["slot"]].append(c)
            assert len(st) == math.prod(len(v) + 1 for v in by.values())
            for s in st:
                assert list(s) == V.kap(rec, lat["rho"], s)                        # canonical order
                for k, v in by.items():                                            # per-speaker prefix
                    mine = [c for c in s if rec["chunks"][c]["slot"] == k]
                    assert mine == v[: len(mine)]
            cov = V.covers(st)
            seen = [j for _, ent in cov for j, _ in ent]
            assert sorted(seen) == list(range(len(st)))                           # each state exactly once
            assert cov[0][0] == tuple(B)                                           # the full continuation first
            for dmax, ent in cov:
                for j, k in ent:
                    assert st[j] == dmax[:k]
            n_nonprefix += len(cov) - 1
    assert n_nonprefix > 0                                                         # extra forwards are exercised


def test_context_written_canonical():
    for rec in S.sample("dev")[:120] + S.sample("train")[:60]:
        lat = V.lattice(rec)
        rho = lat["rho"]
        for r in range(lat["R"]):
            for i, Dp in enumerate(V.prev_states(lat, r)):
                hist, pend = V.context(rec, lat, r, i)
                A = [c for c in range(len(rec["chunks"])) if rho[c] <= r - 2] + list(Dp)
                assert hist == V.kap(rec, rho, A) == V.kap(rec, rho, list(reversed(A))) == S.kappa(rec, rho, A)
                assert set(pend) == set(lat["B"][r - 1] if r else []) - set(Dp)
                for D in lat["states"][r]:
                    w = list(pend) + list(D)
                    assert w == V.kap(rec, rho, w)                                 # canonical order = pending, then D
                    if r < lat["R"] - 1:                                           # = the pre-registered key off the final read
                        assert w == S.kappa(rec, rho, w)


def test_S3a_label_path():
    for split, n in (("dev", 300), ("train", 200)):
        recs = S.sample(split)[:n]
        ex = V.check_label_or_die(recs)
        st = ex["v2_label_path"]
        assert st["reads"] == st["hist_ok"] == st["write_ok"] == st["pend_empty"] == st["label_state0"] > 0
        assert ex["v1_kappa"]["kappa_ids_ok"] == ex["v1_kappa"]["reads"]
    import copy                                                                    # an order that is not S is caught

    def caught(rec):
        try:
            return not V.label_exact_ok(V.label_exact(rec))
        except AssertionError:                                                     # kap != the pre-registered kappa
            return True
    base = multi_windows("dev", 1)[0]
    lat = V.lattice(base)
    r = next(r for r in range(lat["R"] - 1) if len(lat["B"][r]) >= 2)
    a, b = lat["B"][r][:2]
    assert not caught(base)
    for key in ("src_end_ms", "commit_ms"):
        rec = copy.deepcopy(base)
        rec["chunks"][a][key] = rec["chunks"][b][key] + 1                         # b now sorts before a
        assert caught(rec), key


def test_S3b_synthetic_bruteforce():
    rng = np.random.default_rng(7)
    n = 0
    for _ in range(150):
        rec = V.synthetic_rec(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)), int(rng.integers(1, 4)))
        c = V.check_brute(rec)
        assert c["stats"]["pass"] == 1, (rec["id"], c)
        n += c["n_paths"] > 1
    assert n > 50                                                                  # non-trivial lattices


def test_S3b_real_windows_bruteforce():
    recs = multi_windows("dev", 4) + multi_windows("train", 3)
    assert len(recs) == 7
    big = 0
    for rec in recs:
        c = V.check_brute(rec)
        assert c["stats"]["pass"] == 1 and c["max_abs_diff"] < 1e-9, (rec["id"], c)
        big += c["n_paths"] >= 4
    assert big >= 3


def test_bruteforce_detects_wrong_potential():
    """a mis-assembled potential (boundary off by one token) must break the equality."""
    rec = multi_windows("dev", 1)[0]
    lat = V.lattice(rec)
    plan = V.plan_window(rec, lat, skip_forced=False)
    res = V.dummy_res(plan)
    good = V.forward_backward(V.psi_list(V.assemble(plan, res)), 1.0)["logZ"]
    for rd in plan["reads"]:
        for cx in rd["ctx"]:
            for pr in cx["cells"]:
                cx["cells"][pr] = {j: (q, max(0, b - 1)) for j, (q, b) in cx["cells"][pr].items()}
    bad = V.forward_backward(V.psi_list(V.assemble(plan, res)), 1.0)["logZ"]
    paths, scores = V.brute_paths(rec, lat, V.dummy_psi)
    bf = V.brute_force(rec, lat, paths, scores, 1.0)["logZ"]
    assert abs(good - bf) < 1e-9 and abs(bad - bf) > 1e-6


def test_forward_backward_hand():
    # read 0: states {B0 = (a,), ()}; read 1 (final): one state. psi0 = [[log .6, log .2]], psi1 = [[log .5], [log .9]]
    p0 = np.log([[0.6, 0.2]])
    p1 = np.log([[0.5], [0.9]])
    for tau in (0.5, 1.0):
        fb = V.forward_backward([p0, p1], tau)
        w = np.array([(0.6 * 0.5) ** tau, (0.2 * 0.9) ** tau])
        assert abs(fb["logZ"] - math.log(w.sum())) < 1e-12
        assert np.allclose(fb["marg"][0], w / w.sum()) and np.allclose(fb["marg"][1], [1.0])


def test_assemble_matches_direct_potentials():
    for rec in multi_windows("dev", 3):
        lat = V.lattice(rec)
        plan = V.plan_window(rec, lat)
        reads = V.assemble(plan, V.dummy_res(plan))
        for rd, pr in zip(reads, plan["reads"]):
            if rd["psi"] is None:
                assert pr["forced"]
                continue
            M = np.asarray(rd["psi"])
            for cx in pr["ctx"]:
                for j, D in enumerate(lat["states"][rd["r"]]):
                    w = S.ids_of(rec, list(cx["pend"]) + list(D))
                    want = V.dummy_psi(rd["r"], rd["final"], S.ids_of(rec, cx["hist"]), w)
                    assert abs(M[cx["i"], j] - want) < 1e-12
                    if rd["final"]:
                        assert abs(rd["psi_old"][cx["i"]][j] - V.dummy_psi(rd["r"], True, S.ids_of(rec, cx["hist"]), w,
                                                                           "old")) < 1e-12
            for o, cx in zip(rd["ctx"], pr["ctx"]):                               # opener entries = the items in order
                assert [x[0] for x in o["open"]] == list(cx["pend"]) + list(lat["B"][rd["r"]])


def test_final_read_post_rule():
    from experiments.mist_w3po_v1.evaluation.online_decode_finalpost import final_cut_ms

    rec = S.sample("dev")[3]
    post = np.load(rec["post"]).astype(np.float32)
    t = rec["t_ms"][-1]
    assert t == rec["dur_ms"] and len(post) == math.ceil(t / 80)
    tr = V.final_read_post(post, t, "train")
    assert np.array_equal(tr, post)                                               # train: the whole window
    dv = V.final_read_post(post, t, "dev")
    cut = final_cut_ms(t, 10 ** 9)                                                # detector time unknown -> cut = t
    assert cut == t and np.array_equal(dv[: t // 80], post[: t // 80]) and not dv[t // 80:].any()
    assert (len(post) - t // 80) <= 1                                             # only the partial last frame differs


def test_make_items_protocols():
    rec = multi_windows("dev", 1)[0]
    R = rec["n_read"]
    wavs = {}
    reqs = [(rec, R - 1, [], [1, 2], "new"), (rec, R - 1, [], [1, 2], "old"), (rec, 0, [], [3], "new")]
    its = V.make_items(reqs, wavs, "dev")
    assert its[0].full and its[0].final and not its[1].full and its[1].final and not its[2].full and not its[2].final
    assert len({it.akey for it in its}) == 3
    t = rec["t_ms"][-1]
    assert not its[0].post[t // 80:].any() and np.array_equal(its[0].post[: t // 80], its[1].post[: t // 80])
    assert np.array_equal(its[1].post, wavs[rec["id"]][1])                        # the old protocol: engine default
    assert its[0].committed == [1, 2] and its[2].t_ms == rec["t_ms"][0]


def test_score_reqs_v2_fake_engine():
    """score_reqs_v2 with a context-free fake model: log p(forced[t]) = log_softmax(W emb(previous token)) whatever the
    batch / left padding / protocol; <wait> barred at the final read; P protocol per item."""
    import torch

    from experiments.mist_w3po_v1.core.engine import Prompter
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import n_speaker_tokens

    torch.manual_seed(0)
    V_, Hd = 152064, 8

    class Out:
        def __init__(self, h):
            self.last_hidden_state = h

    pr = Prompter.__new__(Prompter)
    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD
    pr.a_ids, pr.b_ids, pr._ctx_cache = [11, 12, AUDIO_PAD, 13], [14, 15, 16], {}
    pr.pad_at = 2
    seen = []

    class Eng:
        dev = torch.device("cpu")
        tok = type("T", (), {"pad_token_id": 0})()
        emb_layer = torch.nn.Embedding(V_, Hd).double()
        head = torch.nn.Linear(Hd, V_).double()

        class thinker:
            @staticmethod
            def model(inputs_embeds, attention_mask, use_cache, return_dict):
                return Out(inputs_embeds)

        def audio_embeds(self, its, dtype):
            out = []
            for it in its:
                seen.append((it.akey, it.full, float(it.post.sum())))
                n = 2 + it.t_ms // 1000
                out.append((n, torch.zeros(n + n_speaker_tokens(n), Hd, dtype=torch.float64)))
            return out

        def logits(self, h):
            return self.head(h).float()

    eng = Eng()
    eng.pr = pr
    cand = multi_windows("dev", 12)
    with_old = [r for r in cand if any(q["proto"] == "old" for q in V.plan_window(r)["reqs"])]
    recs = with_old[:1] + [r for r in cand if r is not with_old[0]][:1]
    reqs = []
    for rec in recs:
        plan = V.plan_window(rec)
        reqs += [(rec, q["r"], q["hist_ids"], q["forced_ids"], q["proto"]) for q in plan["reqs"]]
    assert len({len(q[2]) + len(q[3]) for q in reqs}) > 1 and any(q[4] == "old" for q in reqs)
    for batch in (1, 3, 8):
        res = V.score_reqs_v2(eng, reqs, batch, "dev")
        for (rec, r, hist, forced, proto), (tl, ec) in zip(reqs, res):
            final = r == rec["n_read"] - 1
            prev = ([eng.pr.b_ids[-1]] + list(hist))[-1:] + list(forced)
            with torch.no_grad():
                lg = eng.head(eng.emb_layer(torch.as_tensor(prev))).float()
                if final:
                    lg[:, S.WAIT_ID] = -1e9
                lsm = torch.log_softmax(lg, -1).double().numpy()
            e = S.EOS_ID if final else S.WAIT_ID
            assert np.allclose(tl, lsm[np.arange(len(forced)), forced], atol=1e-5)
            assert np.allclose(ec, lsm[:, e], atol=1e-5) and len(ec) == len(forced) + 1
    fulls = {(k, f) for k, f, _ in seen}
    assert any(k.endswith("|new") and f for k, f in fulls) and any(k.endswith("|old") and not f for k, f in fulls)
    assert all(f == k.endswith("|new") for k, f in fulls)


def test_deferral_class():
    assert V.deferral_class(True, False, 3) == "DUE_FINAL"
    assert V.deferral_class(True, True, 0) == "DUE_FINAL"
    assert V.deferral_class(False, True, 2) == "U_WRITTEN_AT_DUE"
    assert V.deferral_class(False, False, 0) == "NOTHING_ELSE_WRITTEN_AT_DUE"
    assert V.deferral_class(False, False, 1) == "DEFERRAL"
    rows = [{"cls": "ORDER_OMISSION", "le05": True, "why": "DEFERRAL", "deferral": True},
            {"cls": "MISLABEL", "le05": False, "why": "DUE_FINAL", "deferral": False},
            {"cls": "ORDER_OMISSION", "le05": False, "why": "DEFERRAL", "deferral": True},
            {"cls": "OTHER", "le05": False, "why": "NO_WINDOW", "deferral": False}]
    sm = V.s1_summary(rows)
    assert sm["all_ov50_le1_missed"]["n"] == 4 and abs(sm["all_ov50_le1_missed"]["share"] - 0.5) < 1e-12
    assert sm["ORDER_OMISSION"]["share"] == 1.0 and sm["ov50_le05_missed"]["deferral"] == 1


def test_analysis_on_dummy_scores():
    per = []
    for rec in multi_windows("dev", 6):
        plan = V.plan_window(rec)
        row = {"id": rec["id"], "R": plan["lat"]["R"], "n_fw": len(plan["reqs"]),
               "states": [[list(s) for s in st] for st in plan["lat"]["states"]],
               "reads": V.assemble(plan, V.dummy_res(plan))}
        w = V.window_analysis(rec, row)
        for it in w["items"]:
            for k, v in it.items():
                if k.startswith("q_"):
                    assert -1e-12 <= v <= 1 + 1e-12
        assert all(0 < x["q_label"] <= 1 + 1e-12 for x in w["q"].values())
        assert w["opens"] and all(o["ctl_spk_lp"] is not None for o in w["opens"] if o["m_multi"])
        per.append(w)
    rep = V.split_report(per)
    s2 = rep[f"S2[new,tau={V.TAU}]"]
    assert s2["n"] == rep["S2_population"] > 0 and 0 <= s2["mean"] <= 1
    assert rep["lattice_sizes"]["total_forwards"] == sum(w["sizes"]["n_fw"] for w in per)


def test_v1_alignment_keys():
    v1 = V.v1_label_psi("dev")
    assert len(v1) == 167                                                         # the v1 dev multi-item reads
    rec = next(r for r in S.sample("dev") if any((r["id"], x) in v1 for x in range(r["n_read"])))
    lat = V.lattice(rec)
    for (i, r) in [k for k in v1 if k[0] == rec["id"]]:
        assert len(lat["B"][r]) >= 2


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")
