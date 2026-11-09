"""CPU tests of crf/precheck_serial.py (orders, kappa exactness, the read plan, the tempered posterior, P_complete, the
score slicing, P3 and P0) on synthetic data and the real precheck samples; the GPU scoring path is checked by a
--limit smoke run (and P3 against the earlier precheck).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_serial
"""
from __future__ import annotations

import math

import numpy as np

from experiments.mist_w3po_v1.crf import precheck_serial as S


def fake_rec(utts: list[int]) -> dict:
    chunks, ids, t = [], [], 0
    for ci, u in enumerate(utts):
        p = [S.SPK0 + u, S.NEW0 + ci, 1000 + ci, S.NEW0 + ci + 1]
        chunks.append({"utt": u, "tok0": len(ids), "tok1": len(ids) + len(p), "piece": f"<c{ci}>",
                       "src_start_ms": 100 * ci, "src_end_ms": 100 * ci + 50, "commit_ms": 100 * ci + 1050, "slot": u + 1})
        ids += p
    return {"id": "fake", "chunks": chunks, "S_ids": ids, "units": [{"utt": u, "a_ms": 0} for u in set(utts)]}


def test_orders():
    rec = fake_rec([0, 1, 1, 2])
    B = [0, 1, 2, 3]
    od = S.orders(rec, B)
    assert od[0] == tuple(B) and len(od) == len(set(od)) == 12 == S.n_orders(rec, B)
    for p in od:
        assert sorted(p) == B and p.index(1) < p.index(2)            # same-unit chunks keep their order
    od2 = S.orders(rec, B, open_first=0)
    assert all(p[0] == 0 for p in od2) and len(od2) == 3 and od2[0] == tuple(B)
    assert S.n_orders(fake_rec([5, 5, 5]), [0, 1, 2]) == len(S.orders(fake_rec([5, 5, 5]), [0, 1, 2])) == 1
    assert len(S.orders(fake_rec([0, 1, 2, 3]), [0, 1, 2, 3])) == 24


def test_seg_open_matches_engine():
    from experiments.mist_w3po_v1.core.engine import seg_state

    rng = np.random.default_rng(0)
    pool = [S.SPK0, S.SPK0 + 3, S.NEW0, S.NEW0 + 7, 100, 2000, S.WAIT_ID]
    for _ in range(2000):
        ids = list(rng.choice(pool, rng.integers(0, 12)))
        assert S.seg_open(ids) == (seg_state([int(x) for x in ids]) != "out")


def test_kappa_exact_and_plan_on_samples():
    for split, n in (("dev", 300), ("train", 120)):
        recs = S.sample(split)[:n]
        st = S.check_exact_or_die(recs)
        assert st["kappa_ids_ok"] == st["reads"] and st["hist_open"] == 0 and st["kappa_key_ties"] == 0
        for rec in recs:
            rho = S.chunk_reads(rec)
            beta = [0] + list(rec["beta_heur"])
            B = S.read_items(rec, rho)
            assert sum(len(b) for b in B) == len(rec["chunks"])
            for r, b in enumerate(B):                                # the label B_r is S[beta_{r-1}:beta_r]
                assert S.ids_of(rec, b) == rec["S_ids"][beta[r]: beta[r + 1]]
            for rd in S.read_plan(rec):
                r = rd["r"]
                assert rd["hist"] == rec["S_ids"][: beta[r]]
                assert rd["hist"] + S.ids_of(rec, rd["orders"][0]) == rec["S_ids"][: beta[r + 1]]
                if rd["capped"]:
                    assert len(rd["orders"]) == 1 and len(rd["B"]) > S.CAP
                else:
                    assert len(rd["orders"]) == rd["n_pi"] == S.n_orders(rec, rd["B"])
                for p in rd["orders"]:
                    assert sorted(p) == sorted(rd["B"])
                    assert len(S.ids_of(rec, p)) == len(S.ids_of(rec, rd["B"]))


def test_kappa_s13_known_mismatch():
    recs = {r["id"]: r for r in S.sample("dev")}
    rec = recs["IB4004_zh_0110"]
    rho = S.chunk_reads(rec)
    H = S.history(rec, rho, 3)
    assert S.kappa(rec, rho, H) == H                                 # used key: the S prefix
    assert S.kappa(rec, rho, H, "s13") != H                          # §13 as written: not the S prefix
    assert S.kappa(rec, rho, list(reversed(H))) == H                 # a function of the set only
    e = S.exactness_window(rec, keep=-1)
    assert e["stats"]["s13_bad"] >= 1 and e["s13_bad"][0]["r"] == 3


def test_tokenised_and_v8_rows():
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    recs = S.sample("dev")[:25]
    rows = S.v8_rows_of(S.V8_ROWS["dev"], {r["id"] for r in recs})
    st = {}
    for rec in recs:
        for k, v in S.exactness_window(rec, tokenizer(), rows.get(rec["id"], {}))["stats"].items():
            st[k] = st.get(k, 0) + v
    assert st["tok_checked"] == st["reads"] == st["kappa_tok_ok"]
    assert st["v8_rows"] > 0 and st["v8_prefix_ok"] == st["v8_target_ok"] == st["v8_end_ok"] == st["v8_rows"]
    assert S.exactness_ok(st)


def test_tempered():
    q, mu = S.tempered(np.log([0.25, 0.25, 0.25, 0.25]))
    assert mu == 0.0 and np.allclose(q, 0.25)
    psi = np.log([0.97, 0.01, 0.01, 0.01])
    q, mu = S.tempered(psi)
    H = -(q * np.log(q)).sum()
    assert mu > 0 and abs(H - 0.5 * math.log(4)) < 1e-6
    tau = 1 / (1 + mu)                                               # q* ∝ q^{1/(1+mu)}
    ref = np.exp(tau * psi) / np.exp(tau * psi).sum()
    assert np.allclose(q, ref)
    q2, mu2 = S.tempered(np.log([0.7, 0.3]))                         # H = 0.61 >= 0.5 log 2 = 0.35: untouched
    assert mu2 == 0.0 and np.allclose(q2, [0.7, 0.3])
    q3, mu3 = S.tempered(np.log([0.999999, 1e-6]))
    assert mu3 > 0 and abs(-(q3 * np.log(q3)).sum() - 0.5 * math.log(2)) < 1e-6


def test_read_metrics():
    # items A (label first, chunk 0) and B (chunk 1, another unit). p(y_A) = 0.3, p(rest | A) = 0.9;
    # p(y_B) = 0.6, p(rest | B) = 0.5  ->  f = B != label-first, P_complete(f) = 0.5, P_complete(label) = 0.9
    row = {"id": "x", "r": 2, "final": False, "label": [0, 1], "capped": False, "n_pi": 2,
           "items": [{"dur_s": 0.6, "ov50": True}, {"dur_s": 3.0, "ov50": True}],
           "orders": [{"pi": [0, 1], "psi": math.log(0.3 * 0.9), "first_lp": math.log(0.3), "end_at_bounds": [0, -2.0, 0]},
                      {"pi": [1, 0], "psi": math.log(0.6 * 0.5), "first_lp": math.log(0.6), "end_at_bounds": [0, -0.5, 0]}]}
    m = S.read_metrics(row)
    assert m["f_star"] == 1 and not m["f_is_label"] and m["short_ov"]
    assert abs(m["P_complete_f"] - 0.5) < 1e-12 and abs(m["P_complete_label_first"] - 0.9) < 1e-12
    assert abs(m["q_label"] - 0.27 / 0.57) < 1e-12 and abs(m["nonlabel"] - 0.30 / 0.57) < 1e-12
    assert abs(m["p_end_after_f"] - math.exp(-0.5)) < 1e-12
    assert S.read_metrics({**row, "capped": True}) is None
    assert S.read_metrics({**row, "orders": row["orders"][:1]}) is None


def test_slice_scores():
    import torch

    M, V = 6, 11
    lsm = torch.arange((M + 1) * V, dtype=torch.float64).reshape(M + 1, V)
    forced = [3, 5, 7]                                               # m = 3: positions 3, 4, 5 predict them, 6 -> e
    tl, ec = S.slice_scores(lsm, 3, forced, 9)
    assert tl.tolist() == [lsm[3, 3].item(), lsm[4, 5].item(), lsm[5, 7].item()]
    assert ec.tolist() == [lsm[k, 9].item() for k in (3, 4, 5, 6)]
    tl, ec = S.slice_scores(lsm, M, list(range(M)), 2)
    assert tl.tolist() == [lsm[k, k].item() for k in range(M)] and len(ec) == M + 1


def test_score_reqs_fake_engine():
    """score_reqs with a context-free fake model (hidden state = the input embedding): the log-prob of forced[t] must be
    log_softmax(W emb(the token before it)), e_r's after the last forced token, whatever the batch / left padding."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, Prompter
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import n_speaker_tokens

    torch.manual_seed(0)
    V, Hd = 152064, 8

    class Out:
        def __init__(self, h):
            self.last_hidden_state = h

    pr = Prompter.__new__(Prompter)                                   # the real Prompter.ids, without the processor
    pr.a_ids, pr.b_ids, pr._ctx_cache = [11, 12, AUDIO_PAD, 13], [14, 15, 16], {}
    pr.pad_at = 2

    class Eng:
        dev = torch.device("cpu")
        tok = type("T", (), {"pad_token_id": 0})()
        emb_layer = torch.nn.Embedding(V, Hd).double()
        head = torch.nn.Linear(Hd, V).double()

        class thinker:
            @staticmethod
            def model(inputs_embeds, attention_mask, use_cache, return_dict):
                return Out(inputs_embeds)

        def audio_embeds(self, its, dtype):
            out = []
            for it in its:
                n = 2 + it.t_ms // 1000
                out.append((n, torch.zeros(n + n_speaker_tokens(n), Hd, dtype=torch.float64)))
            return out

        def logits(self, h):
            return self.head(h).float()

    eng = Eng()
    eng.pr = pr
    recs = [r for r in S.sample("dev") if any(rd["n_pi"] >= 2 for rd in S.read_plan(r))][:2]
    reqs = []
    for rec in recs:
        for rd in S.read_plan(rec):
            for p in rd["orders"]:
                reqs.append((rec, rd["r"], rd["hist"], S.ids_of(rec, p)))
    assert len({len(q[2]) + len(q[3]) for q in reqs}) > 1                  # mixed lengths -> left padding is exercised
    for batch in (1, 3, 8):
        res = S.score_reqs(eng, reqs, batch, True)
        for (rec, r, hist, forced), (tl, ec) in zip(reqs, res):
            final = r == rec["n_read"] - 1
            prev = ([eng.pr.b_ids[-1]] + list(hist))[-1:] + list(forced)    # the token before each forced one, then e
            with torch.no_grad():
                lg = eng.head(eng.emb_layer(torch.as_tensor(prev))).float()
                if final:
                    lg[:, S.WAIT_ID] = -1e9
                lsm = torch.log_softmax(lg, -1).double().numpy()
            e = S.EOS_ID if final else S.WAIT_ID
            assert np.allclose(tl, lsm[np.arange(len(forced)), forced], atol=1e-5)
            assert np.allclose(ec, lsm[:, e], atol=1e-5) and len(ec) == len(forced) + 1


def test_p3_row():
    rec = {"beta_heur": [0, 3, 7], "n_read": 3}
    tok = np.full((3, 7), -0.1)
    end = np.full((3, 8), -2.0)
    row = {"r": 2, "label": [1, 2], "orders": [{"pi": [1, 2], "psi": 4 * -0.1 + -2.0 + 1e-3, "tok_lp": [-0.1] * 4,
                                                "end_at_bounds": [-2.0, -2.0, -2.0]}]}
    d = S.p3_row(row, (tok, end), rec)
    assert abs(d["abs_diff"] - 1e-3) < 1e-9 and d["tok_abs_diff_max"] == 0.0 and d["end_abs_diff"] == 0.0


def test_counts_and_p0():
    c = S.counts(S.sample("dev"))
    assert c["multi_item_reads"] == sum(c["B_size_dist"].values()) == sum(c["Pi_size_dist"].values())
    assert c["windows"] == 300 and 0 < c["windows_with_multi"] <= 300
    rep, units = S.p0_coverage(sample_ids={r["id"] for r in S.sample("dev")})
    g = rep["le1_ov50"]
    assert g["n"] == 151 and g["found_in_ref_windows"] == 151 and g["multi_ref_windows"] == 0
    assert rep["le1_all"]["n"] == 172 and rep["P0"] == g["P0_share_B_ge2"]
    assert g["share_Pi_ge2_within_cap"] <= g["share_Pi_ge2"] <= g["P0_share_B_ge2"]
    for u in units:
        assert u["n_items"] >= 1


if __name__ == "__main__":
    import time

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    t0 = time.time()
    for name, fn in tests:
        t1 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} TESTS PASSED ({time.time() - t0:.1f}s)")
