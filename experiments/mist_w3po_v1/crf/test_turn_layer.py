"""Tests of the end-to-end TurnCRF layer (turn_layer.py), its reference labels (turn_refs_e2e.py) and the decode
wrapper (turn_decode.py). CPU only; runnable directly (pytest is not installed):

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_turn_layer

  (a) gate: at g = 0 the layer output is bit-identical to P (float32 and float64, random non-neutral CRF and e-MLP
      parameters), a nonzero gradient reaches g, and the CRF / e-MLP parameters get gradient once g != 0 (and from the
      auxiliary loss at g = 0)
  (b) the differentiable forward-backward: log Z and frame / slot marginals equal brute-force enumeration on tiny cases,
      closed and right-censored, with and without the e terms; equal to turn_crf.semi_forward_backward and to
      d log Z / d u on a realistic size; gradcheck of the marginals; potentials equal turn_crf's
  (c) the auxiliary loss (log Z~ - log Z~_y) / tau equals brute force; path_score equals turn_crf.semi_path_score
  (d) the engine: TurnEngine without a layer and with a g = 0 layer gives the same speaker-stream embeddings, the same
      prompts / model inputs and the same greedy tokens as core/engine.Engine (stub Thinker, real feature extractor,
      real SDProjector); with g != 0 the embeddings change; turn_decode.main without --turn-crf is online_decode.main
      with the same argv
  (e) reference labels: segment parsing, the window-level re-map, the top-2 rule
"""
from __future__ import annotations

import itertools
import sys
import tempfile
import types

import numpy as np
import torch
from torch import nn

from experiments.mist_w3po_v1.crf import turn_crf as tc
from experiments.mist_w3po_v1.crf import turn_layer as tl
from experiments.mist_w3po_v1.crf.test_turn_crf import _logZ_y, _summ, brute_semi

TOL = 1e-9


def _layer(K=8, L=257, Q=50, e_dim=0, seed=0, scale=0.5, e_scale=0.3):
    torch.manual_seed(seed)
    m = tl.TurnLayer(K=K, L=L, Q=Q, e_dim=e_dim, e_hidden=8)
    with torch.no_grad():
        for p in m.crf.parameters():
            p.add_(scale * torch.randn_like(p))
        if m.e_mlp is not None:
            m.e_mlp[-1].weight.normal_(0, e_scale)
            m.e_mlp[-1].bias.normal_(0, e_scale)
    return m


def _P(B, T, K, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    P = torch.rand(B, T, K, generator=g, dtype=torch.float64)
    P[P < 0.2] = 0.0
    return P.to(dtype)


def _pot1(p, b=0):
    return {k: (v[b].detach() if k in ("u", "tr") else v.detach()) for k, v in p.items()}


# ---------------------------------------------------------------------------------------------------------------- (a)
def test_gate_identity_and_grads():
    for e_dim in (0, 6):
        m = _layer(e_dim=e_dim, seed=1)
        for dt in (torch.float32, torch.float64):
            P = _P(2, 37, 8, 3, dt)
            e = torch.randn(2, 37, 6) if e_dim else None
            out = m(P, e)
            assert out.dtype == dt and torch.equal(out, P), "g = 0 must give P bit for bit"
            out1 = m(P[0], None if e is None else e[0])
            assert torch.equal(out1, P[0])
        assert torch.equal(m(P[:, :0]), P[:, :0])
        # gradient reaches g at g = 0; the CRF parameters get none from the output at g = 0
        P = _P(1, 25, 8, 4, torch.float32)
        e = torch.randn(1, 25, 6) if e_dim else None
        w = torch.randn(1, 25, 8)
        m.zero_grad()
        (m(P, e) * w).sum().backward()
        assert m.g.grad is not None and float(m.g.grad.abs()) > 1e-6, m.g.grad
        assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.crf_parameters())
        # with g != 0 every CRF parameter that the input reaches gets gradient
        with torch.no_grad():
            m.g.fill_(0.3)
        m.zero_grad()
        (m(P, e) * w).sum().backward()
        for n, p in m.named_parameters():
            if n == "crf.end":                       # the censored prefix has no end score
                continue
            assert p.grad is not None and float(p.grad.abs().max()) > 0, n
        # the auxiliary loss reaches the CRF parameters at g = 0
        with torch.no_grad():
            m.g.zero_()
        m.zero_grad()
        y = torch.randint(0, m.crf.space.C, (1, 25))
        m.aux_loss(P, y, e).sum().backward()
        assert float(m.crf.a.grad.abs().max()) > 0 and float(m.crf.A.grad.abs().max()) > 0
    print("(a) gate identity + gradients ok")


# ---------------------------------------------------------------------------------------------------------------- (b)
def test_potentials_equal_cpu_module():
    m = _layer(seed=2)
    P = _P(3, 41, 8, 5)
    p_new = m.potentials(P)
    p_old = m.crf.potentials(P)
    for k in p_old:
        assert torch.allclose(p_new[k], p_old[k], atol=1e-12, rtol=0), k
    print("(b) potentials == turn_crf.SemiTurnCRF.potentials ok")


def test_marginals_brute():
    n = 0
    for K, T, L, Q, e_dim, seed in [(2, 4, 3, 2, 0, 0), (3, 5, 2, 2, 0, 1), (3, 5, 5, 3, 4, 2), (2, 6, 4, 2, 4, 3),
                                    (3, 1, 3, 2, 4, 4), (3, 3, 6, 2, 0, 5), (2, 5, 2, 1, 4, 6)]:
        m = _layer(K=K, L=L, Q=Q, e_dim=e_dim, seed=seed, scale=0.8)
        P = _P(1, T, K, seed + 10)
        e = torch.randn(1, T, e_dim, generator=torch.Generator().manual_seed(seed)) if e_dim else None
        p = m.potentials(P, e)
        q = _pot1(p)
        C = m.crf.space.C
        for censored in (False, True):
            items = brute_semi(q["u"], q["tr"], q["start_c"], q["end_c"], q["d_c"], q["ds_c"], censored)
            logZ, mu, _, _, _ = _summ(items, T, C)
            lz, mu2 = tl.semi_fb(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], censored)
            assert abs(float(lz[0]) - logZ) < TOL, (K, T, censored, float(lz[0]), logZ)
            assert torch.allclose(mu2[0], mu, atol=1e-9), (K, T, censored, (mu2[0] - mu).abs().max())
            if censored:
                _, _, pi, _ = m.marginals(P, e)
                assert torch.allclose(pi[0], mu @ m.crf.space.M, atol=1e-9)
            n += 1
    print(f"(b) forward-backward == brute force on {n} tiny cases (closed + censored, with / without e) ok")


def test_marginals_realistic():
    m = _layer(seed=7, scale=0.3)
    P = _P(2, 60, 8, 8)
    p = m.potentials(P)
    for censored in (True, False):
        lz, mu = tl.semi_fb(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], censored)
        for b in range(2):
            q = _pot1(p, b)
            lz_ref, mu_ref = tc.semi_forward_backward(q["u"], q["tr"], q["start_c"], q["end_c"], q["d_c"], q["ds_c"],
                                                      censored)
            assert abs(float(lz[b]) - float(lz_ref)) < 1e-8 and torch.allclose(mu[b], mu_ref, atol=1e-9)
        # marginals = d log Z / d u
        u = p["u"].detach().requires_grad_(True)
        lz2, _ = tl.semi_fb(u, p["tr"].detach(), p["start_c"].detach(), p["end_c"].detach(), p["d_c"].detach(),
                            p["ds_c"].detach(), censored)
        (gu,) = torch.autograd.grad(lz2.sum(), u)
        assert torch.allclose(gu, mu.detach(), atol=1e-9)
    print("(b) forward-backward == turn_crf.semi_forward_backward and == dlogZ/du at T = 60, K = 8 ok")


def test_gradcheck():
    torch.manual_seed(0)
    B, T, C, L = 1, 4, 4, 3
    u = torch.randn(B, T, C, dtype=torch.float64, requires_grad=True)
    tr = torch.randn(B, T, C, C, dtype=torch.float64, requires_grad=True)
    st, en = torch.randn(C, dtype=torch.float64), torch.randn(C, dtype=torch.float64)
    d = torch.randn(L, C, dtype=torch.float64, requires_grad=True)
    ds = torch.logcumsumexp(d.detach().flip(0), 0).flip(0).requires_grad_(True)
    f = lambda u_, tr_, d_, ds_: tl.semi_fb(u_, tr_, st, en, d_, ds_, True)[1]   # noqa: E731
    assert torch.autograd.gradcheck(f, (u, tr, d, ds), eps=1e-6, atol=1e-6)
    print("(b) gradcheck of the censored marginals (wrt u, tr, d, ds) ok")


# ---------------------------------------------------------------------------------------------------------------- (c)
def test_aux_loss_brute():
    n = 0
    for K, T, L, Q, e_dim, seed in [(2, 4, 3, 2, 0, 0), (3, 5, 4, 2, 4, 1), (2, 6, 6, 3, 0, 2), (3, 3, 2, 1, 4, 3)]:
        m = _layer(K=K, L=L, Q=Q, e_dim=e_dim, seed=seed, scale=0.8)
        P = _P(1, T, K, seed + 20)
        e = torch.randn(1, T, e_dim, generator=torch.Generator().manual_seed(seed)) if e_dim else None
        q = _pot1(m.potentials(P, e))
        items = brute_semi(q["u"], q["tr"], q["start_c"], q["end_c"], q["d_c"], q["ds_c"], True)
        logZ = _summ(items, T, m.crf.space.C)[0]
        g = torch.Generator().manual_seed(seed + 30)
        for _ in range(4):
            y = items[int(torch.randint(len(items), (1,), generator=g))][1]
            ref = (logZ - _logZ_y(items, list(y))) / T
            got = float(m.aux_loss(P, torch.tensor([y]), e)[0])
            assert abs(got - ref) < TOL, (got, ref)
            # cached path (the training wrapper reuses the forward's log Z and potentials)
            _, cached = m(P, e, return_aux=True)
            got2 = float(m.aux_loss(P, torch.tensor([y]), e, cached=cached)[0])
            assert abs(got2 - ref) < TOL
            n += 1
    # path_score == turn_crf.semi_path_score (batched, K = 8)
    m = _layer(seed=9)
    P = _P(3, 50, 8, 9)
    p = m.potentials(P)
    y = torch.randint(0, m.crf.space.C, (3, 50))
    y[:, 10:25] = 3
    lengths = torch.full((3,), 50)
    for cens in (False, True):
        closed, c2 = tc.semi_path_score(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], y, lengths,
                                        lengths if cens else None)
        ref = c2 if cens else closed
        got = tl.path_score(p["u"], p["tr"], p["start_c"], p["end_c"], p["d_c"], p["ds_c"], y, cens)
        assert torch.allclose(got, ref, atol=1e-9), (cens, got, ref)
    print(f"(c) aux loss == brute force ({n} label sequences); path_score == turn_crf.semi_path_score ok")


# ---------------------------------------------------------------------------------------------------------------- (d)
class _Out:
    def __init__(self, h, pkv):
        self.last_hidden_state, self.past_key_values = h, pkv


class _StubModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d)
        self.seen = []

    def forward(self, inputs_embeds, attention_mask, past_key_values=None, use_cache=True, return_dict=True):
        self.seen.append(inputs_embeds.detach().clone())
        s = inputs_embeds.sum(1) + (0 if past_key_values is None else past_key_values)
        h = torch.tanh(self.W(s / attention_mask.sum(1, keepdim=True)))
        return _Out(h[:, None].expand(-1, inputs_embeds.shape[1], -1), s)


class _StubThinker(nn.Module):
    def __init__(self, d=16, vocab=152000, n_fn=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.model = _StubModel(d)
        self.lm_head = nn.Linear(d, vocab)
        self.aproj = nn.Linear(128, d)
        self.n_fn = n_fn

    def get_input_embeddings(self):
        return self.emb

    def get_audio_features(self, feats, mask):
        n = self.n_fn(mask.sum(1)).tolist()
        out = []
        for b, nb in enumerate(n):
            x = feats[b, :, : int(mask[b].sum())].float()
            out.append(self.aproj(torch.nn.functional.adaptive_avg_pool1d(x[None], nb)[0].T))
        return types.SimpleNamespace(last_hidden_state=torch.cat(out, 0))


def _feat_len_fn():
    """transformers' Qwen3-Omni _get_feat_extract_output_lengths; its module refuses to import without an accelerator,
    then a stub module with the same formula is registered (both engines import the same function)."""
    name = "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"
    try:
        mod = __import__(name, fromlist=["_get_feat_extract_output_lengths"])
        return mod._get_feat_extract_output_lengths
    except Exception:
        def f(input_lengths):
            leave = input_lengths % 100
            feat = (leave - 1) // 2 + 1
            return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        stub = types.ModuleType(name)
        stub._get_feat_extract_output_lengths = f
        sys.modules[name] = stub
        return f


def _engines():
    from transformers import WhisperFeatureExtractor

    from experiments.mist_w3po_v1.core import engine as E
    from experiments.mist_w3po_v1.crf.turn_decode import TurnEngine
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import N_SLOT, SDProjector

    n_fn = _feat_len_fn()

    class _Prompter(E.Prompter):
        def __init__(self):
            self.proc = types.SimpleNamespace(feature_extractor=WhisperFeatureExtractor.from_pretrained(E.RELEASE))
            self.tok = types.SimpleNamespace(pad_token_id=0)
            self.a_ids = [11, 12, E.AUDIO_PAD, 13]
            self.b_ids = [14, 15]
            self.pad_at = 2
            self._ctx_cache = {}

    torch.manual_seed(0)
    th = _StubThinker(n_fn=n_fn).eval()
    sd = SDProjector(N_SLOT, 16)
    with torch.no_grad():
        sd.mlp[-1].weight.normal_(0, 0.5)
        sd.mlp[-1].bias.normal_(0, 0.1)
    sd = sd.to(torch.bfloat16).eval()
    pr = _Prompter()
    dev = torch.device("cpu")
    return E, E.Engine(th, sd, pr, dev), TurnEngine(th, sd, pr, dev), th


def _items(E):
    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.1, 16000 * 6).astype(np.float32)
    post = rng.random((75, 8)).astype(np.float32)
    post[post < 0.6] = 0.0
    wav2 = rng.normal(0, 0.1, 16000 * 4).astype(np.float32)
    post2 = rng.random((50, 8)).astype(np.float32)
    post2[post2 < 0.7] = 0.0
    return [E.ReadItem(wav=wav, post=post, t_ms=1000, final=False, committed=[], akey="a|1000"),
            E.ReadItem(wav=wav, post=post, t_ms=3000, final=False, committed=[E.SPK0, E.NEW0 + 3, 500, 501],
                       akey="a|3000"),
            E.ReadItem(wav=wav2, post=post2, t_ms=3000, final=False, committed=[], akey="b|3000"),
            E.ReadItem(wav=wav2, post=post2, t_ms=4000, final=True, committed=[E.SPK0 + 1], akey="b|4000"),
            E.ReadItem(wav=wav2, post=post2, t_ms=4000, final=True, committed=[], akey="", full=True)]


def test_engine_default_unchanged():
    E, eng, teng, th = _engines()
    items = _items(E)
    ref = eng.audio_embeds(items, torch.float32)
    a = teng.audio_embeds(items, torch.float32)                      # no layer: Engine.audio_embeds itself
    assert all(n1 == n2 and torch.equal(x1, x2) for (n1, x1), (n2, x2) in zip(ref, a))
    for e_dim in (0, 16):
        layer = _layer(e_dim=e_dim, seed=11)
        teng.set_turn(layer)
        b = teng.audio_embeds(items, torch.float32)                  # g = 0 layer
        assert all(n1 == n2 and torch.equal(x1, x2) for (n1, x1), (n2, x2) in zip(ref, b)), e_dim
        with torch.no_grad():
            layer.g.fill_(0.5)
        c = teng.audio_embeds(items, torch.float32)
        assert not all(torch.equal(x1, x2) for (_, x1), (_, x2) in zip(ref, c)), "g != 0 must change the input"
        assert torch.equal(ref[0][1], c[0][1]), "tau = 0 (t <= 1.04 s): nothing to change"
        teng.set_turn(None)
    # whole reads: prompts, model inputs and greedy tokens
    assert type(teng).read is E.Engine.read
    th.model.seen.clear()
    o_ref = eng.read(items, max_new=6)
    seen_ref = list(th.model.seen)
    th.model.seen.clear()
    teng.set_turn(_layer(e_dim=16, seed=12))
    o_new = teng.read(items, max_new=6)
    seen_new = list(th.model.seen)
    assert [o.ids for o in o_ref] == [o.ids for o in o_new] and [o.prompt_len for o in o_ref] == [o.prompt_len for o in o_new]
    assert len(seen_ref) == len(seen_new) and all(torch.equal(x, y) for x, y in zip(seen_ref, seen_new))
    print(f"(d) engine: default and g = 0 identical (embeddings, {len(seen_ref)} model inputs, tokens), g != 0 differs ok")


def test_decode_wrapper_default():
    from experiments.mist_w3po_v1.core import engine as E
    from experiments.mist_w3po_v1.crf import turn_decode as td
    from experiments.mist_w3po_v1.evaluation import online_decode as od

    seen = {}
    orig_main, orig_argv = od.main, sys.argv
    try:
        def fake():
            seen["argv"] = list(sys.argv[1:])
            seen["engine"] = od.Engine
        od.main = fake
        args = ["--thinker", "T", "--windows", "W", "--out", "O", "--context", "2"]
        td.main(list(args))
        assert seen["argv"] == args and seen["engine"] is E.Engine
        d = tempfile.mkdtemp(prefix="test_turn_layer_")          # under $TMPDIR; never deleted (no rm)
        torch.save(_layer(seed=3).export(), f"{d}/{tl.LAYER_FILE}")
        td.main(args[:2] + ["--turn-crf", d] + args[2:])
        assert seen["argv"] == args and issubclass(seen["engine"], td.TurnEngine)
        assert od.Engine is E.Engine
    finally:
        od.main, sys.argv = orig_main, orig_argv
    print("(d) turn_decode.main: no --turn-crf -> online_decode.main, same argv, original Engine ok")


# ---------------------------------------------------------------------------------------------------------------- (e)
def test_refs():
    from experiments.mist_w3po_v1.crf import turn_refs_e2e as R

    s = "<|spk2|><|t3|>Hello there.<|t10|><|spk5|><|t8|>Yes<|t12|>"
    assert R.parse_segments(s) == [(2, 3, 10), (5, 8, 12)]
    post = np.zeros((20, 8), np.float32)
    post[3:10, 0] = 0.9            # speaker of session slot 2 sits on column 0 in this window
    post[8:12, 4] = 0.9            # session slot 5 on its own column
    wm = R.window_map(R.parse_segments(s), post)
    assert wm == {2: 1, 5: 5}, wm
    rows = [{"id": "w1|lowlat|3000", "inst": "gstar", "end": "wait", "prefix": "", "target": "", "preset": "lowlat"},
            {"id": "w1|lowlat|5000", "inst": "gstar", "end": "eos", "prefix": s[:34], "target": s[34:],
             "preset": "lowlat"}]
    for mode, k1 in (("window", 0), ("session", 1)):
        idx = R.RefIndex(rows, mode)
        y = idx.labels(rows[0], 15, post)
        sp = idx.space
        assert sp.labels[y[5]] == (k1,) and sp.labels[y[9]] == tuple(sorted((k1, 4))) and sp.labels[y[11]] == (4,)
        assert y[0] == 0 and y[14] == 0 and len(y) == 15
    # > 2 active: keep the 2 longest-covered
    Rm = np.zeros((6, 8), bool)
    Rm[:, 1] = True
    Rm[2:4, 3] = True
    Rm[1:5, 6] = True
    y, n = R.joint_labels(Rm, R.LabelSpace(8))
    assert n == 2 and R.LabelSpace(8).labels[y[2]] == (1, 6)
    print("(e) reference labels: parsing, window-level re-map, top-2 rule ok")


def main():
    tests = [test_gate_identity_and_grads, test_potentials_equal_cpu_module, test_marginals_brute,
             test_marginals_realistic, test_gradcheck, test_aux_loss_brute, test_refs, test_decode_wrapper_default,
             test_engine_default_unchanged]
    for t in tests:
        t()
    print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
