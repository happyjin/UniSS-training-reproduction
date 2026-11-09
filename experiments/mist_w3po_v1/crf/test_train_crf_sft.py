"""Tests of crf/train_crf_sft.py (scheme 1' trainer and C0), CPU only, a tiny random causal LM in place of v8 (real
tokenizer / feature extractor / prompts, real sched_v1 AMI-train windows with their audio and posteriors):

  (a) C0 = plain SFT: the --objective heuristic loss and gradients (LoRA a/b, projector, new token rows) equal v8's own
      loss (thinker_gstar_model.ThinkerGStarModel.forward) on the same reads as v8 rows built by v8's build_batch
      (pretrain_thinker_gstar.build_batch: prefix S[:beta_{r-1}], target block, end)
  (b) Fisher identity: the --objective sched gradient (h = 1, D = 1, no L_state) equals autograd of -log Z_lam through
      the lattice DP on differentiable log-probs (--final-bar-wait 0); with the bar the lattice's last-read scores are
      the decoding distribution (<wait> barred before the softmax at every position), the loss's stay plain
  (c) post_transform None: the batch tensors are v8's build_batch tensors bit for bit and the forward (embeddings,
      hidden states) is bit-identical to ThinkerGStarModel.embed + thinker.model; the hook path with the identity is
      equal to it, a parametrised hook gets gradients, P_prefix / enc shapes and extra_loss info are as documented
  (d) weights: sum_{r,j} w_tok = N and sum_{r,b} w_end = R per window (sched and C0), so D = sum (sum h + R) for both;
      entropy and q(heuristic) against brute force
  plus: torch post_to_tokens = numpy, lr schedule = Megatron's logged values, step plan, checkpoint round trip (lora.pt
  in the rl/train_w3po format, loadable by lora.load_state_dict on a fresh injection).

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_train_crf_sft
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.mist_w3po_v1.crf import schedule_lattice as sl
from experiments.mist_w3po_v1.crf import train_crf_sft as T
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    ATTN, N_SLOT, ThinkerGStarModel, post_to_tokens)
from experiments.uniss_streaming_p2st_traj_v1.training import lora

torch.set_num_threads(8)
SCHED = T.SCHED
_CACHE = {}


# ---------------------------------------------------------------- tiny model
class TinyThinker(nn.Module):
    """The thinker interface the trainer uses: get_input_embeddings, get_audio_features, model (causal LM body with
    model.layers.N.self_attn.{q,k,v,o}_proj), lm_head, audio_tower. Real vocabulary size, real audio-token counts."""

    def __init__(self, d: int = 32, V: int = 151936, seed: int = 0):
        super().__init__()
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(seed)
        cfg = Qwen3Config(vocab_size=8, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=d // 4, max_position_embeddings=8192)
        cfg._attn_implementation = "eager"
        self.model = Qwen3Model(cfg)
        self.embed = nn.Embedding(V, d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.audio_tower = nn.Linear(128, d)
        with torch.no_grad():
            self.embed.weight.mul_(0.5)
            self.lm_head.weight.mul_(3.0)

    def get_input_embeddings(self):
        return self.embed

    def get_audio_features(self, input_features, feature_attention_mask):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

        lens = feature_attention_mask.sum(1)
        n = _get_feat_extract_output_lengths(lens)
        outs = []
        for i in range(input_features.shape[0]):
            x = self.audio_tower(input_features[i, :, : int(lens[i])].T.to(self.audio_tower.weight.dtype))
            outs.append(F.adaptive_avg_pool1d(x.T[None], int(n[i]))[0].T)
        return SimpleNamespace(last_hidden_state=torch.cat(outs, 0))

    def gradient_checkpointing_enable(self, **kw):
        pass


def tiny_model(post_transform=None, seed: int = 0, b_scale: float = 0.05) -> T.CrfThinker:
    th = TinyThinker(seed=seed)
    sd = T.SDProjector(N_SLOT, 32)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():   # a generic point: non-zero projector output layer and LoRA b
        sd.mlp[-1].weight.copy_(0.05 * torch.randn(sd.mlp[-1].weight.shape, generator=g))
    m = T.CrfThinker(th, sd.state_dict(), 4, 8.0, post_transform)
    with torch.no_grad():
        for ad in m.adapters:
            ad.b.copy_(b_scale * torch.randn(ad.b.shape, generator=g))
    return m


def env():
    if "env" not in _CACHE:
        from experiments.mist_w3po_v1.core.engine import Prompter

        pr = Prompter("zh2en")
        _CACHE["env"] = (pr, pr.proc)
    return _CACHE["env"]


def ctx_of(perturb: float = 0.0, budget: int = 4000, bar: bool = True) -> dict:
    pr, proc = env()
    return {"audio": T.AudioCache(), "proc": proc, "prompter": pr, "perturb": perturb, "seed": 7, "tok_budget": budget,
            "lp_chunk": 64, "final_bar_wait": bar, "n_prompt0": len(pr.a_ids) + len(pr.b_ids) - 1}


def pick_windows(n: int, hier: bool = True, max_n: int = 30, min_r: int = 2, max_r: int = 4, offset: int = 0):
    recs = [json.loads(l) for l in open(Path(SCHED) / "shard_000.jsonl", encoding="utf-8")]
    good = [r for r in recs if min_r <= r["n_read"] <= max_r and 8 <= r["n_tok"] <= max_n
            and len(set(r["beta_heur"])) >= 2 and len(r["chunks"]) >= 2]
    wins = [T.make_window(i, r, hier) for i, r in enumerate(good[offset: offset + n])]
    T.set_reach(wins, 128)
    return wins


def grads(m: T.CrfThinker) -> dict:
    out = {}
    for i, ad in enumerate(m.adapters):
        out[f"a{i}"], out[f"b{i}"] = ad.a.grad.clone(), ad.b.grad.clone()
    for k, p in m.sd_proj.named_parameters():
        out["sd." + k] = p.grad.clone()
    out["new_emb"], out["new_head"] = m.new_emb.grad.clone(), m.new_head.grad.clone()
    return out


def zero(m):
    for p in m.parameters():
        p.grad = None


def cmp_grads(g1: dict, g2: dict, rtol: float, name: str) -> float:
    worst = 0.0
    for k in g1:
        a, b = g1[k].double(), g2[k].double()
        scale = max(float(b.abs().max()), 1e-12)
        err = float((a - b).abs().max()) / scale
        worst = max(worst, err)
        assert err < rtol, f"{name}: grad {k} rel err {err:.3e}"
    return worst


# ---------------------------------------------------------------- v8 reference rows
def v8_items(w: T.Window, audio: T.AudioCache) -> list[dict]:
    """Every read of the window as a v8 row ({"prefix", "target", "end", ...}; pretrain_thinker_gstar.GStarRows)."""
    items = []
    wav, post = audio.get(w)
    prev = 0
    for r, b in enumerate(w.beta.tolist()):
        pre = "".join(c["piece"] for c in w.rec["chunks"] if c["tok1"] <= prev)
        tgt = "".join(c["piece"] for c in w.rec["chunks"] if c["tok0"] >= prev and c["tok1"] <= b)
        t = w.t_ms[r]
        x = wav[: t * 16]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        p = post.copy()
        p[max(0, t - 1040) // 80:] = 0.0
        items.append({"wav": x, "dir": "zh2en", "inst": "gstar", "prefix": pre, "target": tgt,
                      "end": "eos" if r == w.R - 1 else "wait", "post": p, "_r": r, "_b0": prev, "_b1": b})
        prev = b
    return items


def v8_forward_loss(m: T.CrfThinker, items: list[dict]):
    """v8's SFT loss on v8 rows: pretrain_thinker_gstar.build_batch + ThinkerGStarModel.forward (called unbound)."""
    from experiments.uniss_qwen3_omni_v1.training import pretrain_thinker_gstar as pg

    pr, proc = env()
    batch = pg.build_batch([{k: v for k, v in it.items() if not k.startswith("_")} for it in items], T.V8)
    out = ThinkerGStarModel.forward(m, batch["input_ids"], batch["attention_mask"], batch["labels"],
                                    batch["input_features"], batch["feature_attention_mask"], batch["sd_post"],
                                    batch["n_audio"], batch["has_sd"])
    return out[0], batch


# ---------------------------------------------------------------- (a)
def test_c0_equals_v8_sft():
    wins = pick_windows(2)
    ctx = ctx_of()
    m = tiny_model()
    items = [it for w in wins for it in v8_items(w, ctx["audio"])]
    for w, it0 in zip(wins, [v8_items(w, ctx["audio"]) for w in wins]):   # the v8 rows' token ids = S slices
        pr, proc = env()
        for it in it0:
            assert proc.tokenizer(it["prefix"], add_special_tokens=False)["input_ids"] == w.S[: it["_b0"]]
            assert proc.tokenizer(it["target"], add_special_tokens=False)["input_ids"] == w.S[it["_b0"]: it["_b1"]]
    zero(m)
    ref, _ = v8_forward_loss(m, items)
    ref.backward()
    g_ref = grads(m)
    D, A = T.step_totals(wins)
    weights = {w.idx: T.heur_weights(w) for w in wins}
    rows = T.train_rows(wins, weights, D, A, 0.0)
    assert len(rows) == sum(w.R for w in wins)
    for rw in rows:   # C0 rows are v8 rows: S[:beta_r] fed, positions beta_{r-1}..beta_r
        b0 = 0 if rw.r == 0 else int(rw.win.beta[rw.r - 1])
        assert rw.K == int(rw.win.beta[rw.r]) and rw.pos.tolist() == list(range(b0, rw.K + 1))
    res = {}
    for budget in (100000, 700):   # one micro-batch (= v8's batch), then several (accumulated)
        zero(m)
        lg = T.grad_pass(m, rows, {**ctx, "tok_budget": budget}, 0)
        g = grads(m)
        err_l = abs(lg["loss_main"] - float(ref)) / abs(float(ref))
        assert err_l < 1e-6, (budget, lg["loss_main"], float(ref))
        assert lg["loss_state"] == 0.0
        # float32: one batch matches v8 to rounding; accumulating micro-batches changes the summation order
        err_g = cmp_grads(g, g_ref, 1e-5 if lg["mbs"] == 1 else 5e-4, f"C0 vs v8 ({lg['mbs']} micro-batches)")
        res[lg["mbs"]] = (lg["loss_main"], err_l, err_g)
    assert len(res) == 2
    print(f"  (a) C0 loss vs v8 loss {float(ref):.6f}: " + "; ".join(
        f"{k} micro-batch(es): loss {v[0]:.6f} rel {v[1]:.1e}, grads max rel {v[2]:.1e}" for k, v in sorted(res.items())))


# ---------------------------------------------------------------- (b)
def test_fisher_identity():
    wins = pick_windows(2, hier=False, offset=3)
    m = tiny_model(seed=3)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 12.0          # p(<wait>) ~ 0.5, so barring it at the last read matters
    m.logits_at = lambda h: ThinkerGStarModel.logits_at(m, h) + bias
    lam = 0.7
    # reference: autograd of -log Z through the DP on differentiable (plain) log-probs
    ctx = ctx_of(bar=False)
    zero(m)
    sc = T.score_pass(m, wins, ctx, 0, grad=True)
    lat = sl.pack([{"tok_lp": sc[w.idx][0], "end_lp": sc[w.idx][1], "lo": w.lo, "hi": w.hi, "lag": w.lag}
                   for w in wins], W=128)
    _, logZ = sl.forward(lat, lam)
    (-logZ.sum()).backward()
    g_ref = grads(m)
    # the trainer: weights from the lattice (no grad), weighted CE, h = 1, D = 1, no L_state
    zero(m)
    sc2 = T.score_pass(m, wins, ctx, 0)
    mg = sl.marginals(T.lattice_of(wins, sc2, 128, "cpu"), lam)
    weights = {w.idx: (mg["w_tok"][i, : w.R, : w.N].numpy(), mg["w_end"][i, : w.R, : w.N + 1].numpy())
               for i, w in enumerate(wins)}
    rows = T.train_rows(wins, weights, 1.0, 1.0, 0.0)
    T.grad_pass(m, rows, ctx, 0)
    err = cmp_grads(grads(m), g_ref, 2e-4, "Fisher")
    # --final-bar-wait 1: the lattice gets the decoding distribution of the last read (<wait> barred before the
    # softmax at every position, engine.read), all other reads and the loss's pair stay plain
    sc3 = T.score_pass(m, wins, ctx_of(bar=True), 0)
    for w in wins:
        tok, endp, tokl, endl = (x.detach() for x in sc3[w.idx])
        assert torch.equal(tok, sc2[w.idx][0]) and torch.equal(endp, sc2[w.idx][1])
        assert torch.equal(tokl[: w.R - 1], tok[: w.R - 1]) and torch.equal(endl[: w.R - 1], endp[: w.R - 1])
        r = w.R - 1
        rw = T.Row(w, r, w.N, np.arange(w.N + 1))
        b = T.collate([rw], ctx["audio"], ctx["proc"], ctx["prompter"], False, 0.0, 7, 0, "score")
        with torch.no_grad():
            h = m.hidden(b)
            lg = m.logits_at(h[b.sel_row, b.sel_col]).double()
            lp = torch.log_softmax(lg, -1)
            lg[:, T.WAIT_ID] = float("-inf")
            ls = torch.log_softmax(lg, -1)
        S_ = torch.as_tensor(w.S)
        # this row alone vs inside score_pass's micro-batch: float32 batching noise (same for plain and barred), so
        # compare the bar's shift (barred - plain), which must be -log(1 - p(<wait>)) at every position
        assert float((tok[r] - lp[torch.arange(w.N), S_]).abs().max()) < 2e-3
        sh_t = (tokl[r] - tok[r]) - (ls[torch.arange(w.N), S_] - lp[torch.arange(w.N), S_])
        sh_e = (endl[r] - endp[r]) - (ls[:, T.EOS_ID] - lp[:, T.EOS_ID])
        assert float(sh_t.abs().max()) < 1e-4 and float(sh_e.abs().max()) < 1e-4, (float(sh_t.abs().max()),
                                                                                  float(sh_e.abs().max()))
        d_ = tokl[r] - tok[r]                               # barring <wait> raised every last-read score
        assert float(d_.min()) > 0 and float(d_.mean()) > 0.1, (float(d_.min()), float(d_.mean()))
    mg3 = sl.marginals(T.lattice_of(wins, sc3, 128, "cpu"), lam)
    d_w = float((mg3["w_tok"] - mg["w_tok"]).abs().max())
    assert d_w > 1e-3, d_w                                    # ... and moved the posterior (it is not a constant)
    print(f"  (b) Fisher identity (bar off): grads max rel err {err:.1e} (lam {lam}); bar on: last-read lattice "
          f"scores = barred log-softmax, other reads plain, posterior moves by up to {d_w:.2f}")


# ---------------------------------------------------------------- (c)
def test_forward_bit_identical_and_hook():
    wins = pick_windows(2, offset=6)
    ctx = ctx_of()
    m = tiny_model(seed=5)
    items = [it for w in wins for it in v8_items(w, ctx["audio"])]
    _, vb = v8_forward_loss(m, items)
    weights = {w.idx: T.heur_weights(w) for w in wins}
    D, A = T.step_totals(wins)
    rows = T.train_rows(wins, weights, D, A, 0.0)        # same order as items (window, read)
    b = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], False, 0.0, 7, 0, "train")
    for k in ("input_ids", "attention_mask", "input_features", "feature_attention_mask", "sd_post", "n_audio",
              "has_sd"):
        assert torch.equal(getattr(b, k), vb[k]), k
    with torch.no_grad():
        e_ref = ThinkerGStarModel.embed(m, vb["input_ids"], vb["input_features"], vb["feature_attention_mask"],
                                        vb["sd_post"], vb["n_audio"], vb["has_sd"])
        h_ref = m.thinker.model(inputs_embeds=e_ref, attention_mask=vb["attention_mask"],
                                return_dict=True).last_hidden_state
        e_m = m.embed_batch(m.to_device(b))
        h_m = m.hidden(b)
    assert torch.equal(e_m, e_ref) and torch.equal(h_m, h_ref)
    # identity hook = the v8 path
    seen = {}

    def ident(P, enc, meta):
        seen.setdefault("calls", []).append((tuple(P.shape), None if enc is None else tuple(enc.shape), meta["T"],
                                             meta["t_ms"], meta["F"], meta["phase"]))
        return P

    m._post_transform = ident
    bh = T.collate(rows, ctx["audio"], ctx["proc"], ctx["prompter"], True, 0.0, 7, 0, "train")
    with torch.no_grad():
        e_h = m.embed_batch(m.to_device(bh))
        h_h = m.hidden(bh)
    d_e = float((e_h - e_ref).abs().max())
    d_h = float((h_h - h_ref).abs().max())
    assert d_e < 1e-6 and d_h < 1e-5, (d_e, d_h)
    for (ps, es, Tn, t, Fn, ph) in seen["calls"]:
        assert ps == (Tn, 8) and es == (Tn, 32) and Tn == max(0, t - 1040) // 80 and Fn >= Tn and ph == "train"
    # a parametrised hook gets gradients; extra_loss sees the documented info
    class Hook(nn.Module):
        wants_enc = True

        def __init__(self):
            super().__init__()
            self.g = nn.Parameter(torch.tensor(0.3))
            self.w = nn.Linear(32, 8)

        def forward(self, P, enc, meta):
            return P + self.g * torch.sigmoid(self.w(enc)) * (1 - P)

    hk = Hook()
    m2 = tiny_model(post_transform=hk, seed=5)
    info_seen = []

    def extra(info):
        info_seen.append({k: (len(v) if isinstance(v, list) else v) for k, v in info.items() if k != "device"})
        return 0.01 * sum(o.sum() for o in info["P_out"])

    zero(m2)
    lg = T.grad_pass(m2, rows, ctx, 3, extra_loss=extra, n_rows_global=len(rows), n_windows_global=len(wins))
    assert hk.g.grad is not None and float(hk.g.grad.abs()) > 0 and hk.w.weight.grad.abs().sum() > 0
    assert m2.sd_proj.mlp[1].weight.grad.abs().sum() > 0
    assert info_seen and all(s["rows_global"] == len(rows) and s["step"] == 3 and s["metas"] == s["P_in"] for s in info_seen)
    assert lg["loss_extra"] != 0.0
    print(f"  (c) None path: batch = v8 build_batch bit for bit, embeddings and hidden states torch.equal; identity hook "
          f"max |diff| emb {d_e:.1e} hidden {d_h:.1e}; hook params get grads; extra_loss info ok")


def test_post_to_tokens_torch():
    wins = pick_windows(3)
    audio = T.AudioCache()
    worst = 0.0
    exact = True
    for w in wins:
        wav, post = audio.get(w)
        for t in w.t_ms:
            p = post.copy()
            p[max(0, t - 1040) // 80:] = 0
            n = max(1, int(13 * t / 1000))
            a = post_to_tokens(p, n, t / 1000)
            b = T.post_to_tokens_torch(torch.as_tensor(p), n, t / 1000).numpy()
            worst = max(worst, float(np.abs(a.astype(np.float32) - b).max()))
            exact &= np.array_equal(a.astype(np.float32), b)
    assert worst < 1e-6, worst
    print(f"  post_to_tokens torch vs numpy: max |diff| {worst:.1e} (bit-equal: {exact})")


# ---------------------------------------------------------------- (d)
def test_weight_sums():
    recs = [json.loads(l) for l in open(Path(SCHED) / "shard_001.jsonl", encoding="utf-8")][:300]
    wins = [T.make_window(i, r) for i, r in enumerate(recs)]
    T.set_reach(wins, 128)
    assert all(w.feasible for w in wins)
    rng = np.random.default_rng(0)
    sc = {}
    for w in wins:   # (tok, end) for the loss, (tok, end) for the lattice
        t_, e_ = torch.as_tensor(-rng.exponential(1.0, (w.R, w.N))), torch.as_tensor(np.log(rng.uniform(0.01, 1, (w.R, w.N + 1))))
        sc[w.idx] = (t_, e_, t_, e_)
    worst = 0.0
    for i in range(0, len(wins), 64):
        part = wins[i: i + 64]
        lat = T.lattice_of(part, sc, 128, "cpu")
        sol = sl.solve_lambda([lat], 0.8256)
        mg = sl.marginals(lat, sol["lam"])
        ps = T.posterior_stats(lat, mg, part, sol["lam"])
        assert abs(float(np.mean(ps["exp_lat"])) - 0.8256) < 1e-5
        for k, w in enumerate(part):
            wt, we = mg["w_tok"][k, : w.R, : w.N].numpy(), mg["w_end"][k, : w.R, : w.N + 1].numpy()
            ht, he = T.heur_weights(w)
            for a, b, nm in ((wt.sum(), w.N, "sched tok"), (we.sum(), w.R, "sched end"), (ht.sum(), w.N, "C0 tok"),
                             (he.sum(), w.R, "C0 end")):
                worst = max(worst, abs(a - b))
                assert abs(a - b) < 1e-8, (nm, a, b)
            assert np.allclose(wt.sum(0), 1.0) and np.allclose(we.sum(1), 1.0)
            dt = float((w.h[None] * wt).sum() + we.sum())
            assert abs(dt - (w.h.sum() + w.R)) < 1e-8 and abs((w.h[None] * ht).sum() + he.sum() - dt) < 1e-8
            assert ps["H"][k] > -1e-9 and 0 < ps["q_heur"][k] <= 1 + 1e-12
    # entropy / q(heuristic) against brute force on tiny random instances
    from experiments.mist_w3po_v1.crf.test_schedule_lattice import brute, random_instance

    rng = np.random.default_rng(1)
    n_ok = 0
    while n_ok < 60:
        inst = random_instance(rng, int(rng.integers(1, 5)), int(rng.integers(1, 4)), p_ninf=0.0)
        bf = brute(inst)
        if bf is None:
            continue
        lat = sl.Lattice.single(inst["tok_lp"], inst["end_lp"], inst["lo"], inst["hi"], inst["lag"], W=inst["W"])
        mg = sl.marginals(lat, inst["lam"])
        s0 = np.array(bf["scheds"][0])
        fake = SimpleNamespace(R=len(s0), N=lat.N, beta=s0)
        ps = T.posterior_stats(lat, mg, [fake], inst["lam"])
        H_bf = float(-(bf["p"] * np.log(np.clip(bf["p"], 1e-300, None))).sum())
        assert abs(ps["H"][0] - H_bf) < 1e-8, (ps["H"][0], H_bf)
        assert abs(ps["q_heur"][0] - bf["p"][0]) < 1e-10
        n_ok += 1
    print(f"  (d) weight sums N + R per window (300 real windows, both objectives): max err {worst:.1e}; "
          f"H(q), q(heuristic) = brute force on 60 instances")


# ---------------------------------------------------------------- schedule, plan, checkpoints
def test_lr_schedule_and_plan():
    a = SimpleNamespace(lr=1e-4, min_lr=1e-5, warmup=50, steps=3000)
    logged = {1: 2.0e-06, 10: 2.0e-05, 20: 4.0e-05, 2980: 1.001021e-05, 2990: 1.000255e-05, 3000: 1.0e-05}   # v8 log
    for k, v in logged.items():
        assert abs(T.lr_at(k, a) - v) < 1e-11, (k, T.lr_at(k, a), v)
    Rs = list(np.random.default_rng(0).integers(1, 23, 2000))
    p1 = T.plan_steps(Rs, 5, 500, 64)
    p2 = T.plan_steps(Rs, 5, 500, 64)
    assert p1 == p2 and len(p1) == 500 and len([i for s in p1 for i in s]) > 2000
    rows = [sum(Rs[i] for i in s) for s in p1]
    assert abs(np.mean(rows) - 64) < 3, np.mean(rows)
    flat = [i for s in p1 for i in s]
    assert sorted(flat[:2000]) == list(range(2000))   # the first pass is a permutation
    wins = pick_windows(5)
    parts = T.assign(wins, 4)
    assert sorted(w.idx for p in parts for w in p) == sorted(w.idx for w in wins)
    print(f"  lr schedule = v8's logged values; plan: mean {np.mean(rows):.1f} rows/step, deterministic, passes are "
          f"permutations")


def test_checkpoint_roundtrip():
    m = tiny_model(seed=9)
    hk = nn.Linear(3, 3)
    params = m.trainable() + list(hk.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95))
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    args = SimpleNamespace(objective="sched", base="x", lora_rank=4, lora_alpha=8.0)
    d = tempfile.mkdtemp(prefix="test_crf_ckpt_")   # under $TMPDIR; a few KB, left in place (no deletion)
    if True:
        ck = Path(d) / "ckpt" / "step_00003"
        T.save_ckpt(ck, m, opt, [hk], 0.25, 3, args)
        assert T.latest_ckpt(Path(d)) == ck
        m2 = tiny_model(seed=11)
        hk2 = nn.Linear(3, 3)
        opt2 = torch.optim.AdamW(m2.trainable() + list(hk2.parameters()), lr=1e-3, betas=(0.9, 0.95))
        ex = T.load_ckpt(ck, m2, opt2, [hk2])
        assert ex["lam"] == 0.25 and ex["step"] == 3
        for p, q in zip(m.trainable() + list(hk.parameters()), m2.trainable() + list(hk2.parameters())):
            assert torch.equal(p.detach().cpu(), q.detach().cpu())
        assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
        # lora.pt = the rl/train_w3po format: keys model.layers.N.self_attn.X_proj.{a,b}, loadable on a fresh injection
        st = torch.load(ck / "lora.pt")
        assert all(k.startswith("model.layers.") and k.rsplit(".", 1)[1] in "ab" for k in st)
        th = TinyThinker(seed=2)
        ad = lora.inject(th, rank=4, alpha=8.0, targets=ATTN,
                         layer_filter=lambda p: T.LAYER_RE.search(p + ".") is not None)
        lora.load_state_dict(ad, st)
    print("  checkpoint round trip ok (lora.pt / extra.pt / optim.pt, extra_params, lam)")


def test_export_equals_in_memory_apply():
    """crf/export_merged.py (safetensors) and crf/decode_crf_ckpt.apply_ckpt's fold (in memory) give the same bf16
    weights, rows and projector, on a fake two-shard base."""
    import copy

    from safetensors.torch import load_file, save_file

    from experiments.mist_w3po_v1.crf import decode_crf_ckpt as DC
    from experiments.mist_w3po_v1.crf import export_merged as EM

    d = Path(tempfile.mkdtemp(prefix="test_crf_export_"))   # under $TMPDIR, ~40 MB, left in place (no deletion)
    tiny = TinyThinker(seed=4).to(torch.bfloat16)
    names = [n for n, mod in tiny.named_modules() if n.rsplit(".", 1)[-1] in ATTN]
    state = {f"thinker.{n}.weight": dict(tiny.named_modules())[n].weight.detach().clone() for n in names}
    state["thinker.model.embed_tokens.weight"] = tiny.embed.weight.detach().clone()
    state["thinker.lm_head.weight"] = tiny.lm_head.weight.detach().clone()
    state["thinker.other.weight"] = torch.ones(3, dtype=torch.bfloat16)
    base = d / "base"
    base.mkdir()
    keys = sorted(state)
    sh = {k: ("model-00001-of-00002.safetensors" if i % 2 else "model-00002-of-00002.safetensors")
          for i, k in enumerate(keys)}
    for f in set(sh.values()):
        save_file({k: state[k] for k in keys if sh[k] == f}, str(base / f))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": sh}))
    (base / "config.json").write_text("{}")
    m = tiny_model(seed=6)
    with torch.no_grad():
        for ad in m.adapters:
            ad.a.copy_(torch.randn_like(ad.a))
        m.new_emb.add_(torch.randn_like(m.new_emb))
        m.new_head.add_(torch.randn_like(m.new_head))
    torch.save(m.sd_proj.state_dict(), base / "sd_proj.pt")
    args = SimpleNamespace(objective="sched", base=str(base), lora_rank=4, lora_alpha=8.0)
    ck = d / "ckpt" / "step_00001"
    opt = torch.optim.AdamW(m.trainable(), lr=1e-3)
    T.save_ckpt(ck, m, opt, [], 0.1, 1, args)
    EM.main(["--ckpt", str(ck), "--base", str(base), "--out", str(d / "merged")])
    got = {}
    for f in set(sh.values()):
        got.update(load_file(str(d / "merged" / f)))
    ref = copy.deepcopy(tiny)
    n = DC.fold_into(ref, torch.load(ck / "lora.pt"), 4, 8.0)
    assert n == len(names)
    with torch.no_grad():
        ref.embed.weight[T.NEW0:T.NEW_END] = m.new_emb.to(torch.bfloat16)
        ref.lm_head.weight[T.NEW0:T.NEW_END] = m.new_head.to(torch.bfloat16)
    mods = dict(ref.named_modules())
    for nm in names:
        assert torch.equal(got[f"thinker.{nm}.weight"], mods[nm].weight), nm
        assert not torch.equal(got[f"thinker.{nm}.weight"], state[f"thinker.{nm}.weight"]), nm
    assert torch.equal(got["thinker.model.embed_tokens.weight"], ref.embed.weight)
    assert torch.equal(got["thinker.lm_head.weight"], ref.lm_head.weight)
    assert torch.equal(got["thinker.other.weight"], state["thinker.other.weight"])
    sp = torch.load(d / "merged" / "sd_proj.pt")
    assert all(torch.equal(sp[k], v.float()) for k, v in m.sd_proj.state_dict().items())
    print(f"  export_merged = in-memory fold (decode_crf_ckpt) on a fake 2-shard base: {n} adapters, rows, projector")


def main():
    t0 = time.time()
    tests = [test_lr_schedule_and_plan, test_post_to_tokens_torch, test_weight_sums, test_checkpoint_roundtrip,
             test_c0_equals_v8_sft, test_fisher_identity, test_forward_bit_identical_and_hook,
             test_export_equals_in_memory_apply]
    for t in tests:
        t1 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL {len(tests)} PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
