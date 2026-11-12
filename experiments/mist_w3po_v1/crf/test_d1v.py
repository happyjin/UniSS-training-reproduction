"""Tests and the R0 CPU precheck of D1v (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md §2.1), CPU only.

  R0  why D1's GPU selfcheck failed (16/20, 15/20) and that scorer v2 fixes it. A tiny Qwen3-Omni thinker text model
      (transformers' Qwen3OmniMoeThinkerTextModel, the same code path as the 30B model: interleaved M-RoPE with
      theta 1e6, GQA, top-k MoE through transformers' grouped_mm fallback, SDPA; random weights, bf16), the real
      Prompter and real VE1 training windows (crf/sched_t5_merge3_w10_v1), audio embeddings that depend only on the
      read (so the audio side is identical in every path) and a decision head whose <wait> logit is a near-tie with
      one content token (margins ~ 0.1, a stress test of the numerics). Every scorer runs D1's selfcheck: the
      per-boundary prefix loop (simulate, one unpadded forward per boundary) against the one-forward batch driver
      (simulate_rounds, B.batches of up to 16 reads, padded):
        v1        d1_rollin.gpu_batch_scorer: left padding, no position ids (D1's scorer)
        v1+pos    v1 with position ids = cumsum(mask) - 1 (only the RoPE positions changed)
        v1+4d     v1 with an explicit 4-D mask in every forward (only the attention path changed)
        v1+pos+4d both;  right+pos+2d  right padding + positions, transformers' own mask (path differs again)
        v2        d1_rollin.gpu_batch_scorer_v2: right padding, position ids from 0, 4-D mask, shared audio cache
      GO (registered): v2 agrees on 100 % of >= 60 windows; v1 disagrees and v1+pos removes (or clearly reduces) it.
  R0b right_padded's layout; v2's hidden states at a row's real positions = the unpadded forward's, bit for bit.
  R1-R3, M  test_d1's tests on the VE1 data (sched_t5_merge3_w10_v1) and with lattice mode v2p (R3: --d1 with
      mode v2p, alone and with DART).

    PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= python -m experiments.mist_w3po_v1.crf.test_d1v \
        [--r0-out eval_outputs/mist_w3po_v1/crf/d1v_scorer_precheck_v1.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

if not torch.cuda.is_available():
    torch.accelerator.current_accelerator = lambda *a, **k: None

from experiments.mist_w3po_v1.crf import d1_rollin as R
from experiments.mist_w3po_v1.crf import dart_aug as DA
from experiments.mist_w3po_v1.crf import latent_write_crf as LW
from experiments.mist_w3po_v1.crf import test_d1 as TD
from experiments.mist_w3po_v1.crf import test_latent_write_crf as TL
from experiments.mist_w3po_v1.crf import train_crf_sft as T

torch.set_num_threads(8)
VE1_SCHED = "eval_outputs/mist_w3po_v1/crf/sched_t5_merge3_w10_v1"
CONTENT = 1000
VOCAB = 152320


# ------------------------------------------------------------------------------------------------ tiny engine
class TinyEngine:
    """the attributes of core.engine.Engine that d1_rollin's scorers use, on a tiny random thinker text model."""

    def __init__(self, seed: int = 0, layers: int = 2, scale: float = 0.01, bias: float = 0.0):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel

        from experiments.mist_w3po_v1.core.engine import Prompter

        cfg = Qwen3OmniMoeTextConfig(
            vocab_size=VOCAB, hidden_size=128, intermediate_size=64, moe_intermediate_size=64, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2,
            rope_theta=1000000, rms_norm_eps=1e-6,
            rope_scaling={"interleaved": True, "mrope_interleaved": True, "mrope_section": [8, 4, 4],
                          "rope_type": "default", "type": "default"})
        torch.manual_seed(seed)
        m = Qwen3OmniMoeThinkerTextModel(cfg)
        with torch.no_grad():   # weights large enough that position / numerics matter after 2 layers
            for p in m.parameters():
                if p.ndim >= 2:
                    p.normal_(0.0, 0.08)
        self.text_model = m.to(torch.bfloat16).eval()
        self.thinker = SimpleNamespace(model=self.text_model)
        self.emb_layer = self.text_model.embed_tokens
        self.pr = Prompter("zh2en")
        self.tok = self.pr.tok
        self.dev = torch.device("cpu")
        g = torch.Generator().manual_seed(seed + 1)
        self.w = torch.randn(cfg.hidden_size, generator=g)
        self.scale, self.bias = scale, bias
        self.H = cfg.hidden_size

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        lg = torch.full((h.shape[0], VOCAB), -1e4)
        lg[:, CONTENT] = 0.0
        lg[:, R.WAIT_ID] = (h.float() @ self.w) * self.scale + self.bias
        return lg

    def _emb(self, akey: str, t_ms: int):
        from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import n_speaker_tokens

        n = max(1, math.ceil(t_ms / 80))
        seed = int.from_bytes(hashlib.sha1(akey.encode()).digest()[:4], "little")
        g = torch.Generator().manual_seed(seed)
        e = (torch.randn(n + n_speaker_tokens(n), self.H, generator=g) * 0.5).to(torch.bfloat16)
        return n, e

    def audio_embeds(self, items, emb_dtype, lookahead_ms: int = 1040, cache: dict | None = None):
        out = []
        for j, it in enumerate(items):
            k = it.akey or f"__{j}"
            if cache is not None and k in cache:
                out.append(cache[k])
                continue
            v = self._emb(k, it.t_ms)
            if cache is not None and not k.startswith("__"):
                cache[k] = v
            out.append(v)
        return out


def scorer_variant(eng, wavs: dict, budget: int, batch: int, left: bool, pos_fix: bool, mask4: bool):
    """the one-forward scorer with the layout pieces switched one at a time (R0's decomposition):
    left = left padding (v1) or right padding (v2); pos_fix = position ids counting each row's real tokens from 0
    (v1: none, so a left-padded row's tokens are numbered from its pad count); mask4 = an explicit 4-D mask in every
    forward (v2), else the 2-D mask (transformers drops it for an unpadded forward -> is_causal + GQA inside SDPA,
    and builds a 4-D mask for a padded batch -> repeat_kv + attn_mask: two different attention paths)."""
    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B

    def one(part):
        its = [ReadItem(wav=None, post=None, t_ms=rec["t_ms"][r], final=False, committed=list(ids), ctx="",
                        akey=f"{rec['id']}|{rec['t_ms'][r]}") for rec, r, ids, _ in part]
        out = []
        with torch.no_grad():
            ae = eng.audio_embeds(its, eng.emb_layer.weight.dtype)
            prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
            Bn, L = len(its), max(len(p) for p in prompts)
            tid = torch.full((Bn, L), eng.tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((Bn, L), dtype=torch.long)
            for i, p in enumerate(prompts):
                sl = slice(L - len(p), L) if left else slice(0, len(p))
                tid[i, sl] = torch.as_tensor(p)
                mask[i, sl] = 1
            kw = {}
            if pos_fix:
                kw["position_ids"] = (mask.cumsum(1) - 1).clamp(min=0) if left else \
                    torch.arange(L)[None].expand(Bn, L).contiguous()
            am = mask
            if mask4:
                v = mask.bool()
                am = (torch.ones((L, L), dtype=torch.bool).tril()[None, None] & v[:, None, None, :]) | \
                    torch.eye(L, dtype=torch.bool)[None, None]
            emb = eng.emb_layer(tid)
            emb = emb.masked_scatter((tid == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
            hs = eng.thinker.model(inputs_embeds=emb, attention_mask=am, use_cache=False, return_dict=True,
                                   **kw).last_hidden_state
            for i, (rec, r, ids, ps) in enumerate(part):
                M = len(ids)
                base = (L - M - 1) if left else (len(prompts[i]) - M - 1)
                out.append(eng.logits(hs[i, torch.as_tensor([base + p for p in ps])]).argmax(-1).tolist())
        return out

    def scorer(todo):
        res = [None] * len(todo)
        at = {(rec["id"], r): k for k, (rec, r, _, _) in enumerate(todo)}
        for part in B.batches([(rec, r) for rec, r, _, _ in todo], batch, budget):
            ks = [at[(rec["id"], r)] for rec, r in part]
            for k, g in zip(ks, one([todo[k] for k in ks])):
                res[k] = g
        return res
    return scorer


def margins_v1(eng, part):
    """the <wait> margins at every boundary of (rec, r, ids, pos) items for v1's batch layout and for unpadded single
    forwards (the selfcheck's prefix side at the batch's own history)."""
    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD, ReadItem

    def fw(items, left: bool, pos_ids: bool):
        its = [ReadItem(wav=None, post=None, t_ms=rec["t_ms"][r], final=False, committed=list(ids), ctx="",
                        akey=f"{rec['id']}|{rec['t_ms'][r]}") for rec, r, ids, _ in items]
        ae = eng.audio_embeds(its, torch.bfloat16)
        prompts = [eng.pr.ids(n, "", it.committed) for it, (n, _) in zip(its, ae)]
        Bn, L = len(its), max(len(p) for p in prompts)
        tid = torch.full((Bn, L), eng.tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((Bn, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            if left:
                tid[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            else:
                tid[i, : len(p)] = torch.as_tensor(p)
                mask[i, : len(p)] = 1
        emb = eng.emb_layer(tid)
        emb = emb.masked_scatter((tid == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
        kw = {"position_ids": (mask.cumsum(1) - 1).clamp(min=0)} if pos_ids else {}
        with torch.no_grad():
            hs = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False, return_dict=True,
                                   **kw).last_hidden_state
        out = []
        for i, (rec, r, ids, ps) in enumerate(items):
            M, Lp = len(ids), len(prompts[i])
            base = (L - M - 1) if left else (Lp - M - 1)
            lg = eng.logits(hs[i, torch.as_tensor([base + p for p in ps])])
            out.append((lg[:, R.WAIT_ID] - lg[:, CONTENT]).tolist())
        return out
    return fw(part, True, False), fw(part, True, True), [fw([x], True, False)[0] for x in part]


# ------------------------------------------------------------------------------------------------ R0
def selfcheck(recs: list[dict], make_scorer) -> dict:
    """D1's selfcheck with one scorer: literal per-boundary prefix loop vs the one-forward batch driver."""
    sc_lit, sc_bat = make_scorer(), make_scorer()

    def prefix_scorer(rec, r, ids, pos):
        return [sc_lit([(rec, r, ids[:p], [p])])[0][0] for p in pos]
    lit = [R.simulate(rec, prefix_scorer) for rec in recs]
    bat = R.simulate_rounds(recs, sc_bat)
    eq = sum(a["rho"] == b["rho"] for a, b in zip(lit, bat))
    dec = sum(len(a["written"]) for a in lit)
    return {"windows": len(recs), "equal": eq, "reads": dec,
            "differ": [a["id"] for a, b in zip(lit, bat) if a["rho"] != b["rho"]]}


def r0(out: str | None = None, n_windows: int = 80) -> dict:
    recs = R.selfcheck_recs(TD.real_recs(VE1_SCHED, 4000), n_windows, 3)
    assert len(recs) == n_windows
    res = {"windows": n_windows, "sched": VE1_SCHED, "runs": []}
    for seed, scale in ((0, 0.01), (1, 0.01), (2, 0.003)):
        eng = TinyEngine(seed=seed, scale=scale)
        wavs = {rec["id"]: (None, None) for rec in recs}
        # the decision head's margin scale: |<wait> - content| at the first boundary of every window's first read
        probe = [(rec, 0, R.read_input(rec, R.kap_all(rec), 0)[0], R.read_input(rec, R.kap_all(rec), 0)[1])
                 for rec in recs[:16]]
        m_left, m_leftpos, m_single = margins_v1(eng, probe)
        flat = lambda xs: np.concatenate([np.asarray(x) for x in xs])
        d_left = np.abs(flat(m_left) - flat(m_single))
        d_pos = np.abs(flat(m_leftpos) - flat(m_single))
        row = {"seed": seed, "scale": scale, "margin_abs_median": float(np.median(np.abs(flat(m_single)))),
               "probe_margin_absdiff_vs_unpadded": {"v1_left": {"max": float(d_left.max()), "mean": float(d_left.mean())},
                                                    "v1_left_pos": {"max": float(d_pos.max()),
                                                                    "mean": float(d_pos.mean())}}}
        acache: dict = {}
        for name, mk in (("v1", lambda: R.gpu_batch_scorer(eng, wavs, 24000, 16)),
                         ("v1+pos", lambda: scorer_variant(eng, wavs, 24000, 16, True, True, False)),
                         ("v1+4d", lambda: scorer_variant(eng, wavs, 24000, 16, True, False, True)),
                         ("v1+pos+4d", lambda: scorer_variant(eng, wavs, 24000, 16, True, True, True)),
                         ("right+pos+2d", lambda: scorer_variant(eng, wavs, 24000, 16, False, True, False)),
                         ("v2", lambda: R.gpu_batch_scorer_v2(eng, wavs, 24000, 16, acache))):
            t0 = time.time()
            row[name] = selfcheck(recs, mk)
            row[name]["s"] = round(time.time() - t0, 1)
        res["runs"].append(row)
        print("  R0 " + json.dumps({k: (v if not isinstance(v, dict) or "equal" not in v else
                                        f"{v['equal']}/{v['windows']}") for k, v in row.items()}), flush=True)
    v2_all = all(r["v2"]["equal"] == r["v2"]["windows"] for r in res["runs"])
    bad = {k: sum(r[k]["windows"] - r[k]["equal"] for r in res["runs"])
           for k in ("v1", "v1+pos", "v1+4d", "v1+pos+4d", "right+pos+2d", "v2")}
    res["differ_windows"] = bad
    # registered GO (prereg §2.1 R0 (b)): v2 100 %, and v1 disagrees while correcting v1's position ids removes or
    # reduces it; the decomposition says which of the two layout pieces (positions, mask path) carries the effect
    res["GO"] = {"v2_100pct": v2_all, "v1_differ_windows": bad["v1"], "v1pos_differ_windows": bad["v1+pos"],
                 "position_offset_reduces": bool(bad["v1"] > 0 and bad["v1+pos"] < bad["v1"]),
                 "mask_path_reduces": bool(bad["v1+4d"] < bad["v1"]),
                 "pass": bool(v2_all and bad["v1"] > 0 and bad["v1+pos"] < bad["v1"])}
    print("  R0 GO " + json.dumps(res["GO"]), flush=True)
    if out:
        with open(out, "x", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    assert res["GO"]["v2_100pct"], "scorer v2 disagrees with the per-boundary prefix loop"
    return res


def test_r0b():
    ids, m4, pos = R.right_padded([[5, 6, 7], [8, 9], [1, 2, 3, 4]], 0)
    assert ids.tolist() == [[5, 6, 7, 0], [8, 9, 0, 0], [1, 2, 3, 4]] and pos.tolist() == [[0, 1, 2, 3]] * 3
    assert m4.shape == (3, 1, 4, 4) and m4.dtype == torch.bool
    assert m4[1, 0].tolist() == [[True, False, False, False], [True, True, False, False], [True, True, False, False],
                                 [True, True, False, False]]
    assert bool(m4.any(-1).all())   # no fully masked query row
    eng = TinyEngine(seed=5)
    m = eng.text_model
    rng = np.random.default_rng(0)
    seqs = [rng.integers(100, 5000, int(n)).tolist() for n in (37, 64, 50, 12, 64, 120)]

    def fw(rows):
        t, mk, p = R.right_padded(rows, 0)
        with torch.no_grad():
            return m(input_ids=t, attention_mask=mk, position_ids=p, use_cache=False).last_hidden_state.float()
    hb = fw(seqs)
    pre = bat = 0.0
    for i, s in enumerate(seqs):
        h1 = fw([s])[0]
        k = max(1, len(s) // 2)
        pre = max(pre, float((fw([s[:k]])[0] - h1[:k]).abs().max()))   # causality + indexing: prefix == full[:k]
        bat = max(bat, float((hb[i, : len(s)] - h1).abs().max()))
    # the mask path matters: no mask (transformers: is_causal + GQA inside SDPA) vs an explicit mask differ in bf16,
    # and the top-k router turns such differences into large jumps; v2 therefore always passes the 4-D mask
    with torch.no_grad():
        h_none = m(input_ids=torch.as_tensor([seqs[1]]), use_cache=False).last_hidden_state[0].float()
    none_vs_4d = float((h_none - fw([seqs[1]])[0]).abs().max())
    assert pre == 0.0 and bat == 0.0, (pre, bat)
    print(f"  R0b right_padded: layout, positions from 0, causal & key-valid 4-D mask without empty rows; with the "
          f"4-D mask: prefix == full[:k] and batch == single bit for bit (CPU); for contrast, no mask vs 4-D mask on "
          f"the same row: max |dh| {none_vs_4d:.3f} (router jumps)")
    return {"prefix_vs_full": pre, "batch_vs_single": bat, "nomask_vs_4dmask": none_vs_4d}


# ------------------------------------------------------------------------------------------------ R1-R3, M on VE1
def test_r1_r2_m_ve1():
    TD.SCHEDS = [VE1_SCHED]
    for t in (TD.test_manifest_parts, TD.test_r1, TD.test_r2):
        t()


def test_r3_v2p():
    base = TD.TV.small_choice_windows(3, max_paths=60)
    wins = [T.make_window(w.idx, TL.widen(w.rec), True) for w in base]
    T.set_reach(wins, 128)
    m = TD.TT.tiny_model(seed=43)
    bias = torch.zeros(151936)
    bias[T.WAIT_ID] = 9.0
    bias[T.EOS_ID] = 6.0
    m.logits_at = lambda h: TD.TT.ThinkerGStarModel.logits_at(m, h) + bias
    D = T.step_totals(wins)[0]
    v0, l0, g0 = TD._run(m, wins, D, mode="v2p")
    v, lg, g = TD._run(m, wins, D, mode="v2p", d1=None)
    assert len(v["rows"]) == len(v0["rows"]) and lg["loss_main"] == l0["loss_main"]
    assert all(torch.equal(g0[k], g[k]) for k in g0)
    _, bt = TD.fake(0.7, "r3v")
    d1 = {w.idx: (R.simulate_rounds([w.rec], bt)[0]["rho"], None) for w in wins}
    v1, l1, g1 = TD._run(m, wins, D, mode="v2p", d1=d1)
    v1p, l1p, g1p = TD._run(m, wins, D, pf=True, mode="v2p", d1=d1)
    assert l1["loss_main"] == l1p["loss_main"] and all(torch.equal(g1[k], g1p[k]) for k in g1)
    n0 = len(v0["rows"])
    for a, b in zip(v0["rows"], v1["rows"][:n0]):
        assert a.key == b.key and np.array_equal(a.c_tok, b.c_tok) and np.array_equal(a.c_end, b.c_end)
    extra = v1["rows"][n0:]
    assert len(extra) == v1["stats"]["d1_rows"] > 0
    tol = 1e-5 * max(1.0, abs(l1["loss_main"]))
    assert abs(l1["loss_main"] - (l0["loss_main"] + l1["loss_aug"])) < tol
    draws = {w.idx: [1] * len(w.rec["chunks"]) for w in wins}
    v2, l2, _ = TD._run(m, wins, D, mode="v2p", dart=draws, d1=d1)
    kinds = [rw.key[0] for rw in v2["rows"][n0:]]
    assert kinds.count("dart") == v2["stats"]["dart_rows"] > 0 and kinds.count("d1") == v2["stats"]["d1_rows"]
    assert math.isfinite(l2["loss_main"])
    print(f"  R3 (v2p) step: d1 None == absent bit for bit; {len(extra)} D1 rows appended, base rows unchanged, loss = "
          f"base + aug; prefetch 1 == 0; DART + D1: {kinds.count('dart')} + {kinds.count('d1')} rows")


def test_cli():
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-m", "experiments.mist_w3po_v1.crf.d1_rollin", "--help"], capture_output=True,
                       text=True)
    assert p.returncode == 0 and "--scorer" in p.stdout and "v2p" in p.stdout
    print("  CLI: --scorer {v1, v2} (default v1), --selfcheck-min-reads, --selfcheck-tol, --manifest --lw-mode v2p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0-out", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    tests = [("r0b", test_r0b), ("cli", test_cli), ("r3", test_r3_v2p), ("ve1", test_r1_r2_m_ve1),
             ("r0", lambda: r0(a.r0_out or None))]
    for name, t in tests:
        if a.only and name not in a.only.split(","):
            continue
        t1 = time.time()
        t()
        print(f"PASS {name} ({time.time() - t1:.1f}s)", flush=True)
    print(f"ALL PASSED ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
