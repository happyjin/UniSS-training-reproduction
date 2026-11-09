"""Unit tests for evaluation/online_decode_finalpost.py (CPU, the real Prompter / feature extractor, a fake Thinker that
records the model inputs of Engine.read):

    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.evaluation.test_online_decode_finalpost

  off           argv handling: no flag / '--final-read-post off' -> online_decode.main gets the plain argv and
                Engine.read is not patched
  identity      --final-read-post close_detect with close_detect = t (cut = the engine's t - 1.04 s): prompt ids,
                inputs_embeds and attention_mask bit-identical to the unpatched Engine.read
  extended      close_detect = t + 5 s: prompt ids identical; embeddings differ only at the speaker-stream rows of the
                final items; their speaker input equals P masked from frame t // 80; non-final items unchanged
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.evaluation import online_decode as od
from experiments.mist_w3po_v1.evaluation import online_decode_finalpost as fp
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import AUDIO_PAD, N_SLOT, SDProjector, post_to_tokens

D = 16
V = 152064


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond, name


class FakeThinker:
    """get_audio_features: deterministic rows from the features; model: records inputs, every row -> <eos>."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(V, D).to(torch.bfloat16)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, D, generator=g).to(torch.bfloat16))
        self.proj = torch.randn(128, D, generator=g)
        self.calls = []

    def get_input_embeddings(self):
        return self.emb

    def get_audio_features(self, feats, mask):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
        n = _get_feat_extract_output_lengths(mask.sum(1)).tolist()
        rows = []
        for b, k in enumerate(n):
            f = feats[b].float().T[: int(mask[b].sum())]           # [frames, 128]
            ch = torch.tensor_split(f, k, 0)
            rows.append(torch.stack([c.mean(0) for c in ch]) @ self.proj)
        return SimpleNamespace(last_hidden_state=torch.cat(rows, 0))

    def model(self, inputs_embeds, attention_mask, use_cache=True, return_dict=True, past_key_values=None):
        self.calls.append((inputs_embeds.detach().clone(), attention_mask.detach().clone()))
        B, L = attention_mask.shape
        return SimpleNamespace(last_hidden_state=torch.zeros(B, L, D), past_key_values=None)

    def lm_head(self, h):
        lg = torch.zeros(h.shape[0], V)
        lg[:, E.EOS_ID] = 10.0
        return lg


def make_engine(prompter):
    th = FakeThinker()
    torch.manual_seed(0)
    sd = SDProjector(N_SLOT, D, hidden=32)
    with torch.no_grad():   # the real init zeroes the last layer (output 0 for any P); random weights make P visible
        for p in sd.parameters():
            p.normal_(0.0, 0.5)
    sd = sd.to(torch.bfloat16).eval()
    seen = []
    orig_fwd = sd.forward

    def fwd(x):
        seen.append(x.detach().clone())
        return orig_fwd(x)

    sd.forward = fwd
    eng = E.Engine(th, sd, prompter, torch.device("cpu"))
    ids_seen = []
    orig_ids = prompter.ids

    def ids(n, ctx, prefix):
        r = orig_ids(n, ctx, prefix)
        ids_seen.append(list(r))
        return r

    eng.pr = SimpleNamespace(ids=ids, proc=prompter.proc)
    return eng, th, seen, ids_seen


def items():
    rng = np.random.default_rng(0)
    out = []
    for wid, dur, reads in (("wA", 1150, [1000, 1150]), ("wB", 2300, [2300]), ("wC", 5030, [5000, 5030])):
        wav = rng.standard_normal(dur * 16).astype(np.float32) * 0.1
        post = rng.random((-(-dur // 80), N_SLOT)).astype(np.float32)
        for t in reads:
            out.append(E.ReadItem(wav=wav, post=post, t_ms=t, final=t >= dur, committed=[] if t < 2000 else [11, 12],
                                  ctx="", akey=f"{wid}|{t}"))
    return out


def run(eng, th, its, seen, ids_seen):
    th.calls.clear(), seen.clear(), ids_seen.clear()
    eng.read(its, max_new=4)
    return th.calls[0], list(seen), list(ids_seen)


def main():
    # --- off: argv and patching
    base = ["--thinker", "T", "--windows", "W.jsonl", "--out", "o.jsonl", "--loop-guard-k", "6"]
    check("no flag -> off, argv unchanged", fp.split_argv(base) == ("off", base))
    check("'--final-read-post off' -> argv without the flag",
          fp.split_argv(base[:2] + ["--final-read-post", "off"] + base[2:]) == ("off", base))
    check("'--final-read-post=close_detect' parsed", fp.split_argv(base + ["--final-read-post=close_detect"])[0]
          == "close_detect")
    try:
        fp.split_argv(base + ["--final-read-post", "full"])
        check("bad mode rejected", False)
    except SystemExit:
        check("bad mode rejected", True)
    orig_read, orig_main, got = E.Engine.read, od.main, {}
    od.main = lambda: got.update(argv=list(sys.argv), read=E.Engine.read)
    sys.argv = ["x"] + base[:2] + ["--final-read-post", "off"] + base[2:]
    fp.main()
    od.main = orig_main
    check("off: online_decode.main sees the plain argv", got["argv"] == ["x"] + base)
    check("off: Engine.read not patched", got["read"] is orig_read and E.Engine.read is orig_read)

    # --- cut rule
    check("cut: short window, late close -> t", fp.final_cut_ms(1150, 3640) == 1150)
    check("cut: close - 1.04 s inside the window", fp.final_cut_ms(2000, 2500) == 1460)
    check("cut: close = t -> engine default t - 1.04 s", fp.final_cut_ms(5000, 5000) == 3960)
    check("cut: never earlier than the default", fp.final_cut_ms(5000, 1000) == 3960)

    from experiments.mist_w3po_v1.core.engine import Prompter
    pr = Prompter("zh2en")
    eng, th, seen, ids_seen = make_engine(pr)
    its = items()
    (e0, m0), s0, p0 = run(eng, th, its, seen, ids_seen)

    # --- identity: close_detect = t at every final read -> same cut as the engine
    tr = fp.transform(its, {"wA": 1150, "wB": 2300, "wC": 5030})
    check("non-final items are the same objects", all(a is b for a, b in zip(its, tr) if not a.final))
    check("final items: only post/full changed",
          all(dataclasses_eq_except(a, b) for a, b in zip(its, tr) if a.final))
    orig = fp.install({"wA": 1150, "wB": 2300, "wC": 5030})
    try:
        (e1, m1), s1, p1 = run(eng, th, its, seen, ids_seen)
    finally:
        E.Engine.read = orig
    check("identity: prompt ids identical", p1 == p0)
    check("identity: attention mask identical", torch.equal(m1, m0))
    check("identity: inputs_embeds bit-identical", torch.equal(e1, e0))
    check("identity: speaker-stream inputs identical", len(s1) == len(s0) and all(torch.equal(a, b) for a, b in zip(s1, s0)))

    # --- extended: close_detect = t + 5 s
    cd = {"wA": 6150, "wB": 7300, "wC": 10030}
    orig = fp.install(cd)
    try:
        (e2, m2), s2, p2 = run(eng, th, its, seen, ids_seen)
    finally:
        E.Engine.read = orig
    check("extended: prompt ids identical", p2 == p0)
    check("extended: attention mask identical", torch.equal(m2, m0))
    ids0 = torch.tensor([[0] * (m0.shape[1] - len(p)) + p for p in p0])
    diff = (e2 != e0).any(-1)
    check("extended: non-final rows unchanged", not diff[[i for i, it in enumerate(its) if not it.final]].any())
    check("extended: only audio-pad positions changed", not (diff & (ids0 != AUDIO_PAD)).any())
    check("extended: final rows changed", all(bool(diff[i].any()) for i, it in enumerate(its) if it.final))
    # the speaker input of a final item = post_to_tokens(P masked from t // 80); unique akeys -> one sd call per item
    ok = True
    sd_inputs = [s[0] for s in s2]
    if len(sd_inputs) == len(its):
        for it, x in zip(its, sd_inputs):
            p = it.post.copy()
            p[(fp.final_cut_ms(it.t_ms, cd[it.akey.split('|')[0]]) if it.final else max(0, it.t_ms - 1040)) // 80:] = 0
            want = torch.as_tensor(post_to_tokens(p, x.shape[0], len(it.wav[: it.t_ms * 16]) / 16000)).to(torch.bfloat16)
            ok &= torch.equal(x, want)
    else:
        ok = False
    check("extended: speaker inputs = P masked at the final cut", ok)
    check("extended: final cut = t here (whole window visible)",
          all(fp.final_cut_ms(it.t_ms, cd[it.akey.split('|')[0]]) == it.t_ms for it in its if it.final))
    print("ALL PASS")


def dataclasses_eq_except(a, b) -> bool:
    import dataclasses
    for f in dataclasses.fields(a):
        if f.name in ("post", "full"):
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            if va is not vb:
                return False
        elif va != vb:
            return False
    return b.full and not a.full and b.post is not a.post


if __name__ == "__main__":
    main()
