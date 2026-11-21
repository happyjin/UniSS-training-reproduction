"""E10 v3 tuner checks on the real S4 base (one GPU):
  1. v3 base (S4 + merged S5 adapter) + a fresh LoRA (zero B) computes the same logits as the stock S5 model on a
     record without speaker slots (all columns but the 8 re-initialised speaker rows);
  2. the trainable set is exactly thinker attention LoRA + 8 speaker rows + e10_spk_proj (no talker / MLP / encoder);
  3. a backward pass on a streaming record with speaker slots reaches e10_spk_proj, the speaker rows and LoRA;
  4. save_v3 -> load_v3_chain([S5, saved]) reproduces the logits of the in-memory model (slots included).

    CUDA_VISIBLE_DEVICES=0 $SIMULS2S/bin/python e10_multispk/tests/smoke_v3_tuner.py \
        --model checkpoints/S4_1031v2_ckpt93767 --adapter checkpoints/p4mix_2n_20261117/checkpoint-29773 \
        --records e10_multispk/data/li_v3_joint --out e10_multispk/runs/smoke_v3_tuner
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E10 = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(E10), "custom"), E10]

import torch  # noqa: E402


def make_batches(records, model_dir):
    from streaming_si.data import s4_pipeline as P
    from streaming_si.data.s4_pipeline import add_s4_stream_tokens, build_val_rows, s4_env
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from transformers import AutoTokenizer
    R = os.path.abspath(records)
    os.environ.update({"SI_S4_DATA": f"{R}/records/en2zh/train/records", "SI_S4_DATA_ROOT": R,
                       "SI_S4_VAL_DIR": f"{R}/val", "SI_S4_STRATIFIED": "0", "SI_MAX_LENGTH": "1280",
                       "SI_MAX_AUDIO_SEC": "30", "SI_S4_INDEX_DIR": ""})
    tok = AutoTokenizer.from_pretrained(model_dir)
    add_s4_stream_tokens(tok, capacity=int(json.load(open(os.path.join(model_dir, "config.json")))["vocab_size"]))
    enc = P.S4Encoder(tok, VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(model_dir)))
    rows = build_val_rows(s4_env())
    stream = P.collate_s4([enc.encode({k: v[3] for k, v in rows.items()})])
    ds = enc.train_ds()
    i = next(i for i, (fi, li) in enumerate(ds.items) if ds.indexes[fi].read(li)["e10"]["kind"] == "sd")
    sd = P.collate_s4([enc.encode({"s4_split": "train", "s4_index": i, "s4_designated": "", "s4_applied": "",
                                   "s4_seed": 7, "s4_plan": enc._train_digest})])
    return tok, stream, sd


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        elif isinstance(v, list) and v and any(torch.is_tensor(x) for x in v):
            out[k] = [x.to(dev) if torch.is_tensor(x) else x for x in v]
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    from e10 import v3
    v3.install()
    from e10.load import load_base, load_s5
    dev = "cuda:0"
    torch.manual_seed(0)
    tok, stream, sd = make_batches(a.records, a.model)
    slot = v3.slot_id(tok)
    stream, sd = to_dev(stream, dev), to_dev(sd, dev)
    print(f"[v3] stream sample: {stream['input_ids'].shape[1]} tokens, slots {stream['spk_ns']}; sd sample "
          f"{sd['input_ids'].shape[1]} tokens, slots {sd['spk_ns']}")
    assert stream["spk_ns"][0] > 0 and sd["spk_ns"][0] == 0
    plain = {k: v for k, v in sd.items() if k not in ("spk_posts", "spk_ns")}

    ref, _, ids = load_s5(a.model, a.adapter, merge=True, device=dev)
    with torch.no_grad():
        lr = ref(**plain).logits.float()
    del ref
    torch.cuda.empty_cache()

    base, _, _ = load_base(a.model, device=dev)
    raw = v3.build_base(base, [a.adapter], None, slot=slot)
    with torch.no_grad():
        raw.reinit_stream_token_rows(v3.spk_ids(raw), seed=20261026)
    pm = v3.attach_lora(raw, 32, 64, 0.0)
    names = v3.set_trainable(pm)
    spk = v3.spk_ids(raw)
    keep = torch.ones(lr.size(-1), dtype=torch.bool, device=dev)
    keep[spk] = False
    pm.eval()
    with torch.no_grad():
        le = pm(**sd).logits.float()
    d1 = float((le - lr)[..., keep].abs().max())
    print(f"[v3] 1. max |logit diff| vs stock S5 (sd record, non-speaker columns) = {d1:.3e}")
    assert d1 < 5e-2, d1

    bad = [n for n in names if not n.startswith("e10_spk_proj.") and
           ("talker" in n or "mlp" in n or "audio" in n or "encoder" in n)]
    lora = [n for n in names if "lora_A" in n]
    print(f"[v3] 2. trainable tensors {len(names)} (LoRA A {len(lora)}); forbidden {bad[:3]}; e.g. "
          f"{sorted(set(n.split('.lora_A')[0].split('.')[-1] for n in lora))}")
    assert not bad

    pm.train()
    st = dict(stream)
    st["spk_posts"] = [torch.rand_like(p) if p is not None else None for p in st["spk_posts"]]
    out = pm(**st)
    out.loss.backward()
    inner = pm.get_base_model()
    g_proj = inner.e10_spk_proj.mlp[-1].weight.grad
    g_tok = inner.thinker_model.model.embed_tokens.token_adapter.trainable_tokens_delta["default"].grad
    g_lora = sum(float(p.grad.abs().sum()) for n, p in inner.named_parameters() if "lora_B" in n and p.grad is not None)
    print(f"[v3] 3. loss {float(out.loss):.4f}; |grad spk_proj last| {float(g_proj.abs().sum()):.3e}; "
          f"|grad speaker rows| {float(g_tok.abs().sum()):.3e}; |grad lora_B| {g_lora:.3e}")
    assert float(g_proj.abs().sum()) > 0 and float(g_tok.abs().sum()) > 0 and g_lora > 0
    for grp, key in (("spk_proj", "e10_spk_proj."), ("tokens", "trainable_tokens_delta"), ("lora", "lora_")):
        gs = [p.grad.float().norm() for n, p in inner.named_parameters() if key in n and p.grad is not None]
        print(f"[v3]    grad norm {grp}: {float(torch.stack(gs).norm()) if gs else 0:.3e}")
    pm.zero_grad(set_to_none=True)

    with torch.no_grad():
        inner.e10_spk_proj.mlp[-1].weight.normal_(0, 0.02)
        for n, p in inner.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-3)
        g = inner.thinker_model.model.embed_tokens.token_adapter.trainable_tokens_delta["default"]
        g.add_(torch.randn_like(g) * 0.05)
    pm.eval()
    st["spk_posts"] = [torch.rand_like(p) if p is not None else None for p in stream["spk_posts"]]
    os.makedirs(a.out)
    v3.save_v3(pm, a.out)
    print("[v3] saved:", sorted(os.listdir(a.out)))
    merged = pm.merge_and_unload().eval()          # in-memory merge = what load_v3_chain must reproduce
    with torch.no_grad():
        l_mem = merged(**st).logits.float()
    del pm, inner, raw, base, merged
    torch.cuda.empty_cache()

    base2, _, _ = load_base(a.model, device=dev)
    raw2 = v3.load_v3_chain(base2, [a.adapter, a.out], slot).eval()
    with torch.no_grad():
        l_load = raw2(**st).logits.float()
    d4 = float((l_load - l_mem).abs().max())
    rel = d4 / float(l_mem.abs().max())
    print(f"[v3] 4. save -> load_v3_chain max |logit diff| = {d4:.3e} (relative {rel:.2e}); argmax agreement "
          f"{float((l_load.argmax(-1) == l_mem.argmax(-1)).float().mean()):.4f}")
    assert rel < 1e-3, rel
    print("[v3] ALL OK")


if __name__ == "__main__":
    main()
