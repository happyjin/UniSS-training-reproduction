"""E10 tuner checks on the real S4 base (one GPU, ~25 GB):
  1. warm start (e10.tuner.warm_start) with a zero e10_sd_proj computes the same logits as base + the stock S5
     adapter (checkpoint-29773) on a sample without speaker tags (all columns but the 8 speaker rows);
  2. scope `thinker`: exactly thinker LoRA + the 12 token rows + e10_sd_proj are trainable;
  3. a backward pass with a speaker-tagged sample and a non-zero posterior reaches e10_sd_proj and the token rows;
  4. save_e10 -> load_e10 round trip reproduces the logits.

    CUDA_VISIBLE_DEVICES=0 $SIMULS2S/bin/python e10_multispk/tests/smoke_tuner_equivalence.py \
        --model checkpoints/S4_1031v2_ckpt93767 --adapter checkpoints/p4mix_2n_20261117/checkpoint-29773 \
        --records e10_multispk/runs/smoke_records_v1 --out e10_multispk/runs/smoke_tuner
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E10 = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(E10), "custom"), E10]


def batch_of(records, model_dir, want_tags):
    from streaming_si.data import s4_pipeline as P
    from streaming_si.data.s4_pipeline import add_s4_stream_tokens, build_val_rows, s4_env
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from transformers import AutoTokenizer
    R = os.path.abspath(records)
    os.environ.update({"SI_S4_DATA": f"{R}/records/en2zh/train/records", "SI_S4_DATA_ROOT": R,
                       "SI_S4_VAL_DIR": f"{R}/val", "SI_S4_STRATIFIED": "0", "SI_MAX_LENGTH": "1024",
                       "SI_MAX_AUDIO_SEC": "30"})
    tok = AutoTokenizer.from_pretrained(model_dir)
    add_s4_stream_tokens(tok, capacity=int(json.load(open(os.path.join(model_dir, "config.json")))["vocab_size"]))
    enc = P.S4Encoder(tok, VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(model_dir)))
    if want_tags:
        rows = build_val_rows(s4_env())
        row = {k: v[0] for k, v in rows.items()}
    else:   # an sd record (plain instruction, no tags) from the train file
        ds = enc.train_ds()
        i = next(i for i, (fi, li) in enumerate(ds.items) if ds.indexes[fi].read(li)["e10"]["kind"] == "sd")
        row = {"s4_split": "train", "s4_index": i, "s4_designated": "", "s4_applied": "", "s4_seed": 7,
               "s4_plan": enc._train_digest}
    return P.collate_s4([enc.encode(row)])


def to_dev(b, dev):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [x.to(dev) for x in v]
        else:
            out[k] = v
    return out


import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    os.environ.setdefault("SI_S4_TRAIN_SCOPE", "thinker")
    from e10 import patch
    patch.install()
    from e10.load import load_base, load_s5
    from e10.tuner import _finalize, apply_scope, load_e10, save_e10, spk_ids_from_stream, warm_start
    dev = "cuda:0"
    torch.manual_seed(0)
    plain = to_dev(batch_of(a.records, a.model, want_tags=False), dev)
    tagged = to_dev(batch_of(a.records, a.model, want_tags=True), dev)

    ref, _, ids = load_s5(a.model, a.adapter, merge=False, device=dev)
    # a bare PeftModel.from_pretrained leaves SI.lm_head pointing at the unwrapped head (S5's own inference merges
    # first, which hides it); refresh the alias as s4_peft.load_s4_adapter does
    from streaming_si.integration.s4_peft import _refresh_thinker_aliases
    ref = _refresh_thinker_aliases(ref).eval()
    with torch.no_grad():
        lr = ref(**{k: v for k, v in plain.items() if k != "spk_posts"}).logits.float()
    del ref
    torch.cuda.empty_cache()

    base, _, ids2 = load_base(a.model, device=dev)
    pm = warm_start(base, a.adapter, seed=20261026)
    pm = _finalize(pm, "smoke")
    spk = spk_ids_from_stream(ids2)
    keep = torch.ones(lr.size(-1), dtype=torch.bool, device=dev)
    keep[spk] = False
    pm.eval()
    with torch.no_grad():
        le = pm(**plain).logits.float()
    d1 = float((le - lr)[..., keep].abs().max())
    print(f"[tuner] 1. max |logit diff| (non-speaker columns) = {d1:.3e}")
    assert d1 < 1e-3, d1

    names = apply_scope(pm, "thinker")
    bad = [n for n in names if "talker" in n]
    print(f"[tuner] 2. trainable tensors {len(names)}; talker tensors {len(bad)}; e10/token: "
          f"{[n for n in names if 'e10_sd_proj' in n or 'trainable_tokens' in n]}")
    assert not bad and any("e10_sd_proj" in n for n in names) and any("trainable_tokens" in n for n in names)

    pm.train()
    tagged["spk_posts"] = [torch.rand_like(p) for p in tagged["spk_posts"]]
    out = pm(**tagged)
    out.loss.backward()
    raw = pm.get_base_model()
    g_sd = raw.e10_sd_proj.modules_to_save["default"].weight.grad
    g_tok = raw.thinker_model.model.embed_tokens.token_adapter.trainable_tokens_delta["default"].grad
    print(f"[tuner] 3. loss {float(out.loss):.4f}; |grad sd| {float(g_sd.abs().sum()):.3e}; "
          f"|grad speaker rows| {float(g_tok[4:].abs().sum()):.3e}")
    assert float(g_sd.abs().sum()) > 0 and float(g_tok[4:].abs().sum()) > 0
    pm.zero_grad(set_to_none=True)

    with torch.no_grad():
        raw.e10_sd_proj.modules_to_save["default"].weight.normal_(0, 0.01)
    pm.eval()
    with torch.no_grad():
        l_before = pm(**tagged).logits.float()
    save_e10(pm, a.out)
    del pm, raw, base
    torch.cuda.empty_cache()
    base2, _, _ = load_base(a.model, device=dev)
    pm2 = load_e10(base2, a.out, is_trainable=False).eval()
    with torch.no_grad():
        l_after = pm2(**tagged).logits.float()
    d4 = float((l_after - l_before).abs().max())
    print(f"[tuner] 4. save/load round trip max |logit diff| = {d4:.3e}; files {sorted(os.listdir(a.out))}")
    assert d4 < 1e-3, d4
    print("[tuner] OK")


if __name__ == "__main__":
    main()
