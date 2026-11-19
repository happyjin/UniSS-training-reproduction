"""Robust wave 6: export a trained arm (wave6_disen_train.py lora_final.pt) as a plain HF Talker that si_infer loads.

  q/k/v/o  W <- W + (alpha / r) B A                       (fp32 add, stored bf16)
  disen    input embedding rows of the 32-voice vocabulary (GLOBAL_0 .. GLOBAL_0 + 4095) <- sigmoid(m) * row; the output
           head is untied and keeps the ORIGINAL embedding, so logits mean what they meant (inference uses c_id only).
  identity (--identity) the same export path with no LoRA and sigma = 1 (untied head): must reproduce T1c exactly.

Check (--check 5): on 5 rows of the training distribution (no c_irr), logits of the exported model vs the training-time
forward (base + LoRA modules + mask) at every label position: max |diff| and argmax agreement, written to check.json.

    qwen3omni python experiments/mist_w3po_v1/voicecond/wave6_export.py --ckpt <lora_final.pt> --out <dir> [--check 5]
    qwen3omni python experiments/mist_w3po_v1/voicecond/wave6_export.py --identity --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave6_disen_train as W  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device(a.device)
    model = AutoModelForCausalLM.from_pretrained(str(W.TALKER), dtype=torch.bfloat16).to(dev).eval()
    sd = None if a.identity else torch.load(a.ckpt, map_location="cpu")
    E0 = model.model.embed_tokens.weight.detach().clone()
    sig = None
    if sd is not None:
        scale = sd["alpha"] / sd["r"]
        for li, layer in enumerate(model.model.layers):
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(layer.self_attn, nm)
                L = sd["lora"][f"{li}.{nm}"]
                dW = (L["B"].float() @ L["A"].float()) * scale
                lin.weight.data = (lin.weight.data.float() + dW.to(dev)).to(torch.bfloat16)
        if sd["disen"] is not None:
            sig = torch.sigmoid(sd["disen"]["m"].float()).to(dev)
    # disen only: untie (the output head keeps the original embedding) and scale the input rows of the voice vocabulary by
    # sigma(m); tft and identity stay tied, exactly as T1c. The config written is T1c's own config.json with only
    # tie_word_embeddings changed (transformers 5's save_pretrained rewrites it in a new format: rope_parameters, layer_types,
    # dtype, which vLLM read differently in the v0 identity check).
    untie = sig is not None
    if untie:
        head = torch.nn.Linear(E0.shape[1], E0.shape[0], bias=False, dtype=torch.bfloat16, device=dev)
        head.weight = torch.nn.Parameter(E0.clone())
        new_in = E0.clone()
        rows = slice(W.GLOBAL_0, W.GLOBAL_0 + 4096)
        new_in[rows] = (new_in[rows].float() * sig).to(torch.bfloat16)
        model.config.tie_word_embeddings = False
        model.lm_head = head
        model.model.embed_tokens.weight = torch.nn.Parameter(new_in)
    out.mkdir(parents=True)
    model.save_pretrained(str(out), safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        if (W.TALKER / f).exists():
            shutil.copy2(W.TALKER / f, out / f)
    cfg = json.loads((W.TALKER / "config.json").read_text())
    cfg["tie_word_embeddings"] = not untie
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), "pt") as fh:
        keys = set(fh.keys())
        if untie:
            assert "lm_head.weight" in keys and torch.equal(fh.get_tensor("lm_head.weight").to(dev), E0)
            assert not torch.equal(fh.get_tensor("model.embed_tokens.weight").to(dev), E0)
        else:
            assert torch.equal(fh.get_tensor("model.embed_tokens.weight").to(dev), E0)

    # ---- check against the training-time forward
    rep = {"ckpt": a.ckpt, "identity": a.identity, "sigma_mean": float(sig.mean()) if sig is not None else 1.0,
           "sigma_min": float(sig.min()) if sig is not None else 1.0}
    if a.check:
        ex = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16).to(dev).eval()
        ref = AutoModelForCausalLM.from_pretrained(str(W.TALKER), dtype=torch.bfloat16).to(dev).eval()
        loras = W.add_lora(ref)
        disen = None
        if sd is not None:
            for k, m in loras.items():
                m.A.data = sd["lora"][k]["A"].to(dev)
                m.B.data = sd["lora"][k]["B"].to(dev)
            if sd["disen"] is not None:
                disen = W.Disen(ref.config.hidden_size).to(dev)
                disen.load_state_dict(sd["disen"])
        tok = AutoTokenizer.from_pretrained(str(W.QWEN3))
        rows = json.load(open(REPO / "eval_outputs/mist_w3po_v1/wave6_disen/rows_3000x32.json"))
        ds = W.DS(rows, 1)
        samples = [ds[k] for k in (0, 1, 2, 16, 17)][: a.check]
        from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
        from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
        P = load_encoder(BiCodecPhi(str(dev)), str(W.CENC))
        ga = [list(map(int, v)) for v in P.tokens(P.phi_wave([s["a"] for s in samples]))]
        ids, att, lb, lp, ll = W.build(samples, ga, tok, dev)
        with torch.no_grad():
            h_ref = ref.model(inputs_embeds=W.voice_embeds(ref, ids, disen), attention_mask=att).last_hidden_state
            l_ref = ref.lm_head(h_ref[lb, lp]).float()
            l_ex = ex(input_ids=ids, attention_mask=att).logits[lb, lp].float()
        rep.update(n_positions=int(lb.numel()), max_abs_diff=float((l_ref - l_ex).abs().max()),
                   mean_abs_diff=float((l_ref - l_ex).abs().mean()),
                   argmax_agree=float((l_ref.argmax(-1) == l_ex.argmax(-1)).float().mean()),
                   nll_ref=float(torch.nn.functional.cross_entropy(l_ref, ll)),
                   nll_export=float(torch.nn.functional.cross_entropy(l_ex, ll)))
    (out / "export_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
