"""Wave 6: where does the exported Talker's logit mismatch come from? Same 5 rows / C ids as wave6_export.py --check.
Forwards (all eval mode, sdpa, no dropout, no c_irr):
  F2  training-form: bf16 base + LoRA path in fp32 (wave6_disen_train.LoRA)        <- the export check's reference
  F1  bf16 base + LoRA path in bf16
  F3  merged bf16 (the exported model directory)
  F4  merged fp32: fp32 base, W + (alpha/r) B A in fp32, sigma(m) rows in fp32, fp32 forward
  F5  unmerged fp32: fp32 base + fp32 LoRA path
  F0  T1c bf16 without LoRA (lower anchor)
Reports NLL on the labels and argmax agreement / max |diff| of each forward vs F2 and vs F5.
    qwen3omni python experiments/mist_w3po_v1/voicecond/wave6_export_diag.py --arm disen|tft --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave6_disen_train as W  # noqa: E402

WD = REPO / "eval_outputs/mist_w3po_v1/wave6_disen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["disen", "tft"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda:0")
    sd = torch.load(WD / f"train_{a.arm}/lora_final.pt", map_location="cpu")
    tok = AutoTokenizer.from_pretrained(str(W.QWEN3))
    rows = json.load(open(WD / "rows_3000x32.json"))
    ds = W.DS(rows, 1)
    samples = [ds[k] for k in (0, 1, 2, 16, 17)]
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    P = load_encoder(BiCodecPhi(str(dev)), str(W.CENC))
    ga = [list(map(int, v)) for v in P.tokens(P.phi_wave([s["a"] for s in samples]))]
    del P
    ids, att, lb, lp, ll = W.build(samples, ga, tok, dev)
    sig = torch.sigmoid(sd["disen"]["m"].float()).to(dev) if sd["disen"] is not None else None

    def logits_unmerged(dtype, lora_dtype):
        m = AutoModelForCausalLM.from_pretrained(str(W.TALKER), dtype=dtype).to(dev).eval()
        if lora_dtype is not None:
            L = W.add_lora(m)
            for k, mod in L.items():
                mod.A.data = sd["lora"][k]["A"].to(dev)
                mod.B.data = sd["lora"][k]["B"].to(dev)
                if lora_dtype == "bf16":
                    mod.forward = (lambda mm: (lambda x: mm.base(x) + F.linear(F.linear(x, mm.A.to(x.dtype)),
                                                                                mm.B.to(x.dtype)) * mm.scale))(mod)
        with torch.no_grad():
            emb = m.model.embed_tokens(ids)
            if sig is not None and lora_dtype is not None:
                emb = torch.cat([emb[:, :W.VOICE_AT], emb[:, W.VOICE_AT: W.VOICE_AT + W.N_VOICE] * sig.to(emb.dtype),
                                 emb[:, W.VOICE_AT + W.N_VOICE:]], 1)
            h = m.model(inputs_embeds=emb, attention_mask=att).last_hidden_state
            lg = m.lm_head(h[lb, lp]).float()
        del m
        torch.cuda.empty_cache()
        return lg

    def logits_merged_fp32():
        m = AutoModelForCausalLM.from_pretrained(str(W.TALKER), dtype=torch.float32).to(dev).eval()
        scale = sd["alpha"] / sd["r"]
        for li, layer in enumerate(m.model.layers):
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(layer.self_attn, nm)
                L = sd["lora"][f"{li}.{nm}"]
                lin.weight.data += (L["B"].float() @ L["A"].float()).to(dev) * scale
        with torch.no_grad():
            emb = m.model.embed_tokens(ids)
            if sig is not None:
                emb = torch.cat([emb[:, :W.VOICE_AT], emb[:, W.VOICE_AT: W.VOICE_AT + W.N_VOICE] * sig,
                                 emb[:, W.VOICE_AT + W.N_VOICE:]], 1)
            h = m.model(inputs_embeds=emb, attention_mask=att).last_hidden_state
            lg = m.lm_head(h[lb, lp]).float()
        del m
        torch.cuda.empty_cache()
        return lg

    def logits_export():
        m = AutoModelForCausalLM.from_pretrained(str(WD / "eval" / a.arm / "model"), dtype=torch.bfloat16).to(dev).eval()
        with torch.no_grad():
            lg = m(input_ids=ids, attention_mask=att).logits[lb, lp].float()
        del m
        torch.cuda.empty_cache()
        return lg

    L = {"F2_bf16_base_fp32_lora": logits_unmerged(torch.bfloat16, "fp32"),
         "F1_bf16_base_bf16_lora": logits_unmerged(torch.bfloat16, "bf16"),
         "F3_merged_bf16_export": logits_export(),
         "F4_merged_fp32": logits_merged_fp32(),
         "F5_fp32_base_fp32_lora": logits_unmerged(torch.float32, "fp32"),
         "F0_t1c_bf16_no_lora": logits_unmerged(torch.bfloat16, None)}
    rep = {"arm": a.arm, "n_positions": int(ll.numel()), "forwards": {}}
    for k, v in L.items():
        r = {"nll": round(float(F.cross_entropy(v, ll)), 4)}
        for ref in ("F2_bf16_base_fp32_lora", "F5_fp32_base_fp32_lora"):
            r[f"argmax_agree_vs_{ref[:2]}"] = round(float((v.argmax(-1) == L[ref].argmax(-1)).float().mean()), 4)
            r[f"max_abs_vs_{ref[:2]}"] = round(float((v - L[ref]).abs().max()), 3)
            r[f"mean_abs_vs_{ref[:2]}"] = round(float((v - L[ref]).abs().mean()), 4)
        rep["forwards"][k] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
