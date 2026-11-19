#!/usr/bin/env python3
"""Causality assertion for the upstream talker path (regression guard).

Verifies that ``Qwen3TtsTalkerModel.forward`` is CAUSAL on the codec time axis:
changing input at positions > t must NOT change hidden state at position t.
If this fails, the talker leaks future codec tokens (== main-code labels) and
main token_acc collapses to ~1.0 for free.

Run:  python inference/test_talker_causality.py
Exits non-zero on failure (usable as a CI/pre-train guard).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch


def build_tiny_talker(device="cpu"):
    from inference.common import ensure_project_imports
    ensure_project_imports()
    from configuration_qwen_omnispeak import QwenOmniSpeakConfig  # noqa
    from modeling_qwen_omnispeak import Qwen3TtsTalkerModel

    # Minimal talker config: tiny dims, few layers -> fast CPU test.
    import importlib
    cfgmod = importlib.import_module("configuration_qwen_omnispeak")
    TalkerCfg = getattr(cfgmod, "Qwen3TtsTalkerConfig")
    cfg = TalkerCfg(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, rms_norm_eps=1e-6,
        vocab_size=16, codec_vocab_size=64,
    )
    torch.manual_seed(0)
    model = Qwen3TtsTalkerModel(cfg).to(device).eval()
    return model


def main():
    device = "cpu"
    model = build_tiny_talker(device)
    H = model.codec_embedding.embedding_dim
    B, T = 1, 12
    torch.manual_seed(1)
    emb = torch.randn(B, T, H, device=device)
    attn2d = torch.ones(B, T, dtype=torch.long, device=device)

    with torch.no_grad():
        h_full = model(inputs_embeds=emb, attention_mask=attn2d).last_hidden_state  # [B,T,H]

        # Perturb inputs strictly AFTER position t0; a causal model keeps
        # hidden[:, :t0+1] unchanged.
        t0 = T // 2
        emb_pert = emb.clone()
        emb_pert[:, t0 + 1:, :] = torch.randn_like(emb_pert[:, t0 + 1:, :]) * 5.0
        h_pert = model(inputs_embeds=emb_pert, attention_mask=attn2d).last_hidden_state

        max_drift_past = (h_full[:, :t0 + 1] - h_pert[:, :t0 + 1]).abs().max().item()
        max_drift_future = (h_full[:, t0 + 1:] - h_pert[:, t0 + 1:]).abs().max().item()

    print(f"[causality] perturb inputs at positions > {t0}")
    print(f"  max hidden drift at positions <= {t0} (should be ~0): {max_drift_past:.3e}")
    print(f"  max hidden drift at positions >  {t0} (should be > 0): {max_drift_future:.3e}")

    ok_causal = max_drift_past < 1e-4
    ok_effect = max_drift_future > 1e-3
    if ok_causal and ok_effect:
        print("  RESULT: PASS — talker is causal (no future leakage).")
        sys.exit(0)
    else:
        print("  RESULT: FAIL — "
              + ("past positions changed when future inputs changed => NON-CAUSAL LEAKAGE"
                 if not ok_causal else "future perturbation had no effect (test setup issue)"))
        sys.exit(1)


if __name__ == "__main__":
    main()
