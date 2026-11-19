"""M3 dry-run: shape + weight-load sanity check for the upstream talker input path.

Loads an existing ``qwen2_5_3b_omnispeak_*`` checkpoint with
``use_upstream_talker_input=True`` enabled, runs a dummy forward, and prints:

  * missing / unexpected weight keys (should only be ``thinker_adapter.*``)
  * per-tensor shape sanity (thinker signal / adapter / text_projection /
    text_channel / codec_emb / talker_out / main_codec_logits)
  * whether the loss is finite

Usage::

    python scripts/verify_forward_shape.py \
        --model-dir models/qwen2_5_3b_omnispeak_a \
        --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Make ``custom/`` importable so ``modeling_qwen_omnispeak`` resolves.
if str(REPO_ROOT / 'custom') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'custom'))


def _load_config_dict(model_dir: Path) -> Dict[str, Any]:
    with (model_dir / 'config.json').open('r', encoding='utf-8') as f:
        return json.load(f)


def _pick_dtype(name: str) -> torch.dtype:
    return {'bfloat16': torch.bfloat16, 'float16': torch.float16,
            'fp16': torch.float16, 'bf16': torch.bfloat16,
            'float32': torch.float32, 'fp32': torch.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True,
                        help='Path to a qwen2_5_3b_omnispeak_* checkpoint directory.')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--codec-len', type=int, default=64,
                        help='Length of the dummy codec sequence (frames).')
    parser.add_argument('--text-len', type=int, default=24,
                        help='Length of the dummy assistant text (tokens).')
    parser.add_argument('--codec-bos-idx', type=int, default=6,
                        help='Position of codec_bos in the dummy codec_input_ids.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f'Model dir not found: {model_dir}')

    dtype = _pick_dtype(args.dtype)
    device = torch.device(args.device)

    # ── 1) Build config with the upstream-input flag flipped ON ─────────
    from qwen_omnispeak.configuration_qwen_omnispeak import QwenOmniSpeakConfig
    from qwen_omnispeak.modeling_qwen_omnispeak import (
        QwenOmniSpeakForConditionalGeneration,
    )

    cfg_dict = _load_config_dict(model_dir)
    cfg_dict['use_upstream_talker_input'] = True
    cfg_dict.setdefault('thinker_signal_type', 'hidden_plus_embed')
    config = QwenOmniSpeakConfig(**cfg_dict)

    print(f'[cfg] use_upstream_talker_input = {config.use_upstream_talker_input}')
    print(f'[cfg] thinker_signal_type       = {config.thinker_signal_type}')
    print(f'[cfg] thinker_family            = {config.thinker_family}')
    talker_cfg = config.talker_config
    print(f'[cfg] talker.hidden_size        = {talker_cfg.hidden_size}')
    print(f'[cfg] talker.text_hidden_size   = {talker_cfg.text_hidden_size}')
    print(f'[cfg] talker.tts_pad_token_id   = {talker_cfg.tts_pad_token_id}')

    # ── 2) Instantiate model on meta / cpu, then load weights ───────────
    print('[build] instantiating model ...')
    model = QwenOmniSpeakForConditionalGeneration(config)
    print(f'[build] instantiation done, total params '
          f'{sum(p.numel() for p in model.parameters()):,}')

    print('[load] loading state_dict from checkpoint ...')
    from safetensors.torch import load_file
    import glob
    state_dict: Dict[str, torch.Tensor] = {}
    for sf in sorted(glob.glob(str(model_dir / '*.safetensors'))):
        state_dict.update(load_file(sf))
    print(f'[load] state_dict has {len(state_dict)} tensors')

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'[load] missing_keys    : {len(missing)}')
    for k in missing[:20]:
        print(f'    - {k}')
    if len(missing) > 20:
        print(f'    ... and {len(missing) - 20} more')
    print(f'[load] unexpected_keys : {len(unexpected)}')
    for k in unexpected[:20]:
        print(f'    - {k}')
    if len(unexpected) > 20:
        print(f'    ... and {len(unexpected) - 20} more')

    # Sanity: the only missing family should be thinker_adapter.*
    unaccounted = [k for k in missing if 'thinker_adapter' not in k]
    if unaccounted:
        print(f'[WARN] {len(unaccounted)} missing key(s) not covered by '
              f'thinker_adapter (first 5):')
        for k in unaccounted[:5]:
            print(f'    ! {k}')

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # ── 3) Build a dummy batch and run forward ──────────────────────────
    print('\n[dummy] building batch ...')
    B, T_codec, T_text = args.batch_size, args.codec_len, args.text_len
    codec_bos_id = int(getattr(config, 'codec_bos_id', 2149))
    codec_pad_id = int(getattr(config, 'codec_pad_id', 2148))
    tts_pad_id = int(talker_cfg.tts_pad_token_id)

    # thinker input: a plausible chatml-ish sequence (random ids in vocab range).
    thinker_vocab = 151_643
    torch.manual_seed(0)
    input_ids = torch.randint(0, thinker_vocab, (B, T_text + 8), device=device)
    attention_mask = torch.ones_like(input_ids)

    # codec_input_ids: pad × (codec_bos_idx-1) + codec_bos + random main codes.
    codec_input_ids = torch.full((B, T_codec), codec_pad_id, device=device, dtype=torch.long)
    codec_input_ids[:, args.codec_bos_idx] = codec_bos_id
    codec_input_ids[:, args.codec_bos_idx + 1:] = torch.randint(
        0, int(talker_cfg.codec_vocab_size), (B, T_codec - args.codec_bos_idx - 1),
        device=device)
    codec_attention_mask = torch.ones_like(codec_input_ids)
    codec_labels = codec_input_ids.clone()
    # -100 out the control-prefix slots
    codec_labels[:, :args.codec_bos_idx + 1] = -100

    # target_text_ids: random assistant text token ids (padded).
    target_text_ids = torch.randint(0, thinker_vocab, (B, T_text), device=device)
    codec_bos_index = torch.full((B,), args.codec_bos_idx, dtype=torch.long, device=device)

    print(f'[dummy] input_ids        : {tuple(input_ids.shape)}')
    print(f'[dummy] codec_input_ids  : {tuple(codec_input_ids.shape)}')
    print(f'[dummy] target_text_ids  : {tuple(target_text_ids.shape)}')
    print(f'[dummy] codec_bos_index  : {codec_bos_index.tolist()}')

    print('\n[fwd] running forward ...')
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            codec_input_ids=codec_input_ids,
            codec_attention_mask=codec_attention_mask,
            codec_labels=codec_labels,
            target_text_ids=target_text_ids,
            codec_bos_index=codec_bos_index,
            run_thinker=True,
        )

    print(f'[fwd] main_codec_logits  : {tuple(out.main_codec_logits.shape)}')
    if out.residual_codec_logits is not None:
        print(f'[fwd] residual_codec_lg  : {tuple(out.residual_codec_logits.shape)}')
    else:
        print('[fwd] residual_codec_lg  : (skipped, no residual labels in dummy)')
    if out.talker_hidden_states is not None:
        print(f'[fwd] talker_hidden      : {tuple(out.talker_hidden_states.shape)}')
    if out.semantic_block is not None:
        print(f'[fwd] semantic_block     : {tuple(out.semantic_block.shape)}  (text_channel)')
    if out.loss is not None:
        loss_val = out.loss.detach().float().item()
        print(f'[fwd] loss               : {loss_val:.4f}  '
              f'(finite={torch.isfinite(out.loss).item()})')
    else:
        print('[fwd] loss               : None')

    print('\n[dry-run] shape sanity OK.' if out.main_codec_logits.shape[:2] == codec_input_ids.shape
          else '[dry-run] SHAPE MISMATCH!')


if __name__ == '__main__':
    main()
