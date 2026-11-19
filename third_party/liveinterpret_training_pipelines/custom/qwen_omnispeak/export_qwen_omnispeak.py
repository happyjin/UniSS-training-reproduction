#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Export Qwen-OmniSpeak model directories.

This script:

1. Loads upstream Qwen3-Omni or Qwen2.5-Omni (Thinker is the part we keep)
   and Qwen3-TTS weights from the supplied paths or HF model IDs.
2. Builds a Qwen-OmniSpeak composite model for one or more variants
   (A/B/C/D/E/F/G/H/I/J, see ``QWEN_OMNISPEAK_MODEL_DESIGN.md``).
3. Copies the selected Qwen Omni processor (tokenizer, image_processor, etc.)
   into the export directory.

After exporting, the new model directory is *self-sufficient*: re-training
with ms-swift only needs the registration plugin
(``register_qwen_omnispeak.py``) plus the exported directory.

Example::

    python training_pipelines/custom/qwen_omnispeak/export_qwen_omnispeak.py \\
        --thinker_model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --tts_model_path     Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --output_root        ./exported/qwen_omnispeak \\
        --variants           A B C D E F \\
        --dtype              bfloat16
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Make the local ``modeling_qwen_omnispeak`` package importable.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from modeling_qwen_omnispeak import (  # noqa: E402
    QwenOmniSpeakForConditionalGeneration,
)
from configuration_qwen_omnispeak import (  # noqa: E402
    QWEN25_THINKER_FAMILY,
    QWEN3_THINKER_FAMILY,
    QwenOmniSpeakConfig,
    Qwen3TtsTalkerConfig,
    default_hidden_state_selection,
    infer_thinker_family,
)


# ---------------------------------------------------------------------------
# Variants spec (kept in sync with ``QwenOmniSpeakConfig.SUPPORTED_VARIANTS``)
# ---------------------------------------------------------------------------

ALL_VARIANTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Default output-directory / model_type prefix per Thinker family. The prefix
# is combined with the lower-cased variant letter to produce directory names
# such as ``qwen2_5_7b_omnispeak_b`` (kept in sync with the model_types
# registered in ``register_qwen_omnispeak.py``).
DEFAULT_VARIANT_PREFIX = {
    QWEN3_THINKER_FAMILY: 'qwen3_omnispeak',
    QWEN25_THINKER_FAMILY: 'qwen2_5_omnispeak',
}


def _auto_variant_prefix(thinker_family: str, thinker_model_path: Optional[str]) -> str:
    """Best-effort prefix when ``--variant_prefix`` is not supplied.

    For Qwen2.5-Omni we further disambiguate 3B vs 7B from the model path so the
    exported directories match the requested ``qwen2_5_3b_omnispeak_*`` /
    ``qwen2_5_7b_omnispeak_*`` naming.
    """
    if thinker_family == QWEN25_THINKER_FAMILY:
        tag = (thinker_model_path or '').lower()
        if '3b' in tag:
            return 'qwen2_5_3b_omnispeak'
        if '7b' in tag:
            return 'qwen2_5_7b_omnispeak'
        return 'qwen2_5_omnispeak'
    return DEFAULT_VARIANT_PREFIX.get(thinker_family, 'qwen_omnispeak')


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _normalise_thinker_family(value: Optional[str], config_dict: Optional[Dict] = None) -> str:
    return infer_thinker_family(config_dict or {}, value)


def _load_thinker(thinker_model_path: str, dtype: torch.dtype, thinker_family: Optional[str] = None):
    """Load only the *thinker* part of a Qwen Omni checkpoint.

    Supports Qwen3-Omni-MoE and Qwen2.5-Omni. The returned ``full.config`` is
    persisted inside Qwen-OmniSpeak so future training/export can reconstruct
    the correct Thinker class and processor.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(thinker_model_path, trust_remote_code=True)
    family = _normalise_thinker_family(thinker_family, cfg.to_dict())
    if family == QWEN25_THINKER_FAMILY:
        from transformers import Qwen2_5OmniForConditionalGeneration
        print(f'[load] Qwen2.5-Omni from: {thinker_model_path}')
        full = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            thinker_model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    else:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        print(f'[load] Qwen3-Omni from: {thinker_model_path}')
        full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            thinker_model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    thinker = full.thinker
    return full, thinker, family


def _thinker_hidden_size(thinker) -> int:
    text_config = getattr(thinker.config, 'text_config', None)
    if text_config is not None and getattr(text_config, 'hidden_size', None) is not None:
        return int(text_config.hidden_size)
    if getattr(thinker.config, 'hidden_size', None) is not None:
        return int(thinker.config.hidden_size)
    raise AttributeError('Could not infer thinker hidden_size from thinker.config.')


def _build_qwen3_tts_talker_config(tts_state_dict: Optional[Dict[str, torch.Tensor]]) -> Qwen3TtsTalkerConfig:
    """Best-effort inference of the Qwen3-TTS talker config from a state_dict.

    Falls back to the documented Qwen3-TTS-12Hz-1.7B-CustomVoice defaults when
    fields cannot be inferred. Reads from upstream ``talker.model.*`` and
    ``talker.code_predictor.*`` namespaces.
    """
    cfg = Qwen3TtsTalkerConfig()
    if tts_state_dict is None:
        return cfg
    import re
    layers = set()
    cp_layers = set()
    for k in tts_state_dict.keys():
        m = re.match(r'^talker\.model\.layers\.(\d+)\.', k)
        if m:
            layers.add(int(m.group(1)))
        m = re.match(r'^talker\.code_predictor\.model\.layers\.(\d+)\.', k)
        if m:
            cp_layers.add(int(m.group(1)))
    if layers:
        cfg.original_num_hidden_layers = max(layers) + 1
    if cp_layers:
        cfg.code_predictor_num_layers = max(cp_layers) + 1

    def _shape(key):
        return tuple(tts_state_dict[key].shape) if key in tts_state_dict else None

    # text_embedding
    s = _shape('talker.model.text_embedding.weight')
    if s is not None:
        cfg.vocab_size, cfg.hidden_size = s
    # codec_embedding (main)
    s = _shape('talker.model.codec_embedding.weight')
    if s is not None:
        cfg.codec_vocab_size = s[0]
    # backbone intermediate / heads
    s = _shape('talker.model.layers.0.mlp.gate_proj.weight')
    if s is not None:
        cfg.intermediate_size = s[0]
    s_q = _shape('talker.model.layers.0.self_attn.q_proj.weight')
    s_k = _shape('talker.model.layers.0.self_attn.k_proj.weight')
    s_qn = _shape('talker.model.layers.0.self_attn.q_norm.weight')
    if s_qn is not None:
        cfg.head_dim = s_qn[0]
    if s_q is not None and cfg.head_dim:
        cfg.num_attention_heads = s_q[0] // cfg.head_dim
    if s_k is not None and cfg.head_dim:
        cfg.num_key_value_heads = s_k[0] // cfg.head_dim
    # code predictor
    s = _shape('talker.code_predictor.model.layers.0.mlp.gate_proj.weight')
    if s is not None:
        cfg.code_predictor_intermediate_size = s[0]
        cfg.code_predictor_hidden_size = s[1]
    s = _shape('talker.code_predictor.lm_head.0.weight')
    if s is not None:
        cfg.code_predictor_vocab_size = s[0]
    # Count actual residual code groups in the upstream checkpoint.
    cp_lm_head_indices = set()
    cp_codec_emb_indices = set()
    for k in tts_state_dict.keys():
        m = re.match(r'^talker\.code_predictor\.lm_head\.(\d+)\.', k)
        if m:
            cp_lm_head_indices.add(int(m.group(1)))
        m = re.match(r'^talker\.code_predictor\.model\.codec_embedding\.(\d+)\.', k)
        if m:
            cp_codec_emb_indices.add(int(m.group(1)))
    if cp_lm_head_indices:
        cfg.num_code_groups = max(cp_lm_head_indices) + 1
    s = _shape('talker.code_predictor.model.codec_embedding.0.weight')
    cfg.code_predictor_has_codec_embedding = s is not None
    s_q = _shape('talker.code_predictor.model.layers.0.self_attn.q_proj.weight')
    s_k = _shape('talker.code_predictor.model.layers.0.self_attn.k_proj.weight')
    s_qn = _shape('talker.code_predictor.model.layers.0.self_attn.q_norm.weight')
    if s_qn is not None:
        cfg.code_predictor_head_dim = s_qn[0]
    if s_q is not None and cfg.code_predictor_head_dim:
        cfg.code_predictor_num_attention_heads = s_q[0] // cfg.code_predictor_head_dim
    if s_k is not None and cfg.code_predictor_head_dim:
        cfg.code_predictor_num_key_value_heads = s_k[0] // cfg.code_predictor_head_dim
    return cfg


def _load_tts_state_dict(tts_model_path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    """Load a raw state_dict from the Qwen3-TTS checkpoint.

    Strategy: prefer ``safetensors`` shards, fall back to a single
    ``pytorch_model.bin``. We *don't* instantiate the upstream class because
    transformers 5.x doesn't ship a ``Qwen3TTS*`` class yet — we just need
    raw tensors and the shape information.
    """
    if tts_model_path is None:
        return None
    p = Path(tts_model_path)
    if not p.exists():
        # Try resolving as a HF id via huggingface_hub snapshot.
        try:
            from huggingface_hub import snapshot_download
            cache_dir = snapshot_download(repo_id=tts_model_path)
            p = Path(cache_dir)
        except Exception as e:
            print(f'[warn] Could not resolve Qwen3-TTS path {tts_model_path!r}: {e}; '
                  f'will random-init talker.')
            return None
    state: Dict[str, torch.Tensor] = {}
    safes = sorted(p.glob('*.safetensors'))
    if safes:
        from safetensors.torch import load_file
        for sf in safes:
            print(f'[load] tts shard: {sf.name}')
            shard = load_file(str(sf))
            state.update(shard)
        return state
    bins = sorted(list(p.glob('pytorch_model*.bin')) + list(p.glob('model.bin')))
    if bins:
        for bn in bins:
            print(f'[load] tts shard: {bn.name}')
            shard = torch.load(str(bn), map_location='cpu', weights_only=True)
            state.update(shard)
        return state
    print(f'[warn] No model shards found under {p}; will random-init talker.')
    return None


# ---------------------------------------------------------------------------
# State-dict remapping
# ---------------------------------------------------------------------------


# Map raw Qwen3-TTS state_dict keys to talker namespace inside Qwen-OmniSpeak.
#
# Upstream Qwen3-TTS-12Hz-1.7B-Base layout (every key starts with "talker."):
#   talker.model.text_embedding.weight
#   talker.model.codec_embedding.weight
#   talker.model.layers.{0..27}.<rest>
#   talker.model.norm.weight
#   talker.codec_head.weight
#   talker.code_predictor.small_to_mtp_projection.{weight,bias}
#   talker.code_predictor.model.layers.{0..4}.<rest>
#   talker.code_predictor.model.norm.weight
#   talker.code_predictor.model.codec_embedding.{0..15}.weight
#   talker.code_predictor.lm_head.{0..15}.weight
#
#   talker.text_projection.<...>     -- input projection inside upstream talker (we don't use it)
#   speaker_encoder.<...>            -- speaker conditioner (we don't use it)
#
# Our Qwen3TtsTalkerForConditionalGeneration mirrors the *talker.* subtree
# exactly, so the remap is essentially:
#     drop "talker." prefix
#     rewrite "model.layers.{old}" -> "model.layers.{new}" for kept layers
#
# text_projection.* and speaker_encoder.* are returned as a side bucket so
# they can be saved as auxiliary state for potential future use.


def _remap_tts_to_talker(
    raw_state,
    keep_layer_indices,
):
    """Rename raw Qwen3-TTS keys to match Qwen3TtsTalkerForConditionalGeneration.

    Returns (main_state, aux_state):
      main_state -> talker.load_state_dict(strict=False)
      aux_state  -> sidecar containing text_projection / speaker_encoder
    """
    import re as _re
    keep_set = {old: new for new, old in enumerate(keep_layer_indices)}
    out = {}
    aux = {}
    dropped = []

    layer_re = _re.compile(r"^model\.layers\.(\d+)(\..*)?$")

    for raw_key, value in raw_state.items():
        if raw_key.startswith("speaker_encoder."):
            aux[raw_key] = value
            continue
        if raw_key.startswith("talker.text_projection."):
            aux[raw_key] = value
            continue
        if not raw_key.startswith("talker."):
            dropped.append(raw_key)
            continue
        k = raw_key[len("talker."):]
        m = layer_re.match(k)
        if m is not None:
            old_idx = int(m.group(1))
            tail = m.group(2) or ""
            if old_idx not in keep_set:
                continue
            new_idx = keep_set[old_idx]
            out[f"model.layers.{new_idx}{tail}"] = value
            continue
        # Everything else already lives at the same relative path in our talker.
        out[k] = value

    if dropped:
        print(f"[remap] dropped {len(dropped)} unmatched TTS state-dict keys (first 5): {dropped[:5]}")
    return out, aux


# ---------------------------------------------------------------------------
# Building one variant
# ---------------------------------------------------------------------------


def build_one_variant(
    variant: str,
    thinker,
    thinker_full_config,
    tts_state: Optional[Dict[str, torch.Tensor]],
    tts_inferred_config: Qwen3TtsTalkerConfig,
    dtype: torch.dtype,
    thinker_family: str,
    thinker_model_id: Optional[str],
    talker_model_id: Optional[str],
    hidden_state_selection,
) -> Tuple[QwenOmniSpeakForConditionalGeneration, Dict[str, torch.Tensor]]:
    """Assemble a Qwen-OmniSpeak model for a single variant.

    Returns ``(model, aux_state)`` where ``aux_state`` collects raw upstream
    keys we don't load into the composite (kept for sidecar persistence).
    """
    # Build config from variant defaults, then override with the inferred
    # hidden / vocab sizes from the Qwen3-TTS checkpoint when available.
    talker_overrides = {
        'hidden_size': tts_inferred_config.hidden_size,
        'intermediate_size': tts_inferred_config.intermediate_size,
        'num_attention_heads': tts_inferred_config.num_attention_heads,
        'num_key_value_heads': tts_inferred_config.num_key_value_heads,
        'head_dim': tts_inferred_config.head_dim,
        'vocab_size': tts_inferred_config.vocab_size,
        'codec_vocab_size': tts_inferred_config.codec_vocab_size,
        'num_code_groups': tts_inferred_config.num_code_groups,
        'original_num_hidden_layers': tts_inferred_config.original_num_hidden_layers,
        'max_position_embeddings': tts_inferred_config.max_position_embeddings,
        'rms_norm_eps': tts_inferred_config.rms_norm_eps,
        # Code-predictor inner shape — must be propagated when present.
        'code_predictor_hidden_size': tts_inferred_config.code_predictor_hidden_size,
        'code_predictor_intermediate_size': tts_inferred_config.code_predictor_intermediate_size,
        'code_predictor_num_layers': tts_inferred_config.code_predictor_num_layers,
        'code_predictor_num_attention_heads':
            tts_inferred_config.code_predictor_num_attention_heads,
        'code_predictor_num_key_value_heads':
            tts_inferred_config.code_predictor_num_key_value_heads,
        'code_predictor_head_dim': tts_inferred_config.code_predictor_head_dim,
        'code_predictor_vocab_size': tts_inferred_config.code_predictor_vocab_size,
        'code_predictor_has_codec_embedding':
            tts_inferred_config.code_predictor_has_codec_embedding,
    }

    bridge_overrides = {
        'thinker_hidden_size': _thinker_hidden_size(thinker),
        'talker_hidden_size': tts_inferred_config.hidden_size,
        'intermediate_size': max(tts_inferred_config.intermediate_size, 4096),
    }

    cfg = QwenOmniSpeakConfig.from_variant(
        variant=variant,
        thinker_config=thinker_full_config.to_dict(),
        talker_overrides=talker_overrides,
        bridge_overrides=bridge_overrides,
        thinker_family=thinker_family,
        thinker_model_id=thinker_model_id,
        talker_model_id=talker_model_id,
        hidden_state_selection=hidden_state_selection,
    )

    print(f'[variant {variant}] building composite model (meta-init): '
          f'bridge={cfg.bridge_config.bridge_type}, talker_layers={cfg.talker_config.num_hidden_layers}'
          f' (selection={cfg.talker_config.layer_selection_mode})')

    # ---- Build the composite on the meta device for instant construction ----
    # Without this every variant re-initialises 30B fp32 weights (≈ 5 min each
    # and ~120 GB of throwaway RAM). With ``init_empty_weights`` the whole
    # build is < 1 s and uses essentially zero extra RAM.
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = QwenOmniSpeakForConditionalGeneration(cfg)

    # ---- 1. Inject Thinker weights (zero-copy assign into meta tensors) ----
    print(f'[variant {variant}] copying Thinker weights (assign) ...')
    thinker_state = thinker.state_dict()
    # Cast to target dtype lazily — share storage when source is already bf16.
    thinker_state = {k: v.to(dtype=dtype) if v.dtype != dtype else v
                     for k, v in thinker_state.items()}
    missing_t, unexpected_t = model.thinker.load_state_dict(thinker_state, strict=False,
                                                            assign=True)
    print(f'[variant {variant}]   thinker: missing={len(missing_t)} unexpected={len(unexpected_t)}')

    # ---- 2. Inject Talker weights ----
    aux_state: Dict[str, torch.Tensor] = {}
    if tts_state is not None:
        keep = cfg.talker_config.resolved_layer_indices()
        remapped, aux_state = _remap_tts_to_talker(tts_state, keep)
        remapped = {k: v.to(dtype=dtype) if v.dtype != dtype else v
                    for k, v in remapped.items()}
        missing_tts, unexpected_tts = model.talker.load_state_dict(remapped, strict=False,
                                                                   assign=True)
        print(f'[variant {variant}]   talker: loaded={len(remapped)} '
              f'missing={len(missing_tts)} unexpected={len(unexpected_tts)} aux={len(aux_state)}')
        if missing_tts:
            print(f'[variant {variant}]   first missing keys: {missing_tts[:5]}')
        if unexpected_tts:
            print(f'[variant {variant}]   first unexpected keys: {unexpected_tts[:5]}')
    else:
        print(f'[variant {variant}]   talker: no Qwen3-TTS state provided, randomly initialised.')

    # ---- 3. Materialise any submodule still on meta (= the Bridge) ----
    # Bridge is the only thing left without real weights. We instantiate fresh
    # tensors of the requested dtype so save_pretrained can write them.
    n_meta = 0
    for name, param in model.named_parameters():
        if param.is_meta:
            new_p = torch.nn.Parameter(
                torch.empty(param.shape, dtype=dtype, device='cpu'), requires_grad=param.requires_grad)
            # Standard initialisation matching nn.Linear / nn.Embedding defaults.
            with torch.no_grad():
                if 'weight' in name and new_p.dim() >= 2:
                    torch.nn.init.normal_(new_p, mean=0.0, std=0.02)
                elif 'bias' in name:
                    torch.nn.init.zeros_(new_p)
                else:
                    torch.nn.init.normal_(new_p, mean=0.0, std=0.02)
            # Re-bind onto the model.
            parent = model
            *parts, leaf = name.split('.')
            for p in parts:
                parent = getattr(parent, p)
            setattr(parent, leaf, new_p)
            n_meta += 1
    for name, buf in model.named_buffers():
        if buf.is_meta:
            new_b = torch.zeros(buf.shape, dtype=buf.dtype, device='cpu')
            parent = model
            *parts, leaf = name.split('.')
            for p in parts:
                parent = getattr(parent, p)
            parent.register_buffer(leaf, new_b, persistent=True)
            n_meta += 1
    if n_meta:
        print(f'[variant {variant}]   bridge: materialised {n_meta} meta tensors (random init).')
    else:
        print(f'[variant {variant}]   bridge: no meta tensors left.')

    return model, aux_state


# ---------------------------------------------------------------------------
# Self-contained export-directory writer
# ---------------------------------------------------------------------------


def _copy_processor(out_dir: Path, thinker_model_path: str, thinker_family: str) -> None:
    """Copy/save the selected Thinker processor (tokenizer + image/audio processors)."""
    try:
        if thinker_family == QWEN25_THINKER_FAMILY:
            from transformers import Qwen2_5OmniProcessor
            proc = Qwen2_5OmniProcessor.from_pretrained(thinker_model_path, trust_remote_code=True)
            proc.save_pretrained(str(out_dir))
            print(f'[export] saved Qwen2.5-Omni processor to {out_dir}')
        else:
            from transformers import Qwen3OmniMoeProcessor
            proc = Qwen3OmniMoeProcessor.from_pretrained(thinker_model_path, trust_remote_code=True)
            proc.save_pretrained(str(out_dir))
            print(f'[export] saved Qwen3-Omni processor to {out_dir}')
    except Exception as e:
        print(f'[warn] failed to copy Qwen Omni processor ({e}); falling back to tokenizer only.')
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(thinker_model_path, trust_remote_code=True)
            tok.save_pretrained(str(out_dir))
        except Exception as e2:
            print(f'[warn] tokenizer copy also failed: {e2}')


def export_variant(
    model: QwenOmniSpeakForConditionalGeneration,
    out_dir: Path,
    thinker_model_path: str,
    thinker_family: str,
    aux_state: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. Save weights + config via standard HF save_pretrained.
    print(f'[export] saving model to {out_dir} ...')
    model.save_pretrained(str(out_dir), safe_serialization=True)
    # 2. Save the selected Qwen Omni processor.
    _copy_processor(out_dir, thinker_model_path, thinker_family)
    # 3. Optional: save Qwen3-TTS sidecar state (text_projection / speaker_encoder)
    #    that we don't load into the composite but want to keep around.
    if aux_state:
        from safetensors.torch import save_file
        sidecar = out_dir / 'qwen3_tts_extras.safetensors'
        save_file(aux_state, str(sidecar))
        print(f'[export] saved {len(aux_state)} aux Qwen3-TTS tensors to {sidecar}')
    print(f'[export] done: {out_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--thinker_model_path', required=True,
                        help='HF model id or local path to Qwen3-Omni or Qwen2.5-Omni (Thinker side will be used).')
    parser.add_argument('--thinker_family', default=None,
                        choices=[None, QWEN3_THINKER_FAMILY, QWEN25_THINKER_FAMILY, 'qwen3', 'qwen2_5', 'qwen2.5'],
                        help='Optional explicit Thinker family. By default it is inferred from config.json.')
    parser.add_argument('--tts_model_path', default=None,
                        help='HF model id or local path to Qwen3-TTS-12Hz-1.7B-CustomVoice. '
                        'Optional: when omitted the talker is randomly initialised.')
    parser.add_argument('--output_root', required=True,
                        help='Directory under which one sub-directory per variant is created.')
    parser.add_argument('--variant_prefix', default=None,
                        help='Output sub-directory / model_type prefix. Each variant directory is named '
                        '"{variant_prefix}_{a..j}". Defaults to qwen3_omnispeak for Qwen3-Omni and '
                        'qwen2_5_3b_omnispeak / qwen2_5_7b_omnispeak for Qwen2.5-Omni (inferred from the path).')
    parser.add_argument('--variants', nargs='+', default=ALL_VARIANTS, choices=ALL_VARIANTS,
                        help='Which variants to export.')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--hidden_state_selection', default=None,
                        help='Thinker hidden state forwarded to the bridge. Use an integer index or "last". '
                        'Defaults to 24 for Qwen3-Omni and last for Qwen2.5-Omni.')
    parser.add_argument('--accept_hidden_layer', default=None,
                        help='Deprecated alias of --hidden_state_selection.')
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {'float32': torch.float32, 'float16': torch.float16,
             'bfloat16': torch.bfloat16}[args.dtype]

    # 1. Load Thinker once and reuse across variants.
    full_omni, thinker, thinker_family = _load_thinker(args.thinker_model_path, dtype, args.thinker_family)
    thinker_full_cfg = full_omni.config
    hidden_state_selection = args.hidden_state_selection
    if hidden_state_selection is None:
        hidden_state_selection = args.accept_hidden_layer
    if hidden_state_selection is None:
        hidden_state_selection = default_hidden_state_selection(thinker_family)

    variant_prefix = args.variant_prefix or _auto_variant_prefix(thinker_family, args.thinker_model_path)
    print(f'[export] thinker_family={thinker_family} variant_prefix={variant_prefix!r} '
          f'hidden_state_selection={hidden_state_selection!r}')

    # 2. Load Qwen3-TTS state_dict once and reuse across variants.
    tts_state = _load_tts_state_dict(args.tts_model_path)
    tts_inferred_cfg = _build_qwen3_tts_talker_config(tts_state)
    print(f'[export] inferred Qwen3-TTS talker config: '
          f'hidden_size={tts_inferred_cfg.hidden_size}, '
          f'orig_layers={tts_inferred_cfg.original_num_hidden_layers}, '
          f'codec_vocab={tts_inferred_cfg.codec_vocab_size}')

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        print(f'\n=========== variant {variant} ===========')
        model, aux_state = build_one_variant(
            variant=variant,
            thinker=thinker,
            thinker_full_config=thinker_full_cfg,
            tts_state=tts_state,
            tts_inferred_config=tts_inferred_cfg,
            dtype=dtype,
            thinker_family=thinker_family,
            thinker_model_id=args.thinker_model_path,
            talker_model_id=args.tts_model_path,
            hidden_state_selection=hidden_state_selection,
        )
        model.config.hidden_state_selection = hidden_state_selection
        model.config.accept_hidden_layer = hidden_state_selection
        out_dir = output_root / f'{variant_prefix}_{variant.lower()}'
        export_variant(model, out_dir, args.thinker_model_path, thinker_family, aux_state=aux_state)
        del model
        import gc
        gc.collect()

    print('\n[export] all variants exported.')


if __name__ == '__main__':
    main()
