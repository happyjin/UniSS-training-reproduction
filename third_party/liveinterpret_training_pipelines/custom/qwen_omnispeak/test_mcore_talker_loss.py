#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU parity tests for the Megatron talker codec loss.

These tests do NOT require Megatron / transformer_engine / GPUs. They verify
that ``mcore_talker_loss.compute_codec_loss`` reproduces, bit-for-bit (up to
fp tolerance), the codec loss computed by the reference DeepSpeed model
``QwenOmniSpeakForConditionalGeneration.forward`` -- guaranteeing the Megatron
path (variant 3) optimises the SAME objective as variants 1/2.

It also unit-tests the ``_combined_loss_func`` loss-merging math.

Run::

    python liveinterpret_training_pipelines/custom/qwen_omnispeak/test_mcore_talker_loss.py
"""
from __future__ import annotations

import os
import sys
import types

import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_DIR = os.path.dirname(_CUR_DIR)
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)
if _CUSTOM_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_DIR)

from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration  # noqa: E402
from configuration_qwen_omnispeak import QwenOmniSpeakConfig  # noqa: E402
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeThinkerConfig)

import mcore_talker_loss  # noqa: E402


def _make_tiny_config(variant: str) -> QwenOmniSpeakConfig:
    cfg = Qwen3OmniMoeThinkerConfig()
    cfg.text_config.hidden_size = 64
    cfg.text_config.intermediate_size = 128
    cfg.text_config.num_hidden_layers = 2
    cfg.text_config.num_attention_heads = 4
    cfg.text_config.num_key_value_heads = 2
    cfg.text_config.head_dim = 16
    cfg.text_config.vocab_size = 256
    cfg.text_config.num_local_experts = 2
    cfg.text_config.num_experts_per_tok = 1
    cfg.text_config.shared_expert_intermediate_size = 64
    cfg.text_config.moe_intermediate_size = 32
    cfg.text_config.rope_scaling = {'rope_type': 'default'}
    cfg.audio_config.encoder_layers = 1
    cfg.audio_config.num_hidden_layers = 1
    cfg.audio_config.d_model = 64
    cfg.audio_config.encoder_attention_heads = 4
    cfg.audio_config.encoder_ffn_dim = 128
    cfg.audio_config.output_dim = 64
    cfg.vision_config.depth = 1
    cfg.vision_config.num_hidden_layers = 1
    cfg.vision_config.hidden_size = 64
    cfg.vision_config.intermediate_size = 128
    cfg.vision_config.out_hidden_size = 64
    cfg.vision_config.num_heads = 4
    talker_overrides = {
        'hidden_size': 64,
        'intermediate_size': 128,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'head_dim': 16,
        'vocab_size': 256,
        'codec_vocab_size': 64,
        'code_predictor_hidden_size': 32,
        'code_predictor_num_layers': 1,
        'code_predictor_vocab_size': 32,
        'num_code_groups': 4,
        'original_num_hidden_layers': 28,
        'max_position_embeddings': 256,
    }
    bridge_overrides = {
        'thinker_hidden_size': 64,
        'talker_hidden_size': 64,
        'intermediate_size': 128,
        'num_query_tokens': 8,
        'qformer_num_layers': 1,
        'qformer_num_heads': 4,
    }
    return QwenOmniSpeakConfig.from_variant(
        variant=variant,
        thinker_config=cfg.to_dict(),
        talker_overrides=talker_overrides,
        bridge_overrides=bridge_overrides,
        accept_hidden_layer=1,
        codec_eos_id=63,
    )


def test_hidden_layer_resolution():
    print('=== test_hidden_layer_resolution ===')
    # (layer index, apply final norm). ``last``/``final`` must additionally
    # apply the decoder's final_layernorm (HF ``hidden_states[-1]`` is the
    # post-norm output; P0-2).
    cfg = types.SimpleNamespace(hidden_state_selection='last')
    assert mcore_talker_loss._resolve_hidden_layer_index(cfg, 4) == (3, True)
    cfg = types.SimpleNamespace(hidden_state_selection=24)
    assert mcore_talker_loss._resolve_hidden_layer_index(cfg, 48) == (23, False)
    cfg = types.SimpleNamespace(accept_hidden_layer=1)
    assert mcore_talker_loss._resolve_hidden_layer_index(cfg, 2) == (0, False)
    print('  OK')


def _fake_mm(model):
    """A minimal stand-in for MultimodalGPTModel exposing what
    ``compute_codec_loss`` needs: visual.{bridge,talker,thinker_adapter,hf_config}
    and language_model.embedding.word_embeddings.
    """
    visual = types.SimpleNamespace(
        bridge=model.bridge,
        talker=model.talker,
        thinker_adapter=getattr(model, 'thinker_adapter', None),
        hf_config=model.config,
    )
    embedding = types.SimpleNamespace(word_embeddings=model.thinker.get_input_embeddings())
    language_model = types.SimpleNamespace(embedding=embedding)
    return types.SimpleNamespace(visual=visual, language_model=language_model)


def _run_parity(variant: str, has_codec_supervision, tol: float = 1e-4) -> float:
    torch.manual_seed(0)
    model = QwenOmniSpeakForConditionalGeneration(_make_tiny_config(variant))
    model.eval()
    cfg = model.config
    B, T_text, T_codec = 2, 5, 6
    H = cfg.bridge_config.thinker_hidden_size

    input_ids = torch.randint(0, cfg.thinker_config.text_config.vocab_size, (B, T_text))
    text_embeds = model.thinker.get_input_embeddings()(input_ids)
    hidden_states = torch.randn(B, T_text, H)
    codec_ids = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))
    codec_labels = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))
    codec_labels[:, -1] = cfg.codec_eos_id
    residual_labels = torch.randint(
        0, cfg.talker_config.code_predictor_vocab_size, (B, T_codec, cfg.talker_config.num_code_groups))

    ref_kwargs = dict(
        thinker_text_embeds=text_embeds,
        thinker_hidden_states=hidden_states,
        codec_input_ids=codec_ids,
        codec_labels=codec_labels,
        codec_residual_labels=residual_labels,
    )
    if has_codec_supervision is not None:
        ref_kwargs['has_codec_supervision'] = has_codec_supervision

    with torch.no_grad():
        ref_out = model(**ref_kwargs)
    ref_loss = ref_out.loss

    mm = _fake_mm(model)
    codec = {
        'codec_input_ids': codec_ids,
        'codec_labels': codec_labels,
        'codec_residual_labels': residual_labels,
    }
    # NOTE (P2-1): the capture hook now normalises Megatron's [S, B, H] layer
    # output to [B, S, H], so ``compute_codec_loss`` expects [B, S, H].
    with torch.no_grad():
        mine, num_tokens, metrics = mcore_talker_loss.compute_codec_loss(
            mm,
            hf_config=cfg,
            input_ids=input_ids,
            captured_hidden=hidden_states,
            codec=codec,
            thinker_attention_mask=None,
            has_codec_supervision=has_codec_supervision,
        )

    assert mine is not None, f'variant {variant}: codec loss is None'
    err = (mine - ref_loss).abs().max().item()
    assert err < tol, f'variant {variant} ({"gated" if has_codec_supervision is not None else "full"}): |Δ|={err:.3e}'
    assert int(num_tokens) > 0
    assert 'codec_loss' in metrics
    assert 'eos_boundary_top1_acc' in metrics
    assert 'premature_eos_top1_rate' in metrics
    return err


def test_parity_all_variants():
    print('=== test_parity_all_variants ===')
    for variant in QwenOmniSpeakConfig.SUPPORTED_VARIANTS:
        e_full = _run_parity(variant, has_codec_supervision=None)
        e_gated = _run_parity(variant, has_codec_supervision=torch.tensor([True, False]))
        print(f'  variant {variant}: |Δ|_full={e_full:.2e}, |Δ|_gated={e_gated:.2e}  OK')


def _run_streaming_parity(variant: str, tol: float = 1e-4) -> float:
    """Parity for the simultaneous-policy read-write schedule: the Megatron codec
    loss must apply the SAME ``codec_to_sem_visible`` streaming mask as the
    reference forward, and the result must differ from the offline (full-context)
    loss (proving the mask is actually wired through).
    """
    torch.manual_seed(0)
    model = QwenOmniSpeakForConditionalGeneration(_make_tiny_config(variant))
    model.eval()
    cfg = model.config
    B, T_text, T_codec = 2, 5, 6
    H = cfg.bridge_config.thinker_hidden_size

    input_ids = torch.randint(0, cfg.thinker_config.text_config.vocab_size, (B, T_text))
    text_embeds = model.thinker.get_input_embeddings()(input_ids)
    hidden_states = torch.randn(B, T_text, H)
    codec_ids = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))
    codec_labels = torch.randint(0, cfg.talker_config.codec_vocab_size, (B, T_codec))
    residual_labels = torch.randint(
        0, cfg.talker_config.code_predictor_vocab_size, (B, T_codec, cfg.talker_config.num_code_groups))

    # Monotonic non-decreasing per-frame visibility over the (post-bridge) sem
    # block. Derive the true sem length from the bridge output so the schedule's
    # early frames are guaranteed to be < sem_len (i.e. actually gated).
    with torch.no_grad():
        sem_block = model.bridge(
            thinker_text_embeds=text_embeds,
            thinker_hidden_states=hidden_states,
            semantic_route_mask=None,
            thinker_hidden_attention_mask=None,
        )
    sem_len = int(sem_block.size(1))
    assert sem_len > 1, 'need sem_len > 1 to exercise the streaming mask'
    base = torch.linspace(1, sem_len, T_codec).round().long()
    codec_to_sem_visible = base.unsqueeze(0).expand(B, T_codec).contiguous()

    ref_kwargs = dict(
        thinker_text_embeds=text_embeds,
        thinker_hidden_states=hidden_states,
        codec_input_ids=codec_ids,
        codec_labels=codec_labels,
        codec_residual_labels=residual_labels,
        codec_to_sem_visible=codec_to_sem_visible,
    )
    with torch.no_grad():
        ref_loss = model(**ref_kwargs).loss
        # Offline reference (no schedule) to confirm the mask actually changes the loss.
        offline_loss = model(**{k: v for k, v in ref_kwargs.items() if k != 'codec_to_sem_visible'}).loss

    mm = _fake_mm(model)
    codec = {
        'codec_input_ids': codec_ids,
        'codec_labels': codec_labels,
        'codec_residual_labels': residual_labels,
    }
    with torch.no_grad():
        mine, num_tokens, _ = mcore_talker_loss.compute_codec_loss(
            mm,
            hf_config=cfg,
            input_ids=input_ids,
            captured_hidden=hidden_states,
            codec=codec,
            thinker_attention_mask=None,
            has_codec_supervision=None,
            codec_to_sem_visible=codec_to_sem_visible,
        )

    assert mine is not None, f'variant {variant}: streaming codec loss is None'
    err = (mine - ref_loss).abs().max().item()
    assert err < tol, f'variant {variant} (streaming): |Δ|={err:.3e}'
    assert int(num_tokens) > 0
    # The streaming mask must measurably change the loss vs offline full-context.
    assert (ref_loss - offline_loss).abs().item() > 1e-6, (
        f'variant {variant}: streaming schedule did not change the loss -- mask not applied')
    return err


def test_streaming_parity_all_variants():
    print('=== test_streaming_parity_all_variants ===')
    for variant in QwenOmniSpeakConfig.SUPPORTED_VARIANTS:
        e = _run_streaming_parity(variant)
        print(f'  variant {variant}: |Δ|_streaming={e:.2e}  OK')


def test_combined_loss_math():
    print('=== test_combined_loss_math ===')
    trainer = types.SimpleNamespace(args=types.SimpleNamespace(enable_dft_loss=False))

    B, S = 2, 4
    output_tensor = torch.rand(B, S)
    labels = torch.randint(0, 10, (B, S))
    labels[0, 0] = -100
    codec_loss = torch.tensor(0.75)
    codec_tokens = torch.tensor(5)
    loss_mask = labels != -100

    # continue-pretrain mode (P1-10): per-micro-batch MEAN with equal
    # weighting — lm_loss == text_mean + codec_loss and num == 1.
    lm_loss, num, metrics = mcore_talker_loss._combined_loss_func(
        trainer, output_tensor, labels=labels, text_ce_labels=None, loss_scale=None,
        channels=None, packed_seq_params=None, codec_loss=codec_loss,
        codec_tokens=codec_tokens, codec_metrics={}, bridge_mode=False)
    expected = (output_tensor.float() * loss_mask).sum() / loss_mask.sum() + codec_loss
    assert torch.allclose(lm_loss, expected, atol=1e-6), (lm_loss, expected)
    assert int(num) == 1

    # bridge-alignment mode, thinker frozen: codec-only mean, unit count.
    lm_loss_b, num_b, _ = mcore_talker_loss._combined_loss_func(
        trainer, output_tensor, labels=labels, text_ce_labels=None, loss_scale=None,
        channels=None, packed_seq_params=None, codec_loss=codec_loss,
        codec_tokens=codec_tokens, codec_metrics={}, bridge_mode=True)
    assert torch.allclose(lm_loss_b, codec_loss, atol=1e-6)
    assert int(num_b) == 1

    # bridge-alignment mode, thinker unfrozen (P1-5): text_mean + codec_mean.
    ce = torch.rand(B, S)
    lm_loss_t, num_t, _ = mcore_talker_loss._combined_loss_func(
        trainer, ce, labels=labels, text_ce_labels=labels, loss_scale=None,
        channels=None, packed_seq_params=None, codec_loss=codec_loss,
        codec_tokens=codec_tokens, codec_metrics={}, bridge_mode=True)
    expected_t = (ce.float() * loss_mask).sum() / loss_mask.sum() + codec_loss
    assert torch.allclose(lm_loss_t, expected_t, atol=1e-6), (lm_loss_t, expected_t)
    assert int(num_t) == 1
    print('  OK')


# ===========================================================================
#  L1/L2 adapter (upstream reply-add) parity — CPU, tiny config.
#
#  The tests above only exercise the legacy BRIDGE path; these close the gap
#  flagged in docs/CUSTOM_BACKEND_PARITY_REVIEW_SPEC.md §3.1/§3.3 by running
#  the UPSTREAM protocol (thinker_adapter + reply-add layout) through BOTH the
#  DeepSpeed forward and ``compute_codec_loss`` on identical deterministically
#  seeded weights and comparing, stage by stage:
#
#    L1: adapter_out → full_emb → codec_hidden → main_codec_logits → loss
#        (+ speaker-slot substitution parity via a stubbed ECAPA embedder)
#    L2: adapter gradients (cosine / max|Δ|), frozen-module grad-is-None, and
#        the parameter delta of one AdamW step
#
#  Scope note: both ends share the SAME modules here, so this verifies the
#  Megatron-side codec-loss construction itself. The thinker hidden CAPTURE
#  semantics (P0-2 final-norm, real mcore decoder) is not covered on CPU —
#  that requires the node-level probe (review spec §3.2).
# ===========================================================================

def _make_tiny_upstream_config(variant: str) -> QwenOmniSpeakConfig:
    cfg = _make_tiny_config(variant)
    cfg.use_upstream_talker_input = True
    cfg.full_add_layout = False
    cfg.use_speaker_prompt = False
    cfg.adapter_dropout = 0.0  # determinism: no dropout in the parity graph
    # The tiny talker text vocab is 256 while the production tts_pad/bos ids
    # exceed it; remap them into range for the pad/bos embedding lookups.
    cfg.talker_config.tts_pad_token_id = 200
    cfg.talker_config.tts_bos_token_id = 201
    # text_projection maps text_hidden_size -> talker hidden; default 2048
    # does not match the tiny 64-dim talker.
    cfg.talker_config.text_hidden_size = 64
    return cfg


def _build_reply_add_batch(cfg, with_speaker: bool = False, B: int = 2,
                           T_text: int = 6, T_main: int = 8):
    """Deterministic reply-add batch mirroring the template collator output:
    talker control prefix (think/lang/eos [+ speaker slot] pad bos), main
    codes, residual labels, thinker reply labels, codec_bos_index and — for
    the speaker test — the slot index.
    """
    g = torch.Generator().manual_seed(1234)
    V = cfg.talker_config.codec_vocab_size
    RV = cfg.talker_config.code_predictor_vocab_size
    G = cfg.talker_config.num_code_groups
    if with_speaker:
        prefix = [5, 6, 7, 8, 9, 10, 11]   # ..., speaker-slot placeholder, pad, bos
        speaker_slot_index = torch.full((B,), 4, dtype=torch.long)
    else:
        prefix = [5, 6, 7, 8, 9, 10]       # think, think_bos, lang, think_eos, pad, bos
        speaker_slot_index = None
    k = len(prefix)
    main = torch.randint(3, V, (B, T_main), generator=g)
    codec_input_ids = torch.cat([torch.tensor(prefix).expand(B, k), main], dim=1)
    codec_labels = torch.cat([
        torch.full((B, k - 1), -100), main,
        torch.full((B, 1), cfg.codec_eos_id)], dim=1)
    codec_residual_labels = torch.randint(0, RV, (B, k + T_main, G), generator=g)
    codec_attention_mask = torch.ones(B, k + T_main, dtype=torch.long)
    thinker_text_labels = torch.full((B, T_text), -100, dtype=torch.long)
    thinker_text_labels[:, 2:] = torch.randint(3, 40, (B, T_text - 2), generator=g)
    codec_bos_index = torch.full((B,), k - 1, dtype=torch.long)
    return dict(
        codec_input_ids=codec_input_ids, codec_labels=codec_labels,
        codec_residual_labels=codec_residual_labels,
        codec_attention_mask=codec_attention_mask,
        thinker_text_labels=thinker_text_labels,
        codec_bos_index=codec_bos_index,
        speaker_slot_index=speaker_slot_index,
    )


def _thinker_side_inputs(model, batch, seed: int = 7):
    """Pre-computed thinker outputs (hidden + token embeds): keeps the frozen
    thinker out of the graph so both ends start from identical tensors."""
    cfg = model.config
    B, T_text = batch['thinker_text_labels'].shape
    H = cfg.bridge_config.thinker_hidden_size
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, cfg.thinker_config.text_config.vocab_size,
                              (B, T_text), generator=g)
    text_embeds = model.thinker.get_input_embeddings()(input_ids)
    hidden_states = torch.randn(B, T_text, H, generator=g)
    return input_ids, text_embeds, hidden_states


class _StageCapture:
    """Records stage tensors via hooks on the SHARED modules. The two
    backends run the same objects (tiny parity), so clear() between runs."""

    def __init__(self, model):
        self.store = {}
        self._handles = []
        talker_model = model.talker.model
        # ``talker.model(inputs_embeds=..., attention_mask=...)`` is called
        # with keyword args only, so the pre-hook must inspect kwargs.
        self._handles.append(talker_model.register_forward_pre_hook(
            self._pre('full_emb'), with_kwargs=True))
        self._handles.append(talker_model.register_forward_hook(
            self._post('codec_hidden', attr='last_hidden_state')))
        self._handles.append(model.talker.codec_head.register_forward_hook(
            self._post('main_codec_logits')))
        if model.thinker_adapter is not None:
            self._handles.append(model.thinker_adapter.register_forward_hook(
                self._post('adapter_out')))

    def _pre(self, name):
        def fn(_module, args, kwargs):
            t = kwargs.get('inputs_embeds')
            if t is None and args:
                t = args[0]
            self.store[name] = t.detach().clone()
        return fn

    def _post(self, name, attr=None):
        def fn(_module, _args, out):
            t = getattr(out, attr, None) if attr else out
            if t is None and isinstance(out, tuple):
                t = out[0]
            self.store[name] = t.detach().clone()
        return fn

    def clear(self):
        self.store.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


_CODEC_KEYS = ('codec_input_ids', 'codec_labels', 'codec_residual_labels',
               'codec_attention_mask')


def test_upstream_parity_reply_add():
    print('=== test_upstream_parity_reply_add (L1) ===')
    torch.manual_seed(0)
    model = QwenOmniSpeakForConditionalGeneration(_make_tiny_upstream_config('A'))
    model.eval()
    cfg = model.config
    batch = _build_reply_add_batch(cfg)
    input_ids, text_embeds, hidden_states = _thinker_side_inputs(model, batch)

    cap = _StageCapture(model)
    with torch.no_grad():
        ref = model(
            thinker_text_embeds=text_embeds, thinker_hidden_states=hidden_states,
            thinker_text_labels=batch['thinker_text_labels'],
            codec_bos_index=batch['codec_bos_index'],
            codec_input_ids=batch['codec_input_ids'],
            codec_attention_mask=batch['codec_attention_mask'],
            codec_labels=batch['codec_labels'],
            codec_residual_labels=batch['codec_residual_labels'],
        )
    ref_stages = dict(cap.store)
    ref_stages['loss'] = ref.loss.detach().clone()

    cap.clear()
    mm = _fake_mm(model)
    with torch.no_grad():
        mine, num_tokens, _ = mcore_talker_loss.compute_codec_loss(
            mm, hf_config=cfg, input_ids=input_ids,
            captured_hidden=hidden_states,
            codec={k: batch[k] for k in _CODEC_KEYS},
            thinker_attention_mask=None, has_codec_supervision=None,
            use_upstream_input=True,
            text_labels=batch['thinker_text_labels'],
            codec_bos_index=batch['codec_bos_index'])
    mine_stages = dict(cap.store)
    mine_stages['loss'] = mine.detach().clone()
    cap.remove()

    assert int(num_tokens) == int((batch['codec_labels'] != -100).sum())
    for name in ('adapter_out', 'full_emb', 'codec_hidden', 'main_codec_logits', 'loss'):
        assert name in ref_stages and name in mine_stages, f'missing stage {name}'
        d = (ref_stages[name].float() - mine_stages[name].float()).abs().max().item()
        assert d < 1e-5, f'{name}: |Δ|={d:.3e}'
        print(f'  {name}: shape={tuple(ref_stages[name].shape)} |Δ|={d:.2e}  OK')


def test_upstream_parity_speaker():
    print('=== test_upstream_parity_speaker (L1) ===')
    import modeling_qwen_omnispeak as _M

    talker_h = _make_tiny_config('A').talker_config.hidden_size  # codec_emb width

    class _StubEmbedder:
        """Deterministic stand-in for the frozen ECAPA encoder (the real one
        needs node-side weights). Distinct per waveform so per-sample
        substitution is actually exercised."""

        def extract(self, wav, sr, device=None, dtype=None):
            seed = float(torch.as_tensor(wav, dtype=torch.float32).abs().sum())
            return torch.arange(talker_h, dtype=torch.float32) * (1e-3 + (seed % 1e-3))

    orig_embedder = _M.get_speaker_embedder
    _M.get_speaker_embedder = lambda enc_dim=2048: _StubEmbedder()
    prev_env = os.environ.get('QWEN_OMNISPEAK_USE_SPEAKER_PROMPT')
    os.environ['QWEN_OMNISPEAK_USE_SPEAKER_PROMPT'] = 'true'
    try:
        torch.manual_seed(0)
        model = QwenOmniSpeakForConditionalGeneration(_make_tiny_upstream_config('A'))
        model.eval()
        cfg = model.config
        batch = _build_reply_add_batch(cfg, with_speaker=True)
        input_ids, text_embeds, hidden_states = _thinker_side_inputs(model, batch)
        ref_wavs = [torch.randn(100, generator=torch.Generator().manual_seed(11)),
                    torch.randn(120, generator=torch.Generator().manual_seed(12))]
        ref_srs = [24000, 24000]

        cap = _StageCapture(model)
        with torch.no_grad():
            ref = model(
                thinker_text_embeds=text_embeds, thinker_hidden_states=hidden_states,
                thinker_text_labels=batch['thinker_text_labels'],
                codec_bos_index=batch['codec_bos_index'],
                codec_input_ids=batch['codec_input_ids'],
                codec_attention_mask=batch['codec_attention_mask'],
                codec_labels=batch['codec_labels'],
                codec_residual_labels=batch['codec_residual_labels'],
                reference_audio_wav=ref_wavs, reference_audio_sr=ref_srs,
                speaker_slot_index=batch['speaker_slot_index'])
        ref_stages = dict(cap.store)

        cap.clear()
        mm = _fake_mm(model)
        with torch.no_grad():
            mine, _, _ = mcore_talker_loss.compute_codec_loss(
                mm, hf_config=cfg, input_ids=input_ids,
                captured_hidden=hidden_states,
                codec={k: batch[k] for k in _CODEC_KEYS},
                thinker_attention_mask=None, has_codec_supervision=None,
                use_upstream_input=True,
                text_labels=batch['thinker_text_labels'],
                codec_bos_index=batch['codec_bos_index'],
                reference_audio_wav=ref_wavs, reference_audio_sr=ref_srs,
                speaker_slot_index=batch['speaker_slot_index'])
        mine_stages = dict(cap.store)
        cap.remove()

        for name in ('full_emb', 'codec_hidden', 'main_codec_logits'):
            d = (ref_stages[name].float() - mine_stages[name].float()).abs().max().item()
            assert d < 1e-5, f'{name}: |Δ|={d:.3e}'
            print(f'  {name}: |Δ|={d:.2e}  OK')

        # Fail-fast regression (P0-1): speaker config on + fields missing must
        # raise instead of silently training on the codec_pad placeholder.
        raised = False
        try:
            with torch.no_grad():
                mcore_talker_loss.compute_codec_loss(
                    mm, hf_config=cfg, input_ids=input_ids,
                    captured_hidden=hidden_states,
                    codec={k: batch[k] for k in _CODEC_KEYS},
                    thinker_attention_mask=None, has_codec_supervision=None,
                    use_upstream_input=True,
                    text_labels=batch['thinker_text_labels'],
                    codec_bos_index=batch['codec_bos_index'])
        except RuntimeError:
            raised = True
        assert raised, 'missing speaker fields must raise under use_speaker_prompt'
        print('  speaker fail-fast on missing fields  OK')
    finally:
        _M.get_speaker_embedder = orig_embedder
        if prev_env is None:
            os.environ.pop('QWEN_OMNISPEAK_USE_SPEAKER_PROMPT', None)
        else:
            os.environ['QWEN_OMNISPEAK_USE_SPEAKER_PROMPT'] = prev_env


def test_upstream_backward_and_step_parity():
    print('=== test_upstream_backward_and_step_parity (L2) ===')
    import copy

    torch.manual_seed(0)
    base = QwenOmniSpeakForConditionalGeneration(_make_tiny_upstream_config('A'))
    base.eval()
    # Production freeze semantics: everything frozen except the adapter, so
    # the parity target is exactly the trained parameter set.
    for p in base.parameters():
        p.requires_grad_(False)
    for p in base.thinker_adapter.parameters():
        p.requires_grad_(True)

    batch = _build_reply_add_batch(base.config)
    input_ids, text_embeds, hidden_states = _thinker_side_inputs(base, batch)

    def _collect(m):
        return {k: p.detach().clone()
                for k, p in m.thinker_adapter.named_parameters()}

    def _grads(m):
        return {k: p.grad.detach().clone()
                for k, p in m.thinker_adapter.named_parameters()}

    # DeepSpeed/HF side.
    m_ref = copy.deepcopy(base)
    ref_loss = m_ref(
        thinker_text_embeds=text_embeds, thinker_hidden_states=hidden_states,
        thinker_text_labels=batch['thinker_text_labels'],
        codec_bos_index=batch['codec_bos_index'],
        codec_input_ids=batch['codec_input_ids'],
        codec_attention_mask=batch['codec_attention_mask'],
        codec_labels=batch['codec_labels'],
        codec_residual_labels=batch['codec_residual_labels']).loss
    ref_loss.backward()
    ref_grads = _grads(m_ref)
    torch.optim.AdamW(m_ref.thinker_adapter.parameters(), lr=1e-3).step()
    ref_params = _collect(m_ref)

    # Megatron side (same weights via deepcopy, same batch).
    m_mc = copy.deepcopy(base)
    mm = _fake_mm(m_mc)
    mc_loss, _, _ = mcore_talker_loss.compute_codec_loss(
        mm, hf_config=m_mc.config, input_ids=input_ids,
        captured_hidden=hidden_states,
        codec={k: batch[k] for k in _CODEC_KEYS},
        thinker_attention_mask=None, has_codec_supervision=None,
        use_upstream_input=True,
        text_labels=batch['thinker_text_labels'],
        codec_bos_index=batch['codec_bos_index'])
    mc_loss.backward()
    mc_grads = _grads(m_mc)
    torch.optim.AdamW(m_mc.thinker_adapter.parameters(), lr=1e-3).step()
    mc_params = _collect(m_mc)

    d_loss = abs(ref_loss.item() - mc_loss.item())
    assert d_loss < 1e-5, f'loss |Δ|={d_loss:.3e}'
    print(f'  loss: ref={ref_loss.item():.6f} mc={mc_loss.item():.6f} |Δ|={d_loss:.2e}  OK')

    # Frozen modules must stay out of the graph entirely.
    assert all(p.grad is None for p in m_ref.thinker.parameters())
    assert all(p.grad is None for p in m_mc.thinker.parameters())
    print('  frozen thinker grad-is-None on both ends  OK')

    for k in ref_grads:
        a, b = ref_grads[k].float(), mc_grads[k].float()
        assert a.shape == b.shape, k
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        d = (a - b).abs().max().item()
        assert cos > 0.999, f'grad {k}: cosine={cos:.6f}'
        assert d < 1e-5, f'grad {k}: |Δ|={d:.3e}'
    print(f'  adapter grads ({len(ref_grads)} tensors): cosine>0.999, |Δ|<1e-5  OK')

    for k in ref_params:
        d = (ref_params[k] - mc_params[k]).abs().max().item()
        assert d < 1e-5, f'one-step delta {k}: |Δ|={d:.3e}'
    print('  one-step AdamW parameter deltas aligned  OK')


def test_guards():
    print('=== test_guards ===')
    bad = types.SimpleNamespace(pipeline_model_parallel_size=2, context_parallel_size=1,
                                sequence_parallel=False, packing=False, padding_free=False)
    raised = False
    try:
        mcore_talker_loss._check_parallel_guards(bad)
    except ValueError:
        raised = True
    assert raised, 'PP>1 should raise'

    ok = types.SimpleNamespace(pipeline_model_parallel_size=1, context_parallel_size=1,
                               sequence_parallel=False, packing=False, padding_free=False)
    mcore_talker_loss._check_parallel_guards(ok)  # must not raise
    print('  OK')


def test_enable_flag():
    print('=== test_enable_flag ===')
    prev = os.environ.get('QWEN_OMNISPEAK_MCORE_CODEC_LOSS')
    try:
        os.environ.pop('QWEN_OMNISPEAK_MCORE_CODEC_LOSS', None)
        assert mcore_talker_loss.codec_loss_enabled() is True
        os.environ['QWEN_OMNISPEAK_MCORE_CODEC_LOSS'] = 'false'
        assert mcore_talker_loss.codec_loss_enabled() is False
        os.environ['QWEN_OMNISPEAK_MCORE_CODEC_LOSS'] = '1'
        assert mcore_talker_loss.codec_loss_enabled() is True
    finally:
        if prev is None:
            os.environ.pop('QWEN_OMNISPEAK_MCORE_CODEC_LOSS', None)
        else:
            os.environ['QWEN_OMNISPEAK_MCORE_CODEC_LOSS'] = prev
    print('  OK')


if __name__ == '__main__':
    test_enable_flag()
    test_hidden_layer_resolution()
    test_guards()
    test_combined_loss_math()
    test_parity_all_variants()
    test_streaming_parity_all_variants()
    test_upstream_parity_reply_add()
    test_upstream_parity_speaker()
    test_upstream_backward_and_step_parity()
    print('\nAll Megatron talker codec-loss parity tests passed.')
