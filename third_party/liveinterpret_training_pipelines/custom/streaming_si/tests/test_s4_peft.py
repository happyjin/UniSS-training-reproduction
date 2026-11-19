"""S4 PEFT（peft 0.15.2）在真实 StreamingSI 结构上的行为；断言与源 tests/test_s4_peft.py 对齐。

运行：python -m pytest custom/streaming_si/tests/test_s4_peft.py -q（需 peft，用 swift 环境）
"""
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

pytest.importorskip("peft")

from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.integration.s4_peft import (apply_s4_train_scope, attach_s4_peft,  # noqa: E402
                                  expected_s4_lora_modules, load_s4_adapter, save_s4_adapter,
                                  unwrap_s4_model)
from streaming_si.tests.test_streaming_si import tiny_config  # noqa: E402

TOKENS = [500, 501, 502, 503]
LEAVES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _model(seed=0):
    torch.manual_seed(seed)
    return StreamingSIForConditionalGeneration(tiny_config()).eval()


def _trainable(model):
    return {n for n, p in unwrap_s4_model(model).named_parameters() if p.requires_grad}


def _expected_names(n_thinker, n_talker):
    th = {f"thinker_model.model.layers.{i}.{leaf}.lora_{s}.default.weight"
          for i in range(n_thinker) for leaf in LEAVES for s in "AB"}
    th.add("thinker_model.model.embed_tokens.token_adapter.trainable_tokens_delta.default")
    tk = {f"talker.decoder.layers.{i}.{leaf}.lora_{s}.default.weight"
          for i in range(n_talker) for leaf in LEAVES for s in "AB"}
    return th, tk


def test_attach_targets_only_transformer_blocks_and_ties_four_rows():
    pm = attach_s4_peft(_model(), TOKENS)
    raw = unwrap_s4_model(pm)
    lora = sorted(n for n, m in raw.named_modules() if hasattr(m, "lora_A"))
    assert len(lora) == expected_s4_lora_modules(raw) == 7 * (2 + 2)
    assert all(n.startswith(("thinker_model.model.layers.", "talker.decoder.layers.")) for n in lora)
    embed, head = raw.thinker_model.model.embed_tokens, raw.thinker_model.lm_head
    assert embed.token_adapter.token_indices["default"] == TOKENS
    assert head.token_adapter.tied_adapter is embed.token_adapter
    assert raw.lm_head is head and raw.thinker is raw.thinker_model.model
    # talker 的 embed_tokens / talker_text_embed / audio_* 不能被 token adapter 或 LoRA 触及
    names = _trainable(pm)
    assert not any(n.startswith(("audio_proj", "audio_encoder", "talker_text_embed",
                                 "talker.decoder.embed_tokens", "talker.text_proj")) for n in names)
    assert any("thinker_proj.modules_to_save.default" in n for n in names)
    assert any("fusion.modules_to_save.default" in n for n in names)


def test_token_rows_update_both_embedding_and_lm_head():
    pm = attach_s4_peft(_model(), TOKENS)
    raw = unwrap_s4_model(pm)
    delta = raw.thinker_model.model.embed_tokens.token_adapter.trainable_tokens_delta["default"]
    with torch.no_grad():
        delta.copy_(torch.randn_like(delta))
    ids = torch.tensor([TOKENS])
    emb = raw.thinker.embed_tokens(ids)[0]
    assert torch.allclose(emb, delta)
    hidden = torch.randn(1, 3, emb.size(-1))
    logits = raw.lm_head(hidden)
    assert torch.allclose(logits[..., TOKENS], hidden @ delta.T, atol=1e-5)
    # 同一个参数同时接收两条路的梯度
    (raw.thinker.embed_tokens(ids).sum() + raw.lm_head(hidden)[..., TOKENS].sum()).backward()
    assert delta.grad is not None and bool(delta.grad.abs().sum() > 0)


@pytest.mark.parametrize("scope", ["joint", "thinker", "talker"])
def test_scope_selects_exact_groups(scope):
    pm = attach_s4_peft(_model(), TOKENS)
    apply_s4_train_scope(pm, scope)
    th, tk = _expected_names(2, 2)
    tk |= {n for n in _trainable(attach_s4_peft(_model(), TOKENS))
           if "modules_to_save.default" in n}
    expected = {"joint": th | tk, "thinker": th, "talker": tk}[scope]
    assert _trainable(pm) == expected


def test_adapter_round_trip_restores_outputs():
    trained = attach_s4_peft(_model(0), TOKENS)
    with torch.no_grad():
        for i, (_, p) in enumerate((x for x in trained.named_parameters() if x[1].requires_grad)):
            p.add_(0.01 * (i + 1))
    x = torch.tensor([[1, 2, TOKENS[0], 7, TOKENS[3]]])
    ref_logits = unwrap_s4_model(trained).thinker_model(x).logits
    with tempfile.TemporaryDirectory() as d:
        save_s4_adapter(trained, d, {"stream_token_ids": TOKENS})
        files = set(os.listdir(d))
        assert {"adapter_model.safetensors", "adapter_config.json", "s4_metadata.json"} <= files
        loaded = load_s4_adapter(_model(0), d, TOKENS, is_trainable=True)
        with pytest.raises(ValueError, match="不一致"):
            load_s4_adapter(_model(0), d, [TOKENS[1], TOKENS[0]] + TOKENS[2:])
    got = unwrap_s4_model(loaded).thinker_model(x).logits
    assert torch.allclose(got, ref_logits, atol=1e-6)
    assert _trainable(loaded) == _trainable(trained)


def test_untied_config_is_rejected_instead_of_silent():
    m = StreamingSIForConditionalGeneration(tiny_config())
    m.config.tie_word_embeddings = False
    with pytest.raises(RuntimeError, match="未与 lm_head 绑定"):
        attach_s4_peft(m, TOKENS)
