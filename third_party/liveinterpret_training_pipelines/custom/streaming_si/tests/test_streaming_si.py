"""StreamingSI 包装的 CPU 单测（tiny 随机权重，不依赖任何 checkpoint）。

    python -m pytest custom/streaming_si/tests/test_streaming_si.py -q

核心断言：swift 约定（未 shift、forward 内算 loss）与源项目 `train_s3_s2st.py`
（dataset 预 shift、trainer 外算 loss）逐位等价。
"""
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.modeling_streaming_si import (  # noqa: E402
    StreamingSIConfig, StreamingSIForConditionalGeneration)
from streaming_si.model.model_si import SI  # noqa: E402

AUDIO_ID = 500
TINY_VOXTRAL_AUDIO = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                          num_attention_heads=2, head_dim=16, num_mel_bins=128, sliding_window=750)
TINY_VOXTRAL_PROJ = dict(audio_hidden_size=32, text_hidden_size=48, downsample_factor=4)


def tiny_config(**over):
    kw = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              head_dim=16, intermediate_size=128, vocab_size=512, audio_ids=[AUDIO_ID],
              talker_hidden_size=64, talker_num_hidden_layers=2, talker_num_attention_heads=4,
              talker_num_key_value_heads=2, talker_head_dim=16, talker_intermediate_size=128,
              speech_token_size=100, voxtral_audio_config=TINY_VOXTRAL_AUDIO,
              voxtral_projector_config=TINY_VOXTRAL_PROJ)
    kw.update(over)
    return StreamingSIConfig(**kw)


def s3_batch(cfg, n_frames=(3, 2), gen_seed=0):
    """构造 S3 风格样本（未 shift 的完整序列）：audio_pad×n + 指令 + 译文(+eos)。"""
    g = torch.Generator().manual_seed(gen_seed)
    ids, labels, am, speech = [], [], [], []
    for n in n_frames:
        instr = torch.randint(10, 400, (4,), generator=g).tolist()
        tgt = torch.randint(10, 400, (5,), generator=g).tolist() + [cfg.eos_token_id % cfg.vocab_size]
        ids.append([AUDIO_ID] * n + instr + tgt)
        labels.append([-100] * (n + len(instr)) + tgt)
        am.append([0] * (n + len(instr)) + [1] * len(tgt))
        speech.append(torch.randint(0, cfg.speech_token_size, (7,), generator=g))

    def pad(seqs, v):
        L = max(len(s) for s in seqs)
        return torch.tensor([s + [v] * (L - len(s)) for s in seqs])

    mel_T = max(n_frames) * 8
    mel = torch.randn(len(n_frames), 128, mel_T, generator=g)
    for i, n in enumerate(n_frames):
        mel[i, :, n * 8:] = 0
    return dict(input_ids=pad(ids, 0), labels=pad(labels, -100), answer_mask=pad(am, 0),
                attention_mask=pad([[1] * len(s) for s in ids], 0),
                speech_tokens=torch.stack(speech), speech_token_lens=torch.tensor([7] * len(n_frames)),
                audio_inputs=mel, audio_lens=torch.tensor([n * 8 for n in n_frames]))


def reference_s3_loss(model, batch):
    """逐字复现源 train_s3_s2st.py:170-194 的计算（dataset 预 shift + 外部 CE）。"""
    x = batch['input_ids'][:, :-1]
    y = batch['labels'][:, 1:]
    am = batch['answer_mask'][:, :-1]
    att = batch['attention_mask'][:, :-1]
    res = SI.forward(model, x, attention_mask=att, audio_inputs=batch['audio_inputs'],
                     audio_lens=batch['audio_lens'], answer_mask=am,
                     speech_tokens=batch['speech_tokens'],
                     speech_token_lens=batch['speech_token_lens'])
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), y.reshape(-1).clamp_min(0))
    m = (y.reshape(-1) != -100).float()
    text_loss = (raw * m).sum() / (m.sum() + 1e-9)
    sl = res.speech_logits.view(-1, res.speech_logits.size(-1))
    st = res.speech_target.reshape(-1)
    valid = st != -100
    speech_loss = loss_fct(sl[valid], st[valid]).mean()
    return text_loss, speech_loss


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = StreamingSIForConditionalGeneration(tiny_config()).eval()
    return m


def test_structure_and_frozen_audio(model):
    keys = {k.split('.')[0] for k in model.state_dict()}
    assert keys == {'audio_encoder', 'audio_proj', 'thinker_model', 'talker', 'talker_text_embed'}
    assert all(not p.requires_grad for p in model.audio_encoder.parameters())
    assert not model.talker_text_embed.weight.requires_grad
    assert model.lm_head.weight.data_ptr() == model.thinker.embed_tokens.weight.data_ptr()
    assert model.config.audio_hidden_size == TINY_VOXTRAL_PROJ['text_hidden_size']


def test_loss_matches_source_s3(model):
    batch = s3_batch(model.config)
    with torch.no_grad():
        out = model(**batch)
        ref_text, ref_speech = reference_s3_loss(model, batch)
    assert torch.allclose(out.metrics['text_loss'], ref_text, atol=0, rtol=0), \
        (out.metrics['text_loss'], ref_text)
    assert torch.allclose(out.metrics['speech_loss'], ref_speech, atol=1e-7), \
        (out.metrics['speech_loss'], ref_speech)
    assert torch.allclose(out.loss, ref_text + ref_speech, atol=1e-6)


def test_loss_weights_and_skips():
    torch.manual_seed(0)
    m = StreamingSIForConditionalGeneration(tiny_config(text_loss_weight=0.0)).eval()
    with torch.no_grad():
        out = m(**s3_batch(m.config))
    assert out.logits.numel() == 0                  # text_w=0 时跳过 lm_head（0 元素占位）
    assert 'speech_acc' in out.metrics and 'text_acc' not in out.metrics
    assert torch.allclose(out.loss, out.metrics['speech_loss'], atol=1e-6)

    m2 = StreamingSIForConditionalGeneration(tiny_config(speech_loss_weight=0.0)).eval()
    with torch.no_grad():
        out2 = m2(**s3_batch(m2.config))
    assert out2.speech_logits is None               # speech_w=0 时不跑 talker
    assert torch.allclose(out2.loss, out2.metrics['text_loss'], atol=1e-6)


def test_gradient_accumulation_semantics():
    """HF Trainer 据 accepts_loss_kwargs 决定是否做 GA 除法；必须为 False（loss 已是 batch 均值）。"""
    import importlib.util
    # 读源码而非 import Trainer（后者会连带导入 accelerate/datasets，与测试目标无关）。
    # 判定逻辑仍在当前 transformers 版本中；升级后若改名，本断言会提示重新核对。
    path = importlib.util.find_spec('transformers.trainer').origin
    assert 'hasattr(unwrapped_model, "accepts_loss_kwargs")' in open(path, encoding='utf-8').read()
    assert StreamingSIForConditionalGeneration(tiny_config()).accepts_loss_kwargs is False


def test_backward_reaches_trainable_parts(model):
    model.train()
    try:
        model.zero_grad()
        model(**s3_batch(model.config)).loss.backward()
        assert model.thinker.layers[0].self_attn.q_proj.weight.grad is not None
        assert model.talker.decoder.layers[0].self_attn.q_proj.weight.grad is not None
        assert model.audio_proj.mlp[1].weight.grad is not None
        assert all(p.grad is None for p in model.audio_encoder.parameters())
        assert not model.audio_encoder.training     # 冻结塔在 train() 下仍保持 eval
    finally:
        model.eval()
        model.zero_grad()


def test_save_load_roundtrip(model):
    batch = s3_batch(model.config)
    with torch.no_grad():
        ref = model(**batch)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)
        loaded = StreamingSIForConditionalGeneration.from_pretrained(d).eval()
    with torch.no_grad():
        got = loaded(**batch)
    assert torch.allclose(got.loss, ref.loss, atol=1e-6)
    sd_a, sd_b = model.state_dict(), loaded.state_dict()
    assert sd_a.keys() == sd_b.keys()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    assert loaded.lm_head.weight.data_ptr() == loaded.thinker.embed_tokens.weight.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Graph 需要 GPU')
def test_talker_cuda_graph_matches_eager():
    """TalkerDecodeGraph（StaticCache + 单 token CUDA Graph）与 eager DynamicCache 等价（fp32）。

    覆盖：多 token 段与单 token 步交错、graph 在不同位置重放、跨句 reset 复用，以及
    stream_generate_s4 两条路径在同一 seed 下产出相同的 token。
    """
    from streaming_si.model.model_si import TalkerStreamState
    torch.manual_seed(0)
    talker = StreamingSIForConditionalGeneration(tiny_config()).cuda().eval().talker
    D = talker.backbone.config.hidden_size
    plan = [7, 1, 1, 1, 5, 1, 1, 1, 1, 1, 9] + [1] * 20
    with torch.inference_mode():
        for sentence in range(2):
            g = torch.Generator(device='cuda').manual_seed(sentence)
            xs = [torch.randn(1, n, D, device='cuda', generator=g) for n in plan]
            dg = talker.decode_graph(256)
            dg.reset()
            got = [(dg.step(x) if x.size(1) == 1 else dg.prefill(x))[:, -1].clone() for x in xs]
            pkv, ref = None, []
            for x in xs:
                out = talker.backbone(inputs_embeds=x, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                ref.append(out.last_hidden_state[:, -1])
            assert float((torch.cat(got) - torch.cat(ref)).abs().max()) < 1e-4

    cond = torch.randn(1, 6, D, device='cuda', generator=torch.Generator(device='cuda').manual_seed(9))
    toks = {}
    for cg in (False, True):
        torch.manual_seed(1)
        st = TalkerStreamState(talker, cuda_graph=cg)
        assert (st.graph is not None) == cg
        toks[cg] = list(talker.stream_generate_s4(st, cond_chunk=cond, max_speech_tokens=40)) + \
            list(talker.stream_generate_s4(st, finalize=True, max_speech_tokens=40))
    assert toks[True] == toks[False] and len(toks[True]) > 0
