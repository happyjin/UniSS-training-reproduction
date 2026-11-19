#!/usr/bin/env python3
"""单卡手动训练探针：判别 deepspeed 路径 adapter-only 训练卡死的根因层面。

背景：megatron vs deepspeed 双后端对照（adapter-only，thinker/talker 冻结）中，
deepspeed 侧 loss 卡死（22.9→24 平台）而 megatron 正常下降（22.9→13.5）。
静态排查已排除参数集/优化器配置/聚合口径/数据差异。本探针绕开 deepspeed
引擎与 HF Trainer 分布式链路，在单卡上用纯 HF 前向 + 手动 AdamW 训练同一
合成 batch，判别根因层面：

  * loss 正常下降 → HF 前向/优化路径正常，卡死根因在 deepspeed 引擎/HF
    Trainer 的分布式链路（zero2/grad_accum/损失缩放）。
  * loss 卡死（不动或上升） → 根因在 HF 前向/模型路径本身（与分布式无关）。

设计要点（与训练配置对齐）：
  * 优化器：AdamW(lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
  * 梯度裁剪：clip_grad_norm_=1.0（与两侧名义配置一致）
  * warmup：线性 3 步到 1e-4 后恒定（探针只需要早期行为）
  * freeze：仅 model.thinker_adapter 可训练（16.7875M）
  * speaker prompt / ICL 关闭（简化路径；两侧同开已排除其影响）
  * 单一合成 batch 反复过拟合：排除数据变化干扰，判别最锐利
  * codec 内容用真实文本 token id（mod 映射避开特殊 id），含 n-gram 结构
    可学；residual 用随机标签（恒常背景项，观察 main CE 分量即可）

用法（容器内）：
  MODEL_PATH=/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a \
      python3 test_script/parity_probe/single_gpu_train_probe.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "custom"))
sys.path.insert(0, str(REPO_ROOT / "custom" / "qwen_omnispeak"))

os.environ.setdefault('QWEN_OMNISPEAK_USE_SPEAKER_PROMPT', 'false')
os.environ.setdefault('QWEN_OMNISPEAK_USE_ICL', 'false')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration  # noqa: E402

MODEL_PATH = os.environ.get(
    'MODEL_PATH', '/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a')
SEED = 42
STEPS = int(os.environ.get('PROBE_STEPS', '60'))
BATCH = int(os.environ.get('PROBE_BATCH', '4'))
T_TEXT = int(os.environ.get('PROBE_T_TEXT', '192'))
T_CODEC = int(os.environ.get('PROBE_T_CODEC', '640'))
REPLY_LEN = int(os.environ.get('PROBE_REPLY_LEN', '72'))
BOS_IDX = int(os.environ.get('PROBE_BOS_IDX', '8'))
LOG_EVERY = 5


def build_batch(tokenizer, cfg, device):
    """构造一个结构真实的合成 batch（字段与 collator 产出对齐）。"""
    g = torch.Generator().manual_seed(SEED)

    # ---- 文本侧：prompt + reply -------------------------------------------
    prompt_text = (
        '<|im_start|>user\nTranscribe and translate the audio into English.'
        '<|audio_bos|><|AUDIO|><|audio_eos|>\n<|im_end|>\n<|im_start|>assistant\n')
    reply_text = (
        'The quick brown fox jumps over the lazy dog near the river bank. '
        'We discuss the future of machine translation systems and their '
        'evaluation methodology in realistic deployment scenarios today.')
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    reply_ids_full = tokenizer(reply_text, add_special_tokens=False).input_ids
    reply_ids = (reply_ids_full * (REPLY_LEN // max(1, len(reply_ids_full)) + 1))[:REPLY_LEN]

    text_ids_list, text_labels_list = [], []
    for _ in range(BATCH):
        ids = (prompt_ids * (T_TEXT // max(1, len(prompt_ids)) + 1))[:T_TEXT - REPLY_LEN] + reply_ids
        labels = [-100] * (T_TEXT - REPLY_LEN) + reply_ids
        text_ids_list.append(ids)
        text_labels_list.append(labels)
    input_ids = torch.tensor(text_ids_list, dtype=torch.long, device=device)
    thinker_text_labels = torch.tensor(text_labels_list, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    # ---- codec 侧：结构化内容（真实文本 id 映射到安全值域） ----------------
    codec_src = tokenizer(
        ' '.join(['language modeling requires careful calibration of losses '
                  'and optimizers across distributed backends'] * 40),
        add_special_tokens=False).input_ids
    content = [(t % 1500) + 100 for t in codec_src][:T_CODEC - BOS_IDX - 2]
    n_content = len(content)
    bos_id = int(getattr(cfg, 'codec_bos_id', 2149))
    eos_id = int(getattr(cfg, 'codec_eos_id', 2150))
    pad_id = int(getattr(cfg, 'codec_pad_id', 2148))  # codec 域 pad（非文本 tts_pad）

    codec_in_rows, codec_lab_rows, resid_rows = [], [], []
    for _ in range(BATCH):
        # 布局：前 BOS_IDX-1 帧控制前缀（pad 占位）→ BOS → 内容 → EOS → pad
        ids = [pad_id] * (BOS_IDX - 1) + [bos_id] + content + [eos_id]
        labs = [-100] * (BOS_IDX - 1) + content + [eos_id]
        ids = ids + [pad_id] * (T_CODEC - len(ids))
        labs = labs + [-100] * (T_CODEC - len(labs))
        codec_in_rows.append(ids)
        codec_lab_rows.append(labs)
        resid = torch.full((T_CODEC, 15), -100, dtype=torch.long)
        valid_len = BOS_IDX + n_content  # 有效监督帧数（含 EOS 帧）
        resid[BOS_IDX - 1:valid_len] = torch.randint(
            0, 2048, (valid_len - BOS_IDX + 1, 15), generator=g)
        resid_rows.append(resid)

    codec_input_ids = torch.tensor(codec_in_rows, dtype=torch.long, device=device)
    codec_labels = torch.tensor(codec_lab_rows, dtype=torch.long, device=device)
    codec_residual_labels = torch.stack(resid_rows).to(device)
    codec_attention_mask = (codec_labels != -100)
    codec_bos_index = torch.full((BATCH,), BOS_IDX - 1, dtype=torch.long, device=device)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        codec_input_ids=codec_input_ids,
        codec_labels=codec_labels,
        codec_attention_mask=codec_attention_mask,
        codec_residual_labels=codec_residual_labels,
        thinker_text_labels=thinker_text_labels,
        codec_bos_index=codec_bos_index,
        labels=codec_labels,  # codec-aligned（acc 用，对齐 bridge 模板行为）
        run_thinker=True,
    ), (codec_labels != -100)


def main() -> None:
    torch.manual_seed(SEED)
    device = 'cuda'
    print(f'[probe] loading model from {MODEL_PATH}', flush=True)
    model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.cuda().train()

    adp = getattr(model.model, 'thinker_adapter', None)
    print(f'[probe] use_upstream_talker_input={getattr(model.config, "use_upstream_talker_input", None)} '
          f'thinker_adapter built={adp is not None}', flush=True)
    assert adp is not None, 'thinker_adapter not built; check QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT'
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('model.thinker_adapter'))
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f'[probe] trainable={n_train/1e6:.4f}M (expect ~16.79M)', flush=True)
    assert n_train > 0, 'no trainable parameters matched model.thinker_adapter'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cfg = model.config
    batch, valid_mask = build_batch(tokenizer, cfg, device)
    print(f'[probe] batch: T_text={T_TEXT} T_codec={T_CODEC} '
          f'valid_codec_tokens={int(valid_mask.sum().item())}', flush=True)

    opt = torch.optim.AdamW(
        trainable, lr=0.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # 训练前快照（对比训练后 ΔW，判别权重学习速度）
    snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    main_vocab = out_vocab = None
    print('[probe] step | loss | main_ce | token_acc | grad_norm | lr', flush=True)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        lr = 1e-4 * min(1.0, step / 3.0)
        for gp in opt.param_groups:
            gp['lr'] = lr

        out = model(**batch)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step == 1 or step % LOG_EVERY == 0:
            with torch.no_grad():
                logits = out.main_codec_logits
                main_vocab = logits.size(-1)
                labs = batch['codec_labels']
                valid = labs != -100
                main_ce = F.cross_entropy(
                    logits[valid].float(), labs[valid]).item()
                acc = (logits.argmax(-1)[valid] == labs[valid]).float().mean().item()
            print(f'[probe] {step:4d} | {loss.item():8.4f} | {main_ce:8.4f} | '
                  f'{acc:7.4%} | {grad_norm.item():9.4f} | {lr:.2e}', flush=True)

    dt = time.time() - t0
    print(f'[probe] done: {STEPS} steps in {dt:.1f}s, main_vocab={main_vocab}', flush=True)

    # 训练后 ΔW 统计（对比 jc85 checkpoint-500 的「bias 学、权重不学」形态）
    print('[probe] ==== adapter ΔW after training ====', flush=True)
    for n, p in model.named_parameters():
        if p.requires_grad:
            d = (p.detach().cpu().float() - snap[n].float())
            print(f'[probe] {n}: dW_norm={d.norm():.4f} rel={d.norm()/(snap[n].float().norm()+1e-12):.4%} '
                  f'w_std={p.detach().cpu().float().std():.6f}', flush=True)


if __name__ == '__main__':
    main()
