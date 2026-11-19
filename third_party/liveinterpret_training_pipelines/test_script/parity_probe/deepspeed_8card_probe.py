#!/usr/bin/env python3
"""E7：8 卡 deepspeed ZeRO-2 探针 —— 判别多卡分片/参数重组是否破坏 adapter 权重。

背景：E6（单卡 zero2）正常、E4（8 卡 DDP）正常、8 卡 zero1/zero2 卡死。
jc85（zero2）500 步后 adapter.norm.weight 从全 1 变成 mean≈0/std≈0.023 的随机
分布——纯训练不可能达成（Adam 500 步最大位移 ~0.05），疑似 ZeRO 多卡分片/
all-gather 重组错位把权重洗乱。

本探针：torchrun 8 进程 + deepspeed zero2，同一合成 batch（每 rank 相同数据，
梯度一致，排除数据分发变量），训练 20 步，rank0 每步打印 adapter 参数统计。

判别：
  * norm.weight 在 ~20 步内从全 1 快速漂移/乱序 → 重组错位（根因确认）
  * 参数正常演化（全 1 缓慢移动）→ 重组正常，根因在梯度通信数值

用法（容器内，8 卡）：
  torchrun --nproc_per_node 8 test_script/parity_probe/deepspeed_8card_probe.py
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
os.environ.setdefault('QWEN_OMNISPEAK_USE_UPSTREAM_TALKER_INPUT', 'true')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration  # noqa: E402
from single_gpu_train_probe import build_batch  # noqa: E402

MODEL_PATH = os.environ.get(
    'MODEL_PATH', '/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a')
SEED = 42
STEPS = int(os.environ.get('PROBE_STEPS', '20'))
RANK = int(os.environ.get('RANK', '0'))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))


def adapter_stats(module, tag):
    for n, p in module.named_parameters():
        if 'thinker_adapter' in n:
            t = p.detach().float()
            print(f'[e7-r{RANK}] {tag} {n}: norm={t.norm():.4f} mean={t.mean():.4e} '
                  f'std={t.std():.6f}', flush=True)


def main() -> None:
    import deepspeed
    from transformers import AutoTokenizer

    torch.manual_seed(SEED)
    torch.cuda.set_device(LOCAL_RANK)
    print(f'[e7-r{RANK}] loading model', flush=True)
    model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.cuda().train()

    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('model.thinker_adapter'))
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    batch, _ = build_batch(tokenizer, model.config, 'cuda')

    ds_config = {
        'train_batch_size': 32,  # 4/micro × 8 rank（每 rank 同 batch，仅探针）
        'train_micro_batch_size_per_gpu': 4,
        'gradient_accumulation_steps': 1,
        'bf16': {'enabled': True},
        'zero_optimization': {'stage': 2, 'contiguous_gradients': True,
                              'overlap_comm': False, 'reduce_scatter': True},
        'gradient_clipping': 1.0,
        'optimizer': {'type': 'AdamW',
                      'params': {'lr': 1e-4, 'betas': [0.9, 0.95], 'eps': 1e-8,
                                 'weight_decay': 0.1}},
        'steps_per_print': 200000,
    }
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=trainable, config=ds_config)
    if RANK == 0:
        print('[e7-r0] engine ready; adapter stats BEFORE training:', flush=True)
        adapter_stats(engine.module, 'init')

    t0 = time.time()
    for step in range(1, STEPS + 1):
        out = engine(**batch)
        loss = out.loss
        engine.backward(loss)
        engine.step()
        if RANK == 0 and step in (1, 2, 5, 10, 20):
            logits = out.main_codec_logits
            labs = batch['codec_labels']
            valid = labs != -100
            with torch.no_grad():
                main_ce = F.cross_entropy(logits[valid].float(), labs[valid]).item()
            nw = engine.module.model.thinker_adapter.norm.weight.detach().float()
            print(f'[e7-r0] step {step:3d} | loss={loss.item():.4f} main_ce={main_ce:.4f} | '
                  f'norm.weight: norm={nw.norm():.4f} mean={nw.mean():.4e} std={nw.std():.6f}',
                  flush=True)

    if RANK == 0:
        print(f'[e7-r0] done in {time.time()-t0:.1f}s; adapter stats AFTER:', flush=True)
        adapter_stats(engine.module, 'final')


if __name__ == '__main__':
    main()
