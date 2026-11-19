#!/usr/bin/env python3
"""E6：单卡 deepspeed ZeRO-2 引擎探针 —— 定位 adapter-only 训练卡死的引擎内机制。

E4 已证「纯 DDP 健康」、E5 已证「ZeRO-1 卡死」、zero2 复验「卡死」。本探针在
单卡引入完整 deepspeed 引擎（ZeRO-2，单 rank 分片），合成 batch 单 batch 反复，
并直接观测引擎内部状态：

  * bf16 模型参数的每步演化（是否被更新）
  * fp32 master 权重（optimizer.single_partition_of_fp32_groups）的演化
  * 两者差值 → 判别「fp32→bf16 写回舍入吞没」假说

判别：
  * 单卡 zero2 卡死 + bf16 参数不动 + fp32 master 在动 → 写回吞没（根因确认）
  * 单卡 zero2 正常 → 多卡分片/通信环节问题（需 8 卡实验）

用法（容器内，单卡）：
  MODEL_PATH=/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a \
      python3 test_script/parity_probe/deepspeed_engine_probe.py
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
# 单卡 deepspeed 分布式环境
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29517')

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from modeling_qwen_omnispeak import QwenOmniSpeakForConditionalGeneration  # noqa: E402
from single_gpu_train_probe import build_batch  # noqa: E402  复用合成 batch

MODEL_PATH = os.environ.get(
    'MODEL_PATH', '/nvme/origin_models/custom_models/qwen2_5_3b_omnispeak_a')
SEED = 42
STEPS = int(os.environ.get('PROBE_STEPS', '30'))
LOG_EVERY = 5


def main() -> None:
    import deepspeed
    from transformers import AutoTokenizer

    torch.manual_seed(SEED)
    print(f'[e6] loading model from {MODEL_PATH}', flush=True)
    model = QwenOmniSpeakForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.cuda().train()

    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('model.thinker_adapter'))
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f'[e6] trainable={n_train/1e6:.4f}M', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    batch, valid_mask = build_batch(tokenizer, model.config, 'cuda')

    ds_config = {
        'train_batch_size': 4,
        'train_micro_batch_size_per_gpu': 4,
        'gradient_accumulation_steps': 1,
        'bf16': {'enabled': True},
        'zero_optimization': {
            'stage': 2,
            'contiguous_gradients': True,
            'overlap_comm': False,
            'reduce_scatter': True,
        },
        'gradient_clipping': 1.0,
        'optimizer': {
            'type': 'AdamW',
            'params': {'lr': 1e-4, 'betas': [0.9, 0.95], 'eps': 1e-8,
                       'weight_decay': 0.1},
        },
        'steps_per_print': 200000,
    }
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=trainable, config=ds_config)
    print(f'[e6] deepspeed engine initialized: {type(optimizer).__name__}', flush=True)

    # 初始快照：bf16 模型参数 + fp32 master
    def bf16_params():
        return {n: p.detach().cpu().clone()
                for n, p in engine.module.named_parameters() if 'thinker_adapter' in n}

    def fp32_masters():
        gs = getattr(optimizer, 'single_partition_of_fp32_groups', [])
        return {i: g.detach().cpu().clone() for i, g in enumerate(gs)}

    init_bf16 = bf16_params()
    init_fp32 = fp32_masters()
    print(f'[e6] fp32 master groups: {len(init_fp32)}, '
          f'sizes={[tuple(v.shape) for v in init_fp32.values()]}', flush=True)

    print('[e6] step | loss | main_ce | token_acc | dW_bf16_norm | dW_fp32_norm', flush=True)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        out = engine(**batch)
        loss = out.loss
        engine.backward(loss)
        engine.step()

        if step == 1 or step % LOG_EVERY == 0:
            with torch.no_grad():
                logits = out.main_codec_logits
                labs = batch['codec_labels']
                valid = labs != -100
                main_ce = F.cross_entropy(logits[valid].float(), labs[valid]).item()
                acc = (logits.argmax(-1)[valid] == labs[valid]).float().mean().item()
                # bf16 模型参数更新量
                cur_bf16 = bf16_params()
                dw_bf16 = torch.sqrt(sum(
                    ((cur_bf16[n].float() - init_bf16[n].float()) ** 2).sum()
                    for n in cur_bf16)).item()
                # fp32 master 更新量
                cur_fp32 = fp32_masters()
                dw_fp32 = torch.sqrt(sum(
                    ((cur_fp32[i] - init_fp32[i]) ** 2).sum()
                    for i in cur_fp32)).item() if cur_fp32 else float('nan')
            print(f'[e6] {step:4d} | {loss.item():8.4f} | {main_ce:8.4f} | '
                  f'{acc:7.4%} | {dw_bf16:10.6f} | {dw_fp32:10.6f}', flush=True)

    print(f'[e6] done in {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
