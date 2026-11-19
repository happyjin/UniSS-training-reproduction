"""合成 S3-w 起点（不预热时为 S3 起点）：S2 的 thinker + audio_proj  +  S1 的 talker + talker_text_embed。

    python custom/streaming_si/scripts/merge_stage_checkpoints.py \
        --thinker_from output/streaming_si_output/s2_st/.../checkpoint-N \
        --talker_from  output/streaming_si_output/s1_tts/.../checkpoint-M \
        --out_dir models/streaming_si_s3w_init

为什么必须合并（与源 scripts/merge_s1_talker_s2_thinker.py 同理）：S3-w / S3 要消费 S2 定型的
thinker，而 S1 的 checkpoint 带的是原始 Qwen3 thinker；直接从 S1 续训会让 S2 白训。

talker_text_embed 取自 S1（= 原始 Qwen3 embedding 副本，S1 全程冻结）。不能取 S2 的
thinker embed_tokens：它与 lm_head tied，S2 训练后与原始值近正交（源项目实测逐 token 余弦 0.37），
S1 学到的 text->speech 映射会整体失配。
"""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.model.speech_decoder import SPEECH_DECODER_DIR  # noqa: E402

FROM_THINKER = ('thinker_model.', 'audio_proj.', 'audio_encoder.')
FROM_TALKER = ('talker.', 'talker_text_embed.')
SIDE_FILES = ('voxtral_processor_config.json', 'streaming_si_export.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thinker_from', required=True, help='S2 产出目录')
    ap.add_argument('--talker_from', required=True, help='S1 产出目录')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    m_th = StreamingSIForConditionalGeneration.from_pretrained(args.thinker_from, dtype=torch.float32)
    m_tk = StreamingSIForConditionalGeneration.from_pretrained(args.talker_from, dtype=torch.float32)
    for attr in ('thinker_arch', 'thinker_text_config', 'hidden_size', 'num_hidden_layers', 'vocab_size',
                 'talker_hidden_size', 'talker_num_hidden_layers', 'talker_num_attention_heads',
                 'talker_num_key_value_heads', 'talker_head_dim', 'talker_intermediate_size',
                 'speech_token_size', 'audio_ids'):
        a, b = getattr(m_th.config, attr), getattr(m_tk.config, attr)
        if a != b:
            raise RuntimeError(f'两个 checkpoint 结构不一致: {attr} {a} vs {b}')

    sd_th, sd_tk = m_th.state_dict(), m_tk.state_dict()
    merged = {}
    for k in sd_th:
        if k.startswith(FROM_THINKER):
            merged[k] = sd_th[k]
        elif k.startswith(FROM_TALKER):
            merged[k] = sd_tk[k]
        else:
            raise RuntimeError(f'未归类的参数键 {k}（合并规则需要更新）')

    # 不变量检查：S1 全程冻结的部分应与基座一致，S2 不训 talker
    same = lambda a, b: torch.equal(a, b)  # noqa: E731
    if same(sd_tk['talker_text_embed.weight'], sd_th['thinker_model.model.embed_tokens.weight']):
        print('[merge][WARN] S2 的 thinker embedding 与 talker_text_embed 完全相同——S2 是否真的训练过？')
    if not same(sd_th['talker_text_embed.weight'], sd_tk['talker_text_embed.weight']):
        raise RuntimeError('两边 talker_text_embed 不同：它应在 S1/S2 全程冻结，检查训练参数前缀')

    m_th.load_state_dict(merged, strict=True)
    m_th.config.cond_mode = 'full'
    os.makedirs(args.out_dir, exist_ok=True)
    m_th.to(dtype).save_pretrained(args.out_dir, safe_serialization=True)
    for f in os.listdir(args.thinker_from):
        if f.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens', 'added_tokens', 'chat_template')) \
                or f in SIDE_FILES:
            shutil.copy(os.path.join(args.thinker_from, f), args.out_dir)
    # 声码器（仅推理用）：两边任一带有即复制，S3-w 起点仍可直接推理
    for src in (args.thinker_from, args.talker_from):
        dec = os.path.join(src, SPEECH_DECODER_DIR)
        if os.path.isdir(dec):
            shutil.copytree(dec, os.path.join(args.out_dir, SPEECH_DECODER_DIR), dirs_exist_ok=True)
            break
    with open(os.path.join(args.out_dir, 'streaming_si_merge.json'), 'w') as f:
        json.dump({'thinker_from': os.path.abspath(args.thinker_from),
                   'talker_from': os.path.abspath(args.talker_from), 'rule': {
                       'thinker_from': FROM_THINKER, 'talker_from': FROM_TALKER}}, f, indent=2)
    reloaded = StreamingSIForConditionalGeneration.from_pretrained(args.out_dir, dtype=dtype)
    assert reloaded.lm_head.weight.data_ptr() == reloaded.thinker.embed_tokens.weight.data_ptr()
    print(f'[merge] OK -> {args.out_dir}')


if __name__ == '__main__':
    main()
