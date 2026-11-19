"""streaming_si 的模型变体（swift model_type）。两种变体共用 template / 数据集 / 训练脚本，只是结构不同。

    model_type         Thinker                         Talker
    streaming_si       Qwen3-0.6B（28L, 1024d）         0.1B：8 层 Qwen3 风格 decoder（1024d, 16/4 头）
    streaming_si_4b    Qwen3.5-4B（32L, 2560d，3:1       0.6B：Qwen3-0.6B 结构（28 层, 1024d, 16/8 头）
                       Gated DeltaNet / 门控全注意力）

变体由导出目录 config.json 的 `thinker_arch` 唯一决定（`variant_of`）；Talker 规格记在
`talker_preset`，结构以实际 talker_* 字段为准（冒烟夹具可用 --talker_overrides 缩小层数）。
Talker 0.6B 与 Qwen3-0.6B 同形，导出时可用 `--talker_init_dir <Qwen3-0.6B>` 以其 transformer 层初始化
（CosyVoice 系 speech LM 也用预训练文本 LLM 初始化）；embed_tokens / head 是 speech 词表，始终随机初始化。
"""
from .base_model import QWEN3_TALKER_8L

_TALKER_KEYS = ('talker_hidden_size', 'talker_num_hidden_layers', 'talker_num_attention_heads',
                'talker_num_key_value_heads', 'talker_head_dim', 'talker_intermediate_size', 'talker_rope_theta')

TALKER_PRESETS = {
    '0.1b': {k: QWEN3_TALKER_8L[k] for k in _TALKER_KEYS},
    '0.6b': dict(talker_hidden_size=1024, talker_num_hidden_layers=28, talker_num_attention_heads=16,
                 talker_num_key_value_heads=8, talker_head_dim=128, talker_intermediate_size=3072,
                 talker_rope_theta=1e6),
}

VARIANTS = {
    'streaming_si': dict(thinker_arch='qwen3', talker_preset='0.1b',
                         desc='Thinker Qwen3-0.6B / Talker 0.1B（8 层）'),
    'streaming_si_4b': dict(thinker_arch='qwen3_5', talker_preset='0.6b',
                            desc='Thinker Qwen3.5-4B / Talker 0.6B（Qwen3-0.6B 结构）'),
}
THINKER_ARCHES = ('qwen3', 'qwen3_5')


def variant_of(config) -> str:
    arch = getattr(config, 'thinker_arch', 'qwen3') or 'qwen3'
    for name, v in VARIANTS.items():
        if v['thinker_arch'] == arch:
            return name
    raise ValueError(f'未知 thinker_arch={arch!r}（应为 {THINKER_ARCHES}）')


def thinker_arch_of_dir(thinker_dir_config: dict) -> str:
    """基座 config.json -> thinker_arch。"""
    mt = thinker_dir_config.get('model_type')
    if mt == 'qwen3':
        return 'qwen3'
    if mt in ('qwen3_5', 'qwen3_5_text'):
        return 'qwen3_5'
    raise ValueError(f'不支持的 thinker 基座 model_type={mt!r}（支持 qwen3 / qwen3_5）')
