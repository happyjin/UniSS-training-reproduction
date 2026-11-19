"""SimulS2ST-Omni 接入 ms-swift：model_type / template / 数据集 / tuner / checkpoint。

    swift sft --external_plugins custom/plugin.py \
        --model <offline 基座> --model_type simuls2st_omni --template simuls2st_omni \
        --tuner_type simuls2st_omni --dataset simuls2st_omni ...        # 全部参数见 pipelines/simuls2st_omni/

逐项对应源 `scripts/train.py` + `QwenOmniTalkerTrainer`（源代码原样迁入 `simuls2st_omni/train/`，此处只做接线）：

| 源                                      | swift 侧                                                                    |
|-----------------------------------------|-----------------------------------------------------------------------------|
| `load_base_model`                       | `SimulS2STOmniLoader.get_model`（直接调用源函数）                            |
| `STAGE_BUILDERS` + `build_wrapped_model`| `SimulS2STOmniTuner.prepare_model`（`--tuner_type simuls2st_omni`）          |
| batch sampler（token-budget / 任务混合）| `plan.py` 物化为指针行；swift 以 batch_size=1、不打乱读取                   |
| `build_dataset` / collator              | 指针行在 `SimulS2STOmniTemplate._encode` 里取源 dataset 样本，经源 collator  |
| `compute_loss`（跳批阈值 + 分任务 loss）| `SimulS2STOmniTemplate.compute_sft_loss`                                    |
| `save_model`（talker_components.pt 等） | `SimulS2STOmniTemplate.save_callback`（swift `_save_model` 的委托钩子）     |
| `_load_from_checkpoint`                 | `install_resume_patch`（只对本模型生效）                                     |

与源实现的行为差异（均为修正，单卡下不出现；详见 README）：

* 多卡：源 `compute_loss` 调 `model.module(...)` 绕过了 DDP 的 forward，各 rank 的梯度从未同步，
  每张卡独立训练、只存 rank0。这里经 DDP forward，梯度正常 all-reduce。
* 相应地 Stage 2/3 必须 `ddp_find_unused_parameters=true`：文本任务 batch 不经过 talker，
  音频塔里被 LoRA 命中的 q/k/v 在无音频 batch 上不参与计算。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from transformers.modeling_outputs import ModelOutput

from .stage_config import config_path, load_stage_config, train_seed

MODEL_TYPE = 'simuls2st_omni'
TEMPLATE_TYPE = 'simuls2st_omni'
TUNER_TYPE = 'simuls2st_omni'
TRAIN_DATASET = 'simuls2st_omni'
VAL_DATASET = 'simuls2st_omni_val'
META_NAME = 'simuls2st_meta.json'
_MESSAGES = [{'role': 'user', 'content': 'simuls2st'}, {'role': 'assistant', 'content': 'simuls2st'}]
_MARKER = '_simuls2st_omni'

logger = logging.getLogger(__name__)


def _swift_logger():
    from swift.utils import get_logger
    return get_logger()


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def _loader_cls():
    from swift.model import ModelLoader

    class SimulS2STOmniLoader(ModelLoader):
        """加载 offline 基座（Qwen2.5-Omni + DualCodec talker），等价于源 `load_base_model`。

        返回的是未包装的 `Qwen2_5OmniForConditionalGeneration`；阶段策略（合并上一阶段 adapter、
        接入 Stage 1 talker、挂 LoRA、冻结）与 `OmniTalkerQwenOmni` 包装在 tuner 里做，
        与源 `build_wrapped_model` 的顺序一致（加载 → 策略 → 包装）。
        """

        def get_config(self, model_dir: str):
            from transformers import Qwen2_5OmniConfig
            return Qwen2_5OmniConfig.from_pretrained(model_dir)

        def get_processor(self, model_dir: str, config):
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(model_dir)

        def get_model(self, model_dir: str, config, processor, model_kwargs):
            from ..train.model_init import load_base_model
            dtype = model_kwargs.get('dtype', model_kwargs.get('torch_dtype'))
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype) if dtype != 'auto' else torch.bfloat16
            device_map = model_kwargs.get('device_map')
            model, _, codec_tokens = load_base_model(model_dir, dtype=dtype or torch.bfloat16, device_map=device_map)
            model.simuls2st_codec_tokens = codec_tokens
            _swift_logger().info(f'[simuls2st_omni] base loaded from {model_dir}: dtype={dtype} '
                                 f'device_map={device_map} codec tokens={codec_tokens}')
            return model

    return SimulS2STOmniLoader


def register_simuls2st_model() -> None:
    from swift.model import Model, ModelGroup, ModelMeta, register_model
    register_model(
        ModelMeta(
            MODEL_TYPE,
            [ModelGroup([Model(MODEL_TYPE, MODEL_TYPE)], TEMPLATE_TYPE)],
            _loader_cls(),
            template=TEMPLATE_TYPE,
            # 非多模态：样本在 template 里经源 collator 整体编码，不需要 swift 的 post_encode 钩子、
            # vit 冻结与自动梯度检查点（梯度检查点按源做法在 tuner 里对 thinker / talker 分别开启）
            is_multimodal=False,
            architectures=['Qwen2_5OmniForConditionalGeneration'],
            requires=['transformers>=4.52'],
            tags=['audio', 'speech-generation', 'simuls2st'],
        ),
        exist_ok=True)


# ---------------------------------------------------------------------------
# 阶段策略（tuner 插件）
# ---------------------------------------------------------------------------
_COPIED_ATTRS = ('model_info', 'model_meta', 'model_dir', 'config', 'generation_config',
                 'origin_generation_config', 'hf_device_map', 'simuls2st_codec_tokens')


def _build_stage_model(model, cfg: Dict[str, Any]):
    """源 `train.py:main` 中「策略 → 包装」两步；返回 `OmniTalkerQwenOmni`。"""
    from ..train.native_train import resolve_path
    from ..train.stage_policy import STAGE_BUILDERS, build_wrapped_model, describe_trainable
    stage, model_cfg = cfg['stage'], cfg['model']
    kwargs = {}
    if stage == 'stage2' and model_cfg.get('init_talker_from'):
        kwargs['init_talker_from'] = resolve_path(model_cfg['init_talker_from'])
    if stage == 'stage3':
        kwargs['merge_adapters_from'] = resolve_path(model_cfg['merge_adapters_from'])
    policy = STAGE_BUILDERS[stage](**kwargs)
    # 源在挂 LoRA 前没有播种（LoRA A 的初始化随进程而变）；这里固定用训练 seed，结果可复现
    torch.manual_seed(train_seed(cfg))
    wrapper = build_wrapped_model(model, policy)
    for name in _COPIED_ATTRS:   # swift Trainer / 保存逻辑从模型上读这些属性
        if hasattr(model, name):
            setattr(wrapper, name, getattr(model, name))
    wrapper.accepts_loss_kwargs = False          # 源 trainer: model_accepts_loss_kwargs = False（GA 按步数除）
    setattr(wrapper, _MARKER, SimpleNamespace(
        stage=stage, use_thinker_lora=policy.thinker_lora is not None,
        use_talker_lora=policy.talker_lora is not None, config_path=cfg['_path']))
    if bool(cfg['training'].get('gradient_checkpointing', False)):
        wrapper.gradient_checkpointing_enable()  # 与源 HF Trainer 相同：kwargs=None -> use_reentrant=False
    counts = describe_trainable(wrapper)
    _swift_logger().info(
        f'[simuls2st_omni] {stage}: trainable {counts["trainable"]:,} / {counts["total"]:,} params '
        f'(thinker {counts["thinker_trainable"]:,}, talker {counts["talker_trainable"]:,}); '
        f'policy={policy.name} init_talker_from={policy.init_talker_from} merge_from={policy.merge_adapters_from}')
    assert counts['trainable'] > 0, 'no trainable parameters under this policy'
    return wrapper


def load_training_checkpoint(wrapper, checkpoint_dir: str) -> None:
    """源 `QwenOmniTalkerTrainer._load_from_checkpoint`：talker（全参 / LoRA）+ thinker LoRA。"""
    from peft import PeftModel, set_peft_model_state_dict

    from ..train.modeling_omni_talker import THINKER_LORA_DIR_NAME, _load_adapter_weights, load_talker_components
    info = getattr(wrapper, _MARKER)
    load_talker_components(wrapper, checkpoint_dir, merge_lora=False)
    if not info.use_thinker_lora:
        return
    lora_dir = os.path.join(checkpoint_dir, THINKER_LORA_DIR_NAME)
    if not os.path.isdir(lora_dir):
        _swift_logger().warning(f'No thinker_lora/ found in {checkpoint_dir}')
        return
    if not isinstance(wrapper.thinker, PeftModel):
        _swift_logger().warning('Thinker is not a PeftModel, skipping LoRA load')
        return
    set_peft_model_state_dict(wrapper.thinker, _load_adapter_weights(lora_dir))
    _swift_logger().info(f'Thinker LoRA adapter loaded from checkpoint: {lora_dir}')


def save_training_checkpoint(wrapper, output_dir: str, tokenizer=None) -> None:
    """源 `QwenOmniTalkerTrainer.save_model` 的目录格式（推理脚本、下一阶段与源项目都能直接读）。

    `talker_components.pt`（talker 全量 state_dict；有 talker LoRA 时为 PEFT 包装后的 state_dict + 配置）、
    `talker_lora/`、`thinker_lora/`。tokenizer / processor 由 swift `_save` 写入。
    """
    from peft import PeftModel

    from ..train.modeling_omni_talker import TALKER_COMPONENTS_NAME, TALKER_LORA_DIR_NAME, THINKER_LORA_DIR_NAME
    base = wrapper.module if hasattr(wrapper, 'module') else wrapper
    info = getattr(base, _MARKER)
    os.makedirs(output_dir, exist_ok=True)
    talker = base.talker
    talker_state = {'talker': talker.state_dict()}
    if info.use_talker_lora and isinstance(talker, PeftModel):
        talker.save_pretrained(os.path.join(output_dir, TALKER_LORA_DIR_NAME))
        peft_cfg = talker.peft_config['default']
        talker_state['format'] = 'peft_state_dict'
        talker_state['talker_lora_config'] = {
            'r': int(peft_cfg.r), 'lora_alpha': int(peft_cfg.lora_alpha),
            'lora_dropout': float(peft_cfg.lora_dropout), 'target_modules': list(peft_cfg.target_modules),
        }
    torch.save(talker_state, os.path.join(output_dir, TALKER_COMPONENTS_NAME))
    if info.use_thinker_lora and isinstance(base.thinker, PeftModel):
        base.thinker.save_pretrained(os.path.join(output_dir, THINKER_LORA_DIR_NAME))
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, META_NAME), 'w', encoding='utf-8') as f:
        json.dump({'stage': info.stage, 'config': info.config_path, 'use_thinker_lora': info.use_thinker_lora,
                   'use_talker_lora': info.use_talker_lora, 'format': 'simuls2st_omni_source_v1'}, f, indent=2)
    _swift_logger().info(f'[simuls2st_omni] saved {info.stage} checkpoint (source format) to {output_dir}')


def _tuner_cls():
    try:
        from swift.tuner_plugin import Tuner
    except Exception:  # 纯源代码单测环境
        Tuner = object

    class SimulS2STOmniTuner(Tuner):
        """`--tuner_type simuls2st_omni`：按 `SIMULS2ST_CONFIG` 的 stage 套用源阶段策略。

        LoRA 秩 / 目标 / 冻结范围全部来自源 `stage_policy.py`，不读 swift 的 `--lora_*` 参数。
        """

        @staticmethod
        def prepare_model(args, model):
            cfg = load_stage_config()
            if int(args.seed) != train_seed(cfg):
                raise ValueError(f'--seed {args.seed} 与 {cfg["_path"]} 的 training.seed={train_seed(cfg)} 不一致'
                                 f'（源用同一个 seed 驱动 batch sampler 与 Trainer）')
            return _build_stage_model(model, cfg)

        @staticmethod
        def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
            save_training_checkpoint(model, save_directory)

        @staticmethod
        def from_pretrained(model, model_id, **kwargs):
            """续训（--resume_from_checkpoint）/ --adapters：先按阶段策略建好结构，再读入 checkpoint 权重。"""
            wrapper = _build_stage_model(model, load_stage_config())
            load_training_checkpoint(wrapper, model_id)
            return wrapper

    return SimulS2STOmniTuner


def register_simuls2st_tuner() -> None:
    from swift.tuner_plugin import tuners_map
    tuners_map[TUNER_TYPE] = _tuner_cls()


def install_resume_patch() -> None:
    """HF `_load_from_checkpoint` 要求目录里有 model.safetensors / adapter_model.*；本模型的 checkpoint 是源格式，
    改为调用源加载逻辑（只对带 `_simuls2st_omni` 标记的模型生效，其余模型走原实现）。"""
    from swift.trainers.mixin import SwiftMixin
    from transformers import Trainer as HfTrainer
    if getattr(SwiftMixin, '_simuls2st_resume_patched', False):
        return
    original = getattr(SwiftMixin, '_load_from_checkpoint', None) or HfTrainer._load_from_checkpoint

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.accelerator.unwrap_model(self.model) if hasattr(self, 'accelerator') else self.model
        if hasattr(target, _MARKER):
            load_training_checkpoint(target, resume_from_checkpoint)
            return
        return original(self, resume_from_checkpoint, model)

    SwiftMixin._load_from_checkpoint = _load_from_checkpoint
    original_train = SwiftMixin.train

    def train(self, *args, **kwargs):
        """源 train.py 训练结束后 `trainer.save_model(output_dir)`：最终权重导出到 output_dir 根目录，
        阶段配置里的 init_talker_from / merge_adapters_from 默认指向这里。

        用 `should_save`（多节点 + --save_on_each_node 时每个节点的 local rank 0 都为真），使每个节点
        本地都有一份，下一阶段在同一组节点上可直接读取。"""
        out = original_train(self, *args, **kwargs)
        target = self.accelerator.unwrap_model(self.model)
        if hasattr(target, _MARKER) and self.args.should_save:
            save_training_checkpoint(target, self.args.output_dir, tokenizer=self.template.processor)
        return out

    SwiftMixin.train = train
    original_init = SwiftMixin.__init__

    def __init__(self, *args, **kwargs):
        """采样计划里没有验证 batch 时关闭评估（swift 只在 eval_dataset is None 时才这样做）。"""
        from .plan import cached_plan
        model, eval_dataset, targs = kwargs.get('model'), kwargs.get('eval_dataset'), kwargs.get('args')
        plan = cached_plan()
        if hasattr(model, _MARKER) and eval_dataset is not None and plan is not None and not plan.eval_rows:
            from transformers.trainer_utils import IntervalStrategy
            kwargs['eval_dataset'] = None
            if targs is not None:
                targs.eval_strategy = IntervalStrategy.NO
                if hasattr(targs, 'eval_dataset'):
                    targs.eval_dataset = []
        original_init(self, *args, **kwargs)

    SwiftMixin.__init__ = __init__
    SwiftMixin._simuls2st_resume_patched = True


# ---------------------------------------------------------------------------
# Template：指针行 -> 源 dataset 样本 -> 源 collator；loss 与保存
# ---------------------------------------------------------------------------
@dataclass
class SimulS2STLossOutput(ModelOutput):
    """swift `Seq2SeqTrainer.compute_loss` 读 `outputs.logits`（为 None 时跳过 token acc）。"""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None


def _codec_tokens_from_constants() -> Dict[str, int]:
    """源 `resize_talker_vocab` 的控制 token 编号（与基座原词表无关，固定在码本之后）。"""
    from ..train.model_init import N_CONTROL_TOKENS, N_SEMANTIC_CODES
    n = N_SEMANTIC_CODES
    return {'codec_pad': n, 'codec_bos': n + 1, 'codec_eos': n + 2, 'codec_mask': n + 3,
            'vocab_size': n + N_CONTROL_TOKENS}


def _template_cls():
    from swift.template import Template

    class SimulS2STOmniTemplate(Template):
        # swift Seq2SeqTrainer 默认把 model_accepts_loss_kwargs 设为 True（除非 template 声明），之后
        # (1) loss 乘以进程数（average_tokens_across_devices），(2) HF 不再按梯度累积步数除 loss。
        # 源 trainer 是 False：每步 loss 为 batch 内均值，DDP 对梯度取平均，GA 按步数除。
        model_accepts_loss_kwargs = False

        # ---- 编码 ----
        def _stage(self):
            if not hasattr(self, '_s2st_plan'):
                from .plan import build_stage
                self._s2st_plan = build_stage(load_stage_config(), tokenizer=self.tokenizer)
            return self._s2st_plan

        def _collator(self):
            if not hasattr(self, '_s2st_collator'):
                from ..train.native_train import build_collator
                with open(os.path.join(self.model_info.model_dir, 'config.json'), encoding='utf-8') as f:
                    talker_cfg = json.load(f)['talker_config']
                talker = SimpleNamespace(text_bos_token=talker_cfg['tts_text_start_token_id'],
                                         text_eos_token=talker_cfg['tts_text_end_token_id'],
                                         text_pad_token=talker_cfg['tts_text_pad_token_id'])
                self._s2st_collator = build_collator(self._stage().cfg['data'], self.processor,
                                                     _codec_tokens_from_constants(), talker)
            return self._s2st_collator

        def _encode(self, inputs) -> Dict[str, Any]:
            """指针行 -> 源 dataset 样本（对应源 DataLoader 里逐下标的 `dataset[i]`）。

            拼 batch 放在 `data_collator`（对应源 `collate_fn`）：swift 的 `Template.encode` 包在
            `torch.inference_mode()` 里，在这里建的张量是 inference tensor，反传时会报错。
            """
            row = inputs.extra_kwargs
            plan = self._stage()
            if row.get('plan') != plan.digest:
                raise RuntimeError(f'指针行的采样计划 {row.get("plan")} 与当前 {plan.digest} 不一致'
                                   f'（配置 / 数据 / world_size 变了却复用了旧的数据集缓存？加 --load_from_cache_file false）')
            samples = [plan.dataset[int(i)] for i in row['batch']]
            # input_ids 只给 swift 计长度（= batch 行数），不进模型
            return {'s2st_samples': samples, 'input_ids': [0] * len(samples)}

        def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
            if len(batch) != 1:
                raise ValueError(f'simuls2st_omni 一行即一个完整 batch，需 --per_device_train_batch_size 1 '
                                 f'--per_device_eval_batch_size 1，收到 {len(batch)} 行')
            out = self._collator()(batch[0]['s2st_samples'])
            # swift 的外层只用 `labels` 判断「有监督」并做 token acc（logits=None 时跳过）；模型不接收它
            out['labels'] = out['talker_labels'] if out.get('talker_labels') is not None else out.get('text_labels')
            if out['labels'] is None:
                raise RuntimeError(f'batch {out.get("ids")} 没有任何监督（task={out.get("task_type")}）')
            return out

        def print_inputs(self, inputs: Dict[str, Any]) -> None:
            samples = inputs.get('s2st_samples') or []
            _swift_logger().info(f'[simuls2st_omni] first batch: {len(samples)} samples, '
                                 f'task={samples[0].get("task_type") if samples else None} '
                                 f'ids={[s.get("id") for s in samples[:4]]}')

        # ---- loss（源 QwenOmniTalkerTrainer.compute_loss）----
        def compute_sft_loss(self, model, inputs: Dict[str, Any], num_items_in_batch=None, trainer=None):
            return _compute_loss(self, model, inputs, trainer)

        # ---- 保存（swift Trainer._save_model 发现 template.save_callback 时直接委托）----
        def save_callback(self, model, output_dir):
            save_training_checkpoint(model, output_dir)

    return SimulS2STOmniTemplate


def _guard_skip(cfg_training: Dict[str, Any], input_ids, talker_input_ids):
    """源四个跳批阈值；返回原因字符串或 None。"""
    B, T = input_ids.shape
    max_thinker_tokens = int(cfg_training.get('max_thinker_tokens_per_batch', 0))
    max_seq_len = int(cfg_training.get('max_seq_len', 0))
    max_talker_seq_len = int(cfg_training.get('max_talker_seq_len', 0))
    max_talker_tokens = int(cfg_training.get('max_talker_tokens_per_batch', 0))
    if max_thinker_tokens > 0 and B * T > max_thinker_tokens:
        return f'thinker_tokens={B * T} > max_thinker_tokens_per_batch={max_thinker_tokens}, thinker_shape={(B, T)}'
    if max_seq_len > 0 and T > max_seq_len:
        return f'seq_len={T} > max_seq_len={max_seq_len}'
    if talker_input_ids is not None:
        if max_talker_seq_len > 0 and talker_input_ids.shape[1] > max_talker_seq_len:
            return f'talker_len={talker_input_ids.shape[1]} > max_talker_seq_len={max_talker_seq_len}, thinker_len={T}'
        tokens = talker_input_ids.shape[0] * talker_input_ids.shape[1]
        if max_talker_tokens > 0 and tokens > max_talker_tokens:
            return (f'talker_tokens={tokens} > max_talker_tokens_per_batch={max_talker_tokens}, '
                    f'talker_shape={tuple(talker_input_ids.shape)}, thinker_len={T}')
    return None


_SKIP_ROW_KEYS = (
    'input_ids', 'attention_mask', 'text_labels', 'talker_input_ids', 'talker_attention_mask',
    'talker_labels', 'reply_start_positions', 'reply_lengths',
    'talker_thinker_scatter_indices', 'talker_text_cond_indices',
)
_SKIP_AUDIO_KEYS = ('audio_features', 'audio_mask')


def _shrink_skipped_batch(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """多卡跳批时只留第 1 行做 forward（结果乘 0，不贡献梯度），避免超长 batch 白白占满显存。

    源实现直接返回 0、不做 forward；多卡下这样会让其余 rank 卡在 DDP all-reduce，所以这里仍要跑
    一次 forward，但没必要用整个 batch。每行的字段都是 batch-first、行内自洽（scatter 索引指向
    本行 thinker 位置），取 [:1] 后仍是合法输入。音频特征只在条数与行数一致时一起裁剪；
    不一致说明不是一行一段音频的布局，原样返回整个 batch（行为与改动前相同）。
    """
    b = int(inputs['input_ids'].shape[0])
    if b <= 1:
        return inputs
    for k in _SKIP_AUDIO_KEYS:
        v = inputs.get(k)
        if torch.is_tensor(v) and v.shape[0] != b:
            return inputs
    out = dict(inputs)
    for k in _SKIP_ROW_KEYS + _SKIP_AUDIO_KEYS:
        v = out.get(k)
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == b:
            out[k] = v[:1]
    return out


def _metric(trainer, mode: str, key: str, value: float) -> None:
    if trainer is not None:
        trainer.custom_metrics[mode][key].update(torch.tensor(float(value)))


def _compute_loss(template, model, inputs: Dict[str, Any], trainer):
    import torch.distributed as dist
    plan = template._stage()
    tr_cfg = plan.cfg['training']
    task_type = inputs.pop('task_type', 'unknown')
    inputs.pop('ids', None)
    mode = 'train' if model.training else 'eval'
    input_ids = inputs['input_ids']
    talker_input_ids = inputs.get('talker_input_ids')
    reason = _guard_skip(tr_cfg, input_ids, talker_input_ids)
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    step = int(trainer.state.global_step) if trainer is not None else 0
    if reason is not None:
        state = template.__dict__.setdefault('_s2st_skip', {'n': 0})
        state['n'] += 1
        if state['n'] % 10 == 1:
            _swift_logger().warning(f'Skipping batch: {reason}, task={task_type}, total_skipped={state["n"]}')
        _metric(trainer, mode, 'skipped_long_seq', 1.0)
        if not distributed:
            # 源：不做 forward，直接返回需要梯度的 0（本步所有参数无梯度，优化器跳过）
            zero = torch.zeros(1, device=input_ids.device, dtype=torch.float32, requires_grad=True).squeeze()
            return SimulS2STLossOutput(loss=zero)
        # 多卡仍要 forward（见下方乘 0），但只用 1 行：否则阈值只挡梯度、不挡显存，超长 batch 照样 OOM
        inputs = _shrink_skipped_batch(inputs)
        input_ids = inputs['input_ids']
        talker_input_ids = inputs.get('talker_input_ids')
    else:
        _metric(trainer, mode, 'skipped_long_seq', 0.0)

    if step < 3 and (trainer is None or trainer.is_world_process_zero()):
        alloc = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        _swift_logger().info(f'early batch step={step} task={task_type} thinker={tuple(input_ids.shape)} '
                             f'talker={None if talker_input_ids is None else tuple(talker_input_ids.shape)} '
                             f'peak={alloc:.1f}GiB')
    outputs = model(
        input_ids=input_ids,
        attention_mask=inputs['attention_mask'],
        talker_input_ids=talker_input_ids,
        # 源取 inputs.get("text_labels", inputs.get("labels"))，源 batch 里没有 `labels` 键；
        # 这里的 `labels` 是给 swift 外层的占位，不能传进来
        text_labels=inputs.get('text_labels'),
        talker_labels=inputs.get('talker_labels'),
        talker_attention_mask=inputs.get('talker_attention_mask'),
        reply_start_positions=inputs.get('reply_start_positions'),
        reply_lengths=inputs.get('reply_lengths'),
        talker_thinker_scatter_indices=inputs.get('talker_thinker_scatter_indices'),
        talker_text_cond_indices=inputs.get('talker_text_cond_indices'),
        input_features=inputs.get('audio_features'),
        feature_attention_mask=inputs.get('audio_mask'),
        return_dict=True,
    )
    loss = outputs.loss
    if reason is not None:
        # 多卡：被跳过的 rank 仍需参与本步的 DDP all-reduce，否则其余 rank 永久等待；乘 0 后不贡献梯度
        loss = loss * 0.0
        return SimulS2STLossOutput(loss=loss)
    _metric(trainer, mode, f'loss/{task_type}', loss.detach().float().item())
    if mode == 'train':
        # Stage 2 的实际任务配比（每个日志窗口内各任务 batch 占比），对应源日志里的 count/<task>
        for name in _task_names(plan):
            _metric(trainer, mode, f'task_frac/{name}', 1.0 if name == task_type else 0.0)
        if outputs.text_loss is not None:
            _metric(trainer, mode, 'loss/text', outputs.text_loss.detach().float().item())
        if outputs.audio_loss is not None:
            _metric(trainer, mode, 'loss/audio', outputs.audio_loss.detach().float().item())
    if step < 3 and torch.cuda.is_available() and (trainer is None or trainer.is_world_process_zero()):
        _swift_logger().info(f'early peak step={step} task={task_type} {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB')
    return SimulS2STLossOutput(loss=loss, text_loss=outputs.text_loss, audio_loss=outputs.audio_loss)


def _task_names(plan) -> List[str]:
    return list(getattr(plan.dataset, 'tasks', [])) if plan.multitask else []


def register_simuls2st_template() -> None:
    from swift.template import TemplateMeta, register_template
    register_template(
        TemplateMeta(
            template_type=TEMPLATE_TYPE,
            # 不使用 chat 包装：prompt 由源 prompt_formats / collator 生成
            prefix=[], prompt=['{{QUERY}}'], chat_sep=None, suffix=[],
            template_cls=_template_cls(),
        ),
        exist_ok=True)


# ---------------------------------------------------------------------------
# 数据集：dataset_path 即阶段配置 yaml，产出指针行
# ---------------------------------------------------------------------------
def _loader():
    from datasets import Dataset as HfDataset
    from swift.dataset.loader import DatasetLoader

    class SimulS2STPointerLoader(DatasetLoader):

        def _load_dataset_path(self, dataset_path, dataset_meta):
            from .plan import build_stage
            plan = build_stage(load_stage_config(dataset_path))
            rows = plan.eval_rows if dataset_meta.dataset_name == VAL_DATASET else plan.train_rows
            if not rows and dataset_meta.dataset_name == TRAIN_DATASET:
                raise RuntimeError(f'{TRAIN_DATASET} 为空（{dataset_path}）')
            if not rows:
                # 源：eval_samples >= 样本数时没有验证集，不评估。swift 不接受空数据集，这里放 1 行占位，
                # Trainer 初始化时按采样计划关闭评估（见 install_resume_patch），占位行不会被读到
                _swift_logger().warning('[simuls2st_omni] 验证集为空（eval_samples 不小于样本数），本次不评估')
                rows = plan.train_rows[:1]
            n = len(rows)
            return HfDataset.from_dict({'batch': rows, 'plan': [plan.digest] * n, 'messages': [_MESSAGES] * n})

    return SimulS2STPointerLoader


def register_simuls2st_datasets() -> None:
    from swift.dataset import DatasetMeta, register_dataset
    path = config_path()
    if not path:
        return
    loader = _loader()
    for name in (TRAIN_DATASET, VAL_DATASET):
        register_dataset(
            DatasetMeta(dataset_name=name, ms_dataset_id=name, dataset_path=path, loader=loader,
                        tags=['audio', 'simuls2st']),
            exist_ok=True)


def register_all() -> None:
    from ..train.datasets.streaming import StreamingS2STDataset
    from .latency import install as install_stable_latency
    install_stable_latency(StreamingS2STDataset)   # 见 latency.py：Stage 3 的 m 抽取可复现
    register_simuls2st_model()
    register_simuls2st_template()
    register_simuls2st_datasets()
    register_simuls2st_tuner()
    install_resume_patch()
