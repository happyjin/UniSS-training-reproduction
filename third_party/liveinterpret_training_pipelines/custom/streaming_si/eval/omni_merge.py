"""给官方 SimulS2ST-Omni 推理补上「先合并上一阶段 adapter」这一步（评测复现的 Stage 3 用）。

训练时 Stage 3 的起点是 `offline 基座 + 合并的 Stage 2`（`model.merge_adapters_from`，见
custom/simuls2st_omni/train/stage_policy.py::merge_adapters_into_base），Stage 3 的 r8 LoRA 是在这个
合并模型上学的。官方 `OmniTalkerSemanticTTSInference.from_pretrained` 只认一个 checkpoint_path，
没有合并上一阶段的入口。

做法：替换官方模块 `src.inference.infer_omni_talker_tts` 里的模块级名字 `load_talker_components`。
官方 from_pretrained 在 `_resize_talker_vocab` 之后、加载 checkpoint_path 的 talker 与 thinker LoRA 之前
恰好调用它一次；包装后在这次调用之前先执行与 merge_adapters_into_base 相同的两步：
  1. thinker：PeftModel.from_pretrained(stage2/thinker_lora) -> merge_and_unload
  2. talker ：官方 load_talker_components(model, stage2, merge_lora=True)（读 stage2 的 talker_components.pt）
然后照常调用原函数加载 Stage 3。其余推理逻辑一行不改。只在第一次调用时合并（一个进程只建一次模型）。
"""
import logging
import os

logger = logging.getLogger(__name__)


def install_stage2_merge(stage2_dir: str) -> None:
    assert os.path.isdir(stage2_dir), f'Stage 2 目录不存在: {stage2_dir}'
    import src.inference.infer_omni_talker_tts as tts
    from src.train.modeling_omni_talker import THINKER_LORA_DIR_NAME

    orig = tts.load_talker_components
    state = {'done': False}

    def _strip_peft(module):
        if hasattr(module, 'peft_config'):
            delattr(module, 'peft_config')

    def load_with_merge(container, checkpoint_dir, merge_lora=False):
        if not state['done']:
            state['done'] = True
            from peft import PeftModel
            lora = os.path.join(stage2_dir, THINKER_LORA_DIR_NAME)
            assert os.path.isdir(lora), f'Stage 2 缺 {THINKER_LORA_DIR_NAME}: {stage2_dir}'
            print(f'[omni_merge] merging Stage 2 thinker LoRA from {lora}', flush=True)
            container.thinker = PeftModel.from_pretrained(container.thinker, lora).merge_and_unload()
            _strip_peft(container.thinker)
            print(f'[omni_merge] merging Stage 2 talker from {stage2_dir}', flush=True)
            assert orig(container, stage2_dir, merge_lora=True), f'Stage 2 缺 talker 权重: {stage2_dir}'
            _strip_peft(container.talker)
            print(f'[omni_merge] Stage 2 merged; now loading {checkpoint_dir}', flush=True)
        return orig(container, checkpoint_dir, merge_lora=merge_lora)

    tts.load_talker_components = load_with_merge
