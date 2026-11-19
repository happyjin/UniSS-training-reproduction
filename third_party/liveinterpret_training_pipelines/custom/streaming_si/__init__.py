"""streaming_si：Voxtral 音频塔 + Thinker（Qwen3 / Qwen3.5）+ CosyVoice speech-token Talker 的流式同传模型。

目录：

    model/        网络结构、HF 包装、兼容实现（Voxtral / Qwen3.5）、loss、声码器打包
    data/         样本编码（S1-S4）、S5 流式数据集 / 采样计划 / 轨迹工具
    integration/  swift 接入：model_type / template / 数据集注册、S5 PEFT tuner
    inference/    推理：离线 S2ST（S1-S4）、S5 流式同传、TTS
    scripts/      命令行工具：基座导出、阶段合并、桥接门 A、冒烟夹具
    tests/        单测与对拍

旧模块名兼容：2026-10 之前所有模块平铺在本目录（如 `streaming_si.modeling_streaming_si`）。
已有 checkpoint / adapter 的配置（peft `auto_mapping.parent_library`）与仓库外脚本仍可能用旧名，
这里把旧名映射到新位置的**同一个模块对象**（不是副本，`isinstance` / monkeypatch 照常生效）。
新代码请直接用新路径。
"""
import importlib
import importlib.abc
import importlib.util
import sys

_MOVED = {
    'model': ('base_model', 'model_si', 'modeling_streaming_si', 'variants', 'voxtral_compat',
              'qwen3_5_compat', 'speech_decoder', 's4_objective', 's4_losses'),
    'data': ('si_encode', 's4_streaming_dataset', 's4_context_views', 'si_sampler', 'si_traj_ops', 's4_pipeline'),
    'integration': ('register_streaming_si', 'register_streaming_si_s4', 's4_peft'),
    'inference': ('infer_s2st', 'infer_s5_streaming', 'infer_tts'),
    'scripts': ('export_streaming_si', 'merge_stage_checkpoints', 's3w_bridge_gate', 's4w_bridge_gate',
                'make_smoke_fixture'),
}



class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target
        self._spec = None

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module):
        # importlib 会把 __spec__ 改成别名的 spec；恢复成真实模块的，避免影响 reload / pickle
        module.__spec__ = self._spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """把 `<pkg>.<旧模块>` 解析到 `<pkg>.<子目录>.<旧模块>`。

    每个包名一个实例：pytest 会以 `custom.streaming_si` 再导入一次本包，两套名字各自映射。
    """

    def __init__(self, pkg):
        self.pkg = pkg
        self.old_to_new = {f'{pkg}.{m}': f'{pkg}.{sub}.{m}' for sub, mods in _MOVED.items() for m in mods}

    def find_spec(self, fullname, path=None, target=None):
        new = self.old_to_new.get(fullname)
        if new is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new))


if not any(getattr(f, 'pkg', None) == __name__ and type(f).__name__ == '_AliasFinder' for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder(__name__))
