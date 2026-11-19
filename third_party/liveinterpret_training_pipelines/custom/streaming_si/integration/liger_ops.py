"""liger-kernel 融合算子接入（`STREAMING_SI_LIGER_OPS=1` 开启）。

## 为什么不能用 swift / transformers 的 `--use_liger_kernel`

`transformers.Trainer.__init__` 只做一件事：

    if self.args.use_liger_kernel:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model, **kernel_config)

而 `_apply_liger_kernel_to_instance` 是拿 **`model.config.model_type`** 去
`MODEL_TYPE_TO_APPLY_LIGER_FN` 查表的。streaming_si 导出目录的 config 是

    model_type    = streaming_si
    architectures = ['StreamingSIForConditionalGeneration']

查表必然 miss —— liger 只打一条 warning 就返回，**不报错、不生效**（静默空转）。
所以 `--use_liger_kernel true` 对本模型是纯粹的摆设，必须自己挂。

## 为什么本模型挂得上（与 qwen_omnispeak 不同）

`custom/qwen_omnispeak/modeling_qwen_omnispeak.py` 的注释说明：它的 talker/adapter 是
project-local 的 `_RMSNorm` / `_MLPProj` 子类，liger 对官方 HF 类的 monkey-patch 永远
碰不到，所以那边只能用 `QWEN_OMNISPEAK_LIGER_OPS=1` 直接调底层 autograd Function。

streaming_si 相反，两塔的 norm / MLP / attention **全是官方 HF 类**：

  * talker  `model_si.SITalker.decoder = transformers.models.qwen3.Qwen3Model`
            （`model/model_si.py`）—— S1 / S3 / S3-w 训的就是它
  * thinker `model/qwen3_5_compat.py` 全程复用
            `transformers.models.qwen3_next.modeling_qwen3_next` 的
            `Qwen3NextRMSNorm` / `Qwen3NextRMSNormGated` / `Qwen3NextMLP` / `Qwen3NextAttention`

所以只要在**建模型之前**按官方 patch 函数把这两个 HF 模块 patch 掉，两塔都能吃到融合核，
不需要像 omnispeak 那样手写 Function 调用。

## 时机：必须在 from_pretrained 之前

liger 的 `apply_liger_kernel_to_*` 对 RMSNorm 这类是**替换模块里的类对象**
（`modeling_xxx.XxxRMSNorm = LigerRMSNorm`），只影响此后新建的实例；对 MLP/attention 则多是
替换 `forward` 方法（对已有实例也生效）。两者混用意味着"建完模型再 patch"只有一半生效，
且 RMSNorm 那半静默失效 —— 和 `--use_liger_kernel` 一个毛病。
故本模块只在 `StreamingSILoader.get_model()` 调 `from_pretrained` **之前**调用。

## 不开 cross_entropy / fused_linear_cross_entropy

talker 不走 `Qwen3ForCausalLM` 的 loss：它用自己的 `llm_decoder`（`nn.Linear`）出 logits，
loss 在 `model/modeling_streaming_si.py::compute_si_losses` 里算。patch HF 的 CE 路径
对本模型无效；而 speech CE 的词表只有 6565（FSQ 3^8 + 4 special，见 `model/base_model.py`），
融合 CE 的收益主要来自超大词表时省掉 logits 物化，这里意义不大。另外
`compute_si_losses` 要用已物化的 logits 算 `speech_acc` / `speech_acc10`（topk），
与 fused-linear-CE 的"不物化 logits"设计直接冲突。

best-effort：liger 没装、版本没有对应 patch 函数、或 patch 本身抛错，都只告警并退回纯
PyTorch 实现，绝不让训练起不来。
"""
import os
from typing import List

from swift.utils import get_logger

logger = get_logger()

LIGER_OPS_ENV = 'STREAMING_SI_LIGER_OPS'

# 两塔用到的官方 HF 模块 -> liger 对应的 patch 函数名，第三项 = 是否必需。
#   talker  = Qwen3Model      -> apply_liger_kernel_to_qwen3        必需
#   thinker = Qwen3Next* 组件 -> apply_liger_kernel_to_qwen3_next   可选
#
# 为什么 talker 必需、thinker 可选：S1 / S3 / S3-w 的 scope=talker，**只有 talker 在训练**
# （前向+反向都吃融合核）；thinker 全程冻结、只有前向，拿不到融合只少一块前向收益，
# 不足以让整个实验失去意义。
#
# 实测（2026-10-15，镜像 p5-cuda12.8-torch2.8-te-v9-ray2-nsys-liger-streaming-si-v1）：
#   liger-kernel 0.5.10（无 __version__ 属性，版本取自 importlib.metadata）
#   共 19 个 apply_liger_kernel_to_*，有 qwen3 / qwen3_moe，**没有 qwen3_next**
#   => talker 能挂，thinker 挂不上。要让 thinker 也融合需升级 liger 并重建镜像。
_TARGETS = (
    ('talker(Qwen3)', 'apply_liger_kernel_to_qwen3', True),
    ('thinker(Qwen3Next)', 'apply_liger_kernel_to_qwen3_next', False),
)

# 只开纯算子融合；CE 相关一律关闭（见模块 docstring）
_KWARGS = dict(rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)

_applied = False


def required_patch_fns() -> List[str]:
    """必需的 patch 函数名（供 SI_CODE_CHECK 断言用，避免两处硬编码函数名）。"""
    return [fn for _, fn, req in _TARGETS if req]


def _liger_version() -> str:
    """liger-kernel 版本。0.5.10 的包里没有 `__version__` 属性（实测为 `<none>`），
    故优先走 importlib.metadata 的发行版元数据。"""
    try:
        import importlib.metadata as _md
        return _md.version('liger-kernel')
    except Exception:
        pass
    try:
        import liger_kernel
        return getattr(liger_kernel, '__version__', '?')
    except Exception:
        return '?'


def liger_enabled() -> bool:
    return os.environ.get(LIGER_OPS_ENV) == '1'


def apply_liger_to_streaming_si() -> List[str]:
    """按需 patch talker / thinker 的官方 HF 模块。返回实际生效的目标名列表。

    幂等（模块级 `_applied` 标记）：swift 在某些路径下会多次构造 loader，重复 patch
    虽然 liger 自身大多幂等，但重复替换类对象容易把"已 patch 的类"再包一层，故直接拦掉。
    """
    global _applied
    if not liger_enabled():
        return []
    if _applied:
        return []

    try:
        from liger_kernel.transformers import monkey_patch as _mp
    except Exception as e:
        logger.warning(f'streaming_si: {LIGER_OPS_ENV}=1 但 liger-kernel 不可用（{e}），'
                       f'退回纯 PyTorch（需要 ray/liger 底座镜像）')
        _applied = True
        return []

    done = []
    for label, fn_name, required in _TARGETS:
        fn = getattr(_mp, fn_name, None)
        if fn is None:
            lvl = logger.error if required else logger.warning
            lvl(f'streaming_si: liger-kernel {_liger_version()} 无 {fn_name}'
                f'（{"必需" if required else "可选，版本未支持"}），{label} 不融合')
            continue
        # 不同 liger 版本的 patch 函数签名不完全一致，逐个 kwarg 探测后再传
        import inspect
        try:
            allowed = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            allowed = set(_KWARGS)
        kwargs = {k: v for k, v in _KWARGS.items() if k in allowed}
        try:
            fn(**kwargs)
        except Exception as e:
            lvl = logger.error if required else logger.warning
            lvl(f'streaming_si: {fn_name}({kwargs}) 失败（{e}），{label} 不融合')
            continue
        done.append(f'{label}:{",".join(sorted(k for k, v in kwargs.items() if v))}')

    _applied = True
    if done:
        logger.info(f'streaming_si: liger-kernel {_liger_version()} 已挂载 -> ' + ' | '.join(done))
    else:
        logger.warning(f'streaming_si: {LIGER_OPS_ENV}=1 但没有任何目标被融合（见上方告警）')
    return done
