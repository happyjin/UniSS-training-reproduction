# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-swift external plugin for Qwen-OmniSpeak training pipelines.

Registers:

* qwen_omnispeak model variants (A/B/C/D/E/F/G/H) with model arch and template alias.
* Bridge-alignment datasets and template (Stage 1).
* Continue-pretrain datasets and template (Stage 2: speech-to-text /
  speech-to-speech translation, multi-speaker segmentation, and joint).
* Simultaneous-policy datasets and template (Stage 3 SFT: read-write policy
  learning for streaming speech-to-speech translation, supervised via a
  monotonic ``codec_to_sem_visible`` streaming attention schedule).
* normal_translation datasets and template (Stage 3 SFT: streaming
  speech-to-speech translation; mechanically identical to simultaneous-policy
  with isolated ``NORMAL_TRANSLATION_*`` switches).
* bilingual_translation datasets and template (Stage 3 SFT: streaming
  bilingual mutual translation; mechanically identical to normal_translation
  with isolated ``BILINGUAL_TRANSLATION_*`` switches).
* multi_speaker_translation datasets and template (Stage 3 SFT: streaming
  multi-speaker speech-to-speech translation; mechanically identical to
  normal_translation with isolated ``MULTI_SPEAKER_TRANSLATION_*`` switches).
* joint-stage1 datasets and template (Stage 3 SFT joint training over normal,
  bilingual, and multi-speaker translation with isolated ``JOINT_STAGE1_*`` switches).
"""
import importlib
import os
import sys

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

# Register qwen_omnispeak model types first so template subclassing works.
importlib.import_module('qwen_omnispeak.register_qwen_omnispeak')

try:
    importlib.import_module('qwen_omnispeak.mcore_adaptation')
    # Megatron-only: re-freeze codec_predictor's structurally-unused last
    # embedding entry after swift's activate_parameters() blanket-reactivates
    # it (see mcore_codec_embedding_freeze.py for the full story — required
    # for OVERLAP_GRAD_REDUCE=true to not crash on Megatron 0.17+'s
    # grad-ready invariant). No-op / safe if BaseMegatronTrainer isn't
    # importable (deepspeed path).
    importlib.import_module('qwen_omnispeak.mcore_codec_embedding_freeze').install_code_predictor_freeze_patch()
except Exception as e:  # pragma: no cover - mcore deps are optional for plain swift sft.
    if 'megatron' in sys.argv[1:3]:
        # Megatron backend: this registration is the model's only entry point —
        # skipping it would surface much later as an unrelated model-type /
        # first-batch error. Fail at plugin import time instead (P1-2).
        raise
    try:
        from swift.utils import get_logger
        get_logger().warning(f'Failed to register Qwen-OmniSpeak mcore adaptation: {e}')
    except Exception:
        pass

try:
    from .dataset import (
        register_bridge_alignment_datasets,
        register_continue_pretrain_datasets,
        register_simultaneous_policy_datasets,
        register_normal_translation_datasets,
        register_bilingual_translation_datasets,
        register_multi_speaker_translation_datasets,
        register_sft_joint_stage1_datasets,
    )
    from .template import (
        register_bridge_alignment_template,
        register_continue_pretrain_template,
        register_simultaneous_policy_template,
        register_normal_translation_template,
        register_bilingual_translation_template,
        register_multi_speaker_translation_template,
        register_joint_stage1_template,
    )
    from .token_accuracy_metrics import install_qwen_omnispeak_accuracy_patch
    from .wandb_experiment_config import install_wandb_extra_config_patch
    from .dataset_routing_tag_patch import install_dataset_loading_speedup_patches
    from .lance_lazy_batch_patch import install_lance_lazy_batch_patch
    from .ar_eval_callback import register_ar_eval_callback
    from .mfu_logging import install_mfu_logging_patch
    from .fill_logging import install_fill_logging_patch
    from .nsys_step_profile import install_nsys_step_profile
    # 注意：ray 场景下的 install_ray_nsight_patch() 不在这里调用——
    # RayHelper.worker() 装饰器的 new_init 会在 --external_plugins（本文件）
    # 被加载之前就先跑完 _create_workers()（见 ray_nsight_patch.py 头部
    # 完整时序证据），任何在这里调用它都必然太晚、拦不到真正的 actor
    # 创建。改为 ktrain/nsys_early_hook/sitecustomize.py 在解释器启动的
    # 最早时机（比 `import swift` 更早）调用，见该文件与
    # ray_nsight_patch.py 的头部说明。
except ImportError:  # pragma: no cover - imported as a top-level plugin by ms-swift.
    from dataset import (
        register_bridge_alignment_datasets,
        register_continue_pretrain_datasets,
        register_simultaneous_policy_datasets,
        register_normal_translation_datasets,
        register_bilingual_translation_datasets,
        register_multi_speaker_translation_datasets,
        register_sft_joint_stage1_datasets,
    )
    from template import (
        register_bridge_alignment_template,
        register_continue_pretrain_template,
        register_simultaneous_policy_template,
        register_normal_translation_template,
        register_bilingual_translation_template,
        register_multi_speaker_translation_template,
        register_joint_stage1_template,
    )
    from token_accuracy_metrics import install_qwen_omnispeak_accuracy_patch
    from wandb_experiment_config import install_wandb_extra_config_patch
    from dataset_routing_tag_patch import install_dataset_loading_speedup_patches
    from lance_lazy_batch_patch import install_lance_lazy_batch_patch
    from ar_eval_callback import register_ar_eval_callback
    from mfu_logging import install_mfu_logging_patch
    from fill_logging import install_fill_logging_patch
    from nsys_step_profile import install_nsys_step_profile
    # 见上方 try 分支同一位置的说明：ray 场景的 install_ray_nsight_patch()
    # 已迁移到 ktrain/nsys_early_hook/sitecustomize.py，不在此处调用。

install_qwen_omnispeak_accuracy_patch()
install_wandb_extra_config_patch()
# 每步 MFU / TFLOP-s 本地记录（deepspeed HF TrainerCallback + megatron
# BaseMegatronTrainer.train_step 双后端接入；写 mfu_log.jsonl + stdout，
# 不依赖 nsys / online wandb）。必须在 train() 前安装，故此处最早最安全。
install_mfu_logging_patch()
# 每步 FillRate（t_wait / t_step）。默认关，不往别人训练 stdout 打 [fill-log]。
# FILL_LOG_ENABLED=1 时 rank0 写 fill_log.jsonl；FILL_LOG_STDOUT=1 才打印。
install_fill_logging_patch()
# 第 NSYS_PROFILE_STEP 步用 nsys cudaProfilerApi 抓取（未设该 env 时完全无开销）。
# 非 ray（torchrun）路径：launch_train.sh 直接用 nsys profile 包裹训练进程本身。
# ray（USE_RAY=true）路径：nsys 包裹已由 ktrain/nsys_early_hook/
# sitecustomize.py 在解释器启动时（早于本文件被 --external_plugins 加载）
# 完成，此处不再重复调用 install_ray_nsight_patch()——RayHelper.worker()
# 装饰器的 new_init 会在本文件加载之前就先跑完 _create_workers()，在这里
# 调用必然太晚（完整时序证据见 custom/ray_nsight_patch.py 头部文档）。
install_nsys_step_profile()

# 诊断：SIGUSR1 无污染栈 dump（faulthandler 由进程自身在信号上下文打印全部线程栈，
# 不需要 ptrace、不暂停进程——规避 py-spy 的观测者效应）。仅在 env
# DIAG_SIGUSR1_DUMP=1 时注册（默认关闭，不影响其他实验；stall 诊断专用）。
if os.environ.get('DIAG_SIGUSR1_DUMP') == '1':
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    print('[diag] faulthandler registered on SIGUSR1 (all_threads=True)', flush=True)
# 诊断：overlap_grad_reduce 的 grad-ready assert 命中哪个具体 param（打印参数名，
# 不只是数量）。仅在 env DIAG_GRAD_READY=1 时注册（默认关闭，只在排查
# Megatron 0.17+ 「每个 trainable param 每个 microbatch 必须登记 grad-ready」
# 不变量崩溃时打开）。见 custom/megatron_grad_ready_diag.py。
try:
    from megatron_grad_ready_diag import install_grad_ready_diag
    install_grad_ready_diag()
except Exception as e:  # pragma: no cover - diagnostics-only, never fatal
    if os.environ.get('DIAG_GRAD_READY') == '1':
        try:
            from swift.utils import get_logger
            get_logger().warning(f'Failed to install grad_ready diag: {e}')
        except Exception:
            pass
# 必须在 swift 读取数据集之前安装：消除 loader 给数据集打 routing tag 时
# add_column 触发的整表 flatten_indices（unist 1899 万行实测单 rank 约 2h，
# 且 8 个 rank 各跑一遍）。详见 dataset_routing_tag_patch 模块文档。
install_dataset_loading_speedup_patches()
# Install before DataLoader construction. PyTorch will then pass each sampler
# batch to LazyLLMDataset.__getitems__, allowing one Lance take for mel,
# speaker embedding and codec codes before per-sample template encoding.
install_lance_lazy_batch_patch()
register_ar_eval_callback()

# Register group-wise LR optimizer callback (thinker vs rest).
# Activated via `--optimizer grouped_lr` + env THINKER_LR on the swift CLI.
try:
    from swift.optimizers.mapping import optimizers_map
    from grouped_lr_optimizer import GroupedLrOptimizerCallback
    optimizers_map['grouped_lr'] = GroupedLrOptimizerCallback
except Exception as e:  # pragma: no cover - grouped_lr 为可选优化器
    try:
        from swift.utils import get_logger
        get_logger().warning(f'Failed to register grouped_lr optimizer: {e}')
    except Exception:
        pass
register_bridge_alignment_datasets()
register_bridge_alignment_template()
register_continue_pretrain_datasets()
register_continue_pretrain_template()
register_simultaneous_policy_datasets()
register_simultaneous_policy_template()
register_normal_translation_datasets()
register_normal_translation_template()
register_bilingual_translation_datasets()
register_bilingual_translation_template()
register_multi_speaker_translation_datasets()
register_multi_speaker_translation_template()
register_sft_joint_stage1_datasets()
register_joint_stage1_template()

# streaming_si（Qwen3-0.6B Thinker + Voxtral + CosyVoice speech-token Talker，
# 迁移自 streaming-si-0.6b）。独立 model_type / template / 数据集，与 qwen_omnispeak 互不影响；
# 数据集按 SI_S1_DATA / SI_P2_DATA 环境变量注册，未设置时跳过。
importlib.import_module('streaming_si.integration.register_streaming_si').register_all()

# SimulS2ST-Omni（Qwen2.5-Omni-3B + DualCodec talker 三阶段，迁移自 SimulS2ST-Omni-stage123）。
# 独立 model_type / template / tuner，数据集按 SIMULS2ST_CONFIG（阶段配置 yaml）注册，未设置时跳过。
importlib.import_module('simuls2st_omni.integration.register').register_all()

# dataloader worker CPU 亲和绑定（训练进程内、DataLoader 构建前生效）。
# 详见 dataloader_cpu_bind 模块；env DATALOADER_CPU_BIND="0" 可关闭。
try:
    importlib.import_module('dataloader_cpu_bind')
except Exception as e:  # pragma: no cover
    try:
        from swift.utils import get_logger
        get_logger().warning(f'Failed to install dataloader_cpu_bind: {e}')
    except Exception:
        pass
