#!/usr/bin/env bash
# SimulS2ST-Omni（Qwen2.5-Omni-3B + DualCodec talker，迁移自 SimulS2ST-Omni-stage123）swift 训练参数层。
# 由 train_stage{1,2,3}.sh source。
#
# 训练行为的唯一事实源是 configs/stage{1,2,3}.yaml（与源项目 configs/ 逐字相同）：
#   * model / data / sampler 段：由 custom/simuls2st_omni/integration 直接读取（SIMULS2ST_CONFIG）；
#   * training 段：custom/simuls2st_omni/integration/swift_args.py 按源 build_training_args 的同一套规则
#     翻译成 swift 参数（并钉住 swift 改过默认值的项，如 adam_beta2）。
# 本文件只负责：路径解析、运行场景、分布式环境与 swift 的启动。
#
# 环境变量：
#   SIMULS2ST_CONFIG        阶段配置（缺省 configs/<stage>.yaml，由 train_stage*.sh 设定）；
#                           相对路径按本目录的 configs/ 解析（如 stage1_tb24k.yaml），ktrain yaml 不必写死代码挂载路径
#   SIMULS2ST_ROOT          配置里相对路径（models/、data/、outputs/）的根，缺省为本仓库根
#   MODEL_PATH              offline 基座目录，缺省为配置 model.path 解析结果
#   OUTPUT_DIR              输出目录，缺省为配置 training.output_dir 解析结果
#   SIMULS2ST_INIT_TALKER_FROM / SIMULS2ST_MERGE_ADAPTERS_FROM   覆盖 Stage 2 / Stage 3 的交接目录
#   SCENE                   prod（GPU，缺省）| smoke（CPU 冒烟，tiny 夹具，3 步）
#   MAX_STEPS / NUM_TRAIN_EPOCHS / LEARNING_RATE / SAVE_STEPS / LOGGING_STEPS / EVAL_STEPS /
#   DATALOADER_NUM_WORKERS  覆盖配置 training 段（试跑用）
#   PYTHON                  缺省 python
set -euo pipefail

SIMULS2ST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${SIMULS2ST_DIR}/../.." && pwd)}"
PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
PYTHON="${PYTHON:-python}"
: "${SCENE:=prod}"
export SIMULS2ST_ROOT="${SIMULS2ST_ROOT:-${REPO_ROOT}}"

# 源 run_stage*.sh：dataloader worker 很多（8 rank × 8 worker），不限线程时 OMP 线程池把 CPU 打满
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

_swift_args_py() {
    PYTHONPATH="${REPO_ROOT}/custom${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -m simuls2st_omni.integration.swift_args "$@"
}

# run_simuls2st [额外 swift 参数...]
run_simuls2st() {
    : "${SIMULS2ST_CONFIG:?需要 SIMULS2ST_CONFIG}"
    [[ "${SIMULS2ST_CONFIG}" == /* ]] || SIMULS2ST_CONFIG="${SIMULS2ST_DIR}/configs/${SIMULS2ST_CONFIG}"
    [[ -f "${SIMULS2ST_CONFIG}" ]] || { echo "[simuls2st] 阶段配置不存在: ${SIMULS2ST_CONFIG}" >&2; return 1; }
    export SIMULS2ST_CONFIG
    local stage="" model_path="" output_dir="" init_talker_from="" merge_adapters_from="" k v
    while IFS='=' read -r k v; do
        case "${k}" in
            stage) stage="${v}" ;; model_path) model_path="${v}" ;; output_dir) output_dir="${v}" ;;
            init_talker_from) init_talker_from="${v}" ;; merge_adapters_from) merge_adapters_from="${v}" ;;
        esac
    done < <(_swift_args_py --paths "${SIMULS2ST_CONFIG}")
    model_path="${MODEL_PATH:-${model_path}}"
    output_dir="${OUTPUT_DIR:-${output_dir}}"
    [[ -f "${model_path}/config.json" ]] || { echo "[simuls2st] 基座目录没有 config.json: ${model_path}" >&2; return 1; }

    # 阶段交接：源在启动时检查上一阶段产物，缺失直接失败（不要让 Stage 2 静默地从未训练的 talker 起步）
    if [[ "${stage}" == stage2 ]]; then
        local from="${SIMULS2ST_INIT_TALKER_FROM:-${init_talker_from}}"
        if [[ -n "${from}" && ! -f "${from}/talker_components.pt" && ! -d "${from}/talker_lora" ]]; then
            echo "[simuls2st] Stage 2 的 init_talker_from 下没有 talker_components.pt / talker_lora/: ${from}" >&2; return 1
        fi
    fi
    if [[ "${stage}" == stage3 ]]; then
        local from="${SIMULS2ST_MERGE_ADAPTERS_FROM:-${merge_adapters_from}}"
        [[ -d "${from}" ]] || { echo "[simuls2st] Stage 3 的 merge_adapters_from 不存在: ${from}" >&2; return 1; }
    fi

    local scene_args=()
    case "${SCENE}" in
        prod)
            if [[ -z "${NPROC_PER_NODE:-}" ]]; then
                NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
                [[ "${NPROC_PER_NODE}" -ge 1 ]] || { echo "[simuls2st] prod 场景未检测到 GPU" >&2; return 1; }
            fi
            # swift sft 见到 NPROC_PER_NODE / NNODES 即用 torchrun 拉起
            export NPROC_PER_NODE NNODES="${NNODES:-1}" NODE_RANK="${NODE_RANK:-0}" \
                   MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29501}"
            ;;
        smoke)
            # 单进程 CPU：macOS 的 MPS 不支持 c10d broadcast；只验证链路
            export MAX_STEPS="${MAX_STEPS:-3}" DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
            scene_args=(--use_cpu true --device_map cpu)
            unset NPROC_PER_NODE NNODES NODE_RANK
            ;;
        *) echo "[simuls2st] 未知 SCENE=${SCENE}（prod | smoke）" >&2; return 1 ;;
    esac

    # 场景里导出的 MAX_STEPS 等覆盖项要在翻译 yaml 之前设好
    local args=() a
    while IFS= read -r a; do args+=("${a}"); done < <(_swift_args_py "${SIMULS2ST_CONFIG}")
    # 源 run_stage*.sh：RESUME=<checkpoint-N> 续训（目录里需有 trainer_state.json）
    if [[ -n "${RESUME:-}" ]]; then
        [[ -f "${RESUME}/trainer_state.json" ]] || { echo "[simuls2st] RESUME 目录没有 trainer_state.json: ${RESUME}" >&2; return 1; }
        args+=(--resume_from_checkpoint "${RESUME}")
    fi

    mkdir -p "${output_dir}"
    echo "[simuls2st] ${stage}: config=${SIMULS2ST_CONFIG} root=${SIMULS2ST_ROOT}" >&2
    echo "[simuls2st] model=${model_path} output=${output_dir} nproc=${NPROC_PER_NODE:-1}x${NNODES:-1}" >&2
    swift sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${model_path}" \
        "${args[@]}" \
        --output_dir "${output_dir}" \
        ${scene_args[@]+"${scene_args[@]}"} \
        "$@"
}
