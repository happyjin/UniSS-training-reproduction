#!/usr/bin/env bash
# streaming_si（Qwen3-0.6B Thinker + Voxtral + CosyVoice speech-token Talker）训练参数层。
# 由 train_tts.sh / train_st.sh / train_s2st.sh / train_simul.sh source；所有影响训练行为的参数只在本目录脚本里设置。
#
# 阶段 -> 可训参数（括号内为源 streaming-si-0.6b 的阶段名与 trainer）：
#   talker        S1 / S3      只训 talker.*                     (源 S1-0 / S1-1a，train_s1_talker.py --mode talker)
#   talker_bridge S3-w / S4-w  只训 talker.thinker_proj.* + talker.fusion.*  (源 S1-1w / S3-w，train_s3_s2st.py --mode bridge)
#   thinker_audio S2           audio_proj.* + thinker_model.*   (源 S2，train_s2_st.py --mode thinker_audio)
#   joint         S4           audio_proj.* + thinker_model.* + talker.*  (源 S3，train_s3_s2st.py --mode joint)
#   stream_lora   S5           LoRA(thinker/talker) + 4 行流式 token + thinker_proj/fusion
#                              （源 S4；--tuner_type streaming_si_s4，细分由 SI_S4_TRAIN_SCOPE 决定，见 s4_peft.py）
# S1-S4 为 swift 全参模式按前缀冻结；audio_encoder（Voxtral）与 talker_text_embed 始终冻结，
# 模型首个训练步会校验（见 modeling_streaming_si.py）。
#
# 学习率：源 S1-S3（本仓库 S1-S4）get_lr = lr*(0.1+0.45*(1+cos(pi*t/T)))、源 S4（本仓库 S5）
#   s4_cosine_lr = min+(lr-min)/2*(1+cos)，均无 warmup，
#   对应 transformers cosine_with_min_lr（min_lr_rate / min_lr）。曲线形状相同，但步位差一：
#   源第 k 次更新取 t=k（1 起），HF 取 t=k-1（0 起），即首步为峰值 lr、末步略高于下限。
# 优化器：源 AdamW(weight_decay=0.01) + clip_grad_norm 1.0。
set -euo pipefail

STREAMING_SI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO_ROOT:=$(cd "${STREAMING_SI_DIR}/../.." && pwd)}"
: "${SOURCE_ROOT:=${REPO_ROOT}}"
PLUGIN_PATH="${PLUGIN_PATH:-${REPO_ROOT}/custom/plugin.py}"
: "${OUTPUT_ROOT:=${SOURCE_ROOT}/output/streaming_si_output}"

# 模型变体（custom/streaming_si/model/variants.py）：两种 model_type 共用 template / 数据集 / 本目录全部训练脚本
#   streaming_si     Thinker Qwen3-0.6B / Talker 0.1B
#   streaming_si_4b  Thinker Qwen3.5-4B / Talker 0.6B
# 默认按 MODEL_PATH/config.json 的 thinker_arch 自动识别；SI_MODEL_TYPE 可显式指定（loader 会校验两者一致）。
TEMPLATE=streaming_si

# ---- SCENE 预设（运行规模差异只在此处派生，不在任务 yaml 覆盖）----
#   prod  : 集群 GPU（bf16 + DeepSpeed ZeRO-2）
#   smoke : 单机 CPU 连通性冒烟（tiny 模型，make_smoke_fixture.py 产出），3 步即停
: "${SCENE:=prod}"
case "${SCENE}" in
    prod)
        : "${TORCH_DTYPE:=bfloat16}"
        : "${DEEPSPEED:=zero2}"
        : "${MAX_STEPS:=-1}"
        : "${DATALOADER_NUM_WORKERS:=4}"
        : "${SAVE_STEPS:=2000}"
        : "${LOGGING_STEPS:=50}"
        ;;
    smoke)
        TORCH_DTYPE=float32
        DEEPSPEED=none
        MAX_STEPS=3
        DATALOADER_NUM_WORKERS=0
        SAVE_STEPS=3
        LOGGING_STEPS=1
        PER_DEVICE_TRAIN_BATCH_SIZE=2
        SAVE_ONLY_MODEL=true          # 冒烟只验证链路，不存 optimizer 状态（每份约 2x 模型体积）
        export CUDA_VISIBLE_DEVICES=""
        ;;
    *) echo "[ERROR] 未知 SCENE=${SCENE}（prod | smoke）" >&2; exit 1 ;;
esac

: "${SAVE_TOTAL_LIMIT:=5}"
: "${SAVE_ONLY_MODEL:=false}"
: "${REPORT_TO:=none}"
: "${WEIGHT_DECAY:=0.01}"
: "${MAX_GRAD_NORM:=1.0}"
: "${MIN_LR_RATE:=0.1}"
: "${MIN_LR:=}"                    # 非空时用绝对下限（S5：3e-5 -> 2e-6），覆盖 MIN_LR_RATE
: "${EVAL_STRATEGY:=no}"           # S5 有 val 时由 train_simul.sh 设为 steps
: "${DATASET_NUM_PROC:=4}"
: "${WARMUP_RATIO:=0}"
# true = 按 lengths 列分桶组 batch（swift BatchSamplerShard + HF get_length_grouped_indices）。
# 数据集必须带 lengths 列：P2 数据（S2 / S4-w / S4）由 ktrain_entry.sh -> p2_parquet_prep.py --lengths 生成，
# Emilia 数据（S1 / S3-w / S3）由 s1_parquet_prep.py 生成，
# 缺列时插件注册阶段直接报错（register_streaming_si.register_streaming_si_datasets）
: "${GROUP_BY_LENGTH:=false}"
export GROUP_BY_LENGTH

# 分布式：swift sft 只要看到 NPROC_PER_NODE/NNODES 就走 torchrun，故仅在 prod 导出；
# smoke 必须单进程（macOS 的 MPS 不支持 c10d broadcast，torchrun 下首个 collective 即崩）。
# ray（USE_RAY=true，KubeRay framework: ray）：单 driver 进程直跑 swift sft --use_ray，由
# RayHelper 在 RAY_DEVICE_GROUPS 上起 actor。此时绝不能导出 NPROC_PER_NODE/NNODES 等
# （swift/cli/main.py 的 use_torchrun() 只看它们是否存在，套 torchrun 会起 8 个 driver、
# 重复建 placement group 永远 pending；launch_train.sh 已 unset，这里不能再探测导出）。
USE_RAY="${USE_RAY:-false}"
[[ "${USE_RAY}" == 1 ]] && USE_RAY=true
RAY_ARGS=()
if [[ "${USE_RAY}" == true ]]; then
    [[ "${SCENE}" == "prod" ]] || { echo "[ERROR] USE_RAY 只支持 SCENE=prod" >&2; exit 1; }
    unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
    RAY_ARGS+=(--use_ray true)
    [[ -n "${RAY_EXP_NAME:-}" ]] && RAY_ARGS+=(--ray_exp_name "${RAY_EXP_NAME}")
    [[ -n "${RAY_DEVICE_GROUPS:-}" ]] && RAY_ARGS+=(--device_groups "${RAY_DEVICE_GROUPS}")
elif [[ "${SCENE}" == "prod" ]]; then
    if [[ -z "${NPROC_PER_NODE:-}" ]]; then
        NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        [[ "${NPROC_PER_NODE}" -ge 1 ]] || { echo "[ERROR] prod 场景未检测到 GPU" >&2; exit 1; }
    fi
    export NPROC_PER_NODE NNODES="${NNODES:-1}" NODE_RANK="${NODE_RANK:-0}" \
           MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29500}"
else
    unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
fi
export TOKENIZERS_PARALLELISM=false

_scope_args() {
    case "$1" in
        talker)
            printf '%s\n' --tuner_type full --freeze_llm true --freeze_vit true --freeze_aligner true \
                          --trainable_parameters talker. ;;
        talker_bridge)      # S3-w / S4-w：只训 thinker 路桥接与融合层（talker 主干 / text_proj 冻结）
            printf '%s\n' --tuner_type full --freeze_llm true --freeze_vit true --freeze_aligner true \
                          --trainable_parameters talker.thinker_proj. talker.fusion. ;;
        thinker_audio)
            printf '%s\n' --tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false ;;
        joint)
            printf '%s\n' --tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner false \
                          --trainable_parameters talker. ;;
        stream_lora)
            printf '%s\n' --tuner_type streaming_si_s4 --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
                          --lora_dropout "${LORA_DROPOUT}" ;;
        *) echo "[ERROR] 未知训练范围 $1" >&2; return 1 ;;
    esac
}

# run_streaming_si <dataset> <run_label> <scope> [额外 swift 参数...]
run_streaming_si() {
    local dataset="$1" label="$2" scope="$3"
    shift 3
    : "${MODEL_PATH:?需要 MODEL_PATH（export_streaming_si.py / merge_stage_checkpoints.py 的产出目录）}"
    local model_type="${SI_MODEL_TYPE:-}"
    if [[ -z "${model_type}" ]]; then
        model_type="$(python3 -c 'import json,sys
arch = json.load(open(sys.argv[1])).get("thinker_arch", "qwen3")
print({"qwen3": "streaming_si", "qwen3_5": "streaming_si_4b"}.get(arch, "unknown:" + arch))' \
            "${MODEL_PATH}/config.json")" || { echo "[ERROR] 读不到 ${MODEL_PATH}/config.json" >&2; return 1; }
        [[ "${model_type}" == unknown:* ]] && { echo "[ERROR] 未知 thinker_arch（${model_type#unknown:}）" >&2; return 1; }
    fi
    local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${label}}"
    mkdir -p "${output_dir}"

    local scope_args=()
    while IFS= read -r a; do scope_args+=("$a"); done < <(_scope_args "${scope}")
    local ds_args=()
    [[ "${DEEPSPEED}" != "none" ]] && ds_args=(--deepspeed "${DEEPSPEED}")
    local lr_kwargs="{\"min_lr_rate\": ${MIN_LR_RATE}}"
    [[ -n "${MIN_LR}" ]] && lr_kwargs="{\"min_lr\": ${MIN_LR}}"

    echo "[streaming_si] ${label}: model_type=${model_type} scope=${scope} dataset=${dataset} model=${MODEL_PATH}" >&2
    echo "[streaming_si] text_w=${SI_TEXT_LOSS_WEIGHT} speech_w=${SI_SPEECH_LOSS_WEIGHT} cond_mode=${SI_COND_MODE}" \
         "lr=${LEARNING_RATE} bs=${PER_DEVICE_TRAIN_BATCH_SIZE}x${RAY_WORLD_SIZE:-$(( ${NPROC_PER_NODE:-1} * ${NNODES:-1} ))}gpu ga=${GRADIENT_ACCUMULATION_STEPS} ray=${USE_RAY}" \
         "ds=${DEEPSPEED} liger=${STREAMING_SI_LIGER_OPS:-0}" >&2

    swift sft \
        --external_plugins "${PLUGIN_PATH}" \
        --model "${MODEL_PATH}" \
        --model_type "${model_type}" \
        --template "${TEMPLATE}" \
        --dataset "${dataset}" \
        "${scope_args[@]}" \
        --torch_dtype "${TORCH_DTYPE}" \
        ${ds_args[@]+"${ds_args[@]}"} \
        ${RAY_ARGS[@]+"${RAY_ARGS[@]}"} \
        --remove_unused_columns false \
        --load_from_cache_file false \
        --lazy_tokenize true \
        --padding_free false \
        --packing false \
        --max_length "${MAX_LENGTH}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${LEARNING_RATE}" \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs "${lr_kwargs}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --group_by_length "${GROUP_BY_LENGTH}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --max_grad_norm "${MAX_GRAD_NORM}" \
        --split_dataset_ratio 0 \
        --eval_strategy "${EVAL_STRATEGY}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --save_only_model "${SAVE_ONLY_MODEL}" \
        --logging_steps "${LOGGING_STEPS}" \
        --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
        --dataset_num_proc "${DATASET_NUM_PROC}" \
        --report_to "${REPORT_TO}" \
        --output_dir "${output_dir}" \
        "$@"
}
