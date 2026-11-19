#!/usr/bin/env bash
# streaming_si S5：chunk-as-turn 流式同传（640 ms 基础块，每块后 等/说 决策），LoRA 训练。
#
# 起点：S4 的 checkpoint（MODEL_PATH）。训练 Thinker/Talker LoRA + 4 行流式 token（wait/W/chunk/src_end，
# 与 lm_head 共享）+ 全量 talker.thinker_proj / talker.fusion；只保存 adapter（s4_peft.py）。
# audio_proj 与 text_proj 冻结。
#
# 本阶段即源项目的 S4：变量（SI_S4_* / S4_ARM / S4_P_DROP）、数据集名（streaming_si_s4）、
# tuner（streaming_si_s4）与 s4_*.py 模块都沿用源项目命名，指的就是本阶段。
#
# 数据准备（一次性；产出 val 划分与分层索引，均在本仓库 dataset_maker/streaming_si/）：
#   python dataset_maker/streaming_si/build_s4_val_split.py --records $SI_S4/records/en2zh/chunk_640ms/val/records \
#       --direction en2zh --n_val 1500 --out $VAL_DIR --data_root $SI_S4/audio      # zh2en 同理
#   python dataset_maker/streaming_si/build_record_index.py --records $SI_S4/records/en2zh/chunk_640ms/train/records \
#       --direction en2zh --data_root $SI_S4/audio --out $INDEX_DIR --workers 16     # zh2en 同理
#
# 用法：
#   MODEL_PATH=<S4 ckpt> SI_S4=/data/si_s4 SI_S4_VAL_DIR=... SI_S4_INDEX_DIR=... bash pipelines/streaming_si/train_simul.sh
#   S4_ARM=prefix_drop S4_P_DROP=0.25 ...  bash pipelines/streaming_si/train_simul.sh   # H1 实验臂
#   续训：追加 --resume_from_checkpoint <ckpt>；只加载 adapter 权重从 0 步开始：追加 --adapters <adapter 目录>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 数据（源 train_s4_stream.sh：两方向 records 联合训练）----
: "${CHUNK_MS:=640}"
if [[ -z "${SI_S4_DATA:-}" ]]; then
    : "${SI_S4:?需要 SI_S4（si_s4 根目录）或显式 SI_S4_DATA}"
    SI_S4_DATA="${SI_S4}/records/en2zh/chunk_${CHUNK_MS}ms/train/records,${SI_S4}/records/zh2en/chunk_${CHUNK_MS}ms/train/records"
    : "${SI_S4_DATA_ROOT:=${SI_S4}/audio}"
fi
: "${SI_S4_DATA_ROOT:?需要 SI_S4_DATA_ROOT（记录中相对音频路径的根，si_s4/audio）}"
: "${SI_S4_VAL_DIR:=}"
: "${SI_S4_INDEX_DIR:=}"
export SI_S4_DATA SI_S4_DATA_ROOT SI_S4_VAL_DIR SI_S4_INDEX_DIR

# ---- 损失（源 COMMON：text 1 / speech 1 / decision 1）----
export SI_TEXT_LOSS_WEIGHT=1
export SI_SPEECH_LOSS_WEIGHT=1
export SI_S4_DECISION_WEIGHT=1        # 调参区间 [0.5, 2.0]；5 会冲掉翻译（源 docs/S4_TODO.md §2）
export SI_COND_MODE=full
export SI_HIDDEN_NOISE=0.0

# ---- 样本口径（源 COMMON + 默认值）----
# 与源项目的差异：源音频上限 20s -> 30s（与 S2/S4 一致），序列上限按比例 640 -> 1024
# （audio_pad 12.5/s、<|chunk|> 每 640ms 一个、WAIT/W 与译文都随时长线性增长：640×1.5=960，取整到 1024）。
# 源项目 S4 本来就是超长跳过、不截断（s4_streaming_dataset: duration_out_of_range / too_long）。
# speech token 上限 750（30s @25Hz）不变。
: "${MAX_LENGTH:=1024}"
export SI_MAX_LENGTH="${MAX_LENGTH}"
: "${SI_MAX_AUDIO_SEC:=30}"
export SI_MAX_AUDIO_SEC
export SI_MAX_SPEECH_LEN=750
export SI_S4_USE_SRC_END=1
export SI_S4_STRICT_TEXT=0
export SI_S4_LATENCY_MULTIPLIER=1     # 'sample' 为论文的延迟倍数随机抽样（M_MAX / M_SCHEDULE 生效）
export SI_S4_M_MAX=4
export SI_S4_M_SCHEDULE=uniform
export SI_S4_STRATIFIED=1             # NIR × 句长分层加权采样（需 SI_S4_INDEX_DIR）
export SI_S4_MAX_REPEAT=2.0
export SI_S4_MIN_SOURCE_MS=0
export SI_S4_SEED=1337
export SI_S4_TRAIN_SCOPE=joint
export SI_S4_P_DROP=0
export SI_S4_PAIRED_CONTROL=0
export SI_S4_PREFIX_CONTEXT_TURNS=16

# ---- 实验臂（源 --p_drop / --paired_crop_control；约束：scope=thinker、speech_w=0、m=1）----
: "${S4_ARM:=default}"
case "${S4_ARM}" in
    default) ;;
    prefix_drop|prefix_drop_control)
        : "${S4_P_DROP:?S4_ARM=${S4_ARM} 需要 S4_P_DROP（Thinker crop 指定比例，(0,1]）}"
        export SI_S4_P_DROP="${S4_P_DROP}" SI_S4_TRAIN_SCOPE=thinker SI_SPEECH_LOSS_WEIGHT=0
        [[ "${S4_ARM}" == prefix_drop_control ]] && export SI_S4_PAIRED_CONTROL=1
        ;;
    *) echo "[ERROR] 未知 S4_ARM=${S4_ARM}（default | prefix_drop | prefix_drop_control）" >&2; exit 1 ;;
esac

# ---- 优化（源 COMMON：LoRA r8/α16/dropout0，3e-5 -> 2e-6 余弦，BS 48/GPU，1 epoch）----
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=48}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${LEARNING_RATE:=3e-5}"
: "${MIN_LR:=2e-6}"
: "${NUM_TRAIN_EPOCHS:=1}"
: "${SAVE_STEPS:=500}"
: "${VAL_INTERVAL:=2000}"
: "${VAL_BATCHES:=20}"
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0

source "${SCRIPT_DIR}/common_args.sh"

WORLD=$(( ${NPROC_PER_NODE:-1} * ${NNODES:-1} ))
export SI_S4_GLOBAL_BATCH=$(( PER_DEVICE_TRAIN_BATCH_SIZE * WORLD * GRADIENT_ACCUMULATION_STEPS ))
export SI_S4_MAX_STEPS="${MAX_STEPS}"
# 源 evaluate：每个方向取前 val_batches × batch_size × world 条
export SI_S4_VAL_LIMIT=$(( VAL_BATCHES * PER_DEVICE_TRAIN_BATCH_SIZE * WORLD ))

# --strict true：样本编码失败即中止，与源一致（源 __getitem__ 有界扫描 / 视图替补耗尽后 raise）。
# swift 默认会吞掉异常并随机换一行，这会静默破坏 prefix-drop 的精确 crop 配额与调度。
EXTRA=(--seed "${SI_S4_SEED}" --dataloader_drop_last true --strict true)
if compgen -G "${SI_S4_VAL_DIR:-/nonexistent}/*_val_keys.json" > /dev/null; then
    EVAL_STRATEGY=steps
    [[ "${SCENE}" == smoke ]] && VAL_INTERVAL="${SAVE_STEPS}"
    # 主判据为决策 macro-F1（两类严重不平衡，acc 会把「恒 W」选成 best）
    EXTRA+=(--val_dataset streaming_si_s4_val --eval_steps "${VAL_INTERVAL}"
            --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
            --metric_for_best_model dec_macro_f1 --greater_is_better true)
else
    echo "[streaming_si][WARN] SI_S4_VAL_DIR 下没有 *_val_keys.json —— 本轮没有 held-out 指标，也不会剔除 val 源文本组" >&2
fi
if [[ "${SI_S4_STRATIFIED}" == 1 || "${SI_S4_P_DROP}" != 0 ]]; then
    [[ -n "${SI_S4_INDEX_DIR}" ]] || { echo "[ERROR] 分层采样 / prefix-drop 需要 SI_S4_INDEX_DIR" >&2; exit 1; }
fi
echo "[streaming_si] S5 arm=${S4_ARM} scope=${SI_S4_TRAIN_SCOPE} p_drop=${SI_S4_P_DROP} decision_w=${SI_S4_DECISION_WEIGHT}" \
     "global_batch=${SI_S4_GLOBAL_BATCH} val_limit/dir=${SI_S4_VAL_LIMIT}" >&2

run_streaming_si streaming_si_s4 "s5_simul_${S4_ARM}_${CHUNK_MS}" stream_lora "${EXTRA[@]}" "$@"
