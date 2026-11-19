#!/usr/bin/env bash
# =============================================================================
# training-ops / pytorchjob-ops / estimate_mfu.sh
#
# 通用 MFU（Model FLOPs Utilization）估算脚本（纯 kubectl，不依赖特定集群/项目）。
#
# 适用场景：ms-swift 封装的 Megatron 训练日志里【没有】原生的 "elapsed time per
# iteration (ms)" / TFLOP 字段（tqdm + wandb 上报），无法直接读 MFU。本脚本用
# 【带时间戳的 checkpoint 保存日志】反推每步平均耗时，再结合模型规模/序列长/有效
# batch/DP/单卡峰值算力，估算 MFU。
#
# 核心公式：
#   step_sec        = Δt(相邻两次 checkpoint 保存) / SAVE_STEPS
#   tokens_per_step = GLOBAL_BATCH_SIZE × SEQ_LEN
#   flops_per_step  = FACTOR × PARAM_COUNT × tokens_per_step   (FACTOR=6：fwd+bwd)
#   per_gpu_flops_s = flops_per_step / step_sec / DP           (DP=数据并行度=总GPU数/TP/PP/CP)
#   MFU             = per_gpu_flops_s / GPU_PEAK_FLOPS
#
# ⚠️ 估算上界说明：FACTOR=6×N 假设【全参】前反向。若训练 freeze 了主干（如仅训
#    bridge/talker），真实可训练 FLOPs 更小，实际 MFU 会低于本脚本上界值。脚本会
#    同时输出一个"上界 MFU"，供判断喂数瓶颈用（data-starved 时 MFU 远低于上界）。
# =============================================================================
# 用法：
#   bash estimate_mfu.sh <JOB_NAME> [NAMESPACE] [选项通过环境变量]
#     JOB_NAME    必填，PyTorchJob 名称
#     NAMESPACE   可选，k8s 命名空间；也可用环境变量 K8S_NS 传入
#
#   可覆盖环境变量（不传则尝试从日志自动解析 / 用默认）：
#     GLOBAL_BATCH_SIZE   有效 batch（=MBS×DP×grad_accum）。默认从日志解析，失败需手传
#     SEQ_LEN             序列长（= max_length）。默认从日志 "max_length: N" 解析
#     PARAM_COUNT         模型参数量（个）。默认 3e9（3B）；MoE 用激活参数量
#     TOTAL_GPUS          总 GPU 数。默认从 pod 数 × 每 pod GPU 解析，失败默认 16
#     TP / PP / CP        并行度，用于算 DP=TOTAL_GPUS/(TP·PP·CP)。默认从日志解析或 1
#     GPU_PEAK_FLOPS      单卡峰值算力(FLOP/s)。默认 989e12（H200 bf16 稠密）
#     SAVE_STEPS          相邻 checkpoint 间隔步数。默认从日志 "eval_steps/save" 或 checkpoint-N 差值解析
#     FLOP_FACTOR         FLOPs 系数。默认 6（fwd+bwd 全参上界）
#
# 输出（stdout，KEY=VALUE）：
#   STEP_SEC / TOKENS_PER_STEP / FLOPS_PER_STEP / PER_GPU_TFLOPS /
#   MFU_UPPER_BOUND_PCT / SAMPLES_PER_SEC / DP / NOTES
# =============================================================================
# 注：不使用 set -e —— 本脚本大量用 grep/date 解析日志，无匹配时返回非零属正常，
# 已用 `|| echo 默认值` 和 :- 兜底；set -e 会在这些解析处误杀脚本。
set -uo pipefail

JOB_NAME="${1:-}"
NS="${2:-${K8S_NS:-}}"

if [[ -z "${JOB_NAME}" || -z "${NS}" ]]; then
  echo "usage: estimate_mfu.sh <JOB_NAME> [NAMESPACE]" >&2
  echo "  (NAMESPACE 也可通过环境变量 K8S_NS 传入)" >&2
  exit 2
fi

OUT="/tmp/pytorchjob-mfu/${JOB_NAME}"
mkdir -p "${OUT}"
LOG="${OUT}/logs.txt"

echo "[mfu] job=${JOB_NAME} ns=${NS}"
echo "[mfu] out=${OUT}"

# ---- 1. 拉取带时间戳的 master 日志 ----
MASTER_POD=$(kubectl -n "${NS}" get pods -l "training.kubeflow.org/job-name=${JOB_NAME}" -o name 2>/dev/null | grep master-0 | head -1 || echo "")
if [[ -z "${MASTER_POD}" ]]; then
  echo "ERROR: master-0 pod not found for job ${JOB_NAME} in ns ${NS}" >&2
  exit 3
fi
kubectl -n "${NS}" logs "${MASTER_POD}" -c pytorch --timestamps > "${LOG}" 2>&1 || true
echo "[mfu] log lines: $(wc -l < "${LOG}")"

# ---- 2. 解析参数（环境变量优先，否则从日志自动解析，最后用默认） ----
SEQ_LEN="${SEQ_LEN:-$(grep -oE 'max_length: [0-9]+' "${LOG}" | head -1 | grep -oE '[0-9]+' || echo '')}"
SEQ_LEN="${SEQ_LEN:-4096}"

# 并行度：从 MegatronSftArguments 解析 tensor/pipeline/context parallel size
TP="${TP:-$(grep -oE 'tensor_model_parallel_size=[0-9]+' "${LOG}" | head -1 | grep -oE '[0-9]+' || echo 1)}"
PP="${PP:-$(grep -oE 'pipeline_model_parallel_size=[0-9]+' "${LOG}" | head -1 | grep -oE '[0-9]+' || echo 1)}"
CP="${CP:-$(grep -oE 'context_parallel_size=[0-9]+' "${LOG}" | head -1 | grep -oE '[0-9]+' || echo 1)}"

# 总 GPU 数：pod 数 × 每 pod GPU（nvidia.com/gpu）。解析失败默认 16。
NUM_PODS=$(kubectl -n "${NS}" get pods -l "training.kubeflow.org/job-name=${JOB_NAME}" --no-headers 2>/dev/null | grep -cE 'master-0|worker-' || echo 0)
GPU_PER_POD=$(kubectl -n "${NS}" get pod "${MASTER_POD#pod/}" -o jsonpath='{.spec.containers[0].resources.limits.nvidia\.com/gpu}' 2>/dev/null || echo "")
GPU_PER_POD="${GPU_PER_POD:-8}"
if [[ "${NUM_PODS}" -gt 0 ]] 2>/dev/null; then
  TOTAL_GPUS="${TOTAL_GPUS:-$(( NUM_PODS * GPU_PER_POD ))}"
else
  TOTAL_GPUS="${TOTAL_GPUS:-16}"
fi

DP=$(( TOTAL_GPUS / (TP * PP * CP) ))
if [[ "${DP}" -lt 1 ]]; then DP=1; fi

# GBS：优先环境变量，否则从日志 global_batch_size 解析
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(grep -oE "'?global_batch_size'?[=: ]+[0-9]+" "${LOG}" | head -1 | grep -oE '[0-9]+' || echo '')}"

PARAM_COUNT="${PARAM_COUNT:-3000000000}"          # 3B 默认
GPU_PEAK_FLOPS="${GPU_PEAK_FLOPS:-989000000000000}" # H200 bf16 稠密 ~989 TFLOP/s
FLOP_FACTOR="${FLOP_FACTOR:-6}"

# ---- 3. SAVE_STEPS：相邻 checkpoint 的步数差（从 checkpoint-N 标记推断） ----
mapfile -t CKPT_STEPS < <(grep -aoE 'checkpoint-[0-9]+' "${LOG}" | grep -oE '[0-9]+' | awk '!seen[$0]++' | sort -n)
if [[ -z "${SAVE_STEPS:-}" && "${#CKPT_STEPS[@]}" -ge 2 ]]; then
  SAVE_STEPS=$(( CKPT_STEPS[1] - CKPT_STEPS[0] ))
fi
SAVE_STEPS="${SAVE_STEPS:-500}"

# ---- 4. 从带时间戳的 "Successfully saved Megatron" 行取相邻间隔，算 step_sec ----
mapfile -t SAVE_TS < <(grep -aE 'Successfully saved Megatron' "${LOG}" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+' | tail -6)
STEP_SEC=""
NOTES=""
if [[ "${#SAVE_TS[@]}" -ge 2 ]]; then
  # 用最后两段间隔的中位/平均；这里取相邻两两差的平均
  total=0; cnt=0; prev=""
  for ts in "${SAVE_TS[@]}"; do
    e=$(date -u -d "${ts}" +%s 2>/dev/null || echo "")
    if [[ -z "${e}" ]]; then continue; fi
    if [[ -n "${prev}" ]]; then
      d=$(( e - prev )); total=$(( total + d )); cnt=$(( cnt + 1 ))
    fi
    prev="${e}"
  done
  if [[ "${cnt}" -gt 0 ]]; then
    AVG_INTERVAL=$(( total / cnt ))
    STEP_SEC=$(awk -v a="${AVG_INTERVAL}" -v s="${SAVE_STEPS}" 'BEGIN{printf "%.4f", a/s}')
    NOTES="step_sec 由 ${cnt} 段 checkpoint 间隔平均得到（含 ckpt 保存/eval 开销，为上界耗时）"
  fi
else
  NOTES="checkpoint 保存日志不足 2 条，无法反推 step_sec；需等更多 checkpoint 或手传 STEP_SEC"
fi

# 允许手动覆盖 STEP_SEC
if [[ -n "${STEP_SEC_OVERRIDE:-}" ]]; then
  STEP_SEC="${STEP_SEC_OVERRIDE}"
  NOTES="step_sec 由 STEP_SEC_OVERRIDE=${STEP_SEC_OVERRIDE} 手动传入"
fi

# ---- 5. 计算 MFU ----
if [[ -z "${STEP_SEC}" || -z "${GLOBAL_BATCH_SIZE}" ]]; then
  {
    echo "================================================================="
    echo "  STRUCTURED OUTPUT (INCOMPLETE)"
    echo "================================================================="
    echo "STEP_SEC=${STEP_SEC:-N/A}"
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-N/A}"
    echo "SEQ_LEN=${SEQ_LEN}"
    echo "DP=${DP} (TOTAL_GPUS=${TOTAL_GPUS} / TP=${TP} PP=${PP} CP=${CP})"
    echo "SAVE_STEPS=${SAVE_STEPS}"
    echo "NOTES=${NOTES:-需手传 GLOBAL_BATCH_SIZE 或 STEP_SEC_OVERRIDE}"
    echo "================================================================="
  } | tee "${OUT}/mfu.txt"
  exit 0
fi

TOKENS_PER_STEP=$(( GLOBAL_BATCH_SIZE * SEQ_LEN ))
read -r FLOPS_PER_STEP PER_GPU_TFLOPS MFU_PCT SAMPLES_PER_SEC < <(awk \
  -v f="${FLOP_FACTOR}" -v n="${PARAM_COUNT}" -v tok="${TOKENS_PER_STEP}" \
  -v sec="${STEP_SEC}" -v dp="${DP}" -v peak="${GPU_PEAK_FLOPS}" \
  -v gbs="${GLOBAL_BATCH_SIZE}" 'BEGIN{
    fps = f * n * tok;
    per_gpu = fps / sec / dp;
    tflops = per_gpu / 1e12;
    mfu = per_gpu / peak * 100;
    sps = gbs / sec;
    printf "%.3e %.1f %.1f %.2f", fps, tflops, mfu, sps;
  }')

{
  echo "================================================================="
  echo "  STRUCTURED OUTPUT (MFU upper-bound estimate)"
  echo "================================================================="
  echo "JOB=${JOB_NAME}"
  echo "STEP_SEC=${STEP_SEC}"
  echo "SAVE_STEPS=${SAVE_STEPS}"
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "PARAM_COUNT=${PARAM_COUNT}"
  echo "TOTAL_GPUS=${TOTAL_GPUS}"
  echo "DP=${DP} (TP=${TP} PP=${PP} CP=${CP})"
  echo "GPU_PEAK_TFLOPS=$(awk -v p="${GPU_PEAK_FLOPS}" 'BEGIN{printf "%.0f", p/1e12}')"
  echo "TOKENS_PER_STEP=${TOKENS_PER_STEP}"
  echo "FLOPS_PER_STEP=${FLOPS_PER_STEP}"
  echo "SAMPLES_PER_SEC=${SAMPLES_PER_SEC}"
  echo "PER_GPU_TFLOPS=${PER_GPU_TFLOPS}"
  echo "MFU_UPPER_BOUND_PCT=${MFU_PCT}"
  echo "NOTES=${NOTES}"
  echo "WARN=MFU 为全参 6N 上界；若 freeze 主干则真实 MFU 更低。MFU 远低于上界通常=GPU data-starved（dataloader 喂数瓶颈）"
  echo "================================================================="
} | tee "${OUT}/mfu.txt"

exit 0
