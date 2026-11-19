#!/usr/bin/env bash
# SimulS2ST-Omni 流式同传推理（SimulEval agent，迁移自源 scripts/run_infer.sh，参数与源相同）。
# 先把 Stage 2 的 adapter 合进 offline 基座，再加载 Stage 3 的 adapter。
#
#   SOURCE=source.txt TARGET=target.txt bash pipelines/simuls2st_omni/run_infer.sh
#
#   BASE     offline 基座（缺省 models/SimulS2ST-Omni/offline）
#   MERGE    Stage 2 产出（swift 训练的 output_dir 根目录或 checkpoint-N，或源项目的 stage2_multitask_full）
#   CKPT     Stage 3 产出（同上；或源项目的 stage3_mix_v2）
#   SOURCE   每行一条 wav 路径；TARGET 每行一条参考译文
#   SIMULS2ST_MODEL_ROOT  含 voicebox/ dualcodec/ w2v/ 的发布仓库目录（缺省 models/SimulS2ST-Omni）
#
# 依赖：仓库 requirements.txt 之外还要 custom/simuls2st_omni/requirements-infer.txt。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/custom:${REPO_ROOT}/vendor/SimulEval${PYTHONPATH:+:${PYTHONPATH}}"
export SIMULS2ST_MODEL_ROOT="${SIMULS2ST_MODEL_ROOT:-${REPO_ROOT}/models/SimulS2ST-Omni}"

BASE="${BASE:-${SIMULS2ST_MODEL_ROOT}/offline}"
CKPT="${CKPT:-models/Qwen2.5_3B_DualCodec/stage3_mix_v2}"
MERGE="${MERGE:-models/Qwen2.5_3B_DualCodec/stage2_multitask_full}"
SOURCE="${SOURCE:?set SOURCE to a utf-8 file with one wav path per line}"
TARGET="${TARGET:?set TARGET to a utf-8 file with one reference line per wav}"
OUT="${OUT:-outputs/simuls2st_infer}"
SRC_LANG="${SRC_LANG:-English}"
TGT_LANG="${TGT_LANG:-Chinese}"
DEVICE="${DEVICE:-cuda:0}"
M="${M:-2}"
PYTHON="${PYTHON:-python}"

if [[ "$TGT_LANG" == "Chinese" ]]; then
  TOK=zh
  UNIT=char
else
  TOK=13a
  UNIT=word
fi

mkdir -p "$OUT"
"${PYTHON}" -m simuleval.cli \
  --agent custom/simuls2st_omni/agents/simuleval_omni_talker_s2st_agent.py \
  --source-segment-size 1000 \
  --model-name-or-path "$BASE" \
  --merge-adapters-from "$MERGE" \
  --checkpoint-path "$CKPT" \
  --source "$SOURCE" \
  --target "$TARGET" \
  --output "$OUT" \
  --source-lang "$SRC_LANG" --target-lang "$TGT_LANG" \
  --latency-multiplier "$M" \
  --sacrebleu-tokenizer "$TOK" --eval-latency-unit "$UNIT" \
  --history-window-turns 28 --history-overlap-turns 16 \
  --prompt-source-chunks 2 --prompt-generated-chunks 1 \
  --thinker-max-new-tokens 256 --talker-max-new-tokens 500 \
  --thinker-no-sample --talker-no-sample \
  --thinker-num-beams 4 \
  --thinker-repetition-penalty 1.2 --thinker-no-repeat-ngram-size 5 \
  --talker-repetition-penalty 1.4 --talker-no-repeat-ngram-size 5 \
  --computation-aware \
  --no-scoring \
  --device "$DEVICE" \
  "$@"
