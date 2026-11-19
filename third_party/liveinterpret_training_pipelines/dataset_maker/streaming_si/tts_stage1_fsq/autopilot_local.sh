#!/bin/bash
# 本地侧自动驾驶（nohup 后台跑）：
#   1) QA 报告一生成就拉回本地
#   2) 过滤完成（FILTERED_READY.json）后立即提交 stage1 S1 训练（4 台 p5e）
# 用法：nohup bash autopilot_local.sh > /tmp/autopilot_local.log 2>&1 &
set -u
REPO=/Users/stevehguo/Documents/coding-project/liveinterpret_training_pipelines
LOCAL_DIR=/Users/stevehguo/Documents/coding-project/TTS_FSQ_QA
REGION=ap-southeast-3
SRC=s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS_FSQ/_QA
YAML=ktrain/ktrainjob_streaming_si_s1_4node_filtered.yaml
mkdir -p "$LOCAL_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 1) 拉回 QA 报告 --------------------------------------------------------
for i in $(seq 1 720); do   # 最多等 12h
    if aws s3 ls "$SRC/report.md" --region "$REGION" >/dev/null 2>&1; then
        aws s3 cp "$SRC/report.md" "$LOCAL_DIR/report.md" --region "$REGION" >/dev/null
        aws s3 cp "$SRC/report.json" "$LOCAL_DIR/report.json" --region "$REGION" >/dev/null 2>&1 || true
        aws s3 cp "$SRC/utmos.json" "$LOCAL_DIR/utmos.json" --region "$REGION" >/dev/null 2>&1 || true
        log "已拉回报告 -> $LOCAL_DIR"
        break
    fi
    sleep 60
done

# ---- 2) 等过滤完成 ----------------------------------------------------------
for i in $(seq 1 720); do
    if aws s3 ls "$SRC/FILTERED_READY.json" --region "$REGION" >/dev/null 2>&1; then
        aws s3 cp "$SRC/FILTERED_READY.json" "$LOCAL_DIR/FILTERED_READY.json" \
            --region "$REGION" >/dev/null
        log "过滤完成：$(cat "$LOCAL_DIR/FILTERED_READY.json" | tr -d '\n')"
        break
    fi
    sleep 60
done

# ---- 3) 提交 stage1 训练 ----------------------------------------------------
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
log "开始提交训练：$YAML"
bash .codebuddy/training-ops-plugin/skills/ktrain-job-ops/scripts/submit.sh \
    --yaml "$YAML" \
    --repo . \
    --launcher ktrain/launch_train.sh \
    --aws-creds-secret aws-local-creds \
    --job-name-prefix si-s1 \
    --task-deadline 'si-s1-data=21600' 2>&1 | tee /tmp/autopilot_submit.log
log "提交流程结束，退出码 ${PIPESTATUS[0]}"

# ---- 4) 巡检训练状态 --------------------------------------------------------
for i in $(seq 1 480); do   # 最多再盯 8h
    sleep 300
    st=$(kubectl -n speech-h200 get ktrainjobs --no-headers 2>/dev/null | grep '^si-s1' | head -3)
    log "训练状态: ${st:-（无 si-s1 任务）}"
done
