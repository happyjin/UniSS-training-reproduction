#!/bin/bash
# 把集群产出的 stage1 质量评估报告拉回本地
# 用法：bash pull_report.sh [本地目录]
set -eu
DIR=${1:-/Users/stevehguo/Documents/coding-project/TTS_FSQ_QA}
REGION=ap-southeast-3
SRC=s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS_FSQ/_QA

mkdir -p "$DIR"
for f in report.md report.json utmos.json; do
    if aws s3 ls "$SRC/$f" --region "$REGION" >/dev/null 2>&1; then
        aws s3 cp "$SRC/$f" "$DIR/$f" --region "$REGION" >/dev/null
        echo "已拉取: $DIR/$f"
    else
        echo "尚未生成: $f"
    fi
done
echo "完成 -> $DIR"
