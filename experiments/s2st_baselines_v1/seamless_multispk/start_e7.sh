#!/usr/bin/env bash
# E7 (reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md): start e7_queue.py detached. DRY_RUN=1 prints the
# job commands and the current card pick only. Checked with bash -n and DRY_RUN=1 only.
#   [DRY_RUN=1] bash experiments/s2st_baselines_v1/seamless_multispk/start_e7.sh
set -euo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
Q=experiments/s2st_baselines_v1/seamless_multispk/e7_queue.py
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
mkdir -p logs/mist_w3po_v1/e7
if [ "${DRY_RUN:-0}" = 1 ]; then
  PYTHONPATH=. $PYE $Q --dry-run
  exit 0
fi
TS=$(date -u +%Y%m%dT%H%M%SZ)
L=logs/mist_w3po_v1/e7/e7_queue_stdout_$TS.log
[ -e $L ] && { echo "拒绝覆盖已有输出: $L"; exit 1; }
setsid nohup env PYTHONPATH=. $PYE -u $Q > $L 2>&1 < /dev/null &
echo "E7 queue pid $! log $L"
