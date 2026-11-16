#!/usr/bin/env bash
# E8 level variant queue (prereg v2 amendment 2): start e8_level_queue.py detached (setsid nohup).
# DRY_RUN=1 prints the server plan, placeholder stops, job commands and the post chain only. Checked with bash -n and DRY_RUN=1 only.
#   [DRY_RUN=1] bash experiments/s2st_baselines_v1/xtranslator_multispk/start_e8_level.sh
set -euo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
Q=experiments/s2st_baselines_v1/xtranslator_multispk/e8_level_queue.py
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
mkdir -p logs/mist_w3po_v1/e8_level
if [ "${DRY_RUN:-0}" = 1 ]; then
  PYTHONPATH=. $PYE $Q --dry-run
  exit 0
fi
TS=$(date -u +%Y%m%dT%H%M%SZ)
L=logs/mist_w3po_v1/e8_level/e8_level_queue_stdout_$TS.log
[ -e $L ] && { echo "拒绝覆盖已有输出: $L"; exit 1; }
setsid nohup env PYTHONPATH=. $PYE -u $Q > $L 2>&1 < /dev/null &
echo "E8 level queue pid $! log $L"
