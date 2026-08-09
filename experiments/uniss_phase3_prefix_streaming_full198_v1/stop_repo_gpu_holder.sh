#!/usr/bin/env bash
set -euo pipefail
pattern='/opt/dlami/nvme/jasonleeeli/projects/UniSS/scripts/gpu_load/target_gpu_util.py'
mapfile -t pids < <(pgrep -f "${pattern}" || true)
if (( ${#pids[@]} == 0 )); then
  echo "No repository GPU holder is running"
  exit 0
fi
printf 'Stopping repository GPU holder PIDs:'
printf ' %s' "${pids[@]}"
printf '\n'
kill -TERM "${pids[@]}"
for _ in $(seq 1 30); do
  remaining=()
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && remaining+=("${pid}")
  done
  (( ${#remaining[@]} == 0 )) && exit 0
  sleep 1
done
echo "GPU holder did not exit after 30 seconds" >&2
exit 1

