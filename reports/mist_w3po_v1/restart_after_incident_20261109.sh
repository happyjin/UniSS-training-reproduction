#!/usr/bin/env bash
# One-step restart after the 2026-11-09 04:48-04:53 UTC node clear -- ONLY after the user approves
# (reports/mist_w3po_v1/incident_20261109_node_clear*.txt). Starts the three W3PO daemons detached (setsid nohup);
# gpuq_w3 then retries rl_pilot_ecf_v8_b01_w128 / rl_pilot_hica_v8_b01_w128 (rc 1, retries 20) with RESUME=1
# (run_rl.sh: the out dir exists -> --resume from ckpt step_00031 / step_00014), and the queued evals follow.
# Refuses if any daemon is already running.
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
Q=logs/mist_w3po_v1/queue_v2
SD=experiments/mist_w3po_v1/scripts
for s in gpuq_w3 idle_fallback yield_guard; do
  if ps -eo args | awk -v s="$SD/$s.py" '$1 ~ /python$/ && $2 == s {f=1} END {exit !f}'; then
    echo "refuse: $s already running" >&2; exit 1
  fi
done
[ -e "$Q/NO_RESURRECT" ] && echo "note: $Q/NO_RESURRECT exists (yield_guard will not resurrect)" >&2
setsid nohup "$PY" $SD/gpuq_w3.py $Q >> $Q/queue.log 2>&1 < /dev/null &
echo "gpuq_w3 pid $!"
sleep 5
setsid nohup "$PY" $SD/idle_fallback.py $Q >> $Q/idle_fallback.log 2>&1 < /dev/null &
echo "idle_fallback pid $!"
sleep 5
setsid nohup "$PY" $SD/yield_guard.py $Q >> $Q/yield_guard.log 2>&1 < /dev/null &
echo "yield_guard pid $!"
