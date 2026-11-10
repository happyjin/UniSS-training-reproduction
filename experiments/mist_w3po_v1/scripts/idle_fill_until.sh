#!/usr/bin/env bash
# Per-card idle filler that re-arms itself until a stop file exists (user rule 2026-11-10: no idle GPU, filler must be the
# no-checkpoint training workload). One loop per card; the loop's own PID goes to $DIR/card<k>.pid so
# start_v2p_after_c0eval.sh sees the loop + its supervisor as one workload tree and stops it by PID.
# The supervisor (run_mist_workload.sh --wait-idle) steps aside and exits when a foreign GPU process appears during its
# load; this loop then waits RESPAWN_S and arms a new one, unless STOP_FILE exists.
#   DIR=logs/mist_w3po_v1/workload_at_<UTC> STOP_FILE=<run>/events/DONE CAP_MIB=80000 \
#     setsid nohup bash experiments/mist_w3po_v1/scripts/idle_fill_until.sh <card> > $DIR/loop<card>.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
k="${1:?card index}"
DIR="${DIR:?set DIR}"; STOP_FILE="${STOP_FILE:?set STOP_FILE}"
CAP_MIB="${CAP_MIB:-80000}"; RESPAWN_S="${RESPAWN_S:-30}"
echo $$ > "$DIR/card$k.pid"
n=0
while [ ! -e "$STOP_FILE" ]; do
  n=$((n + 1))
  echo "$(date -u +%FT%TZ) card $k: arm #$n (cap $CAP_MIB MiB)"
  bash scripts/gpu_load/run_mist_workload.sh --devices "$k" --wait-idle --idle-seconds 30 --poll-seconds 5 \
    --max-used-mib "$CAP_MIB" >> "$DIR/card$k.log" 2>&1 < /dev/null 9>&-
  echo "$(date -u +%FT%TZ) card $k: supervisor exited rc=$?"
  sleep "$RESPAWN_S"
done
echo "$(date -u +%FT%TZ) card $k: stop file present, loop done"
