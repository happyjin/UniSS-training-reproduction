#!/usr/bin/env bash
# Idea 1 (ZNE): run ONE small GPU command (BiCodec encode / decode, TitaNet) on a slot-A card (0-3): checks that the card
# has >= NEED_MIB MiB free after its no-checkpoint placeholder (train_mist_workload supervisor + children) is stopped by
# PID, then runs the command with CUDA_VISIBLE_DEVICES=<card>; writes <marker> (rc in it) when it exits. The guard refills
# the card afterwards. Never stops the guard / gpu_watch, never uses pattern kills.
#   [DRY_RUN=1] [NEED_MIB=20000] bash experiments/mist_w3po_v1/scripts/idea1_gpu_run.sh <card> <marker> <command...>
# Run detached: setsid nohup bash ... > <log> 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
c=$1; MARK=$2; shift 2
NEED_MIB=${NEED_MIB:-20000}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
case " 0 1 2 3 " in *" $c "*) ;; *) log "card $c not in slot A (0-3): refuse"; exit 1;; esac
[ -e "$MARK" ] && { log "marker $MARK exists: refuse"; exit 1; }
sup=$(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2==1 && index($0,"train_mist_workload.py") && index($0,c) {print $1}')
kids=""; [ -n "$sup" ] && kids=$(ps -eo pid=,ppid= | awk -v p="$sup" '$2==p {print $1}')
if [ -n "${DRY_RUN:-}" ]; then log "[dry] card $c: would stop placeholder ${sup:-none} (children $kids); then: $*"; exit 0; fi
if [ -n "$sup" ]; then
  kill -TERM $sup $kids 2>/dev/null
  log "card $c: stopped placeholder $sup (children $kids)"
  for _ in $(seq 1 60); do [ -z "$(ps -o pid= -p $sup $kids 2>/dev/null)" ] && break; sleep 1; done
fi
for _ in $(seq 1 30); do
  free=$(nvidia-smi -i $c --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
  [ "${free:-0}" -ge "$NEED_MIB" ] && break; sleep 2
done
[ "${free:-0}" -ge "$NEED_MIB" ] || { log "card $c: only ${free} MiB free"; echo "rc=nomem" > "$MARK"; exit 1; }
log "card $c: ${free} MiB free; start: $*"
CUDA_VISIBLE_DEVICES=$c "$@"
rc=$?
echo "rc=$rc" > "$MARK"
log "exit rc=$rc"
