#!/usr/bin/env bash
# Idea 6 end-to-end training (reports/mist_w3po_v1/voicecond_idea6_bayes_prereg.zh-CN.md "端到端训练预登记"), one setting:
# 4-GPU DDP on cards 4-7, 16 units / GPU / step, 3000 steps; trainer experiments/mist_w3po_v1/voicecond/idea6_e2e_train.py.
# Modelled on experiments/mist_w3po_v1/scripts/idea2_e2e_train_run.sh:
#   1. claims training slot B (logs/mist_w3po_v1/slots/slotB.owner: who / task / utc / pid; sync_train_exclude.sh then puts
#      4 5 6 7 into logs/mist_w3po_v1/talker_crf_v1/exclude_cards, so no placeholder / eval job is placed there). An owner
#      file that already names this run ("run <RUN>", written by the same executor for the smoke test) is adopted;
#      any other owner -> refuse. Waits (<= 60 s) until the exclusion file lists 4 5 6 7.
#   2. waits until cards 4-7 hold no compute process other than placeholders (train_mist_workload.py); eval jobs are
#      never stopped, only awaited.
#   3. stops the placeholder supervisors (ppid 1, --devices k) and their children on 4-7 BY PID (cmdline checked).
#   4. torchrun --nproc_per_node 4; on exit releases the slot (owner renamed slotB.released_<UTC>) and writes
#      <out>/LAUNCHER_RC. Refuses an existing output directory.
#   [RUN=idea6_e2e_main_v1] [OUT=<dir>] [DRY_RUN=1] bash experiments/mist_w3po_v1/voicecond/idea6_e2e_train_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea6/e2e_<RUN>.launcher.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PYB=$J/conda_envs/uniss-eval/bin
CARDS=4,5,6,7
CL="4 5 6 7"
RUN=${RUN:-idea6_e2e_main_v1}
OUT=${OUT:-eval_outputs/mist_w3po_v1/idea6_e2e/$RUN}
LOGD=logs/mist_w3po_v1/voicecond/idea6
SL=logs/mist_w3po_v1/slots
EXCL=logs/mist_w3po_v1/talker_crf_v1/exclude_cards
PORT=${PORT:-29661}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
[ -e "$OUT" ] && { log "refuse: $OUT exists"; exit 1; }
mkdir -p $LOGD

uuid_of() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v k="$1" '$1 == k {print $2}'; }
apps_on() {
  local u
  u=$(uuid_of $1)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1 == u {print $2}'
}
non_placeholder_on() {
  local p a
  for p in $(apps_on $1); do
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$a" in *train_mist_workload.py*) ;; *) echo "$p:${a:0:120}";; esac
  done
}
free_card() {
  local k=$1 sup kids s a
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    a=$(tr '\0' ' ' < /proc/$s/cmdline 2>/dev/null)
    case "$a" in *train_mist_workload.py*"--devices $k "*) ;; *) log "card $k: pid $s cmdline changed, skipped"; continue;; esac
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
  done
}

OWN=$SL/slotB.owner
if [ -e $OWN ]; then
  if grep -q "run $RUN" $OWN; then log "slot B already claimed for run $RUN: adopted"
  else log "slot B owned by someone else: $(sed -n 1,2p $OWN | tr '\n' ' ')"; exit 1; fi
elif [ -n "${DRY_RUN:-}" ]; then log "[dry] would claim slot B"
else
  T=$SL/.slotB.owner.new_$$
  ( set -o noclobber; printf 'who: Idea 6 e2e training executor (voicecond, run %s)\ntask: Idea 6 end-to-end training (prereg 端到端训练预登记, 4-GPU DDP, 3000 steps) on cards %s; released by renaming to slotB.released_<UTC>\nutc: %s\npid: %s\n' \
    "$RUN" "$CARDS" "$(date -u +%FT%TZ)" "$$" > $T ) && command mv -n $T $OWN && [ ! -e $T ] || { log "slot claim failed"; exit 1; }
  log "claimed slot B"
fi
release() {
  [ -e "$OWN" ] && grep -q "run $RUN" "$OWN" && command mv -n "$OWN" $SL/slotB.released_$(date -u +%Y%m%dT%H%M%SZ) && log "slot B released"
}
if [ -z "${DRY_RUN:-}" ]; then
  for i in $(seq 1 12); do case " $(cat $EXCL 2>/dev/null) " in *" 4 5 6 7 "*) break;; esac; sleep 5; done
  case " $(cat $EXCL 2>/dev/null) " in *" 4 5 6 7 "*) log "exclusion: '$(cat $EXCL)'";; *) log "exclusion file does not list 4-7: '$(cat $EXCL 2>/dev/null)'"; release; exit 1;; esac
fi

log "waiting until cards $CL hold no eval / non-placeholder process"
while :; do
  busy=""
  for k in $CL; do b=$(non_placeholder_on $k); [ -n "$b" ] && busy="$busy [card $k: $(echo $b | tr '\n' ' ')]"; done
  [ -z "$busy" ] && break
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] busy:$busy"; break; fi
  log "busy:$busy"
  sleep 60
done
for k in $CL; do free_card $k; done
if [ -z "${DRY_RUN:-}" ]; then
  for i in $(seq 1 60); do
    left=""; for k in $CL; do [ -n "$(apps_on $k)" ] && left="$left $k"; done
    [ -z "$left" ] && break
    sleep 5
  done
  [ -n "$left" ] && { log "cards still hold processes after placeholder stop:$left"; release; exit 1; }
fi
CMD="env TZ=Europe/Prague OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$CARDS \
$PYB/torchrun --nproc_per_node 4 --master_port $PORT -m experiments.mist_w3po_v1.voicecond.idea6_e2e_train --out $OUT"
log "launch: $CMD"
if [ -n "${DRY_RUN:-}" ]; then log "[dry] done"; exit 0; fi
bash -c "$CMD" > $LOGD/e2e_$RUN.log 2>&1
rc=$?
log "torchrun exit $rc (log $LOGD/e2e_$RUN.log)"
[ -d $OUT ] && echo "$rc $(date -u +%FT%TZ)" >> $OUT/LAUNCHER_RC
release
exit $rc
