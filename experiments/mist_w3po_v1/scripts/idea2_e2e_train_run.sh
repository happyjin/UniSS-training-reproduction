#!/usr/bin/env bash
# Idea 2 end-to-end training, MAIN setting only (reports/mist_w3po_v1/voicecond_idea2_e2e_prereg.zh-CN.md v1 §6):
# 4-GPU DDP (torchrun), 64 sequences / step, 3000 steps; trainer experiments/mist_w3po_v1/voicecond/idea2_e2e_train.py.
#   1. CARDS=4,5,6,7 (default): claims training slot B (logs/mist_w3po_v1/slots/slotB.owner; sync_train_exclude.sh then puts
#      4 5 6 7 into logs/mist_w3po_v1/talker_crf_v1/exclude_cards, so no orchestrator / guard places a NEW job there).
#      Other card sets: no slot claim (the exclusion file is owned by sync_train_exclude.sh and would be overwritten).
#   2. waits until every training card holds no compute process other than guard placeholders (train_mist_workload.py);
#      eval jobs are NEVER stopped, only awaited (user rule: nothing shares a card with the training).
#   3. stops the placeholder supervisors + children on those cards BY PID (cmdline checked; as idea2_teacher_chain_v6.sh
#      free_card, local loop variables), waits for their memory to be released.
#   4. torchrun --nproc_per_node 4; on exit releases the slot (owner file renamed slotB.released_<UTC>) and writes
#      <out>/LAUNCHER_RC. Refuses an existing output directory.
# v2 (2026-11-19 ~08:00 Prague): CARDS=0,1,2,3 claims slot A the same way; OUT / EXTRA (extra trainer flags, debug runs).
#   [CARDS=4,5,6,7] [RUN=idea2_e2e_main_v1] [OUT=<dir>] [EXTRA=<flags>] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_e2e_train_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/e2e_<RUN>.launcher.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
VB=$J/venvs/vllm_omni_030/bin
CARDS=${CARDS:-4,5,6,7}
RUN=${RUN:-idea2_e2e_main_v1}
OUT=${OUT:-eval_outputs/mist_w3po_v1/voicecond_idea2/e2e/$RUN}
EXTRA=${EXTRA:-}
LOGD=logs/mist_w3po_v1/voicecond/idea2
SL=logs/mist_w3po_v1/slots
PORT=${PORT:-29651}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
[ -e "$OUT" ] && { log "refuse: $OUT exists"; exit 1; }
mkdir -p $LOGD
CL=$(echo $CARDS | tr ',' ' ')
NP=$(echo $CL | wc -w)
[ "$NP" = 4 ] || { log "need 4 cards, got '$CARDS'"; exit 1; }

uuid_of() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v k="$1" '$1 == k {print $2}'; }
apps_on() {   # pids of compute processes on card $1
  local u
  u=$(uuid_of $1)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1 == u {print $2}'
}
non_placeholder_on() {   # compute processes on card $1 that are not guard placeholders
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

OWN=""; SLOT=""
[ "$CARDS" = "4,5,6,7" ] && SLOT=B
[ "$CARDS" = "0,1,2,3" ] && SLOT=A
if [ -n "$SLOT" ]; then
  if [ -e $SL/slot$SLOT.owner ]; then log "slot $SLOT already owned: $(sed -n 1,2p $SL/slot$SLOT.owner | tr '\n' ' ')"; exit 1; fi
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] would claim slot $SLOT ($SL/slot$SLOT.owner)"
  else
    T=$SL/.slot$SLOT.owner.new_$$
    ( set -o noclobber; printf 'who: Idea 2 e2e training executor (voicecond, run %s)\ntask: Idea 2 end-to-end training (prereg v1 main setting, 4-GPU DDP, 3000 steps) on cards %s; released by renaming to slot%s.released_<UTC>\nutc: %s\n' \
      "$RUN" "$CARDS" "$SLOT" "$(date -u +%FT%TZ)" > $T ) && command mv -n $T $SL/slot$SLOT.owner && [ ! -e $T ] || { log "slot claim failed"; exit 1; }
    OWN=$SL/slot$SLOT.owner
    log "claimed slot $SLOT; exclude_cards will include cards $CL within 10 s"
  fi
fi
release() {
  [ -n "$OWN" ] && [ -e "$OWN" ] && grep -q "run $RUN" "$OWN" && command mv -n "$OWN" $SL/slot$SLOT.released_$(date -u +%Y%m%dT%H%M%SZ) && log "slot $SLOT released"
}

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
  [ -n "$left" ] && { log "cards still hold processes after placeholder stop:$left"; for k in $left; do log "card $k: $(apps_on $k | tr '\n' ' ')"; done; release; exit 1; }
fi
CMD="env TZ=Europe/Prague TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
PATH=$VB:$PATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$CARDS $VB/torchrun --nproc_per_node 4 --master_port $PORT \
-m experiments.mist_w3po_v1.voicecond.idea2_e2e_train --out $OUT --ablation main $EXTRA"
log "launch: $CMD"
if [ -n "${DRY_RUN:-}" ]; then log "[dry] done"; exit 0; fi
bash -c "$CMD" > $LOGD/e2e_$RUN.log 2>&1
rc=$?
log "torchrun exit $rc (log $LOGD/e2e_$RUN.log)"
[ -d $OUT ] && echo "$rc $(date -u +%FT%TZ)" >> $OUT/LAUNCHER_RC
release
exit $rc
