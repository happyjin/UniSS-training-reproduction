#!/usr/bin/env bash
# Idea 2 e2e prereg §4 ablations on cards 0-3, sequentially (coordinator 2026-11-19 ~08:00 Prague; user: keep the cards on
# Idea 2): attnpool -> frozen_talker -> constlam, each the main_v2 setup (same trainer and fixes: cuDNN SDPA off,
# non-finite guard; same data, seed 20261119, 3000 steps, 4-GPU DDP, 64 sequences / step), only --ablation differs.
#   run names idea2_e2e_abl_{attnpool,frozentalker,constlam}_v1 -> eval_outputs/mist_w3po_v1/voicecond_idea2/e2e/<run>/
#   logs logs/mist_w3po_v1/voicecond/idea2/e2e_<run>.log
# Slot A (logs/mist_w3po_v1/slots/slotA.owner -> exclude_cards 0 1 2 3) is claimed ONCE for the whole chain and released
# at the end (also on failure / signal). Before each run: wait until cards 0-3 hold no eval process, then stop the guard
# placeholders on cards 0-3 BY PID (cmdline checked; never anything else, never cards 4-7 / main_v2). A run that fails
# (no DONE) is logged and the chain continues with the next run. Outputs that exist are refused (run skipped).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_e2e_ablation_chain.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/e2e_ablation_chain.log 2>&1 < /dev/null 9>&- &
# Marker logs/mist_w3po_v1/voicecond/idea2/E2E_ABLATION_CHAIN_DONE (one line per run: name rc DONE|FAILED).
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
VB=$J/venvs/vllm_omni_030/bin
CARDS=0,1,2,3
CL="0 1 2 3"
LOGD=logs/mist_w3po_v1/voicecond/idea2
SL=logs/mist_w3po_v1/slots
E=eval_outputs/mist_w3po_v1/voicecond_idea2/e2e
PORT=29661
RUNS="attnpool:idea2_e2e_abl_attnpool_v1 frozen_talker:idea2_e2e_abl_frozentalker_v1 constlam:idea2_e2e_abl_constlam_v1"
MARK=$LOGD/E2E_ABLATION_CHAIN_DONE
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
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
  case " $CL " in *" $k "*) ;; *) log "refuse to free card $k (not in $CL)"; return;; esac
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    a=$(tr '\0' ' ' < /proc/$s/cmdline 2>/dev/null)
    case "$a" in *train_mist_workload.py*"--devices $k "*) ;; *) log "card $k: pid $s cmdline changed, skipped"; continue;; esac
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}' | tr '\n' ' ')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
  done
}

OWN=""
release() {
  [ -n "$OWN" ] && [ -e "$OWN" ] && grep -q "Idea 2 e2e ablation chain" "$OWN" && \
    command mv -n "$OWN" $SL/slotA.released_$(date -u +%Y%m%dT%H%M%SZ) && log "slot A released"
  OWN=""
}
trap 'release' EXIT
trap 'log "signal received"; exit 1' INT TERM HUP

if [ -e $SL/slotA.owner ]; then log "slot A already owned: $(sed -n 1,2p $SL/slotA.owner | tr '\n' ' ')"; exit 1; fi
if [ -n "${DRY_RUN:-}" ]; then log "[dry] would claim slot A"
else
  T=$SL/.slotA.owner.new_$$
  ( set -o noclobber; printf 'who: Idea 2 e2e ablation chain executor (voicecond)\ntask: Idea 2 e2e ablation chain (prereg §4: attnpool, frozen_talker, constlam; 4-GPU DDP, 3000 steps each) on cards %s; released by renaming to slotA.released_<UTC>\nutc: %s\n' \
    "$CARDS" "$(date -u +%FT%TZ)" > $T ) && command mv -n $T $SL/slotA.owner && [ ! -e $T ] || { log "slot claim failed"; exit 1; }
  OWN=$SL/slotA.owner
  log "claimed slot A (Idea 2 e2e ablation chain); exclude_cards will include 0 1 2 3 within 10 s"
  sleep 15
fi

for x in $RUNS; do
  ab=${x%%:*}; run=${x#*:}; OUT=$E/$run
  if [ -e "$OUT" ]; then log "$run: $OUT exists, refused (skipped)"; echo "$run - EXISTS" >> $MARK; continue; fi
  log "$run: waiting until cards $CL hold no eval / non-placeholder process"
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
    left=""
    for i in $(seq 1 60); do
      left=""; for k in $CL; do [ -n "$(apps_on $k)" ] && left="$left $k"; done
      [ -z "$left" ] && break
      sleep 5
    done
    if [ -n "$left" ]; then
      log "$run: cards still hold processes after placeholder stop:$left -> run not started"
      echo "$run 1 FAILED_CARDS" >> $MARK; continue
    fi
  fi
  CMD="env TZ=Europe/Prague TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch \
XDG_CACHE_HOME=$J/cache PATH=$VB:$PATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$CARDS $VB/torchrun --nproc_per_node 4 \
--master_port $PORT -m experiments.mist_w3po_v1.voicecond.idea2_e2e_train --out $OUT --ablation $ab"
  log "$run: launch: ${CMD##*torchrun}"
  if [ -n "${DRY_RUN:-}" ]; then continue; fi
  bash -c "$CMD" > $LOGD/e2e_$run.log 2>&1
  rc=$?
  st=FAILED; [ -e $OUT/DONE ] && st=DONE
  [ -d $OUT ] && echo "$rc $(date -u +%FT%TZ)" >> $OUT/LAUNCHER_RC
  echo "$run $rc $st" >> $MARK
  log "$run: torchrun exit $rc, $st (log $LOGD/e2e_$run.log)"
done
log "chain finished"
