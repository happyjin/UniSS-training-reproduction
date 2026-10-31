#!/usr/bin/env bash
# Paper v6 (MIST) trainings after prereg_v6 amendment 4 (seed, turn-trained cascade; the target control is dropped;
# replaces run_v6_train3.sh, stopped by PID while it was only waiting) (replaces run_v6_train.sh, whose chain was stopped by PID while
# its first run kept training): waits for that run to finish, then trains one after the other (prereg_v6 §3-§5): each waits until enough cards are free (< 30 GB
# used: only a colleague's small job), reserves them in the gpuq7 ledger (so neither branch's queue starts a job there),
# then trains on 8 cards if all eight are free, else on 4 free cards (GBS 64 either way: MBS 8 x 8 ranks, or MBS 8 x 4 ranks x 2 accumulation steps). Before each
# start the plan is written to logs/gpu_coord/events.log; per-card memory after iteration 30 to train_mem.txt.
# run_thinker_gstar.sh refuses existing outputs. Launch: setsid nohup bash reports/gstar_si_v6/run_v6_train.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
L=logs/multispk/v6
exec >> $L/train.log 2>&1
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
ev() { echo "[$(date -u +%F' '%T) UTC] MIST(v6): $*" >> logs/gpu_coord/events.log; }
EV=eval_outputs/uniss_qwen3_omni_v1; D=data/ami_multispk_si_v1
O2=${EV}/thinker_gstar_o2_20261029b/thinker_merged

TRAIN_CARDS=${TRAIN_CARDS:-2,3,4,5,6,7}   # cards 0-1 are the single-speaker branch's (announced split)
free_cards() {   # cards of TRAIN_CARDS with < 30,000 MiB used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v ok=",$TRAIN_CARDS," '$2 < 30000 && index(ok, "," $1 ",") {print $1}' | tr '\n' ',' | sed 's/,$//'
}

LEDGER=logs/gpuq7_ledger/$$.json   # gpuq7 queues count our reservation here as their own (OUR_MAX)
reserve() {   # $1 comma-separated cards ("" clears)
  python3 -c "import json,sys; c=[x for x in sys.argv[1].split(',') if x]; json.dump({x: 80000 for x in c}, open(sys.argv[2] + '.tmp', 'w'))" "$1" $LEDGER
  command mv -f $LEDGER.tmp $LEDGER
}
reserve ""

wait_cards() {   # sets CARDS / NP: all eight if free, else four free cards
  while :; do
    f=$(free_cards); n=$(echo "$f" | tr ',' '\n' | grep -c .)
    if [ "$n" -ge 8 ]; then CARDS=$f; NP=8; return; fi
    if [ "$n" -ge 4 ]; then CARDS=$(echo "$f" | tr ',' '\n' | tail -n 4 | tr '\n' ',' | sed 's/,$//'); NP=4; return; fi
    sleep 60
  done
}

train() {   # $1 run id, then env assignments for run_thinker_gstar.sh
  local rid=$1; shift
  if [ -e ${EV}/${rid}/thinker_merged ]; then log "$rid exists, skipped"; return; fi
  wait_cards
  reserve "$CARDS"   # from now on no gpuq7 queue starts a job of ours or the other branch's on these cards
  sleep 120          # a job that a queue started just before must show its memory
  f=",$(free_cards),"
  for c in ${CARDS//,/ }; do
    case "$f" in *",$c,"*) ;; *) reserve ""; train "$rid" "$@"; return ;; esac
  done
  log "train $rid on cards $CARDS ($NP ranks): $*"
  ev "training $rid on cards $CARDS ($NP ranks, ~2-4 h; ~75 GB per card)"
  ( until grep -qE "iteration +30/" $L/$rid.log 2>/dev/null; do grep -qE "THINKER_GSTAR_(FAILED|DONE)|拒绝覆盖" $L/$rid.log 2>/dev/null && exit; sleep 20; done
    { echo "[$(date -u +%F' '%T) UTC] $rid stable"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; } >> logs/gpu_coord/train_mem.txt ) &
  env CUDA_VISIBLE_DEVICES=$CARDS NPROC=$NP MBS=8 GBS=64 RUN_ID=$rid "$@" \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > $L/$rid.log 2>&1
  reserve ""
  if grep -q THINKER_GSTAR_DONE $L/$rid.log; then log "$rid done"; ev "training $rid done"; else log "$rid FAILED: $(tail -3 $L/$rid.log | tr '\n' ' ' | cut -c1-300)"; ev "training $rid failed"; fi
}

# the running E-zh training (started by run_v6_train.sh on cards 4-7): keep its cards reserved until it ends
RUN0=thinker_gstar_s2dlchar_zh2en_3k_20261031
reserve "4,5,6,7"
until grep -qE "THINKER_GSTAR_(DONE|FAILED|MERGE_FAILED)" $L/$RUN0.log 2>/dev/null; do sleep 60; done
reserve ""
if grep -q THINKER_GSTAR_DONE $L/$RUN0.log; then log "$RUN0 done"; ev "training $RUN0 done"; else log "$RUN0 FAILED"; ev "training $RUN0 failed"; fi
# E-seed (pre-registered, amendment 2: one more seed)
train thinker_gstar_s2dl_seed20261101 BASE=${O2} INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_v1/rows_train.jsonl \
  TRAIN_ITERS=3000 TRAIN_SEED=20261101 GSTAR_SEED=20261101 MASTER_PORT=29872
# cascade translator trained on its own turns (as O2 without speaker stream otherwise)
train thinker_gstar_cturn_20261031 NOSD=1 ROWS=${D}/e2e_gstar_turns_v6/rows_train.jsonl TRAIN_ITERS=2000 MASTER_PORT=29874
log "V6_TRAIN4_DONE"
