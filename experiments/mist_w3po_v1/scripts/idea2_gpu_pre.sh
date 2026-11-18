#!/usr/bin/env bash
# Idea 2 preprocessing on cards 4-7 (prereg §2): per card k (shard i of n): SIGTERM that card's placeholder tree by PID
# (supervisor = ppid 1 + "train_mist_workload.py --devices k", then its children by ppid; the guard is never touched),
# then at once: Nemotron lowlat diarizer (2 processes per card, shards i and i+n of 2n) in parallel with HDemucs
# (shard i), then BiCodec window encodes (shard i; needs that shard's HDemucs vocals). After all cards: CPU features.
# Marker data/open_voicecond_v1/idea2/pre/PRE_<WHICH>_DONE (or PRE_<WHICH>_FAILED). The guard refills each card itself.
#   WHICH=pilot|all LS="2 4" [CARDS="4 5 6 7"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_gpu_pre.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/pre_<which>.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
WHICH=${WHICH:?}
LS=${LS:-2}
CARDS=${CARDS:-"4 5 6 7"}
V=/opt/dlami/nvme/jasonleeeli/venvs
PY=$V/vllm_omni_030/bin/python
NPY=$V/nemo_diar/bin/python
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
export PATH=$V/vllm_omni_030/bin:$PATH
LOGD=logs/mist_w3po_v1/voicecond/idea2
PRE=data/open_voicecond_v1/idea2/pre
mkdir -p $LOGD $PRE
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
set -- $CARDS; N=$#

free_card() {   # $1 card
  local k=$1 sup kids
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
  done
}

card_chain() {   # $1 card, $2 shard index
  local k=$1 i=$2 rc=0
  free_card $k
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] card $k: diar shards $i,$((i + N))/$((2 * N)), demucs $i/$N, windows L=$LS $i/$N"; return 0; fi
  CUDA_VISIBLE_DEVICES=$k $NPY experiments/mist_w3po_v1/voicecond/idea2_diarize.py --which $WHICH --shard $i --num-shards $((2 * N)) \
    > $LOGD/diar_${WHICH}_$i.log 2>&1 &
  local d1=$!
  CUDA_VISIBLE_DEVICES=$k $NPY experiments/mist_w3po_v1/voicecond/idea2_diarize.py --which $WHICH --shard $((i + N)) --num-shards $((2 * N)) \
    > $LOGD/diar_${WHICH}_$((i + N)).log 2>&1 &
  local d2=$!
  CUDA_VISIBLE_DEVICES=$k $PY -I experiments/mist_w3po_v1/voicecond/idea2_demucs.py --which $WHICH --shard $i --num-shards $N \
    --device cuda:0 > $LOGD/demucs_${WHICH}_$i.log 2>&1 || rc=1
  CUDA_VISIBLE_DEVICES=$k $PY -I experiments/mist_w3po_v1/voicecond/idea2_windows.py win --which $WHICH --L $LS --shard $i \
    --num-shards $N --device cuda:0 > $LOGD/win_${WHICH}_$i.log 2>&1 || rc=1
  wait $d1 || rc=1
  wait $d2 || rc=1
  log "card $k chain done rc=$rc"
  return $rc
}

log "start WHICH=$WHICH LS=$LS CARDS=$CARDS"
pids=""; i=0
for k in $CARDS; do
  card_chain $k $i &
  pids="$pids $!"; i=$((i + 1))
done
fail=0
for p in $pids; do wait $p || fail=1; done
[ -n "${DRY_RUN:-}" ] && exit 0
if [ $fail = 0 ]; then
  $PY -I experiments/mist_w3po_v1/voicecond/idea2_feats.py --which $WHICH --L $LS --workers 32 > $LOGD/feats_$WHICH.log 2>&1 || fail=1
fi
if [ $fail = 0 ]; then date -u +%FT%TZ >> $PRE/PRE_${WHICH^^}_DONE; log "PRE_${WHICH^^}_DONE"
else date -u +%FT%TZ >> $PRE/PRE_${WHICH^^}_FAILED; log "PRE_${WHICH^^}_FAILED"; fi
