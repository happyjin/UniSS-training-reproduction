#!/usr/bin/env bash
# Robust wave 3, V1 second seed for the V1 / V2 choice (robust_wave3_single_prereg.zh-CN.md, "V1 第二种子（选型用）"):
# seed 1, arm v1_s1 (V1 token table; B_s1 reused) x {en2zh, zh2en} x shards 0..3 on cards $CARDS (4 5), then ASR, SQUIM PESQ
# and TitaNet similarity (parallel), then wave3_v1s1_select.py -> wave3_combos/V/summary_v1s1_select.json; DONE_v1s1 / FAILED_v1s1.
# Workers listed in logs/mist_w3po_v1/wave3/v1s1_workers.pids (watcher instance with PIDS / STOP of its own).
# Detached: setsid nohup bash .../wave3_v1s1_orch.sh > logs/mist_w3po_v1/wave3/v1s1_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
V=eval_outputs/mist_w3po_v1/wave3_combos/V
S1=eval_outputs/mist_w3po_v1/wave2_S1
RUN=experiments/mist_w3po_v1/voicecond/wave3_v_run.sh
export PIDS=logs/mist_w3po_v1/wave3/v1s1_workers.pids
CARDS=${CARDS:-"4 5"}
NEED_MIB=${NEED_MIB:-24000}
CAP_MIB=${CAP_MIB:-138000}
GAP=${GAP:-30}
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $V/FAILED_v1s1; exit 1; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
holder_mib() {   # memory of the no-ckpt placeholders on card $1 (the launcher stops them before a job starts)
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" | tr -d ' ')
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | tr -d ' ' |
    awk -F, -v u="$uuid" '$1==u{print $2" "$3}' | while read -r p m; do
      a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/*) echo "$m";; esac
    done | awk '{s+=$1} END {print s+0}'
}
pick_card() {   # a card with room (placeholders count as free), or empty
  for g in $CARDS; do
    u=$(( $(used $g) - $(holder_mib $g) )); t=$(total $g)
    [ $((t - u)) -ge "$NEED_MIB" ] && [ $((u + NEED_MIB)) -le "$CAP_MIB" ] && { echo $g; return; }
  done
}
alive() { kill -0 "$1" 2>/dev/null; }
lastpid() { tail -1 "$PIDS" | cut -d' ' -f2; }

log "orch $$ start"
log "token tables ready"

# talker queue
declare -A JOBPID
QUEUE=()
for arm in v1_s1; do for d in en2zh zh2en; do for s in 0 1 2 3; do QUEUE+=("$arm $d $s"); done; done; done
for j in "${QUEUE[@]}"; do
  read -r arm d s <<< "$j"
  name=talk_${arm}_${d}_${s}
  out=$V/$arm/talker_$d
  [ -s $out/part_$s.jsonl ] && { log "skip $name (done)"; continue; }
  while :; do g=$(pick_card); [ -n "$g" ] && break; sleep 20; done
  tab=$V/tokens/tokens_v1_$d.jsonl
  STAGE=talker GPU=$g DIR=$d OUTD=$out TABLE=$tab SEED=1 SHARD=$s NSHARDS=4 NAME=$name \
    bash $RUN > /dev/null 2>&1 || fail "launch $name"
  JOBPID[$name]=$(lastpid)
  log "launched $name pid ${JOBPID[$name]} card $g"
  sleep "$GAP"
done

# ASR per run dir once its 4 shards are done
declare -A ASRPID
pending="v1_s1:en2zh v1_s1:zh2en"
while [ -n "${pending// /}" ]; do
  left=""
  for rd in $pending; do
    arm=${rd%%:*}; d=${rd##*:}; out=$V/$arm/talker_$d
    ok=1
    for s in 0 1 2 3; do
      p=${JOBPID[talk_${arm}_${d}_${s}]:-}
      if [ -n "$p" ] && alive "$p"; then ok=0; elif [ ! -s $out/part_$s.jsonl ]; then fail "talker $arm $d shard $s ended without part file"; fi
    done
    if [ $ok = 1 ]; then
      g=$(pick_card)
      if [ -n "$g" ]; then
        STAGE=asr GPU=$g RUN=$out NAME=asr_${arm}_$d bash $RUN > /dev/null 2>&1 || fail "launch asr $arm $d"
        ASRPID[$rd]=$(lastpid); log "launched asr $arm $d pid ${ASRPID[$rd]} card $g"; sleep 20
      else left="$left $rd"; fi
    else left="$left $rd"; fi
  done
  pending=$left
  [ -n "${pending// /}" ] && sleep 30
done
for rd in "${!ASRPID[@]}"; do
  while alive "${ASRPID[$rd]}"; do sleep 30; done
  arm=${rd%%:*}; d=${rd##*:}
  [ -s $V/$arm/talker_$d/asr_speech.jsonl ] || fail "asr $rd produced nothing"
done
log "arm ASR done"

# metrics (parallel), then score
mkdir -p $V/metrics
metric() {   # arm d rundir
  local arm=$1 d=$2 r=$3 g=""
  [ -s $V/metrics/${arm}_${d}_squim.jsonl ] || PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq \
      --run $r --out $V/metrics/${arm}_${d}_squim.jsonl --procs 24 || { log "pesq $arm $d failed"; return 1; }
  if [ ! -s $V/metrics/${arm}_${d}_spksim.jsonl ]; then
    while [ -z "$g" ]; do g=$(pick_card); [ -z "$g" ] && sleep 20; done
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir $d --run $r \
      --out $V/metrics/${arm}_${d}_spksim.jsonl --device cuda:0 || { log "spksim $arm $d failed"; return 1; }
  fi
  log "metrics $arm $d"
}
MP=()
for arm in v1_s1; do for d in en2zh zh2en; do metric $arm $d $V/$arm/talker_$d & MP+=($!); sleep 5; done; done
for p in "${MP[@]}"; do wait $p || fail "metrics"; done
$PYQ -m experiments.mist_w3po_v1.voicecond.wave3_v1s1_select --out $V/summary_v1s1_select.json > logs/mist_w3po_v1/wave3/v1s1_select.log 2>&1 \
  || fail "score"
touch $V/DONE_v1s1
log "orch done"
