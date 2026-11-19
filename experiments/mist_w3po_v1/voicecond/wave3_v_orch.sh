#!/usr/bin/env bash
# Robust wave 3 V group orchestrator (reports/mist_w3po_v1/robust_wave3_single_prereg.zh-CN.md, V 组 + 变更 1).
# 1. waits for the token tables (wave3_v_tokens.py, both directions);
# 2. talkers: arms v1 v2 v3 x {en2zh, zh2en} x shards 0..3 (S1's B layout, seed 20261021) via wave3_v_run.sh on cards
#    $CARDS, a job placed only when the card has >= $NEED_MIB free and stays <= $CAP_MIB used, $GAP s between starts;
# 3. per run dir, once its 4 shards are done: ASR (speech-only protocol) on a card with room;
# 4. metrics (CPU / small GPU): SQUIM PESQ and TitaNet similarity of the 6 arm runs and of S1's B runs (once S1's B
#    asr_speech.jsonl exists), written to wave3_combos/V/metrics/;
# 5. wave3_v_score.py -> wave3_combos/V/summary.json, then touches wave3_combos/V/DONE (FAILED on error).
# Detached: setsid nohup bash .../wave3_v_orch.sh > logs/mist_w3po_v1/wave3/v_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
V=eval_outputs/mist_w3po_v1/wave3_combos/V
S1=eval_outputs/mist_w3po_v1/wave2_S1
RUN=experiments/mist_w3po_v1/voicecond/wave3_v_run.sh
PIDS=logs/mist_w3po_v1/wave3/v_workers.pids
CARDS=${CARDS:-"3 4"}
NEED_MIB=${NEED_MIB:-24000}
CAP_MIB=${CAP_MIB:-138000}
GAP=${GAP:-45}
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $V/FAILED; exit 1; }
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
for d in en2zh zh2en; do
  until [ -s $V/tokens/tokens_$d.meta.jsonl ]; do
    grep -q Traceback logs/mist_w3po_v1/wave3/v_tokens_$d.log 2>/dev/null && fail "tokens $d"
    sleep 30
  done
done
log "token tables ready"

# talker queue
declare -A JOBPID
QUEUE=()
for arm in v1 v2 v3; do for d in en2zh zh2en; do for s in 0 1 2 3; do QUEUE+=("$arm $d $s"); done; done; done
for j in "${QUEUE[@]}"; do
  read -r arm d s <<< "$j"
  name=talk_${arm}_${d}_${s}
  out=$V/$arm/talker_$d
  [ -s $out/part_$s.jsonl ] && { log "skip $name (done)"; continue; }
  while :; do g=$(pick_card); [ -n "$g" ] && break; sleep 20; done
  STAGE=talker GPU=$g DIR=$d OUTD=$out TABLE=$V/tokens/tokens_${arm}_$d.jsonl SHARD=$s NSHARDS=4 NAME=$name \
    bash $RUN > /dev/null 2>&1 || fail "launch $name"
  JOBPID[$name]=$(lastpid)
  log "launched $name pid ${JOBPID[$name]} card $g"
  sleep "$GAP"
done

# ASR per run dir once its 4 shards are done
declare -A ASRPID
pending="v1:en2zh v1:zh2en v2:en2zh v2:zh2en v3:en2zh v3:zh2en"
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

# metrics of the arms, then of B once S1 has its ASR
mkdir -p $V/metrics
metric() {   # arm d rundir
  local arm=$1 d=$2 r=$3
  [ -s $V/metrics/${arm}_${d}_squim.jsonl ] || PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq \
      --run $r --out $V/metrics/${arm}_${d}_squim.jsonl --procs 24 || fail "pesq $arm $d"
  if [ ! -s $V/metrics/${arm}_${d}_spksim.jsonl ]; then
    g=""; while [ -z "$g" ]; do g=$(pick_card); [ -z "$g" ] && sleep 20; done
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir $d --run $r \
      --out $V/metrics/${arm}_${d}_spksim.jsonl --device cuda:0 || fail "spksim $arm $d"
  fi
}
for arm in v1 v2 v3; do for d in en2zh zh2en; do metric $arm $d $V/$arm/talker_$d; log "metrics $arm $d"; done; done
for d in en2zh zh2en; do
  until [ -s $S1/talker_${d}_s20261021/asr_speech.jsonl ]; do sleep 60; done
  sleep 30
  metric B $d $S1/talker_${d}_s20261021; log "metrics B $d"
done
$PYQ -m experiments.mist_w3po_v1.voicecond.wave3_v_score --out $V/summary.json > logs/mist_w3po_v1/wave3/v_score.log 2>&1 \
  || fail "score"
touch $V/DONE
log "orch done"
