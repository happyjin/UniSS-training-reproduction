#!/usr/bin/env bash
# Robust wave 6 orchestrator (reports/mist_w3po_v1/robust_wave6_disen_prereg.zh-CN.md). Modelled on wave3_v_orch.sh.
# 0. waits for TRAIN_DONE of both arms; exports each (wave6_export.py, logits check) and an identity export; the identity
#    export and T1c must give identical codes in si_infer (8 noisy CoVoST items, raw voice ids, one shard) -> else FAILED;
# 1. talkers: {tft, disen} x {en2zh, zh2en} x shards 0..3 with wave3's V1 C-token table (S1 blocks, seed 20261021);
# 2. gate-1 duration runs: {t1c, tft, disen} x {raw, c, v2, cclean} x dirs on the 300 noisy CoVoST items (1 shard);
# 3. ASR of the 4 main runs; SQUIM PESQ + TitaNet similarity -> wave6_disen/metrics/;
# 4. wave6_score.py -> wave6_disen/summary.json; touches wave6_disen/DONE (FAILED on error).
# Cards $CARDS (default 4..7), a job placed when the card has >= $NEED_MIB free (placeholders count as free).
# Detached: setsid nohup bash .../wave6_orch.sh > logs/mist_w3po_v1/wave6/orch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
W=eval_outputs/mist_w3po_v1/wave6_disen
V=eval_outputs/mist_w3po_v1/wave3_combos/V
RUN=experiments/mist_w3po_v1/voicecond/wave6_run.sh
PIDS=logs/mist_w3po_v1/wave6/workers.pids
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
CARDS=${CARDS:-"4 5 6 7"}
NEED_MIB=${NEED_MIB:-24000}
CAP_MIB=${CAP_MIB:-138000}
GAP=${GAP:-40}
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; touch $W/FAILED; exit 1; }
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
holder_mib() {
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" | tr -d ' ')
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | tr -d ' ' |
    awk -F, -v u="$uuid" '$1==u{print $2" "$3}' | while read -r p m; do
      a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/*) echo "$m";; esac
    done | awk '{s+=$1} END {print s+0}'
}
pick_card() {
  for g in $CARDS; do
    u=$(( $(used $g) - $(holder_mib $g) )); t=$(total $g)
    [ $((t - u)) -ge "$NEED_MIB" ] && [ $((u + NEED_MIB)) -le "$CAP_MIB" ] && { echo $g; return; }
  done
}
wait_card() { local g=""; while [ -z "$g" ]; do g=$(pick_card); [ -z "$g" ] && sleep 20; done; echo $g; }
alive() { kill -0 "$1" 2>/dev/null; }
lastpid() { tail -1 "$PIDS" | cut -d' ' -f2; }
stop_holders() {   # placeholders off card $1 before a foreground GPU step (by PID, argv re-checked)
  local g=$1
  while read -r pid args; do
    case " 2726951 52045 1222453 $$ " in *" $pid "*) continue;; esac
    case "$args" in *scripts/gpu_load/*) ;; *) continue;; esac
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in *scripts/gpu_load/*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g";; esac
  done < <(ps -eo pid=,args=)
  sleep 8
}

log "orch $$ start"
for arm in disen tft; do
  until [ -e $W/train_$arm/TRAIN_DONE ]; do
    grep -q Traceback logs/mist_w3po_v1/wave6/train_$arm.log 2>/dev/null && fail "train $arm"
    sleep 30
  done
done
log "training done"

# 0. exports + checks (foreground, one card)
mkdir -p $W/eval
g=$(wait_card); stop_holders $g
echo "$g $$ orch_export" >> $PIDS
for arm in disen tft; do
  [ -e $W/eval/$arm/model/export_check.json ] && continue
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYQ experiments/mist_w3po_v1/voicecond/wave6_export.py \
    --ckpt $W/train_$arm/lora_final.pt --out $W/eval/$arm/model --check 5 > logs/mist_w3po_v1/wave6/export_$arm.log 2>&1 \
    || fail "export $arm"
  log "export $arm: $(cat $W/eval/$arm/model/export_check.json | tr -d '\n ')"
done
if [ ! -e $W/eval/identity/model/export_check.json ]; then
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYQ experiments/mist_w3po_v1/voicecond/wave6_export.py --identity \
    --out $W/eval/identity/model --check 5 > logs/mist_w3po_v1/wave6/export_identity.log 2>&1 || fail "export identity"
  log "export identity: $(cat $W/eval/identity/model/export_check.json | tr -d '\n ')"
fi
head -8 $W/durtest/items_a_noisy_en2zh.jsonl > $W/eval/identity/items8.jsonl
for m in t1c identity; do
  M=$T1C; [ $m = identity ] && M=$W/eval/identity/model
  out=$W/eval/identity/si_$m
  [ -s $out/part_0.jsonl ] && continue
  g2=$(wait_card)
  STAGE=talker GPU=$g2 DIR=en2zh OUTD=$out MODEL=$M TABLE=$W/durtest/tokens_raw_en2zh.jsonl SHARD=0 NSHARDS=1 \
    ITEMS=$W/eval/identity/items8.jsonl NAME=ident_$m bash $RUN > /dev/null 2>&1 || fail "launch ident $m"
  p=$(lastpid); while alive $p; do sleep 10; done
  [ -s $out/part_0.jsonl ] || fail "ident $m produced nothing"
done
$PYE - <<EOF || fail "identity export differs from T1c in si_infer"
import json
a = {r["id"]: r.get("codes") for r in map(json.loads, open("$W/eval/identity/si_t1c/part_0.jsonl"))}
b = {r["id"]: r.get("codes") for r in map(json.loads, open("$W/eval/identity/si_identity/part_0.jsonl"))}
same = sum(a[i] == b.get(i) for i in a)
print(f"identity check: {same}/{len(a)} items with identical code counts")
txt = {r["id"]: r.get("blocks") for r in map(json.loads, open("$W/eval/identity/si_t1c/part_0.jsonl"))}
assert same == len(a) == 8
EOF
log "identity check passed"

# 1 + 2. talker queue
declare -A JOBPID
QUEUE=()
for arm in disen tft; do for d in en2zh zh2en; do for s in 0 1 2 3; do QUEUE+=("main $arm $d $s"); done; done; done
for model in t1c tft disen; do for cond in raw c v2 cclean; do for d in en2zh zh2en; do QUEUE+=("dur $model $cond $d"); done; done; done
for j in "${QUEUE[@]}"; do
  read -r kind a1 a2 a3 <<< "$j"
  if [ $kind = main ]; then
    arm=$a1; d=$a2; s=$a3; name=talk_${arm}_${d}_${s}; out=$W/eval/$arm/talker_$d
    [ -s $out/part_$s.jsonl ] && { log "skip $name"; continue; }
    g=$(wait_card)
    STAGE=talker GPU=$g DIR=$d OUTD=$out MODEL=$W/eval/$arm/model TABLE=$V/tokens/tokens_v1_$d.jsonl SHARD=$s NSHARDS=4 \
      NAME=$name bash $RUN > /dev/null 2>&1 || fail "launch $name"
  else
    model=$a1; cond=$a2; d=$a3; name=dur_${model}_${cond}_${d}; out=$W/durtest/${model}_${cond}_$d
    [ -s $out/part_0.jsonl ] && { log "skip $name"; continue; }
    M=$T1C; [ $model != t1c ] && M=$W/eval/$model/model
    g=$(wait_card)
    STAGE=talker GPU=$g DIR=$d OUTD=$out MODEL=$M TABLE=$W/durtest/tokens_${cond}_$d.jsonl SHARD=0 NSHARDS=1 \
      ITEMS=$W/durtest/items_a_noisy_$d.jsonl NAME=$name bash $RUN > /dev/null 2>&1 || fail "launch $name"
  fi
  JOBPID[$name]=$(lastpid)
  log "launched $name pid ${JOBPID[$name]} card $g"
  sleep "$GAP"
done

# 3. ASR of the main runs
declare -A ASRPID
pending="disen:en2zh disen:zh2en tft:en2zh tft:zh2en"
while [ -n "${pending// /}" ]; do
  left=""
  for rd in $pending; do
    arm=${rd%%:*}; d=${rd##*:}; out=$W/eval/$arm/talker_$d
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
  [ -s $W/eval/$arm/talker_$d/asr_speech.jsonl ] || fail "asr $rd produced nothing"
done
log "ASR done"
for name in "${!JOBPID[@]}"; do
  case $name in dur_*) while alive "${JOBPID[$name]}"; do sleep 20; done
    model_cond_d=${name#dur_}; [ -s $W/durtest/$model_cond_d/part_0.jsonl ] || fail "$name produced nothing"
    [ -s $W/durtest/$model_cond_d/results.jsonl ] || $PYE reports/uniss_si_v4/speech_only.py $W/durtest/$model_cond_d > /dev/null;;
  esac
done
log "duration runs done"

mkdir -p $W/metrics
for arm in disen tft; do for d in en2zh zh2en; do
  r=$W/eval/$arm/talker_$d
  [ -s $W/metrics/${arm}_${d}_squim.jsonl ] || PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq \
      --run $r --out $W/metrics/${arm}_${d}_squim.jsonl --procs 24 || fail "pesq $arm $d"
  if [ ! -s $W/metrics/${arm}_${d}_spksim.jsonl ]; then
    g=$(wait_card)
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir $d --run $r \
      --out $W/metrics/${arm}_${d}_spksim.jsonl --device cuda:0 || fail "spksim $arm $d"
  fi
  log "metrics $arm $d"
done; done
$PYQ -m experiments.mist_w3po_v1.voicecond.wave6_score --out $W/summary.json > logs/mist_w3po_v1/wave6/score.log 2>&1 || fail "score"
touch $W/DONE
log "orch done"
