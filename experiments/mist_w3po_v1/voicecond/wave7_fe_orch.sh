#!/usr/bin/env bash
# Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): phase-2 orchestrator on cards 6-7 only.
# Waits for the pseudo-target workers (all 32 units final), then: train FE-pseudo (card 6) and FE-clean (card 7);
# process the S1 items with each front-end; RTF of FE-pseudo on card 6 (exclusive); si_thinker_stream la2 w1 per arm and
# direction (one vLLM job per card); si_infer talkers (V1 C tokens, original items, 4 shards, seed 20261021; 4 jobs per
# card); speech-only ASR; scoring. Every job goes through wave7_fe_run.sh (placeholders stopped by PID; the watcher
# wave7_fe_noplaceholder.sh keeps them off). A step whose expected output is missing stops the orchestrator (no retry).
# Detached: setsid nohup bash .../wave7_fe_orch.sh > logs/mist_w3po_v1/wave7/fe_orch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=experiments/mist_w3po_v1/voicecond/wave7_fe_run.sh
FE=eval_outputs/mist_w3po_v1/wave7/FE
PIDS=logs/mist_w3po_v1/wave7/fe_workers.pids
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "STOP: $*"; exit 1; }
launch() {   # env assignments..., prints the pid
  env "$@" bash "$R" >> logs/mist_w3po_v1/wave7/fe_orch_launch.log 2>&1 || die "launch failed: $*"
  tail -1 "$PIDS" | awk '{print $2}'
}
waitpids() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }

log "orchestrator $$ start"
# 1. pseudo targets
while :; do
  n=$(ls $FE/runs/*.jsonl 2>/dev/null | grep -vc partial)
  [ "$n" -ge 32 ] && break
  alive=0
  for p in $(awk '$3 ~ /^pseudo_/ {print $2}' "$PIDS"); do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 1 ] || die "pseudo workers gone with $n/32 units final"
  sleep 60
done
log "pseudo targets: 32/32 units"

# 2. training (both arms in parallel)
p1=$(launch STAGE=train GPU=6 ARM=pseudo NAME=train_pseudo)
p2=$(launch STAGE=train GPU=7 ARM=clean NAME=train_clean)
log "train pids $p1 $p2"; waitpids "$p1" "$p2"
for a in pseudo clean; do [ -f "$FE/train_$a/final.pt" ] || die "no final.pt for $a"; done
log "training done"

# 3. process + rtf
p1=$(launch STAGE=process GPU=6 ARM=pseudo NAME=process_pseudo)
p2=$(launch STAGE=process GPU=7 ARM=clean NAME=process_clean)
waitpids "$p1" "$p2"
for a in pseudo clean; do for d in zh2en en2zh; do [ -s "$FE/eval_$a/items_$d.jsonl" ] || die "no items $a $d"; done; done
p1=$(launch STAGE=rtf GPU=6 ARM=pseudo NAME=rtf_pseudo); waitpids "$p1"
log "process + rtf done"

# 4. thinker: one vLLM job per card, two rounds
for d in zh2en en2zh; do
  p1=$(launch STAGE=thinker GPU=6 ARM=pseudo DIR=$d NAME=thinker_pseudo_$d)
  p2=$(launch STAGE=thinker GPU=7 ARM=clean DIR=$d NAME=thinker_clean_$d)
  waitpids "$p1" "$p2"
  for a in pseudo clean; do [ -s "$FE/eval_$a/thinker_$d/blocks_${d}_w1.jsonl" ] || die "no blocks $a $d"; done
done
log "thinker done"

# 5. talker: per arm 8 jobs (2 dirs x 4 shards), 4 per card
for a in pseudo clean; do
  ps_=()
  for d in zh2en en2zh; do
    for s in 0 1 2 3; do
      g=$([ "$d" = zh2en ] && echo 6 || echo 7)
      ps_+=("$(launch STAGE=talker GPU=$g ARM=$a DIR=$d SHARD=$s NAME=talker_${a}_${d}_$s)")
    done
  done
  waitpids "${ps_[@]}"
  for d in zh2en en2zh; do
    for s in 0 1 2 3; do [ -s "$FE/eval_$a/talker_$d/part_$s.jsonl" ] || die "no talker part $a $d $s"; done
  done
  log "talker $a done"
done

# 6. asr
ps_=()
for a in pseudo clean; do
  ps_+=("$(launch STAGE=asr GPU=6 ARM=$a DIR=zh2en NAME=asr_${a}_zh2en)")
  ps_+=("$(launch STAGE=asr GPU=7 ARM=$a DIR=en2zh NAME=asr_${a}_en2zh)")
done
waitpids "${ps_[@]}"
for a in pseudo clean; do for d in zh2en en2zh; do [ -s "$FE/eval_$a/talker_$d/asr_speech.jsonl" ] || die "no asr $a $d"; done; done
log "asr done"

# 7. score (CPU)
/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -I experiments/mist_w3po_v1/voicecond/wave7_fe_eval.py score \
  --out $FE/summary.json > logs/mist_w3po_v1/wave7/fe_score.log 2>&1 || die "score failed"
log "score done: $FE/summary.json"
touch logs/mist_w3po_v1/wave7/fe_noplaceholder.stop
log "orchestrator done"
