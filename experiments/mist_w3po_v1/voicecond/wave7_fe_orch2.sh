#!/usr/bin/env bash
# Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): continuation of wave7_fe_orch.sh after its stop
# at the first talker (wave 3's V1 table did not match the arm's prompts, which follow the arm's own first writes):
# V1 (C) voice tokens on each arm's first writes (wave7_fe_v1tokens.py), then talker, asr and score as in wave7_fe_orch.sh.
# Original header:
# Waits for the pseudo-target workers (all 32 units final), then: train FE-pseudo (card 6) and FE-clean (card 7);
# process the S1 items with each front-end; RTF of FE-pseudo on card 6 (exclusive); si_thinker_stream la2 w1 per arm and
# direction (one vLLM job per card); si_infer talkers (V1 C tokens, original items, 4 shards, seed 20261021; 4 jobs per
# card); speech-only ASR; scoring. Every job goes through wave7_fe_run.sh (placeholders stopped by PID; the watcher
# wave7_fe_noplaceholder.sh keeps them off). A step whose expected output is missing stops the orchestrator (no retry).
# Detached: setsid nohup bash .../wave7_fe_orch2.sh > logs/mist_w3po_v1/wave7/fe_orch2.log 2>&1 < /dev/null 8>&- 9>&- &
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

log "orchestrator2 $$ start"
for a in pseudo clean; do [ -s "$FE/eval_$a/thinker_zh2en/blocks_zh2en_w1.jsonl" ] && [ -s "$FE/eval_$a/thinker_en2zh/blocks_en2zh_w1.jsonl" ] || die "no blocks for $a"; done
# 4b. V1 voice tokens on each arm's own first writes (2 jobs per card)
ps_=()
for a in pseudo clean; do
  ps_+=("$(launch STAGE=v1tok GPU=6 ARM=$a DIR=zh2en NAME=v1tok_${a}_zh2en)")
  ps_+=("$(launch STAGE=v1tok GPU=7 ARM=$a DIR=en2zh NAME=v1tok_${a}_en2zh)")
done
waitpids "${ps_[@]}"
for a in pseudo clean; do for d in zh2en en2zh; do [ -s "$FE/eval_$a/tokens/tokens_v1_$d.jsonl" ] || die "no v1 tokens $a $d"; done; done
log "v1 tokens done"

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
