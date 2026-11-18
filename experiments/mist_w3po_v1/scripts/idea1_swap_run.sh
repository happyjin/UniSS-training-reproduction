#!/usr/bin/env bash
# Idea 1 (ZNE) E1.2a talker arms (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md §6): the game_swap_<dir>_O items
# / blocks (same order -> same per-item seeds), voice tokens overridden by a ZNE token table
# (voicecond/idea1_si_infer_vo.py), otherwise exactly game_swap_run.sh's si_infer command (frozen T1c, --speed-policy
# fixed --history-ban 8, window 1, gpu-mem 0.1, one shard). The no-checkpoint placeholder of a card is stopped by PID
# (supervisor + children) right before the first engine there; engines start 60 s apart.
# SPECS="<card> <arm> <dir> <token table>;..."  ARM dir: eval_outputs/mist_w3po_v1/voicecond_idea1/speech/<arm>
# Marker logs/mist_w3po_v1/voicecond_idea1/SWAP_<TAG>_DONE when all exit.
#   SPECS=... TAG=... [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_swap_run.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_idea1/swap_<tag>.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
SPD=eval_outputs/mist_w3po_v1/voicecond_idea1/speech
MODEL=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
ENV="export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:\$PATH OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0;"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${SPECS:?SPECS required}" "${TAG:?TAG required}"
STOPPED=" "
stop_placeholder() {   # <card>: SIGTERM the train_mist_workload tree of that card, by PID (once per card)
  local c=$1 sup kids
  case "$STOPPED" in *" $c "*) return 0;; esac
  STOPPED="$STOPPED$c "
  case " 0 1 2 3 " in *" $c "*) ;; *) log "card $c not in slot A (0-3): refuse"; exit 1;; esac
  sup=$(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2==1 && index($0,"train_mist_workload.py") && index($0,c) {print $1}')
  [ -z "$sup" ] && { log "card $c: no placeholder supervisor"; return 0; }
  kids=$(ps -eo pid=,ppid= | awk -v p="$sup" '$2==p {print $1}')
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] card $c: would stop placeholder $sup (children $kids)"; return 0; fi
  kill -TERM $sup $kids 2>/dev/null
  log "card $c: stopped placeholder $sup (children $kids)"
  for _ in $(seq 1 60); do [ -z "$(ps -o pid= -p $sup $kids 2>/dev/null)" ] && break; sleep 1; done
}
PIDS=""
IFS=';' read -ra SPL <<< "$SPECS"
for spec in "${SPL[@]}"; do
  set -- $spec
  c=$1; arm=$2; dir=$3; tab=$4; R=$SPD/$arm
  [ -s $R/talker_in/items.jsonl ] && [ -s $tab ] || { log "missing inputs for $arm"; exit 1; }
  CMD="$ENV CUDA_VISIBLE_DEVICES=$c $J/venvs/vllm_omni_030/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $tab --model $MODEL --direction $dir --window 1 --output $R/talker --text-policy file --blocks-file $R/talker_in/blocks.jsonl --items-file $R/talker_in/items.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --shard 0 --num-shards 1 > $R/talk.log 2>&1"
  if [ -n "${DRY_RUN:-}" ]; then stop_placeholder $c; echo "$CMD"; continue; fi
  [ -e $R/talker ] && { log "$R/talker exists (not overwritten)"; continue; }
  stop_placeholder $c
  bash -c "$CMD" < /dev/null 9>&- &
  PIDS="$PIDS $!"
  log "card $c $arm pid $!"
  sleep 60
done
[ -n "${DRY_RUN:-}" ] && exit 0
rc=0; for p in $PIDS; do wait $p || rc=1; done
date -u +%FT%TZ > logs/mist_w3po_v1/voicecond_idea1/SWAP_${TAG}_DONE
log "SWAP_${TAG}_DONE rc=$rc"
