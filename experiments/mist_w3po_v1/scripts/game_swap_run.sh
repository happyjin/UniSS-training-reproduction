#!/usr/bin/env bash
# Prompt-swap causal check (reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md; inputs from
# evaluation/game_prompt_swap.py): si_infer with baseline B's frozen T1c talker flags (thinker_t_eval_orch --talker t1c:
# --speed-policy fixed --history-ban 8, window 1, gpu-mem 0.1), ONE shard per arm, on cards 0 and 2: the no-checkpoint
# placeholder of each card is stopped by PID (its supervisor and children, as the orchestrators do) right before the
# first engine starts there; engines start 60 s apart. Marker logs/mist_w3po_v1/game/SWAP_DONE when all six exit.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/game_swap_run.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/game/swap.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
SPD=eval_outputs/mist_w3po_v1/speech
MODEL=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
ENV="export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:\$PATH OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0;"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_placeholder() {   # <card>: SIGTERM the run_mist_workload / train_mist_workload tree of that card, by PID
  local c=$1 sup kids
  sup=$(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2==1 && index($0,"train_mist_workload.py") && index($0,c) {print $1}')
  [ -z "$sup" ] && { log "card $c: no placeholder supervisor"; return 0; }
  kids=$(ps -eo pid=,ppid= | awk -v p="$sup" '$2==p {print $1}')
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] card $c: would stop placeholder $sup (children $kids)"; return 0; fi
  kill -TERM $sup $kids 2>/dev/null
  log "card $c: stopped placeholder $sup (children $kids)"
  for _ in $(seq 1 60); do [ -z "$(ps -o pid= -p $sup $kids 2>/dev/null)" ] && break; sleep 1; done
}
PIDS=""
k=0
# SPECS="<card> <arm> <dir>;..." (default: the six arms below); STOP_AT="<k ...>" = spec indices that first stop the card's placeholder
SPECS=${SPECS:-"0 zh2en_O zh2en;0 zh2en_V zh2en;0 amien_O en2zh;2 en2zh_O en2zh;2 en2zh_V en2zh;2 amien_N en2zh"}
STOP_AT=${STOP_AT:-"0 3"}
IFS=';' read -ra SPL <<< "$SPECS"
for spec in "${SPL[@]}"; do
  set -- $spec
  c=$1; arm=$2; dir=$3; R=$SPD/game_swap_$arm
  case " $STOP_AT " in *" $k "*) stop_placeholder $c;; esac
  CMD="$ENV CUDA_VISIBLE_DEVICES=$c $J/venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model $MODEL --direction $dir --window 1 --output $R/talker --text-policy file --blocks-file $R/talker_in/blocks.jsonl --items-file $R/talker_in/items.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --shard 0 --num-shards 1 > $R/talk.log 2>&1"
  if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; k=$((k + 1)); continue; fi
  [ -e $R/talker ] && { log "$R/talker exists (not overwritten)"; k=$((k + 1)); continue; }
  bash -c "$CMD" < /dev/null 9>&- &
  PIDS="$PIDS $!"
  log "card $c $arm pid $!"
  k=$((k + 1))
  sleep 60
done
[ -n "${DRY_RUN:-}" ] && exit 0
rc=0; for p in $PIDS; do wait $p || rc=1; done
date -u +%FT%TZ > logs/mist_w3po_v1/game/SWAP_DONE${MARK:-}
log "SWAP_DONE rc=$rc"
