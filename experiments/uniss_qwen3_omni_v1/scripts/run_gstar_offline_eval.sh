#!/usr/bin/env bash
# The offline experiments of docs/multi_speaker_si_plan_v7.zh-CN.md on the reviewed AMI test windows, one after another:
#   wait for O-G1's merge -> O-G1 (Nemotron posteriors), O-oracle (reference posteriors), B-off (the original Thinker,
#   no speaker stream) -> train O-nosd (speaker stream zeroed) -> O-nosd. Each inference runs as eight shards, one per
#   card, then gstar_score.py. Launch detached (setsid nohup ... &); DRY_RUN=1 prints the commands.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
O1="${O1:?O-G1 RUN_ID}"
DATA=data/ami_multispk_si_v1/e2e_gstar_v1
WIN="${WIN:-${DATA}/windows_test_ref.jsonl}"
EV=eval_outputs/uniss_qwen3_omni_v1/gstar_offline_eval_${O1#thinker_gstar_}
read -r -a G <<<"${GPUS:-0 1 2 3 4 5 6 7}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else eval "$*"; fi; }

infer() {   # infer <name> <thinker> <extra args>
  local name=$1 thinker=$2 extra=$3 out="${EV}/$1"
  if [ -e "${out}/hyp.jsonl" ]; then log "skip ${name} (done)"; return 0; fi
  run "mkdir -p ${out}"
  log "infer ${name}"
  for k in "${!G[@]}"; do
    run "CUDA_VISIBLE_DEVICES=${G[$k]} PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream --thinker ${thinker} --windows ${WIN} --out ${out}/hyp_s${k}.jsonl --shard ${k} --num-shards ${#G[@]} ${extra} > ${out}/infer_s${k}.log 2>&1 &"
  done
  run wait
  run "cat ${out}/hyp_s*.jsonl > ${out}/hyp.jsonl"
  run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows ${WIN} --hyp ${out}/hyp.jsonl --out ${out}/score.json > ${out}/score.log 2>&1"
  log "scored ${name}: $(tr -d '\n ' < ${out}/score.json 2>/dev/null | cut -c1-200)"
}

M1=eval_outputs/uniss_qwen3_omni_v1/${O1}/thinker_merged
log "wait for ${M1}"
[ -n "${DRY_RUN:-}" ] || until [ -f "${M1}/sd_proj.pt" ] && grep -q "THINKER_GSTAR_DONE" "logs/multispk/${O1}.log" 2>/dev/null; do
  grep -q "THINKER_GSTAR_FAILED\|THINKER_GSTAR_MERGE_FAILED" "logs/multispk/${O1}.log" 2>/dev/null && { log "O-G1 failed"; exit 1; }
  sleep 60
done
infer O-G1 "${M1}" "--inst gstar --post-key offline"
infer O-oracle "${M1}" "--inst gstar --post-key oracle_offline"
infer B-off "pretrained_models/Qwen3-Omni-30B-A3B-Instruct" "--inst sd"
if [ -z "${SKIP_NOSD:-}" ]; then
  N1=thinker_gstar_o1nosd_${O1#thinker_gstar_o1_}
  if [ ! -d "eval_outputs/uniss_qwen3_omni_v1/${N1}/thinker_merged" ]; then
    log "train O-nosd ${N1}"
    run "NOSD=1 RUN_ID=${N1} ROWS=${DATA}/rows_train.jsonl TRAIN_ITERS=${TRAIN_ITERS:-1000} bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${N1}.log 2>&1"
  fi
  infer O-nosd "eval_outputs/uniss_qwen3_omni_v1/${N1}/thinker_merged" "--inst gstar --post-key offline --zero-sd"
fi
log "GSTAR_OFFLINE_EVAL_DONE ${EV}"
if [ -n "${HOLDER:-}" ]; then bash "${HOLDER}" 0,1,2,3,4,5,6,7; fi
