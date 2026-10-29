#!/usr/bin/env bash
# v7 of the launcher: takes over from run_v3_eval6.sh (stopped) and its gpuq6 queue (stopped by PID). The rest of the v3
# evaluation and the AliMeeting fix chain run as one gpuq6 queue (jobs/v3_multi_jobs.json, make_v3_multi_jobs.py): the HF
# decoding runs go four to a card through gstar_multi (one copy of the Thinker; outputs checked identical on 64 windows),
# runs the old queue left going are waited for. Then, as in eval6, A2 trains if it has not, then the GPU holders.
# Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
A2_ROWS=data/si_thinker_oracle_prefix_v7_spont; A2_RUN=thinker_op_v7sp_20261029
LEDGER=logs/gpuq7_ledger/$$.json

train_a2() {
  log "A2: wait for the cards to empty"
  for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 2000' | wc -l)
    [ "$busy" -eq 0 ] && break
    sleep 30
  done
  [ "$busy" -eq 0 ] || { log "A2: cards still busy, skipped"; return 1; }
  echo '{"0": 62000, "1": 62000, "2": 62000, "3": 62000, "4": 62000, "5": 62000, "6": 62000, "7": 62000}' > ${LEDGER}
  log "A2: train ${A2_RUN}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_ID=${A2_RUN} ROWS=${A2_ROWS}/rows_train.jsonl NPROC=8 MBS=32 GBS=256 \
    NUM_WORKERS=12 TRAIN_ITERS=800 SAVE_INTERVAL=200 EXIT_INTERVAL=200 WARMUP_ITERS=30 MASTER_PORT=29836 \
    MERGED=$PWD/eval_outputs/uniss_qwen3_omni_v1/${A2_RUN}/thinker_merged_i200 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_sd.sh > logs/si_v4/${A2_RUN}.train.log 2>&1
  echo '{}' > ${LEDGER}
  log "A2: trained -> $(tail -1 logs/si_v4/${A2_RUN}.train.log | cut -c1-200)"
  touch ${A2_ROWS}/TRAINED
}

[ -e data/alimeeting_si_v1/mt.fail1.log ] || cp -p data/alimeeting_si_v1/mt.log data/alimeeting_si_v1/mt.fail1.log
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq6.py \
  reports/gstar_si_paper_v1/jobs/v3_multi_jobs.json logs/gstar_v3_eval_multi
log "QUEUE_DONE (v3 rest + AliMeeting, shared Thinker)"
[ -f ${A2_ROWS}/READY ] && [ ! -f ${A2_ROWS}/TRAINED ] && train_a2
sleep 15
bash scripts/gpu_queue/holder_on.sh 0,1,3,4
bash scripts/gpu_queue/holder_on.sh 2,5,6,7
log "V3_EVAL_DONE"
