#!/usr/bin/env bash
# Replaces run_v3_eval3.sh while its S2-dl training runs (that training, PID given as $1, continues on its own): when it
# ends, the single-speaker paper's candidate A2 (ABC with spontaneous audio; reports/uniss_si_v4) trains on the 8 cards
# for its 200 steps (about 20 min) if its rows are ready (data/si_thinker_oracle_prefix_v7_spont/READY), then the
# multi-speaker evaluation queue (make_v3_jobs.py, gpuq6) and the GPU holders as before. If the rows are not ready by
# then, A2 trains after the queue instead. While A2 trains, a gpuq7 ledger entry reserves every card, so the stacked
# single-speaker queues place nothing beside it. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
TS=20261029b; TRAIN_PID="${1:?PID of the S2-dl training}"
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

while kill -0 "${TRAIN_PID}" 2>/dev/null; do sleep 60; done
grep -q THINKER_GSTAR_DONE logs/multispk/thinker_gstar_s2dl_${TS}.log || log "S2-dl failed; its decode jobs will fail"
log "S2-dl finished"
[ -f ${A2_ROWS}/READY ] && train_a2
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq6.py \
  reports/gstar_si_paper_v1/jobs/v3_jobs.json logs/gstar_v3_eval
log "QUEUE_DONE"
[ -f ${A2_ROWS}/READY ] && [ ! -f ${A2_ROWS}/TRAINED ] && train_a2
sleep 15
bash scripts/gpu_queue/holder_on.sh 0,1,3,4
bash scripts/gpu_queue/holder_on.sh 2,5,6,7
log "V3_EVAL_DONE"
