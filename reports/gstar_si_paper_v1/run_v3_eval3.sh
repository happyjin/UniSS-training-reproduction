#!/usr/bin/env bash
# Paper v3: after run_bidir_pipeline.sh (O2, S2, O2 without speaker stream), the label-ablation model S2-dl (O2 continued on
# v2's deadline-label rows, en->zh, 3,000 steps as S2), then the evaluation queue (make_v3_jobs.py, gpuq6, 8 cards); GPU holders
# back at the end. Replaces run_v3_eval2.sh (S2-dl had 2,000 steps there; now the same 3,000 as S2). Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; TS=20261029b
[ -e ${EV}/gstar_v3_20261029 ] && { log "拒绝覆盖已有输出: gstar_v3_20261029"; exit 1; }
until grep -q -E "BIDIR_PIPELINE_DONE|FAILED|failed|A8_FAIL" logs/bidir_pipeline_v8b.log; do sleep 60; done
grep -q BIDIR_PIPELINE_DONE logs/bidir_pipeline_v8b.log || { log "training chain did not finish"; exit 1; }
log "train thinker_gstar_s2dl_${TS}"
RUN_ID=thinker_gstar_s2dl_${TS} BASE=${EV}/thinker_gstar_o2_${TS}/thinker_merged INIT_MERGED=1 \
  ROWS=data/ami_multispk_si_v1/e2e_gstar_stream_v1/rows_train.jsonl TRAIN_ITERS=3000 \
  bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/thinker_gstar_s2dl_${TS}.log 2>&1
grep -q THINKER_GSTAR_DONE logs/multispk/thinker_gstar_s2dl_${TS}.log || log "S2-dl failed; its decode jobs will fail"
GPUQ_CARDS=0,1,2,3,4,5,6,7 /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_queue/gpuq6.py \
  reports/gstar_si_paper_v1/jobs/v3_jobs.json logs/gstar_v3_eval
log "QUEUE_DONE"
sleep 15
bash scripts/gpu_queue/holder_on.sh 0,1,3,4
bash scripts/gpu_queue/holder_on.sh 2,5,6,7
log "V3_EVAL_DONE"
