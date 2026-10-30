#!/usr/bin/env bash
# Paper v4, second chain on cards 4-7 (after run_v4.sh, whose PID is $1): the reviewer's multi-interval training.
# S2-dl-m: O2 continued on deadline-label reads at intervals drawn from {0.5, 1, 2} s per window
# (build_ami_gstar_stream_rows_multid), 3,000 steps as S2-dl, 4 cards; decoded at 0.5 / 1 / 2 s; then the
# diarizer-defined En->Zh windows for the v4 models (make_v4_diar_jobs.py); holder back on 4-7.
# Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; OUT=${EV}/gstar_v4_20261030; J=reports/gstar_si_paper_v1/jobs
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python; PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation; D=data/ami_multispk_si_v1
RUN=thinker_gstar_s2dlm_20261030; S2DLM=${EV}/${RUN}/thinker_merged; WE=${D}/e2e_gstar_stream_v1/windows_test_ref.jsonl
PREV="${1:?PID of run_v4.sh}"
while kill -0 "${PREV}" 2>/dev/null; do sleep 60; done
log "run_v4.sh finished; holder off 4-7"
bash scripts/gpu_queue/holder_off.sh 4,5,6,7
sleep 30
log "train ${RUN}"
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 MBS=16 GBS=64 RUN_ID=${RUN} BASE=${EV}/thinker_gstar_o2_20261029b/thinker_merged \
  INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_multid_v1/rows_train.jsonl TRAIN_ITERS=3000 MASTER_PORT=29853 \
  bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${RUN}.log 2>&1
if ! grep -q THINKER_GSTAR_DONE logs/multispk/${RUN}.log; then
  log "train: MBS 16 failed ($(grep -m1 -E 'OutOfMemory|out of memory|FAILED' logs/multispk/${RUN}.log | cut -c1-120)); MBS 8"
  CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 MBS=8 GBS=64 RUN_ID=${RUN} BASE=${EV}/thinker_gstar_o2_20261029b/thinker_merged \
    INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_multid_v1/rows_train.jsonl TRAIN_ITERS=3000 MASTER_PORT=29854 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${RUN}.mbs8.log 2>&1
fi
if [ -d ${S2DLM} ]; then
  for w in 0.5 1 2; do mkdir -p ${OUT}/en2zh_S2dlm_d$w; done
  pids=()
  for c in 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker ${S2DLM} \
      ${J}/v4_runs/p5_card$c.json > logs/multispk/v4/p5_card$c.log 2>&1 &
    pids+=($!); sleep 20
  done
  for p in "${pids[@]}"; do wait $p; done
  for w in 0.5 1 2; do
    d=${OUT}/en2zh_S2dlm_d$w
    cat $d/hyp_s*.jsonl > $d/hyp.jsonl
    PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows ${WE} --hyp $d/hyp.jsonl --post-key lowlat --out $d/score.json > $d/score.log 2>&1
    PYTHONPATH=. ${PYE} -m ${M}.gstar_analyze --windows ${WE} --hyp $d/hyp.jsonl --out $d/analysis.json > /dev/null 2>&1 || true
    log "en2zh_S2dlm_d$w: $(grep -m1 cpBLEU_session $d/score.json) $(grep -m1 LAAL_s $d/score.json)"
  done
else
  log "train FAILED"
fi
log "diarizer-defined En->Zh windows for the v4 models (make_v4_diar_jobs.py)"
GPUQ_CARDS=4,5,6,7 ${PYE} -u scripts/gpu_queue/gpuq6.py ${J}/v4_diar_jobs.json logs/multispk/v4/diar
sleep 15
bash scripts/gpu_queue/holder_on.sh 4,5,6,7
log "V4B_DONE"
