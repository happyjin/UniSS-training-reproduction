#!/usr/bin/env bash
# Paper v4 (multi-speaker, no Bayesian commit rule) on cards 4-7 only (0-3 belong to another job): stop the 4-7 holder,
#   p1  En->Zh S2-dl at 0.5 / 2 s (gstar_multi, 4 runs per card)
#   sa  speech of the existing systems (make_v4_speech_jobs.py set a, gpuq6 on 4-7)
#   tr  Zh->En S2-dl: O2 continued on build_ami_gstar_stream_rows_zh2en rows, 3,000 steps as En->Zh, 4 cards
#   p3  Zh->En S2-dl at 1 s on AMI and AliMeeting;  sb  its speech;  p4  Zh->En S2-dl at 0.5 / 2 s
# then the 4-7 holder again. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; OUT=${EV}/gstar_v4_20261030; J=reports/gstar_si_paper_v1/jobs
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python; PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation; D=data/ami_multispk_si_v1
S2DL=${EV}/thinker_gstar_s2dl_20261029b/thinker_merged; RUNZ=thinker_gstar_s2dl_zh2en_20261030; S2DLZ=${EV}/${RUNZ}/thinker_merged
[ -e ${OUT} ] && { log "拒绝覆盖已有输出: ${OUT}"; exit 1; }
mkdir -p ${OUT} logs/multispk/v4

multi() {   # phase thinker systems...
  local ph=$1 th=$2; shift 2
  for s in "$@"; do mkdir -p ${OUT}/$s; done
  local pids=()
  for c in 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker ${th} \
      ${J}/v4_runs/${ph}_card$c.json > logs/multispk/v4/${ph}_card$c.log 2>&1 &
    pids+=($!); sleep 20
  done
  local ok=0
  for p in "${pids[@]}"; do wait $p && ok=$((ok+1)); done
  log "${ph}: ${ok}/4 cards done"
}
score() {   # system windows
  local d=${OUT}/$1
  cat $d/hyp_s*.jsonl > $d/hyp.jsonl
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows $2 --hyp $d/hyp.jsonl --post-key lowlat --out $d/score.json > $d/score.log 2>&1
  PYTHONPATH=. ${PYE} -m ${M}.gstar_analyze --windows $2 --hyp $d/hyp.jsonl --out $d/analysis.json > /dev/null 2>&1 || true
  log "$1: $(grep -m1 cpBLEU_session $d/score.json) $(grep -m1 LAAL_s $d/score.json)"
}
WE=${D}/e2e_gstar_stream_v1/windows_test_ref.jsonl; WZ=${D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl; WA=data/alimeeting_si_v1/windows_test_ref.jsonl

log "p1"
multi p1 ${S2DL} en2zh_S2dl_d0.5 en2zh_S2dl_d2
score en2zh_S2dl_d0.5 ${WE}; score en2zh_S2dl_d2 ${WE}

log "sa: speech of existing systems"
GPUQ_CARDS=4,5,6,7 ${PYE} -u scripts/gpu_queue/gpuq6.py ${J}/v4_speech_jobs_a.json logs/multispk/v4/speech_a

log "tr: train ${RUNZ}"
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 MBS=16 GBS=64 RUN_ID=${RUNZ} BASE=${EV}/thinker_gstar_o2_20261029b/thinker_merged \
  INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_zh2en_v1/rows_train.jsonl TRAIN_ITERS=3000 MASTER_PORT=29851 \
  bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${RUNZ}.log 2>&1
if ! grep -q THINKER_GSTAR_DONE logs/multispk/${RUNZ}.log; then
  log "tr: MBS 16 failed ($(grep -m1 -E 'OutOfMemory|out of memory|FAILED' logs/multispk/${RUNZ}.log | cut -c1-120)); MBS 8"
  CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 MBS=8 GBS=64 RUN_ID=${RUNZ} BASE=${EV}/thinker_gstar_o2_20261029b/thinker_merged \
    INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_zh2en_v1/rows_train.jsonl TRAIN_ITERS=3000 MASTER_PORT=29852 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${RUNZ}.mbs8.log 2>&1
fi
if [ -d ${S2DLZ} ]; then
  log "tr: done"
  log "p3"
  multi p3 ${S2DLZ} zh2en_S2dl_d1 ali_S2dl_d1
  score zh2en_S2dl_d1 ${WZ}; score ali_S2dl_d1 ${WA}
  log "sb: speech of zh2en_S2dl_d1"
  GPUQ_CARDS=4,5,6,7 ${PYE} -u scripts/gpu_queue/gpuq6.py ${J}/v4_speech_jobs_b.json logs/multispk/v4/speech_b
  log "p4"
  multi p4 ${S2DLZ} zh2en_S2dl_d0.5 zh2en_S2dl_d2
  score zh2en_S2dl_d0.5 ${WZ}; score zh2en_S2dl_d2 ${WZ}
else
  log "tr: FAILED, no Zh->En S2-dl"
fi
sleep 15
bash scripts/gpu_queue/holder_on.sh 4,5,6,7
log "V4_DONE"
