#!/usr/bin/env bash
# Paper v4 under the compute deadline (GPU work must end by 10:15 UTC): Zh->En S2-dl for 1,000 steps on cards 4-7 (as the
# stopped 3,000-step run, same rows, MBS 16, GBS 64), then its decoding at 1 s on AMI and AliMeeting (make_v4c_runs.py).
# Anything still decoding at 10:12 UTC is stopped by PID and reported as unfinished. Launch detached; check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; OUT=${EV}/gstar_v4_20261030; J=reports/gstar_si_paper_v1/jobs
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python; PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation; D=data/ami_multispk_si_v1
RUN=thinker_gstar_s2dl_zh2en_1k_20261030; S2DLZ=${EV}/${RUN}/thinker_merged
DEADLINE=$(date -u -d "10:12" +%s)
log "train ${RUN}"
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 MBS=16 GBS=64 RUN_ID=${RUN} BASE=${EV}/thinker_gstar_o2_20261029b/thinker_merged \
  INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_zh2en_v1/rows_train.jsonl TRAIN_ITERS=1000 MASTER_PORT=29855 \
  bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${RUN}.log 2>&1
[ -d ${S2DLZ} ] || { log "train FAILED"; exit 1; }
log "trained; decode"
for s in zh2en_S2dl_d1 ali_S2dl_d1; do mkdir -p ${OUT}/$s; done
pids=()
for c in 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker ${S2DLZ} \
    ${J}/v4_runs/p3k_card$c.json > logs/multispk/v4/p3k_card$c.log 2>&1 &
  pids+=($!); sleep 10
done
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
  [ $alive -eq 0 ] && break
  if [ $(date -u +%s) -ge ${DEADLINE} ]; then
    log "deadline: stopping unfinished decoding by PID"
    for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && { pkids=$(ps -o pid= --ppid $p); kill $p $pkids; log "stopped $p $pkids"; }; done
    break
  fi
  sleep 20
done
score() {
  local d=${OUT}/$1
  local n=$(cat $d/hyp_s*.jsonl 2>/dev/null | wc -l)
  [ "$n" -eq "$3" ] || { log "$1: unfinished ($n / $3 windows), not scored"; return; }
  cat $d/hyp_s*.jsonl > $d/hyp.jsonl
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows $2 --hyp $d/hyp.jsonl --post-key lowlat --out $d/score.json > $d/score.log 2>&1
  log "$1: $(grep -m1 cpBLEU_session $d/score.json) $(grep -m1 LAAL_s $d/score.json)"
}
score zh2en_S2dl_d1 ${D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl 3317
score ali_S2dl_d1 data/alimeeting_si_v1/windows_test_ref.jsonl 4243
log "V4C_DONE"
