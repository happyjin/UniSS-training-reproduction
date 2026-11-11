#!/usr/bin/env bash
# User order 2026-11-11 12:15 UTC: when the T1 training exits, T1-quick (dev5 text-only decode of T1 step_03000 with
# 0.5 s reads; thinker_t_eval_orch.py --talker none, 4 window shards, cards 0-3 excluded = T2's training) on cards 4-7,
# then T3 training on 4-7 at once (takeover_train.sh: our standby placeholders stopped by PID, supervisor first, then
# run_crf_v2p.sh). Also scores A' on the same dev5 windows with the same text-latency tool (CPU). bash -n / DRY_RUN only.
#   T1_RUN=<id> T3_RUN=<id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t1quick_then_t3.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
TR=eval_outputs/mist_w3po_v1/crf/train
T1_RUN=${T1_RUN:?}
T3_RUN=${T3_RUN:?}
W5=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
NAME=dev5_t1_piece_t0_s3000_w05
RUN=logs/mist_w3po_v1/t_eval/t1quick
OUT=eval_outputs/mist_w3po_v1/crf/t_eval/$NAME
X3="--sched eval_outputs/mist_w3po_v1/crf/sched_piece_t0_v2 --objective joint_speech --talker-model eval_outputs/mist_w3po_v1/crf/t3_talker_model_v1.json"
log() { echo "$(date -u +%FT%TZ) $*"; }
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $RUN --name $NAME --ckpt $TR/$T1_RUN/ckpt/step_03000 --talker none --win $W5 --shards 4 --shard-by window --scorer-port0 18851 --exclude 0 1 2 3"
LAUNCH="CARDS=4,5,6,7 RUN_ID=$T3_RUN PORT=29981 EXTRA=\"$X3\" bash experiments/mist_w3po_v1/scripts/run_crf_v2p.sh"
if [ -n "${DRY_RUN:-}" ]; then echo "$ORCH"; echo "$LAUNCH"; exit 0; fi
L1=logs/mist_w3po_v1/$T1_RUN.log
until grep -q CRF_RUN_EXIT "$L1"; do sleep 5; done
log "T1 exit: $(grep CRF_RUN_EXIT $L1)"
if ! grep -q 'CRF_RUN_EXIT rc=0' "$L1" || [ ! -e $TR/$T1_RUN/ckpt/step_03000/DONE ]; then
  log "T1 did not finish cleanly: no T1-quick; launching T3 directly"
else
  mkdir -p logs/mist_w3po_v1/t_eval
  setsid nohup $ORCH < /dev/null > $RUN.launch.log 2>&1 &
  log "T1-quick orchestrator pid $!"
  mkdir -p $OUT
  $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule \
    --text-only --win $W5 --out $OUT/textlat_aprime_dev5.json > $OUT/textlat_aprime_dev5.log 2>&1
  log "A' dev5 text latency done rc=$?"
  until [ -e $RUN/events/DONE ]; do sleep 10; done
  log "T1-quick DONE: $(cat $RUN/status.json 2>/dev/null | head -c 300)"
fi
CARDS=4,5,6,7 LAUNCH="$LAUNCH" bash experiments/mist_w3po_v1/scripts/takeover_train.sh
