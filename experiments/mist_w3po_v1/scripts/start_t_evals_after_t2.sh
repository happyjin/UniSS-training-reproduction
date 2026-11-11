#!/usr/bin/env bash
# After the T2 training exits (CRF_RUN_EXIT in its log), slot A (0-3) becomes the eval pool (4-7 stay excluded: T3
# trains there). Launches thinker_t_eval_orch.py orchestrators 60 s apart (reports/mist_w3po_v1/
# thinker_crf_t_prereg_v1.zh-CN.md T1/T2 评测, T1-ctrl, T3 评测's T2 + PL-EM pairing):
#   1. T1       c1b1s2, gate vs A'                         (always: T1 finished before T2)
#   2. T1-ctrl  A' Thinker (v2' step_03000) with 0.5 s reads, c1b1s2, gate vs A'
#   3. T2       c1b1s2, gate vs A'                         (only if T2 exited rc=0 with step_03000/DONE)
#   4. T2+PL-EM speech only on T2's text with the PL-EM talker, gate vs A' + PL-EM (after T2's score_text.json)
# slotA.owner is released (renamed) once T2's processes are gone. bash -n / DRY_RUN=1 only.
#   T1_RUN=<id> T2_RUN=<id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t_evals_after_t2.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
PY=$J/conda_envs/qwen3omni/bin/python
O=experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py
TR=eval_outputs/mist_w3po_v1/crf/train
T1_RUN=${T1_RUN:?}
T2_RUN=${T2_RUN:?}
APRIME=dev_crf_v2p_s3000_finalpost_s8_c2g06
PLEMREF=${APRIME}__plem_c1b1s2
RUNS=logs/mist_w3po_v1/t_eval
log() { echo "$(date -u +%FT%TZ) $*"; }
orch() {   # run-name decode-name port0 args...
  local r=$1 n=$2 p=$3; shift 3
  local cmd="$PY $O --run $RUNS/$r --name $n --scorer-port0 $p --exclude 4 5 6 7 $*"
  if [ -n "${DRY_RUN:-}" ]; then echo "$cmd"; return; fi
  mkdir -p $RUNS
  setsid nohup $cmd < /dev/null > $RUNS/$r.launch.log 2>&1 &
  log "orchestrator $r ($n) pid $!"
}
T2L=logs/mist_w3po_v1/$T2_RUN.log
if [ -z "${DRY_RUN:-}" ]; then
  until grep -q CRF_RUN_EXIT "$T2L"; do sleep 15; done
  log "T2 exit: $(grep CRF_RUN_EXIT $T2L)"
  until ! pgrep -f "train_crf_sft.py .*--out $TR/$T2_RUN" > /dev/null; do sleep 5; done
  if [ -e logs/mist_w3po_v1/slots/slotA.owner ]; then
    command mv -f logs/mist_w3po_v1/slots/slotA.owner logs/mist_w3po_v1/slots/slotA.released_$(date -u +%Y%m%dT%H%M%SZ)
    log "slotA.owner released"
  fi
fi
orch t1 dev_t1_piece_t0_s3000_w05 18801 --ckpt $TR/$T1_RUN/ckpt/step_03000 --talker c1b1s2 --gate-ref $APRIME ${APRIME}_tts2
[ -z "${DRY_RUN:-}" ] && sleep 60
orch t1ctrl ${APRIME}_w05ctrl 18811 --ckpt $TR/crf_v2p_20261110T075131Z/ckpt/step_03000 --talker c1b1s2 \
  --gate-ref $APRIME ${APRIME}_tts2
if grep -q 'CRF_RUN_EXIT rc=0' "$T2L" 2>/dev/null && [ -e $TR/$T2_RUN/ckpt/step_03000/DONE ] || [ -n "${DRY_RUN:-}" ]; then
  [ -z "${DRY_RUN:-}" ] && sleep 60
  orch t2 dev_t2_latent_t0_s3000_w05 18821 --ckpt $TR/$T2_RUN/ckpt/step_03000 --talker c1b1s2 \
    --gate-ref $APRIME ${APRIME}_tts2
  if [ -z "${DRY_RUN:-}" ]; then
    until [ -e eval_outputs/mist_w3po_v1/decode/dev_t2_latent_t0_s3000_w05/score_text.json ]; do sleep 30; done
  fi
  orch t2plem dev_t2_latent_t0_s3000_w05__plem 18831 --ckpt - --speech-from dev_t2_latent_t0_s3000_w05 \
    --talker plem --gate-ref $PLEMREF ${PLEMREF}_tts2 --gate-ref $APRIME ${APRIME}_tts2
else
  log "T2 did not finish cleanly: no T2 eval"
fi
