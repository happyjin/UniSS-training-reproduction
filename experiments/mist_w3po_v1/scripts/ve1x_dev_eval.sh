#!/usr/bin/env bash
# VE1 EXPLORATORY dev evaluation (coordinator for the user, 2026-11-12 ~05:05 Prague). A documented deviation: VE1's
# registered dev5 quick check is NOT GO (text ov 23.28 < 23.92) and stays NOT GO; these results are descriptive only
# and change no verdict. t_after_train_rl.sh's post-GO stages (that script is untouched) with the coordinator's changes:
#   1. dev decode (8 shards, 1.0 s reads, --talker none: decode + text score + text latency)   run t_eval/ve1x
#   2. read-level merged copy (crf/merge_read_segments.py) -> <decode>_rl
#   3. speech chain on the merged copy with TWO talkers, 2 TTS seeds each, started 60 s apart:
#        PL-EM (primary)   run t_eval/ve1xrlplem   name dev_ve1_s3000__rlplem
#        c1+b1+s2          run t_eval/ve1xrl       name dev_ve1_s3000__rl
#      gates (gate_2seed) vs A' (c1b1s2), A' + PL-EM and A'rl + PL-EM
#   No per-segment ablation chain. Every stage --cap135 --exclude 4 5 6 7 (cards 0-3 only; RC5 trains on 4-7 and its
#   quick check there has priority, so nothing of this run is placed on 4-7 even after RC5's slot is released).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/ve1x_dev_eval.sh
# Run detached: setsid nohup bash .../ve1x_dev_eval.sh > logs/mist_w3po_v1/ve1/ve1x_dev_eval.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/t_eval
DEC=eval_outputs/mist_w3po_v1/decode
CK=eval_outputs/mist_w3po_v1/crf/train/ve1_20261111T214454Z/ckpt/step_03000
F=dev_ve1_s3000
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
EX="--cap135 --exclude 4 5 6 7"
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2"
DCMD="$ORCH --run $E/ve1x --name $F --ckpt $CK --talker none --window 1 --scorer-port0 19301 $EX"
PCMD="$ORCH --run $E/ve1xrlplem --name ${F}__rlplem --ckpt - --speech-from ${F}_rl --talker plem --window 1 $REFS --scorer-port0 19311 $EX"
CCMD="$ORCH --run $E/ve1xrl --name ${F}__rl --ckpt - --speech-from ${F}_rl --talker c1b1s2 --window 1 $REFS --scorer-port0 19321 $EX"
MCMD="$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl"
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
clean_done() { grep -q 'DONE; failed/skipped \[\]' $1/orch.log; }
EXIST=""
for d in $E/ve1x $E/ve1xrl $E/ve1xrlplem $DEC/$F $DEC/${F}_rl $DEC/${F}__rl $DEC/${F}__rlplem; do [ -e $d ] && EXIST="$EXIST $d"; done
REFOK=1
for r in $A ${A}_tts2 ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 ${A}__rlplem ${A}__rlplem_tts2; do
  [ -s $DEC/$r/score_asr.json ] || { REFOK=0; echo "missing gate reference $DEC/$r/score_asr.json"; }
done
if [ -n "${DRY_RUN:-}" ]; then
  echo "checkpoint: $(ls $CK/DONE $CK/lora.pt $CK/extra.pt 2>&1 | tr '\n' ' ')"
  echo "existing outputs (must be none):${EXIST:- none}; gate references ok: $REFOK"
  echo "ports 19301-19324 in use: $(ss -ltn 2>/dev/null | grep -cE ':(19[3][0-2][0-9]) ')"
  echo "decode: $DCMD"; echo "merge: $MCMD"; echo "speech PL-EM: $PCMD"; echo "speech c1b1s2: $CCMD"
  $DCMD --dry | tail -n 3
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $REFOK = 1 ] || { log "gate references missing: not started"; exit 1; }
mkdir -p $E
setsid nohup $DCMD < /dev/null > $E/ve1x.launch.log 2>&1 &
log "dev decode orchestrator pid $!: $DCMD"
sleep 20
[ -e $E/ve1x/orch.log ] || { log "decode orchestrator did not start (see $E/ve1x.launch.log)"; exit 1; }
wait_done $E/ve1x
clean_done $E/ve1x || { log "decode orchestrator DONE with failed/skipped jobs: $(grep 'DONE;' $E/ve1x/orch.log); stop"; exit 1; }
log "decode DONE"
$MCMD > $E/ve1x_merge.log 2>&1 || { log "merge failed (see $E/ve1x_merge.log)"; exit 1; }
log "merged: $(tail -n 1 $E/ve1x_merge.log | cut -c1-300)"
setsid nohup $PCMD < /dev/null > $E/ve1xrlplem.launch.log 2>&1 &
log "PL-EM speech orchestrator pid $!: $PCMD"
sleep 60
setsid nohup $CCMD < /dev/null > $E/ve1xrl.launch.log 2>&1 &
log "c1b1s2 speech orchestrator pid $!: $CCMD"
wait_done $E/ve1xrlplem
log "PL-EM speech DONE: $(grep 'DONE;' $E/ve1xrlplem/orch.log)"
wait_done $E/ve1xrl
log "c1b1s2 speech DONE: $(grep 'DONE;' $E/ve1xrl/orch.log)"
log "VE1X_ALL_DONE"
