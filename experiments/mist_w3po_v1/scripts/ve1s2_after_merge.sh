#!/usr/bin/env bash
# VE1-s2: the rest of ve1s2_after_train.sh after its merge step (that script stopped at 2026-11-12T09:05:31Z because
# crf/merge_read_segments.py exits 1 when pair_lost > 0; the merged copy decode/dev_ve1s2_s3000_rl was written in full:
# pair_lost 2 windows (IB4010_on_0067, TS3004c_on_0052: a final-read segment's write_ms differs from its block's), pair_ok
# 3599 / 3618 = the same count as VE1rl (3599) and above A'rl (3589); the speech chain already tolerates unpaired windows).
# Unchanged tool and commands; only the stop is bypassed after the coordinator's decision:
#   1. dev5 cut of the full decode (t_dev5_cut.sh, CPU, descriptive) in the background
#   2. PL-EM speech chain on the merged copy, 2 TTS seeds, gates vs A', A' + PL-EM, A'rl + PL-EM   run t_eval/ve1s2rlplem
#   --cap135 --exclude 4 5 6 7 (cards 0-3; 4-7 = PMC training). ve1s2_listen_after.sh (already waiting) follows.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/ve1s2_after_merge.sh
# Run detached: setsid nohup bash .../ve1s2_after_merge.sh > logs/mist_w3po_v1/ve1s2/after_merge.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/t_eval
DEC=eval_outputs/mist_w3po_v1/decode
F=dev_ve1s2_s3000
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
EX="--cap135 --exclude 4 5 6 7"
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
REFS="--gate-ref $A ${A}_tts2 --gate-ref ${A}__plem_c1b1s2 ${A}__plem_c1b1s2_tts2 --gate-ref ${A}__rlplem ${A}__rlplem_tts2"
PCMD="$ORCH --run $E/ve1s2rlplem --name ${F}__rlplem --ckpt - --speech-from ${F}_rl --talker plem --window 1 $REFS --scorer-port0 19341 $EX"
log() { echo "$(date -u +%FT%TZ) $*"; }
EXIST=""
for d in $E/ve1s2rlplem $DEC/${F}__rlplem $DEC/${F}__rlplem_tts2 $DEC/dev5cut_$F eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$F logs/mist_w3po_v1/ve1s2/dev5cut.log; do
  [ -e $d ] && EXIST="$EXIST $d"; done
MJ=$DEC/${F}_rl/merge_read_segments.json
PL=$(python3 -c "import json; print(json.load(open('$MJ'))['pair_lost'])" 2>/dev/null || echo missing)
DONE_OK=0; grep -q 'DONE; failed/skipped \[\]' $E/ve1s2/orch.log 2>/dev/null && DONE_OK=1
if [ -n "${DRY_RUN:-}" ]; then
  echo "decode run clean: $DONE_OK; merged copy pair_lost: $PL (want 2); existing outputs (must be none):${EXIST:- none}"
  echo "ports 19341-19344 in use: $(ss -ltn 2>/dev/null | grep -cE ':1934[1-4] ')"
  echo "dev5 cut: bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F"
  echo "speech PL-EM: $PCMD"
  exit 0
fi
[ -z "$EXIST" ] || { log "拒绝覆盖已有输出:$EXIST"; exit 1; }
[ $DONE_OK = 1 ] && [ "$PL" = 2 ] || { log "decode clean $DONE_OK / pair_lost $PL: not started"; exit 1; }
setsid nohup bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh $F < /dev/null > logs/mist_w3po_v1/ve1s2/dev5cut.log 2>&1 &
log "dev5 cut pid $!"
setsid nohup $PCMD < /dev/null > $E/ve1s2rlplem.launch.log 2>&1 &
log "PL-EM speech orchestrator pid $!: $PCMD"
until [ -e $E/ve1s2rlplem/events/DONE ]; do sleep 30; done
log "PL-EM speech DONE: $(grep 'DONE;' $E/ve1s2rlplem/orch.log)"
log "VE1S2_ALL_DONE"
