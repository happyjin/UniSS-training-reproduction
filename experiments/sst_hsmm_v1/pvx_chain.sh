#!/usr/bin/env bash
# change 59+: generic method chain. env M (name), PT (pipeline tag), ON (Thinker module), XF (Thinker flags); optional TR (LoRA training dir:
# wait TR/TRAIN_DONE -> G0 with OFFMOD -> on PASS continue), optional WAITF (file to wait for instead). Streaming = pf_lane_lib2 run
# (rollout + refit hsmm_anchor + dev with the official BLEU + test p3$PT). Touches $D0/pvx/CHAIN_${M}_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${M:?}" "${PT:?}" "${ON:?}" "${XF:?}"
D0=eval_outputs/sst_hsmm_v1
TR=${TR:-}; OFFMOD=${OFFMOD:-dev_offline_plain}; WAITF=${WAITF:-}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fin() { touch $D0/pvx/CHAIN_${M}_DONE; log "chain $M done: $*"; exit 0; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] $M: TR=$TR OFFMOD=$OFFMOD PT=$PT ON=$ON XF=$XF WAITF=$WAITF"; exit 0; fi
mkdir -p $D0/pvx
[ -e $D0/pvx/CHAIN_${M}_DONE ] && { echo "exists CHAIN_${M}_DONE" >&2; exit 1; }
if [ -n "$TR" ]; then
  until [ -e $D0/$TR/TRAIN_DONE ]; do { [ -e $D0/$TR/DIVERGED ] || [ -e $D0/$TR/FAILED ]; } && fin "training did not finish"; sleep 30; done
  M=$M TR=$TR OFFMOD=$OFFMOD bash experiments/sst_hsmm_v1/pv4_g0.sh > logs/sst_hsmm_v1/pv4_g0_${M}_orch.log 2>&1 < /dev/null 8>&- 9>&-
  [ -e $D0/pv4/G0_${M}_PASS ] || fin "G0 not passed ($(ls $D0/pv4 | grep G0_${M}_ | tr '\n' ' '))"
fi
[ -n "$WAITF" ] && until [ -e "$WAITF" ]; do sleep 30; done
bash experiments/sst_hsmm_v1/make_pf_report.sh $PT $ON $D0/$PT/h/params.json "$XF" eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 || fin "report script"
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
run $PT ONLINE=$ON ROLL=1 FIT=hsmm_anchor XF="$XF"
fin "$(ls $D0/$PT 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ') test $(ls $D0/p3$PT 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"
