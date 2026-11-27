#!/usr/bin/env bash
# change 58 chain v2 (C2' only): M=g2 -> wait pv5/HCG2_SMOKE_OK -> streaming pfg2 with hsc24. Was: change 58 chain for one round-2 method M
# streaming pfg1 with hsc23). Streaming = pf_lane_lib2 run (rollout + refit hsmm_anchor + dev with the official BLEU + test p3$PT).
# Touches pv5/CHAIN_${M}_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${M:?}"
D0=eval_outputs/sst_hsmm_v1
ON=experiments.sst_hsmm_v1.hsc22_si_thinker
case $M in
  c1) TR=c1tr; OFFMOD=dev_offline_plain; PT=pfi1; XF="--draft-lora $D0/c1tr/peft/a1.0000";;
  g2) TR=""; PT=pfg2; ON=experiments.sst_hsmm_v1.hsc24_si_thinker; XF="--draft-lora $D0/p2ft/peft/a1.0000 --draft-tag --mid-mbr 4 --mbr-util hcg";;
  *) echo "bad M" >&2; exit 1;;
esac
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fin() { touch $D0/pv5/CHAIN_${M}_DONE; log "chain $M done: $*"; exit 0; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] $M: $TR $PT $ON XF=$XF"; exit 0; fi
mkdir -p $D0/pv5
if [ -n "$TR" ]; then
  until [ -e $D0/$TR/TRAIN_DONE ]; do { [ -e $D0/$TR/DIVERGED ] || [ -e $D0/$TR/FAILED ]; } && fin "training did not finish"; sleep 30; done
  M=$M TR=$TR OFFMOD=$OFFMOD bash experiments/sst_hsmm_v1/pv4_g0.sh > logs/sst_hsmm_v1/pv4_g0_${M}_orch.log 2>&1 < /dev/null 8>&- 9>&-
  [ -e $D0/pv4/G0_${M}_PASS ] || fin "G0 not passed ($(ls $D0/pv4 | grep G0_${M}_ | tr '\n' ' '))"
else
  until [ -e $D0/pv5/HCG2_SMOKE_OK ]; do [ -e $D0/pv5/hcgsmoke2/FAILED ] && fin "smoke failed"; sleep 30; done
fi
bash experiments/sst_hsmm_v1/make_pf_report.sh $PT $ON $D0/$PT/h/params.json "$XF" eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 || fin "report script"
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
run $PT ONLINE=$ON ROLL=1 FIT=hsmm_anchor XF="$XF"
fin "$(ls $D0/$PT 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ') test $(ls $D0/p3$PT 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"
