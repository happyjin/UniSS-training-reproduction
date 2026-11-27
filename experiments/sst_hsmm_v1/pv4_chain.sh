#!/usr/bin/env bash
# change 57 chain for one method M (b1 | b2): wait $TR/TRAIN_DONE -> G0 (pv4_g0.sh) -> on PASS: test script (make_pf_report.sh) and the
# streaming pipeline (pf_lane_lib2 run: rollout + refit hsmm_anchor + dev with the official BLEU + test p3$PT) -> pv4/CHAIN_${M}_DONE.
# On FAIL (or divergence) the method is reported from G0 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${M:?}"
D0=eval_outputs/sst_hsmm_v1
case $M in
  b1) TR=b1tr; OFFMOD=dev_offline_tag; PT=pfn1; XF="--draft-lora $D0/b1tr/peft/a1.0000 --draft-tag";;
  b2) TR=b2tr; OFFMOD=dev_offline_plain; PT=pff1; XF="--draft-lora $D0/b2tr/peft/a1.0000";;
  *) echo "bad M" >&2; exit 1;;
esac
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fin() { touch $D0/pv4/CHAIN_${M}_DONE; log "chain $M done: $*"; exit 0; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] $M: $TR $OFFMOD $PT XF=$XF"; exit 0; fi
mkdir -p $D0/pv4
until [ -e $D0/$TR/TRAIN_DONE ]; do { [ -e $D0/$TR/DIVERGED ] || [ -e $D0/$TR/FAILED ]; } && fin "training did not finish"; sleep 30; done
M=$M TR=$TR OFFMOD=$OFFMOD bash experiments/sst_hsmm_v1/pv4_g0.sh > logs/sst_hsmm_v1/pv4_g0_${M}_orch.log 2>&1 < /dev/null 8>&- 9>&-
[ -e $D0/pv4/G0_${M}_PASS ] || fin "G0 not passed ($(ls $D0/pv4 | grep G0_${M}_ | tr '\n' ' '))"
bash experiments/sst_hsmm_v1/make_pf_report.sh $PT experiments.sst_hsmm_v1.hsc22_si_thinker $D0/$PT/h/params.json "$XF" eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 || fin "report script"
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
run $PT ROLL=1 FIT=hsmm_anchor XF="$XF"
fin "$(ls $D0/$PT 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ') test $(ls $D0/p3$PT 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"
