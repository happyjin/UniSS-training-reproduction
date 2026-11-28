#!/usr/bin/env bash
# change 67 chain: TAF prechecks (taf_precheck.sh, K=8; if only P0 fails: one retry with K=4 as preregistered) -> on GO the streaming
# dev + test pipeline TAG pft1 (pf_pipe2, ROLL=0, FIT=none, h14, ONLINE=hsc30, XF="--taf K") -> $D0/taf/CHAIN_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D0=eval_outputs/sst_hsmm_v1
ON=experiments.sst_hsmm_v1.hsc30_si_thinker
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fin() { touch $D0/taf/CHAIN_DONE; log "taf chain done: $*"; exit 0; }
if [ -n "${DRY_RUN:-}" ]; then DRY_RUN=1 K=8 bash experiments/sst_hsmm_v1/taf_precheck.sh; echo "[dry] then pft1 ON=$ON XF=--taf K"; exit 0; fi
mkdir -p $D0/taf
[ -e $D0/taf/CHAIN_DONE ] && { echo "exists CHAIN_DONE" >&2; exit 1; }
K=8 bash experiments/sst_hsmm_v1/taf_precheck.sh > logs/sst_hsmm_v1/taf_pre_K8_orch.log 2>&1 < /dev/null 8>&- 9>&-
K_=8
if [ ! -e $D0/taf/PRE_K8_GO ]; then
  [ -e $D0/taf/precheck_K8.json ] || fin "K8 precheck failed to finish"
  if python3 -c "import json,sys; r=json.load(open('$D0/taf/precheck_K8.json')); sys.exit(0 if (r['go_p1'] and not r['go_p0']) else 1)"; then
    log "K8: P1 GO, P0 NOGO -> preregistered retry with K=4"
    K=4 bash experiments/sst_hsmm_v1/taf_precheck.sh > logs/sst_hsmm_v1/taf_pre_K4_orch.log 2>&1 < /dev/null 8>&- 9>&-
    [ -e $D0/taf/PRE_K4_GO ] || fin "K4 precheck NOGO"
    K_=4
  else
    fin "K8 precheck NOGO (P1)"
  fi
fi
log "precheck GO with K=$K_ -> streaming pft1"
bash experiments/sst_hsmm_v1/make_pf_report.sh pft1 $ON $D0/h14/params.json "--taf $K_" eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05 || fin "report script"
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
run pft1 ROLL=0 FIT=none ONLINE=$ON XF="--taf $K_"
fin "$(ls $D0/pft1 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ') test $(ls $D0/p3pft1 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')"
