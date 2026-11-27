#!/usr/bin/env bash
# change 63 HCD driver: smoke (pv9_hcdsmoke.sh, E1 p2ft adapter) -> generic chain M=k1 (no G0; WAITF smoke ok; streaming pfk1 with hsc27
# --draft-lora p2ft --draft-tag --hcd -> test p3pfk1). A failed smoke touches pvx/CHAIN_k1_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D0=eval_outputs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] pv9_hcd"; exit 0; fi
LORA=$D0/p2ft/peft/a1.0000 bash experiments/sst_hsmm_v1/pv9_hcdsmoke.sh > logs/sst_hsmm_v1/pv9_hcdsmoke_orch.log 2>&1 < /dev/null 8>&- 9>&-
[ -e $D0/pv9/HCD_SMOKE_OK ] || { log "smoke failed"; mkdir -p $D0/pvx; touch $D0/pvx/CHAIN_k1_DONE; exit 0; }
M=k1 PT=pfk1 ON=experiments.sst_hsmm_v1.hsc27_si_thinker XF="--draft-lora $D0/p2ft/peft/a1.0000 --draft-tag --hcd" \
  bash experiments/sst_hsmm_v1/pvx_chain.sh
