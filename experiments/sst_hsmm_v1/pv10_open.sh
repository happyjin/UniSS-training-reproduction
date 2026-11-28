#!/usr/bin/env bash
# change 66 HCG-open driver: smoke + degeneracy rule (pv10_opensmoke.sh) -> generic chain M=o1 (no G0; streaming pfo1 with hsc28
# --draft-lora p2ft --draft-tag --mid-mbr 4 --mbr-util hcg --hcg-open -> test p3pfo1). Degenerate or failed smoke touches pvx/CHAIN_o1_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D0=eval_outputs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] pv10_open"; exit 0; fi
bash experiments/sst_hsmm_v1/pv10_opensmoke.sh > logs/sst_hsmm_v1/pv10_opensmoke_orch.log 2>&1 < /dev/null 8>&- 9>&-
[ -e $D0/pv10/OPEN_SMOKE_OK ] || { log "smoke failed or degenerate ($(ls $D0/pv10 2>/dev/null | tr '\n' ' '))"; mkdir -p $D0/pvx; touch $D0/pvx/CHAIN_o1_DONE; exit 0; }
M=o1 PT=pfo1 ON=experiments.sst_hsmm_v1.hsc28_si_thinker XF="--draft-lora $D0/p2ft/peft/a1.0000 --draft-tag --mid-mbr 4 --mbr-util hcg --hcg-open" \
  bash experiments/sst_hsmm_v1/pvx_chain.sh
