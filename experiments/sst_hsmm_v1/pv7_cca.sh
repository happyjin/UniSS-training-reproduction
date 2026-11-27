#!/usr/bin/env bash
# change 61 CCA driver: smoke (pv7_ccasmoke.sh, tail adapter = EGA's, pv6/ega_tail_adapter.txt) -> waits for CCA_SMOKE_OK and the G0-EGA
# result (CCA has no G0 of its own: the forced first tail token exists only on non-final reads, so on complete audio CCA = EGA) ->
# on G0_q1_PASS the generic chain M=r1 (no TR; streaming pfr1 with hsc26 --ega --ega-cont -> test p3pfr1). Touches pvx/CHAIN_r1_DONE.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D0=eval_outputs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] pv7_cca"; exit 0; fi
sel=$(cat $D0/pv6/ega_tail_adapter.txt)
log "CCA tail adapter: $sel"
LORA=$D0/$sel/peft/a1.0000 bash experiments/sst_hsmm_v1/pv7_ccasmoke.sh > logs/sst_hsmm_v1/pv7_ccasmoke_orch.log 2>&1 < /dev/null 8>&- 9>&-
[ -e $D0/pv7/CCA_SMOKE_OK ] || { log "smoke failed"; mkdir -p $D0/pvx; touch $D0/pvx/CHAIN_r1_DONE; exit 0; }
until ls $D0/pv4/G0_q1_* > /dev/null 2>&1; do sleep 30; done
[ -e $D0/pv4/G0_q1_PASS ] || { log "G0-EGA not passed"; touch $D0/pvx/CHAIN_r1_DONE; exit 0; }
M=r1 PT=pfr1 ON=experiments.sst_hsmm_v1.hsc26_si_thinker XF="--draft-lora $D0/$sel/peft/a1.0000 --ega --ega-cont" \
  bash experiments/sst_hsmm_v1/pvx_chain.sh
