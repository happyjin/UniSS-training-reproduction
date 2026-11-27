#!/usr/bin/env bash
# change 60 EGA driver: waits for the SAA G0 (pv4/g0_s1.json or pvx/CHAIN_s1_DONE), picks the tail adapter by the prereg rule (higher G0
# en2zh of C1 = c1tr / SAA = c2tr, tie -> SAA; SAA without a G0 -> C1), runs the EGA smoke (pv6_egasmoke.sh) in the background and the
# generic chain M=q1 (G0-EGA with dev_offline_ega -> waits EGA_SMOKE_OK -> streaming pfq1 with hsc25 --ega -> test p3pfq1).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D0=eval_outputs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ -n "${DRY_RUN:-}" ]; then echo "[dry] pv6_ega"; exit 0; fi
mkdir -p $D0/pv6
until [ -e $D0/pv4/g0_s1.json ] || [ -e $D0/pvx/CHAIN_s1_DONE ]; do sleep 30; done
sel=$(python3 - <<PY
import json, os
a = json.load(open("$D0/pv4/g0_c1.json"))["en2zh"]["model"]
p = "$D0/pv4/g0_s1.json"
b = json.load(open(p))["en2zh"]["model"] if os.path.exists(p) else -1
print("c2tr" if b >= a else "c1tr")
PY
)
log "EGA tail adapter: $sel"
echo "$sel" > $D0/pv6/ega_tail_adapter.txt
LORA=$D0/$sel/peft/a1.0000 setsid nohup bash experiments/sst_hsmm_v1/pv6_egasmoke.sh > logs/sst_hsmm_v1/pv6_egasmoke_orch.log 2>&1 < /dev/null 8>&- 9>&- &
M=q1 PT=pfq1 ON=experiments.sst_hsmm_v1.hsc25_si_thinker XF="--draft-lora $D0/$sel/peft/a1.0000 --ega" TR=$sel OFFMOD=dev_offline_ega \
  WAITF=$D0/pv6/EGA_SMOKE_OK bash experiments/sst_hsmm_v1/pvx_chain.sh
