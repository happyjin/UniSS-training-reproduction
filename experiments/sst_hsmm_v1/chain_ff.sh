#!/usr/bin/env bash
# change 49+: train a drafter LoRA (train_orch.sh) -> rollouts + HSMM fit on its merged copy (drafter_pipe2.sh STOP_AFTER_FIT=1,
# incl. whole-audio dev decodes) -> faithful-finalisation dev + test (ff_pipe.sh, TAG ${TAG}ff, --draft-lora $R/peft/a1.0000).
#   TAG=.. MOD=.. PORT=.. WAIT=.. setsid nohup bash experiments/sst_hsmm_v1/chain_ff.sh > logs/sst_hsmm_v1/${TAG}_chain.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
: "${TAG:?}" "${MOD:?}" "${PORT:?}"
R=eval_outputs/sst_hsmm_v1/$TAG
L=logs/sst_hsmm_v1
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
TAG=$TAG MOD=$MOD PORT=$PORT WAIT=${WAIT:-} bash experiments/sst_hsmm_v1/train_orch.sh > $L/${TAG}_train.log 2>&1 < /dev/null
[ -e $R/TRAIN_DONE ] || { log "train did not finish"; exit 1; }
log "train done"
TAG=$TAG DTH=$R/merged STOP_AFTER_FIT=1 bash experiments/sst_hsmm_v1/drafter_pipe2.sh > $L/${TAG}.log 2>&1 < /dev/null
[ -e $R/h/FIT_DONE ] || { log "fit did not finish"; exit 1; }
log "fit done"
bash experiments/sst_hsmm_v1/make_ff_report.sh ${TAG}ff $R/peft/a1.0000 $R/h/params.json || { log "make report"; exit 1; }
TAG=${TAG}ff DL=$R/peft/a1.0000 PRM=$R/h/params.json bash experiments/sst_hsmm_v1/ff_pipe.sh > $L/${TAG}ff.log 2>&1 < /dev/null
log "ff done rc=$?"
