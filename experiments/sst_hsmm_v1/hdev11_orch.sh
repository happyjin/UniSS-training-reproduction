#!/usr/bin/env bash
# prereg change 38 dev wave (copy of hdev10_orch.sh with hsc10_si_thinker --samples $KS --samp-max $MT, defaults 6 / 24). Was: prereg change 37 dev wave (copy of hdev8_orch.sh; DATA / VIEWS / VER / CARDS from the environment, defaults = hdev8). Was: prereg change 35 dev wave (copy of hdev6_orch.sh; theta grid from $THS, thinker hsc8). Was: prereg change 33 dev wave (copy of hdev5_orch.sh): hsc8_si_thinker --variant $VAR (params $PRM) on the dev items (wave2_S1, wave15 views) for theta in
# {0.80, 0.90, 0.95, 0.98} x 2 directions = 8 jobs on cards 0-7 -> eval_outputs/sst_hsmm_v1/$OUTN/dev_<theta>_<d>.
#   VAR=soft PRM=eval_outputs/sst_hsmm_v1/h3/params.json OUTN=h6soft bash experiments/sst_hsmm_v1/hdev_orch.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${VAR:?}" "${PRM:?}" "${OUTN:?}" "${THS:?}"
E=eval_outputs/sst_hsmm_v1/$OUTN
R=$E/devrun
DATA=${DATA:-eval_outputs/mist_w3po_v1/wave2_S1/data}
VIEWS=${VIEWS:-eval_outputs/mist_w3po_v1/wave15/views}
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=${VER:-eval_outputs/sst_hsmm_v1/v/lora/a1.0000}
LOGD=logs/sst_hsmm_v1/${OUTN}dev
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -s $PRM ] || { echo "params missing" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
ps_=(); CA=(${CARDS:-0 1 2 3 4 5 6 7}); ci=0
for th in $THS; do for d in en2zh zh2en; do
  ps_+=("$(launch ${CA[$ci]} dev_${th}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=${CA[$ci]} $PYV -m experiments.sst_hsmm_v1.hsc10_si_thinker --mode hsc --variant $VAR --samples ${KS:-6} --samp-max ${MT:-24} \
    --params $PRM --theta $th --verifier $VER --second-view enh --views $VIEWS/$d --thinker $NTH --direction $d --gpu-mem 0.6 \
    --items-file $DATA/items_$d.jsonl --out $E/dev_${th}_$d)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for th in $THS; do for d in en2zh zh2en; do [ -s $E/dev_${th}_$d/blocks_${d}_w1.jsonl ] || die "dev $th $d"; done; done
touch $STOP; touch $R/DONE; log "$OUTN done"
