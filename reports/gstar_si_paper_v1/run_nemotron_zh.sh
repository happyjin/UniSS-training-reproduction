#!/usr/bin/env bash
# Nemotron-3 posteriors of a Chinese AMI mix (MIX=zh_mix_v4 -> diar/nemotron_<preset>_<TAG>), both presets, on the cards
# in CARDS (default 0 2 3 4 5 6 7). The GPU holders are not touched here; the caller restores them. Launch detached
# (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
MIX=${MIX:-zh_mix_v4}; TAG=${TAG:-zh4}
read -r -a G <<<"${CARDS:-0 2 3 4 5 6 7}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
for preset in offline lowlat; do
  for k in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$k]} ${PY} experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py --preset ${preset} \
      --audio ${MIX} --out-name nemotron_${preset}_${TAG} --shard ${k} --num-shards ${#G[@]} \
      > logs/multispk/nemotron_${TAG}_${preset}_shard${k}.log 2>&1 &
  done
  wait
  log "preset ${preset}: $(ls data/ami_multispk_si_v1/diar/nemotron_${preset}_${TAG}/*.post.pt 2>/dev/null | wc -l) sessions"
done
log "NEMOTRON_ZH_DONE"
