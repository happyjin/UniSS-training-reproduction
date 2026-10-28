#!/usr/bin/env bash
# Nemotron-3 posteriors of the Chinese AMI meetings (session_audio/zh_mix_v3), both presets, on cards 2,5,6,7 (their
# holder stopped first, restored at the end). Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
G=(2 5 6 7)
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
bash scripts/gpu_queue/holder_off.sh 2567
sleep 15
for preset in offline lowlat; do
  for k in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$k]} ${PY} experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py --preset ${preset} \
      --audio zh_mix_v3 --out-name nemotron_${preset}_zh3 --shard ${k} --num-shards ${#G[@]} \
      > logs/multispk/nemotron_zh3_${preset}_shard${k}.log 2>&1 &
  done
  wait
  log "preset ${preset}: $(ls data/ami_multispk_si_v1/diar/nemotron_${preset}_zh3/*.post.pt 2>/dev/null | wc -l) sessions"
done
sleep 10
bash scripts/gpu_queue/holder_on.sh 2,5,6,7
log "NEMOTRON_ZH_DONE"
