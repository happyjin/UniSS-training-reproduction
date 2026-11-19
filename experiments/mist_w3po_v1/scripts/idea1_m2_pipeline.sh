#!/usr/bin/env bash
# Idea 1 M2 data + training pipeline (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v3): waits for the AISHELL-1
# download (.DONE), extracts it (outer tgz, then the per-speaker tar.gz files) into data/open_voicecond_v1/idea1/aishell1,
# builds train_zh (idea1_build_train.py), runs GPU HDemucs on it (idea1_hdemucs_gpu.py via idea1_gpu_run.sh on the slot-A
# card with the most free memory), waits until train_en and train_zh both have vhat, then starts the two pre-registered
# trainings (ZNE-in-the-loop and the denoising control; idea1_e2e_train.py), each via idea1_gpu_run.sh on the freest card,
# 60 s apart. Markers: data/open_voicecond_v1/idea1/train_zh.{BUILT,HDEMUCS}, eval_outputs/.../m2/train_{zne,ctrl}.DONE.
# Refuses existing outputs (each python step refuses its own; extraction refuses an existing target dir).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_m2_pipeline.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_idea1/m2_pipeline.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
I=data/open_voicecond_v1/idea1
L=logs/mist_w3po_v1/voicecond_idea1
M=eval_outputs/mist_w3po_v1/voicecond_idea1/m2
PYE=$J/conda_envs/uniss-eval/bin/python
PYV=$J/venvs/vllm_omni_030/bin/python
TGZ=data/open_voicecond_v1/raw/aishell1/data_aishell.tgz
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then log "[dry] $*"; else "$@"; fi; }
freest() {   # slot-A card with the most free memory
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1<=3 {print $2, $1}' | sort -nr | head -1 | awk '{print $2}'
}
waitf() { [ -n "${DRY_RUN:-}" ] && { log "[dry] wait for $1"; return 0; }; until [ -e "$1" ]; do sleep 20; done; }
# 1. AISHELL-1
waitf $TGZ.DONE
if [ ! -e $I/aishell1.EXTRACTED ]; then
  [ -e $I/aishell1 ] && [ -z "${DRY_RUN:-}" ] && { log "$I/aishell1 exists without marker: refuse"; exit 1; }
  run mkdir -p $I/aishell1
  run tar -xzf $TGZ -C $I/aishell1 --no-same-owner data_aishell/wav
  if [ -z "${DRY_RUN:-}" ]; then
    n=0; for t in $I/aishell1/data_aishell/wav/*.tar.gz; do tar -xzf $t -C $I/aishell1 --no-same-owner || { log "FAIL $t"; exit 1; }; n=$((n+1)); done
    log "aishell1: $n speaker archives, $(ls $I/aishell1/train $I/aishell1/dev $I/aishell1/test 2>/dev/null | grep -c .^S.) speakers"
    date > $I/aishell1.EXTRACTED
  fi
fi
# 2. train_zh
if [ ! -e $I/train_zh.BUILT ]; then
  run env PYTHONPATH=. OMP_NUM_THREADS=1 $PYE -m experiments.mist_w3po_v1.voicecond.idea1_build_train --lang zh --speech $I/aishell1 \
    --musan $I/musan/musan/music --fsd-audio $I/fsd50k_eval/FSD50K.eval_audio --fsd-gt $I/fsd50k_meta/FSD50K.ground_truth/eval.csv \
    --out $I/train_zh --n 20000 --procs 40 || { log "FAIL build train_zh"; exit 1; }
  [ -n "${DRY_RUN:-}" ] || date > $I/train_zh.BUILT
fi
# 3. HDemucs zh
if [ ! -e $I/train_zh.HDEMUCS ]; then
  c=$(freest); log "HDemucs zh on card $c"
  run env PYTHONPATH=. OMP_NUM_THREADS=8 NEED_MIB=30000 bash experiments/mist_w3po_v1/scripts/idea1_gpu_run.sh $c $I/train_zh.HDEMUCS \
    $PYE -m experiments.mist_w3po_v1.voicecond.idea1_hdemucs_gpu --dir $I/train_zh --bs 8
fi
# 4. both separated
waitf $I/train_en.HDEMUCS; waitf $I/train_zh.HDEMUCS
if [ -z "${DRY_RUN:-}" ]; then
  grep -q 'rc=0' $I/train_en.HDEMUCS && grep -q 'rc=0' $I/train_zh.HDEMUCS || { log "HDemucs failed"; exit 1; }
fi
# 5. trainings
run mkdir -p $M
for mode in zne ctrl; do
  c=$(freest); log "train $mode on card $c"
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] gpu_run $c $M/train_$mode.DONE $PYV -m ...idea1_e2e_train --mode $mode --data $I/train_en $I/train_zh --dev $I/diag_en $I/diag_zh --out $M/$mode"; continue; fi
  ( export PYTHONPATH=. HF_HOME=$J/cache/huggingface PATH=$J/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=8 NEED_MIB=40000
    bash experiments/mist_w3po_v1/scripts/idea1_gpu_run.sh $c $M/train_$mode.DONE $PYV -m experiments.mist_w3po_v1.voicecond.idea1_e2e_train \
      --mode $mode --data $I/train_en $I/train_zh --dev $I/diag_en $I/diag_zh --out $M/$mode --device cuda:0 > $L/train_$mode.log 2>&1 ) &
  sleep 60
done
wait
log "M2 pipeline done: $(cat $M/train_zne.DONE 2>/dev/null) $(cat $M/train_ctrl.DONE 2>/dev/null)"
