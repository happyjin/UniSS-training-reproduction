#!/usr/bin/env bash
# Idea 2 e2e prereg §5 (no training): GPU part of the end-to-end data, started only after the M2 test runs on cards 4-7
# have finished (each of: bench I2_<cond>_<dir>_{DONE,FAILED} x 6, ood3 OOD3_<ss>_{DONE,FAILED} x 3,
# GAME_VFBF_{DONE,FAILED}) and the cvt CPU build (cvt/CPU_DONE):
#   1. cvt stream preprocessing on cards 4-7 (idea2_gpu_pre_v2.sh, IDEA2_ROOT=cvt, L = 2 s: diarizer, HDemucs, windows,
#      nuisance features, v~);
#   2. clean reference codes u* / g* of the cvt speakers (idea2_windows.py refs, card 4) -> teacher voice table;
#   3. teacher targets: the frozen T1c (same flags as the T1c talk jobs) on the cvt teacher items with the voice tokens
#      overridden by g* (idea1_si_infer_vo), --dump-seq --dump-codes, 8 shards (2 per card, engines >= 60 s apart).
# Marker data/open_voicecond_v1/idea2/cvt/TEACHER_DONE (or TEACHER_FAILED_<step>). Placeholders are stopped by PID.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_teacher_chain.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/teacher_chain.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
VPY=$J/venvs/vllm_omni_030/bin/python
CV=data/open_voicecond_v1/idea2/cvt
M=logs/mist_w3po_v1/voicecond/idea2
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED $1"; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ >> $CV/TEACHER_FAILED_$1; exit 1; }
ended() { [ -e $1_DONE ] || [ -e $1_FAILED ]; }
free_card() {
  local k=$1 sup kids
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
  done
}
if [ -z "${DRY_RUN:-}" ]; then
  log "waiting for the M2 test runs and the cvt CPU build"
  while true; do
    ok=1
    for c in id2 ood1 clean; do for d in zh2en en2zh; do ended $M/bench/I2_${c}_${d} || ok=0; done; done
    for s in ami en ali; do ended $M/ood3/OOD3_$s || ok=0; done
    ended $M/GAME_VFBF || ok=0
    [ -e $CV/CPU_DONE ] || ok=0
    ls $CV/CPU_FAILED_* > /dev/null 2>&1 && fail cvt_cpu
    [ $ok = 1 ] && break
    sleep 60
  done
fi
log "step 1: cvt preprocessing"
if [ -n "${DRY_RUN:-}" ]; then log "[dry] IDEA2_ROOT=$CV WHICH=all LS=2 bash experiments/mist_w3po_v1/scripts/idea2_gpu_pre_v2.sh"
else IDEA2_ROOT=$CV TAG=cvt WHICH=all LS=2 bash experiments/mist_w3po_v1/scripts/idea2_gpu_pre_v2.sh || fail pre
  [ -e $CV/pre/PRE_ALL_DONE ] || fail pre_marker; fi
log "step 2: u* / g* and the teacher voice table"
free_card 4
if [ -n "${DRY_RUN:-}" ]; then log "[dry] refs + idea2_teacher_table.py"
else IDEA2_ROOT=$CV CUDA_VISIBLE_DEVICES=4 HF_HOME=$J/cache/huggingface $VPY -I experiments/mist_w3po_v1/voicecond/idea2_windows.py refs \
    --device cuda:0 > $M/teacher_refs.log 2>&1 || fail refs
  python3 experiments/mist_w3po_v1/voicecond/idea2_teacher_table.py > $M/teacher_table.log 2>&1 || fail table; fi
log "step 3: teacher generation"
ENVS="TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
TRITON_CACHE_DIR=$J/cache/triton VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 PYTHONPATH=.:third_party/Megatron-LM \
PATH=$J/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_FLASHINFER_SAMPLER=0"
pids=""; n=0
for set in zh2en_train en2zh_train zh2en_dev en2zh_dev; do
  d=${set%_*}; sp=${set#*_}
  O=$CV/teacher_out/$set; mkdir -p $O
  for s in 0 1; do
    card=$((4 + (n % 4))); n=$((n + 1))
    CMD="env $ENVS CUDA_VISIBLE_DEVICES=$card $VPY -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $CV/teacher_in/voice_gstar.jsonl \
--model $T1C --direction $d --window 1 --output $O --text-policy file --blocks-file $CV/teacher_in/blocks_$set.jsonl \
--items-file $CV/teacher_in/items_$set.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 \
--shard $s --num-shards 2 --dump-seq --dump-codes"
    if [ -n "${DRY_RUN:-}" ]; then log "[dry] $CMD"; continue; fi
    [ -s $O/part_$s.jsonl ] && { log "$set shard $s exists"; continue; }
    free_card $card
    bash -c "$CMD" > $O/talk$s.log 2>&1 &
    pids="$pids $!"
    log "$set shard $s on card $card pid $!"
    sleep 60
  done
done
[ -n "${DRY_RUN:-}" ] && exit 0
rc=0; for p in $pids; do wait $p || rc=1; done
[ $rc = 0 ] || fail teacher
date -u +%FT%TZ >> $CV/TEACHER_DONE
log TEACHER_DONE
