#!/usr/bin/env bash
# v4 (2026-11-19 06:17 Prague, coordinator, user: "非训练任务尽量叠满显存跑，并行加快速度"): step 3 (teacher generation,
# inference only) is spread over ALL 8 cards with 18 shards instead of 8 shards on cards 4-7: zh2en_train / en2zh_train
# 8 shards each (500 items per shard), zh2en_dev / en2zh_dev 1 shard each (400 items); shard k goes to the k-th card in
# order of free memory at step-3 start (round robin, <= 3 shards = ~45 GB per card, every card stays < 135 GiB), engines
# 10 s apart (fixed KV cache SI_KV_CACHE_GIB=7.73, no vLLM memory profiling). Outputs keep v3's layout
# teacher_out/<set>/part_<k>.jsonl (consumers glob part_*.jsonl); only the number of parts changes. Step 2 runs on the
# freest card instead of card 4. Step 1 is NOT rerun: v3's running preprocessing (idea2_gpu_pre_v2.sh, PID PRE_PID) is
# awaited, then cvt/pre/PRE_ALL_DONE is required. Flags of every job are otherwise identical to v3.
# Marker data/open_voicecond_v1/idea2/cvt/TEACHER_DONE (or TEACHER_FAILED_<step>). Placeholders are stopped by PID.
#   PRE_PID=<pid> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_teacher_chain_v4.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/teacher_chain_v4.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
VPY=$J/venvs/vllm_omni_030/bin/python
CV=data/open_voicecond_v1/idea2/cvt
M=logs/mist_w3po_v1/voicecond/idea2
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PRE_PID=${PRE_PID:?}
SPEC="zh2en_train:8 en2zh_train:8 zh2en_dev:1 en2zh_dev:1"
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED $1"; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ >> $CV/TEACHER_FAILED_$1; exit 1; }
free_card() {
  local k=$1 sup kids
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
  done
}
cards_by_free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | awk -F, '{print $1}'; }
log "step 1: waiting for v3's cvt preprocessing PID $PRE_PID"
if [ -z "${DRY_RUN:-}" ]; then
  while kill -0 $PRE_PID 2>/dev/null; do sleep 30; done
  [ -e $CV/pre/PRE_ALL_DONE ] || fail pre_marker
fi
log "step 2: u* / g* and the teacher voice table"
c2=$(cards_by_free | head -1)
if [ -n "${DRY_RUN:-}" ]; then log "[dry] refs on card $c2 + idea2_teacher_table.py"
else free_card $c2
  IDEA2_ROOT=$CV CUDA_VISIBLE_DEVICES=$c2 HF_HOME=$J/cache/huggingface $VPY -I experiments/mist_w3po_v1/voicecond/idea2_windows.py refs \
    --device cuda:0 > $M/teacher_refs.log 2>&1 || fail refs
  python3 experiments/mist_w3po_v1/voicecond/idea2_teacher_table.py > $M/teacher_table.log 2>&1 || fail table; fi
log "step 3: teacher generation, shards $SPEC"
ENVS="TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache \
TRITON_CACHE_DIR=$J/cache/triton VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 PYTHONPATH=.:third_party/Megatron-LM \
PATH=$J/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_FLASHINFER_SAMPLER=0 SI_KV_CACHE_GIB=7.73"
ORDER=($(cards_by_free)); NC=${#ORDER[@]}
log "card order by free memory: ${ORDER[*]}"
pids=""; n=0
for x in $SPEC; do
  set=${x%:*}; ns=${x#*:}; d=${set%_*}
  O=$CV/teacher_out/$set; mkdir -p $O
  for s in $(seq 0 $((ns - 1))); do
    card=${ORDER[$((n % NC))]}; n=$((n + 1))
    CMD="env $ENVS CUDA_VISIBLE_DEVICES=$card $VPY -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $CV/teacher_in/voice_gstar.jsonl \
--model $T1C --direction $d --window 1 --output $O --text-policy file --blocks-file $CV/teacher_in/blocks_$set.jsonl \
--items-file $CV/teacher_in/items_$set.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 \
--shard $s --num-shards $ns --dump-seq --dump-codes"
    if [ -n "${DRY_RUN:-}" ]; then log "[dry] $set shard $s/$ns card $card"; continue; fi
    [ -e $O/part_$s.jsonl ] && { log "$set shard $s exists"; continue; }
    free_card $card
    bash -c "$CMD" > $O/talk$s.log 2>&1 &
    pids="$pids $!"
    log "$set shard $s/$ns on card $card pid $!"
    sleep 10
  done
done
[ -n "${DRY_RUN:-}" ] && exit 0
rc=0; for p in $pids; do wait $p || rc=1; done
for x in $SPEC; do
  set=${x%:*}; O=$CV/teacher_out/$set
  log "$set: $(cat $O/part_*.jsonl 2>/dev/null | wc -l) output lines / $(wc -l < $CV/teacher_in/items_$set.jsonl) items, parts: $(ls $O | grep -c '^part_')"
done
[ $rc = 0 ] || fail teacher
date -u +%FT%TZ >> $CV/TEACHER_DONE
log TEACHER_DONE
