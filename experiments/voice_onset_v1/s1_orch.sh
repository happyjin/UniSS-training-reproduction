#!/usr/bin/env bash
# voice_onset_v1 S1 (reports/voice_onset_v1/S0_S1_S2_prereg.zh-CN.md section 3 + S2 dev traces), cards 4-7 ONLY.
# Built from p3_orch.sh (same launch / placeholder handling / watcher p3_noplaceholder.sh).
#  1. Talker T1c --speed-policy queue (defaults 3.0 / 6 / final 2), seed 20261021:
#     test N+S1 = UE blocks + UE_mbr tokens -> test_robust/UE_mbrq/mbrq/talker_<d>
#     dev traces (S2 training) = wave17 eval_enc_wise05 blocks, source voice -> eval_outputs/voice_onset_v1/s1/dev/talker_<d>
#  2. speech-only merge (both), ASR (test only) ; 3. TitaNet + ECAPA2 + PESQ (test only)
#   setsid nohup bash experiments/voice_onset_v1/s1_orch.sh > logs/voice_onset_v1/s1_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
T=eval_outputs/mist_w3po_v1/test_robust
UE=$T/UE
M=$T/UE_mbr
Q=$T/UE_mbrq
DEVB=eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05
DV=eval_outputs/voice_onset_v1/s1/dev
R=eval_outputs/voice_onset_v1/s1
LOGD=logs/voice_onset_v1/s1
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
export S1_DATA=/opt/dlami/nvme/jasonleeeli/projects/UniSS/$T/data
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
NENV=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface S1_DATA=$S1_DATA)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "$DRY" ] || { mkdir -p $R; touch $R/FAILED; touch $STOP; }; exit 1; }
launch() {   # card name cmd... ; prints pid
  local g=$1 name=$2; shift 2
  local lf=$LOGD/$name.log
  if [ -n "$DRY" ]; then printf '[dry] card %s %s: ' "$g" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  case " 4 5 6 7 " in *" $g "*) ;; *) die "card $g not allowed";; esac
  for pat in run_mist_workload.sh train_mist_workload.py; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g" >&2;; esac
    done < <(ps -eo pid=,args=)
  done
  sleep 8
  local k left=""
  for k in $(seq 1 24); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && break
    sleep 5
  done
  [ -n "$left" ] && log "WARNING: placeholders still on card $g after 120 s:$left" >&2
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p $name" >> "$PIDS"
  log "$name card $g pid $p" >&2
  echo $p
}
waitp() { for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }

if [ -z "$DRY" ]; then
  for x in $Q/mbrq $DV $R/DONE; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  for d in en2zh zh2en; do [ -s $M/tokens/tokens_mbr_$d.jsonl ] || { echo "no tokens $d" >&2; exit 1; }; done
  mkdir -p "$LOGD" $R $Q/metrics $DV
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/voice_onset_v1/p3_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
  git rev-parse HEAD > $R/commit.txt
fi

# 1. Talkers: 16 jobs, 4 per card (test en2zh / zh2en, dev en2zh / zh2en; shard s on card 4+s)
COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy queue --gpu-mem 0.1 --seed 20261021)
ps_=()
for job in "test en2zh" "test zh2en" "dev en2zh" "dev zh2en"; do
  read -r kind d <<< "$job"
  for s in 0 1 2 3; do
    g=$((4 + s))
    if [ $kind = test ]; then
      ps_+=("$(launch $g talk_test_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
        -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $M/tokens/tokens_mbr_$d.jsonl \
        "${COMMON[@]}" --direction $d --output $Q/mbrq/talker_$d --shard $s --num-shards 4 \
        --blocks-file $UE/thinker_$d/blocks_${d}_w1.jsonl --items-file $T/UE_abs/data/items_$d.jsonl)")
    else
      ps_+=("$(launch $g talk_dev_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
        -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer \
        "${COMMON[@]}" --direction $d --output $DV/talker_$d --shard $s --num-shards 4 \
        --blocks-file $DEVB/thinker_$d/blocks_${d}_w1.jsonl --items-file $DEVB/data/items_$d.jsonl)")
    fi
    [ -n "$DRY" ] || sleep 30
  done
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for s in 0 1 2 3; do
  [ -n "$DRY" ] || [ -s $Q/mbrq/talker_$d/part_$s.jsonl ] || die "talker test $d $s"
  [ -n "$DRY" ] || [ -s $DV/talker_$d/part_$s.jsonl ] || die "talker dev $d $s"
done; done
log "talkers done"

# 2. merge (both) + ASR (test)
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do $PYE reports/uniss_si_v4/speech_only.py $DV/talker_$d > $LOGD/merge_dev_$d.log 2>&1 || die "merge dev $d"; done
fi
ps_=(); k=0
for d in en2zh zh2en; do
  g=$((4 + k)); k=$((k + 1)); O=$Q/mbrq/talker_$d
  ps_+=("$(launch $g asr_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$O' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$O/results_speechonly.jsonl' --output '$O/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$O/asr_speech.jsonl'")")
done
waitp "${ps_[@]}"
log "asr done"

# 3. metrics (test)
ps_=(); k=0
for d in en2zh zh2en; do
  O=$Q/mbrq/talker_$d; [ -n "$DRY" ] || [ -s $O/asr_speech.jsonl ] || die "asr $d"
  g=$((4 + k)); k=$((k + 2))
  ps_+=("$(launch $g spk_$d env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. S1_DATA=$S1_DATA $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $O --out $Q/metrics/mbrq_${d}_spksim.jsonl --device cuda:0)")
  ps_+=("$(launch $((g + 1)) ecapa2_$d env CUDA_VISIBLE_DEVICES=$((g + 1)) "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.test_abs_ecapa2 \
    --dir $d --run $O --out $Q/metrics/ecapa2_mbrq_${d}_spksim.jsonl --device cuda:0)")
  if [ -z "$DRY" ]; then PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $O \
    --out $Q/metrics/mbrq_${d}_squim.jsonl --procs 16 > $LOGD/pesq_$d.log 2>&1 & ps_+=($!); fi
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for m in spksim squim; do [ -n "$DRY" ] || [ -s $Q/metrics/mbrq_${d}_$m.jsonl ] || die "no $m $d"; done
  [ -n "$DRY" ] || [ -s $Q/metrics/ecapa2_mbrq_${d}_spksim.jsonl ] || die "no ecapa2 $d"; done
[ -n "$DRY" ] && exit 0
touch $STOP
touch $R/DONE
log "orch done"
