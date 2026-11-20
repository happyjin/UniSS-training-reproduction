#!/usr/bin/env bash
# Robust wave 14, full-system combination (reports/mist_w3po_v1/robust_wave14_unist_prereg.zh-CN.md, run only if both
# primary rules pass): a copy of wave8_aul_combo_orch.sh with the Thinker blocks of unist_wise05 (eval_unist_wise05):
# V1 voice tokens on its own first writes (wave5_m_v1tokens.py, unchanged) + T1c Talker, S1 talker flags (4 shards, seed
# 20261021). Cards 0-3 (instead of 4-7): v1 tokens (0, 1) -> 8 talkers (2 per card) -> ASR (0, 1) -> SQUIM PESQ (CPU) +
# TitaNet similarity (2, 3) for uw_v1 and uw_b (= UniST WiSE + raw voice, the eval_unist_wise05 talker) ->
# wave14_combo_score.py (vs B and vs the wave-5 ERM-WiSE combo). Placeholders stopped by PID per launch; pids go to
# w14_workers.pids, kept off by wave14_combo_noplaceholder.sh (STOP w14_combo_noplaceholder.stop, touched
# after the metrics). Never touches guard 2726951, gpu_watch 52045, sync 1222453.
# Detached: setsid nohup bash .../wave14_combo_orch.sh > logs/mist_w3po_v1/wave14/combo_orch.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
EW=eval_outputs/mist_w3po_v1/wave14/eval_unist_wise05
C=eval_outputs/mist_w3po_v1/wave14/combo
LOGD=logs/mist_w3po_v1/wave14
export PIDS=$LOGD/w14_workers.pids
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "STOP: $*"; touch $C/FAILED; exit 1; }
PROTECT=" 2726951 52045 1222453 $$ "
launch() {   # card name cmd... ; prints pid
  local g=$1 name=$2; shift 2
  local lf=$LOGD/combo_$name.log
  [ -e "$lf" ] && die "log exists: $lf"
  for pat in run_mist_workload.sh train_mist_workload.py; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g" >&2;; esac
    done < <(ps -eo pid=,args=)
  done
  sleep 8
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p $name" >> "$PIDS"
  log "$name card $g pid $p" >&2
  echo $p
}
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }

mkdir -p $C/metrics $C/uw_b
[ -e $C/FAILED ] && die "previous FAILED"
for d in en2zh zh2en; do
  [ -s $EW/talker_${d}_s20261021/asr_speech.jsonl ] || die "no UniST WiSE talker $d"
  [ -e $C/uw_b/talker_$d ] || ln -s ../../eval_unist_wise05/talker_${d}_s20261021 $C/uw_b/talker_$d
done
setsid nohup bash experiments/mist_w3po_v1/voicecond/wave14_combo_noplaceholder.sh > $LOGD/combo_noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "orch $$ start; watcher $!"

# 1. V1 tokens on UniST WiSE's first writes
p1=$(launch 0 v1tok_en2zh env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PYQ -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py \
     --dir en2zh --blocks $EW/thinker_en2zh/blocks_en2zh_w1.jsonl --out-root $C)
p2=$(launch 1 v1tok_zh2en env PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PYQ -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py \
     --dir zh2en --blocks $EW/thinker_zh2en/blocks_zh2en_w1.jsonl --out-root $C)
waitp $p1 $p2
for d in en2zh zh2en; do [ -s $C/tokens/tokens_v1_$d.jsonl ] || die "no tokens $d"; done
log "tokens done"

# 2. talkers: 2 per card on 4-7
ps_=(); k=0
for d in en2zh zh2en; do for s in 0 1 2 3; do
  g=$((0 + k / 2)); k=$((k + 1))
  ps_+=($(launch $g talk_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
      -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $C/tokens/tokens_v1_$d.jsonl \
      --model $TALKER --direction $d --window 1 --output $C/uw_v1/talker_$d --shard $s --num-shards 4 \
      --text-policy file --blocks-file $EW/thinker_$d/blocks_${d}_w1.jsonl --speed-policy block --gpu-mem 0.1 \
      --items-file $DATA/items_$d.jsonl --seed 20261021))
  sleep 25
done; done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for s in 0 1 2 3; do [ -s $C/uw_v1/talker_$d/part_$s.jsonl ] || die "no part $d $s"; done; done
log "talkers done"

# 3. ASR (paper speech-only protocol)
ps_=()
for d in en2zh zh2en; do
  g=$([ $d = en2zh ] && echo 0 || echo 1); R=$C/uw_v1/talker_$d
  ps_+=($(launch $g asr_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$R' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$R/results_speechonly.jsonl' --output '$R/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$R/asr_speech.jsonl'"))
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do [ -s $C/uw_v1/talker_$d/asr_speech.jsonl ] || die "no asr $d"; done
log "asr done"

# 4. PESQ (CPU, 4 runs in parallel) + similarity (cards 6, 7)
ps_=()
for arm in uw_v1 uw_b; do for d in en2zh zh2en; do
  PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $C/$arm/talker_$d \
    --out $C/metrics/${arm}_${d}_squim.jsonl --procs 24 > $LOGD/combo_pesq_${arm}_$d.log 2>&1 & ps_+=($!)
done; done
k=0
for arm in uw_v1 uw_b; do
  g=$((2 + k)); k=$((k + 1))
  ps_+=($(launch $g spksim_$arm bash -c "set -e; for d in en2zh zh2en; do CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir \$d --run $C/$arm/talker_\$d --out $C/metrics/${arm}_\${d}_spksim.jsonl --device cuda:0; done"))
done
waitp "${ps_[@]}"
for arm in uw_v1 uw_b; do for d in en2zh zh2en; do for m in squim spksim; do
  [ -s $C/metrics/${arm}_${d}_$m.jsonl ] || die "no $m $arm $d"; done; done; done
log "metrics done"
touch $LOGD/w14_combo_noplaceholder.stop

# 5. score
$PYQ -m experiments.mist_w3po_v1.voicecond.wave14_combo_score --out $C/summary.json > $LOGD/combo_score.log 2>&1 || die "score"
touch $C/DONE
log "orch done: $C/summary.json"
