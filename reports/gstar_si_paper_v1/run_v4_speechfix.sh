#!/usr/bin/env bash
# Paper v4 speech, Zh->En post-processing: the speech jobs' Whisper step ran ~3 lines/s under a fully loaded CPU (load
# ~320 on 192 cores) and held the cards for hours, so it was stopped by PID; this finishes it beside the training.
# Per system ($@ = names, e.g. zh2en_O2): the remaining clips (--completed-input the partial asr_speech.jsonl) in
# NSH shards on the cards in $CARDS with few CPU threads each, appended to asr_speech.jsonl, then the same stitch /
# score / listen / voice steps as make_v4_speech_jobs.py, and UTMOS from the cached hub (TORCH_HOME).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
SP=eval_outputs/uniss_qwen3_omni_v1/gstar_v4_speech_20261030; V3=eval_outputs/uniss_qwen3_omni_v1/gstar_v3_20261029
V4=eval_outputs/uniss_qwen3_omni_v1/gstar_v4_20261030; W=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python; M=experiments.uniss_qwen3_omni_v1.evaluation
CARDS=(${CARDS:-4 5 6 7}); NSH=${NSH:-2}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
pids=(); i=0
for name in "$@"; do
  t=${SP}/${name}/talker
  for k in $(seq 0 $((NSH-1))); do
    c=${CARDS[$((i % ${#CARDS[@]}))]}; i=$((i+1))
    [ -e ${t}/asr_speech.part${k}.jsonl ] && { log "refuse: ${t}/asr_speech.part${k}.jsonl exists"; exit 1; }
    CUDA_VISIBLE_DEVICES=$c PYTHONPATH=. ${PYE} -m evaluation.asr_transcribe --input ${t}/results_speechonly.jsonl \
      --output ${t}/asr_speech.part${k}.jsonl --completed-input ${t}/asr_speech.jsonl --num-shards ${NSH} --shard-index $k \
      --device cuda:0 --batch-size 16 > ${SP}/${name}/asr.part${k}.log 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait $p || log "an ASR shard failed (pid $p)"; done
for name in "$@"; do
  o=${SP}/${name}; t=${o}/talker
  case ${name} in *LA2*|*S2dl*) mode=stream; pk=lowlat ;; *) mode=offline; pk=offline ;; esac
  case ${name} in *S2dl*) hyp=${V4}/${name} ;; *) hyp=${V3}/${name} ;; esac
  for k in $(seq 0 $((NSH-1))); do cat ${t}/asr_speech.part${k}.jsonl >> ${t}/asr_speech.jsonl; done
  log "${name}: $(wc -l < ${t}/asr_speech.jsonl) / $(wc -l < ${t}/results_speechonly.jsonl) transcribed"
  PYTHONPATH=. ${PYE} -m ${M}.gstar_talker stitch --results ${t}/results.jsonl --prep ${o}/talker_in --windows ${W} \
    --mode ${mode} --out ${o}/stitched
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows ${W} --hyp ${hyp}/hyp.jsonl --asr ${t}/asr_speech.jsonl \
    --post-key ${pk} --out ${o}/score.json > ${o}/score.log 2>&1
  PYTHONPATH=. ${PYE} -m ${M}.gstar_listen --transcript ${o}/stitched/transcript.jsonl --out ${o}/listen.json
  CUDA_VISIBLE_DEVICES=${CARDS[0]} PYTHONPATH=. ${PYE} -m ${M}.gstar_voice --windows ${W} --hyp ${hyp}/hyp.jsonl \
    --results ${t}/results.jsonl --out ${o}/voice.json
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=${CARDS[0]} PYTHONPATH=. ${PYE} -m evaluation.utmos_metrics \
    --input ${t}/results_speechonly.jsonl --output-dir ${o}/utmos --device cuda:0 > ${o}/utmos.v4fix.log 2>&1 || log "utmos failed"
  log "${name}: $(grep -m1 cpASR_BLEU_session ${o}/score.json)"
done
log "SPEECHFIX_DONE"
