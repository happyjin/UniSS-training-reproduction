#!/usr/bin/env bash
# Paper v5, Zh->En speech post-processing on the new server: v4's Whisper step stopped at ~40% (asr_speech.jsonl) and its
# interrupted v4 fix left two small part files never appended. Per system ($@, e.g. zh2en_O2):
#   asr_speech.v5base.jsonl = asr_speech.jsonl + asr_speech.part{0,1}.jsonl (dedup by id; the originals are left as they are)
#   the remaining clips in NSH shards (--completed-input v5base) on the cards in CARDS -> asr_speech.v5part{k}.jsonl
#   asr_speech.v5full.jsonl = base + parts, then stitch / score / listen / voice / UTMOS as make_v4_speech_jobs.py.
# Cards: training holds ~72 GB/card; another user's jobs sit on 0,1,2,4,5; the single-speaker branch stacks vLLM on 3,6,7,
# so the ASR shards go to 0,1,2,4,5. v2: the first try (batch 32) let Whisper's cache grow to ~48 GB per process and
# cards reached 138 GB, so it was stopped by PID; now batch 8 and a hard per-process cap (set_per_process_memory_fraction,
# ASR_FRAC x 139.8 GiB), so an overrun fails only that shard; shards resume (--resume) from their own partial output.
# The v5base files of the first try are reused. Refuses any other output that already exists.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
exec >> logs/multispk/v5_speechfix2.log 2>&1
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
SP=eval_outputs/uniss_qwen3_omni_v1/gstar_v4_speech_20261030; V3=eval_outputs/uniss_qwen3_omni_v1/gstar_v3_20261029
W=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python; M=experiments.uniss_qwen3_omni_v1.evaluation
CARDS=(${CARDS:-0 1 2 4 5}); NSH=${NSH:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for name in "$@"; do
  o=${SP}/${name}; t=${o}/talker
  for f in ${t}/asr_speech.v5full.jsonl ${o}/stitched ${o}/score.json ${o}/listen.json ${o}/voice.json ${o}/utmos; do
    [ -e $f ] && { log "refuse: $f exists"; exit 1; }
  done
  [ -e ${t}/asr_speech.v5base.jsonl ] && { log "reuse $(wc -l < ${t}/asr_speech.v5base.jsonl) lines of ${t}/asr_speech.v5base.jsonl"; continue; }
  ${PYE} -I - ${t}/asr_speech.v5base.jsonl ${t}/asr_speech.jsonl ${t}/asr_speech.part0.jsonl ${t}/asr_speech.part1.jsonl <<'EOF'
import json, sys
seen, out = set(), open(sys.argv[1], "x")
for f in sys.argv[2:]:
    for l in open(f):
        if l.strip() and (i := json.loads(l)["id"]) not in seen:
            seen.add(i); out.write(l if l.endswith("\n") else l + "\n")
print(f"{sys.argv[1]}: {len(seen)} done")
EOF
done

pids=(); i=0
for name in "$@"; do
  t=${SP}/${name}/talker
  for k in $(seq 0 $((NSH-1))); do
    c=${CARDS[$((i % ${#CARDS[@]}))]}; i=$((i+1))
    CUDA_VISIBLE_DEVICES=$c PYTHONPATH=. ${PYE} -c "import runpy, sys, torch; torch.cuda.set_per_process_memory_fraction(${ASR_FRAC:-0.21}); sys.argv = ['asr_transcribe'] + sys.argv[1:]; runpy.run_module('evaluation.asr_transcribe', run_name='__main__')" \
      --input ${t}/results_speechonly.jsonl --output ${t}/asr_speech.v5part${k}.jsonl --completed-input ${t}/asr_speech.v5base.jsonl \
      --num-shards ${NSH} --shard-index $k --device cuda:0 --batch-size 8 --resume >> ${SP}/${name}/asr.v5part${k}.log 2>&1 &
    pids+=($!); log "${name} shard $k on card $c (pid $!)"; sleep 5
  done
done
fail=0; for p in "${pids[@]}"; do wait $p || { log "an ASR shard failed (pid $p)"; fail=1; }; done
[ $fail -eq 0 ] || { log "ASR incomplete: not merging or scoring; rerun this script (shards resume)"; exit 1; }

for name in "$@"; do
  o=${SP}/${name}; t=${o}/talker
  case ${name} in *LA2*|*S2dl*) mode=stream; pk=lowlat ;; *) mode=offline; pk=offline ;; esac
  cat ${t}/asr_speech.v5base.jsonl ${t}/asr_speech.v5part*.jsonl > ${t}/asr_speech.v5full.jsonl
  log "${name}: $(wc -l < ${t}/asr_speech.v5full.jsonl) / $(wc -l < ${t}/results_speechonly.jsonl) transcribed"
  PYTHONPATH=. ${PYE} -m ${M}.gstar_talker stitch --results ${t}/results.jsonl --prep ${o}/talker_in --windows ${W} \
    --mode ${mode} --out ${o}/stitched
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows ${W} --hyp ${V3}/${name}/hyp.jsonl --asr ${t}/asr_speech.v5full.jsonl \
    --post-key ${pk} --out ${o}/score.json > ${o}/score.log 2>&1
  PYTHONPATH=. ${PYE} -m ${M}.gstar_listen --transcript ${o}/stitched/transcript.jsonl --out ${o}/listen.json
  CUDA_VISIBLE_DEVICES=${CARDS[0]} PYTHONPATH=. ${PYE} -m ${M}.gstar_voice --windows ${W} --hyp ${V3}/${name}/hyp.jsonl \
    --results ${t}/results.jsonl --out ${o}/voice.json
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=${CARDS[0]} PYTHONPATH=. ${PYE} -m evaluation.utmos_metrics \
    --input ${t}/results_speechonly.jsonl --output-dir ${o}/utmos --device cuda:0 > ${o}/utmos.v5.log 2>&1 || log "utmos failed"
  log "${name}: $(grep -m1 cpASR_BLEU_session ${o}/score.json)"
done
log "V5_SPEECHFIX_DONE"
