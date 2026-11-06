#!/usr/bin/env bash
# Speech (ASR-BLEU) of one decode run, the v7/v8 protocol (reports/gstar_si_v7/make_v7_speech_jobs.py, after_v7.sh):
# the UniSS talker T1c@1500 speaks every output speaker, voice prompted causally from the diarizer's solo frames,
# each read's block at its commit time (gstar_talker_v6 prep --causal --blocks); streams over 400 blocks are dropped
# (count as empty); speech-only ASR (Whisper large-v3, asr_speech.jsonl, memory si-asr-on-speech-only); per window and
# speaker the transcript becomes one segment -> <run>/hyp_asr.jsonl -> score_run --asr -> <run>/score_asr.json.
#   RUN=<decode dir with hyp.jsonl> WINDOWS=<online windows> REF=<ref windows> NAME=<speech name> CARDS="0 1 2 3 4 5 6 7"
#   [SET=zh2en|ali] setsid nohup bash run_speech.sh > log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache
export VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 TRITON_CACHE_DIR=$J/cache/triton
PYE=$J/conda_envs/uniss-eval/bin/python; PYV=$J/venvs/vllm_omni_030/bin/python; PYO=$J/conda_envs/qwen3omni/bin/python
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
M=experiments.uniss_qwen3_omni_v1.evaluation
RUN=${RUN:?}; WINDOWS=${WINDOWS:?}; REF=${REF:?}; NAME=${NAME:?}; CARDS=${CARDS:?}; SET=${SET:-zh2en}
if [ "$SET" = ali ]; then DATA=data/alimeeting_si_v1; SESS=near_mix; DIAR=nemotron_lowlat; else DATA=data/ami_multispk_si_v1; SESS=zh_mix_v8; DIAR=nemotron_lowlat_zh8; fi
SP=eval_outputs/mist_w3po_v1/speech/$NAME
if [ -e "$SP" ]; then echo "拒绝覆盖已有输出: $SP" >&2; exit 1; fi
[ -e "$RUN/hyp.jsonl" ] || { echo "no $RUN/hyp.jsonl" >&2; exit 1; }
mkdir -p $SP/talker
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log prep
GSTAR_DATA=$DATA PYTHONPATH=. CUDA_VISIBLE_DEVICES= $PYE -m $M.gstar_talker_v6 prep --hyp $RUN/hyp.jsonl --windows $WINDOWS \
  --causal --blocks --preset lowlat --session $SESS --diar $DIAR --sep ' ' --out $SP/talker_in_all > $SP/prep.log 2>&1 \
  && python3 reports/gstar_si_v7/drop_long_streams.py $SP/talker_in_all $SP/talker_in 400 >> $SP/prep.log 2>&1 \
  || { log PREP_FAILED; exit 1; }
set -- $CARDS; N=$#; k=0; pids=()
for c in $CARDS; do
  log "talker shard $k on card $c"
  CUDA_VISIBLE_DEVICES=$c PATH=$J/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=. OMP_NUM_THREADS=16 \
    $PYV -m $M.si_infer --model $T1C --direction zh2en --window 1 --output $SP/talker --text-policy file \
    --blocks-file $SP/talker_in/blocks.jsonl --items-file $SP/talker_in/items.jsonl --speed-policy fixed --history-ban 8 \
    --max-model-len 32768 --gpu-mem 0.1 --shard $k --num-shards $N > $SP/talk$k.log 2>&1 &
  pids+=($!); k=$((k+1))
  [ $k -lt $N ] && sleep 60
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { log TALKER_FAILED; exit 1; }
cat $SP/talker/part_*.jsonl > $SP/talker/results.jsonl
$PYE scripts/gpu_queue/p80_speechonly.py $SP/talker
set -- $CARDS
log "asr on card $1"
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input $SP/talker/results_speechonly.jsonl \
  --output $SP/talker/asr_speech.jsonl --device cuda:0 --batch-size 8 > $SP/asr.log 2>&1 || { log ASR_FAILED; exit 1; }
[ -e $RUN/hyp_asr.jsonl ] || python3 - $RUN/hyp.jsonl $SP/talker/asr_speech.jsonl $RUN/hyp_asr.jsonl <<'PY'
import json, sys
from collections import defaultdict
hyp, asr, out = sys.argv[1:4]
A = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(asr, encoding="utf-8"))}
with open(out, "x", encoding="utf-8") as fh:
    for h in map(json.loads, open(hyp, encoding="utf-8")):
        by = defaultdict(list)
        for s in h["segments"]:
            by[s["spk"]].append(s)
        segs = []
        for k, ss in by.items():
            t = A.get(f"{h['id']}__S{k}", "").strip()
            if t:
                segs.append({"spk": k, "start_s": min(s["start_s"] for s in ss), "end_s": max(s["end_s"] for s in ss),
                             "text": t, "write_ms": max(s.get("write_ms", 0) for s in ss)})
        fh.write(json.dumps({"id": h["id"], "segments": segs}, ensure_ascii=False) + "\n")
PY
PYTHONPATH=.:third_party/Megatron-LM $PYO -m experiments.mist_w3po_v1.evaluation.score_run --run $RUN --windows $WINDOWS \
  --ref-windows $REF --set $SET --asr $RUN/hyp_asr.jsonl > $SP/score.log 2>&1
log "SPEECH_DONE $(tail -n 1 $SP/score.log | cut -c1-300)"
