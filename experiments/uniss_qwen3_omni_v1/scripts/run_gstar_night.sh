#!/usr/bin/env bash
# Tonight's queue for docs/multi_speaker_si_plan_v7 + v8, streaming first once the offline model can be listened to:
#   wait for O-G1's merge -> O-G1 offline test -> GPU smoke of the streaming inference (wait / la2, a few windows)
#   -> train S-T on O-G1 (streaming rows, INIT_MERGED) -> S-T test -> streaming demo (one card, in the background)
#   -> offline O-oracle, B-off -> streaming S-LA2, B-LA2, S-T-oracle -> O-nosd (train, test) -> GPU holder on all cards.
# A failed step is logged and the queue goes on with the next one. Inference runs as eight shards (one per card) and
# is scored with gstar_score.py; a step whose hyp.jsonl exists is skipped. Launch detached; DRY_RUN=1 prints.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
O1="${O1:?the offline run thinker_gstar_o1_...}"
TS="${O1#thinker_gstar_o1_}"
M1=eval_outputs/uniss_qwen3_omni_v1/${O1}/thinker_merged
S1=thinker_gstar_s1_${TS}
N1=thinker_gstar_o1nosd_${TS}
OFF=data/ami_multispk_si_v1/e2e_gstar_v1
SDATA=data/ami_multispk_si_v1/e2e_gstar_stream_v1
EV_OFF=eval_outputs/uniss_qwen3_omni_v1/gstar_offline_eval_o1_${TS}
EV_ST=eval_outputs/uniss_qwen3_omni_v1/gstar_stream_eval_o1_${TS}
HOLDER="${HOLDER:-/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad/holder_on.sh}"
read -r -a G <<<"${GPUS:-0 1 2 3 4 5 6 7}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else eval "$*"; fi; }

infer() {   # infer <offline|online> <eval dir> <name> <thinker> <windows> <post key> <extra args>
  local kind=$1 ev=$2 name=$3 thinker=$4 win=$5 pk=$6 extra=$7 out="$2/$3" mod=gstar_si_stream
  [ "${kind}" = online ] && mod=gstar_si_stream_online
  if [ -e "${out}/hyp.jsonl" ]; then log "skip ${name} (done)"; return 0; fi
  [ -n "${DRY_RUN:-}" ] || [ -d "${thinker}" ] || { log "FAILED ${name}: no ${thinker}"; return 1; }
  run "mkdir -p ${out}"
  log "infer ${name}"
  for k in "${!G[@]}"; do
    run "CUDA_VISIBLE_DEVICES=${G[$k]} PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m experiments.uniss_qwen3_omni_v1.evaluation.${mod} --thinker ${thinker} --windows ${win} --out ${out}/hyp_s${k}.jsonl --shard ${k} --num-shards ${#G[@]} --post-key ${pk} ${extra} > ${out}/infer_s${k}.log 2>&1 &"
  done
  run wait
  local n=$(cat ${out}/hyp_s*.jsonl 2>/dev/null | wc -l)
  if [ -z "${DRY_RUN:-}" ] && [ "${n}" -lt "$(wc -l < ${win})" ]; then log "FAILED ${name}: ${n} of $(wc -l < ${win}) windows (see ${out}/infer_s*.log)"; return 1; fi
  run "cat ${out}/hyp_s*.jsonl > ${out}/hyp.jsonl"
  run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows ${win} --hyp ${out}/hyp.jsonl --post-key ${pk} --out ${out}/score.json > ${out}/score.log 2>&1"
  run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_analyze --windows ${win} --hyp ${out}/hyp.jsonl --preset ${pk%_*} --slot-map $( [ "${kind}" = online ] && echo ${SDATA} || echo ${OFF} )/slot_map.json --out ${out}/analysis.json > /dev/null 2>&1"
  log "scored ${name}: $(tr -d '\n ' < ${out}/score.json 2>/dev/null | cut -c1-240)"
}

log "wait for ${M1}"
[ -n "${DRY_RUN:-}" ] || until [ -f "${M1}/sd_proj.pt" ] && grep -q "THINKER_GSTAR_DONE" "logs/multispk/${O1}.log" 2>/dev/null; do
  grep -q "THINKER_GSTAR_FAILED\|THINKER_GSTAR_MERGE_FAILED" "logs/multispk/${O1}.log" 2>/dev/null && { log "O-G1 failed"; exit 1; }
  sleep 60
done
infer offline "${EV_OFF}" O-G1 "${M1}" "${OFF}/windows_test_ref.jsonl" offline "--inst gstar"

# streaming: GPU smoke of the online decoder on a few windows, then S-T
[ -n "${DRY_RUN:-}" ] || until [ -f "${SDATA}/summary.json" ]; do sleep 30; done
run "mkdir -p ${EV_ST}/smoke"
for p in wait la2; do
  run "CUDA_VISIBLE_DEVICES=${G[0]} PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online --thinker ${M1} --windows ${SDATA}/windows_test_ref.jsonl --out ${EV_ST}/smoke/hyp_${p}.jsonl --policy ${p} --post-key lowlat --limit 6 --batch 6 > ${EV_ST}/smoke/${p}.log 2>&1" \
    && log "online smoke ${p}: ok $(head -c 200 ${EV_ST}/smoke/hyp_${p}.jsonl | tr -d '\n')" || log "FAILED online smoke ${p} (see ${EV_ST}/smoke/${p}.log)"
done
if [ ! -d "eval_outputs/uniss_qwen3_omni_v1/${S1}/thinker_merged" ]; then
  N=$(wc -l < "${SDATA}/rows_train.jsonl" 2>/dev/null || echo 128000)
  ITERS=${S_ITERS:-$(( (N + 63) / 64 ))}
  [ "${ITERS}" -gt 2000 ] && ITERS=2000
  log "train S-T ${S1}: ${ITERS} iterations on ${N} rows, from ${M1}"
  run "BASE=${M1} INIT_MERGED=1 RUN_ID=${S1} ROWS=${SDATA}/rows_train.jsonl TRAIN_ITERS=${ITERS} bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${S1}.log 2>&1" || log "FAILED S-T training (logs/multispk/${S1}.log)"
fi
MS=eval_outputs/uniss_qwen3_omni_v1/${S1}/thinker_merged
infer online "${EV_ST}" S-T_d1 "${MS}" "${SDATA}/windows_test_ref.jsonl" lowlat "--policy wait --window 1"
if [ -e "${EV_ST}/S-T_d1/score.json" ]; then
  log "streaming demo in the background (card ${G[7]})"
  run "MODE=stream POST_KEY=lowlat WIN=${SDATA}/windows_test_ref.jsonl SYS=${EV_ST}/S-T_d1 GPU=${G[7]} bash experiments/uniss_qwen3_omni_v1/scripts/run_gstar_demo.sh > logs/multispk/gstar_demo_stream.log 2>&1 &"
  DEMO_PID=$!
fi

# the comparisons
infer offline "${EV_OFF}" O-oracle "${M1}" "${OFF}/windows_test_ref.jsonl" oracle_offline "--inst gstar"
infer offline "${EV_OFF}" B-off pretrained_models/Qwen3-Omni-30B-A3B-Instruct "${OFF}/windows_test_ref.jsonl" offline "--inst sd"
infer online "${EV_ST}" S-LA2_d1 "${M1}" "${SDATA}/windows_test_ref.jsonl" lowlat "--policy la2 --window 1"
infer online "${EV_ST}" S-T-oracle_d1 "${MS}" "${SDATA}/windows_test_ref.jsonl" oracle_lowlat "--policy wait --window 1"
if [ ! -e "${EV_ST}/B-LA2_d1/hyp.jsonl" ]; then   # the original Thinker, no speakers, LocalAgreement-2 (vLLM)
  out="${EV_ST}/B-LA2_d1"; run "mkdir -p ${out}"
  log "infer B-LA2_d1"
  run "${PYE} -c \"import json;[print(json.dumps({'id':w['id'],'audio':w['audio'],'dur_ms':w['dur_ms']})) for w in map(json.loads, open('${SDATA}/windows_test_ref.jsonl'))]\" > ${out}/items.jsonl"
  for k in "${!G[@]}"; do
    run "${PYE} -c \"import sys;L=open('${out}/items.jsonl').readlines();sys.stdout.writelines(L[${k}::${#G[@]}])\" > ${out}/items_s${k}.jsonl"
    run "CUDA_VISIBLE_DEVICES=${G[$k]} VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. ${PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream --direction en2zh --items-file ${out}/items_s${k}.jsonl --windows 1 --policy la2 --thinker pretrained_models/Qwen3-Omni-30B-A3B-Instruct --out ${out}/s${k} > ${out}/infer_s${k}.log 2>&1 &"
    [ -n "${DRY_RUN:-}" ] || sleep 60   # vLLM engines profiling memory at the same moment fail; stagger them
  done
  run wait
  run "${PYE} -c \"
import json, glob
with open('${out}/hyp.jsonl', 'w') as fh:
    for f in sorted(glob.glob('${out}/s*/blocks_en2zh_w1.jsonl')):
        for r in map(json.loads, open(f)):
            bl = [b for b in r['blocks'] if b['text'].strip()]
            seg = [{'spk': 1, 'start_s': 0.0, 'end_s': 0.1, 'text': r['translation']}] if r['translation'].strip() else []
            fh.write(json.dumps({'id': r['id'], 'segments': seg,
                                 'blocks': [{'write_ms': b['write_ms'], 'spk': 1, 'start_s': 0.0, 'text': b['text']} for b in bl]}, ensure_ascii=False) + '\n')
\""
  run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows ${SDATA}/windows_test_ref.jsonl --hyp ${out}/hyp.jsonl --post-key lowlat --out ${out}/score.json > ${out}/score.log 2>&1"
  log "scored B-LA2_d1: $(tr -d '\n ' < ${out}/score.json 2>/dev/null | cut -c1-240)"
fi
if [ ! -d "eval_outputs/uniss_qwen3_omni_v1/${N1}/thinker_merged" ]; then
  log "train O-nosd ${N1}"
  run "NOSD=1 RUN_ID=${N1} ROWS=${OFF}/rows_train.jsonl TRAIN_ITERS=1000 bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${N1}.log 2>&1" || log "FAILED O-nosd training"
fi
infer offline "${EV_OFF}" O-nosd "eval_outputs/uniss_qwen3_omni_v1/${N1}/thinker_merged" "${OFF}/windows_test_ref.jsonl" offline "--inst gstar --zero-sd"
[ -n "${DEMO_PID:-}" ] && [ -z "${DRY_RUN:-}" ] && wait "${DEMO_PID}"
log "GSTAR_NIGHT_DONE"
until [ -n "${DRY_RUN:-}" ] || ! pgrep -f "run_gstar_demo.sh" > /dev/null; do sleep 30; done   # the offline demo, if still speaking
run "bash ${HOLDER} 0,1,2,3,4,5,6,7"
