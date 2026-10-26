#!/usr/bin/env bash
# The streaming stage of docs/multi_speaker_si_plan_v8_streaming.zh-CN.md, started by hand once the offline O-G1 has
# passed its gate. One after another (each inference as eight shards, one per card, then gstar_score.py):
#   streaming rows -> S-LA2 (O-G1 as is + LocalAgreement-2; DELTAS, default 1 2 3 s) -> B-LA2 (the original Thinker
#   + LocalAgreement-2, no speaker stream) -> train S-T on O-G1 (INIT_MERGED) -> S-T (policy wait) and S-T-oracle.
# Launch detached (setsid nohup ... &); DRY_RUN=1 prints the commands. STEPS (default "rows la2 b0 train st") picks.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
O1="${O1:?the offline run (thinker_gstar_o1_...)}"
M1=eval_outputs/uniss_qwen3_omni_v1/${O1}/thinker_merged
SDATA=data/ami_multispk_si_v1/e2e_gstar_stream_v1
WIN="${SDATA}/windows_test_ref.jsonl"
S1="${S1:-thinker_gstar_s1_${O1#thinker_gstar_o1_}}"
EV=eval_outputs/uniss_qwen3_omni_v1/gstar_stream_eval_${O1#thinker_gstar_}
read -r -a G <<<"${GPUS:-0 1 2 3 4 5 6 7}"
STEPS="${STEPS:-rows la2 b0 train st}"
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else eval "$*"; fi; }
has() { [[ " ${STEPS} " == *" $1 "* ]]; }

infer() {   # infer <name> <thinker> <extra args>
  local name=$1 thinker=$2 extra=$3 out="${EV}/$1"
  if [ -e "${out}/hyp.jsonl" ]; then log "skip ${name} (done)"; return 0; fi
  run "mkdir -p ${out}"
  log "infer ${name}"
  for k in "${!G[@]}"; do
    run "CUDA_VISIBLE_DEVICES=${G[$k]} PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online --thinker ${thinker} --windows ${WIN} --out ${out}/hyp_s${k}.jsonl --shard ${k} --num-shards ${#G[@]} ${extra} > ${out}/infer_s${k}.log 2>&1 &"
  done
  run wait
  run "cat ${out}/hyp_s*.jsonl > ${out}/hyp.jsonl"
  score "${name}" "${extra}"
}
score() {
  local out="${EV}/$1" pk=lowlat
  [[ "$2" == *oracle_lowlat* ]] && pk=oracle_lowlat
  run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows ${WIN} --hyp ${out}/hyp.jsonl --post-key ${pk} --out ${out}/score.json > ${out}/score.log 2>&1"
  log "scored $1: $(tr -d '\n ' < ${out}/score.json 2>/dev/null | cut -c1-220)"
}

[ -f "${M1}/sd_proj.pt" ] || { log "no merged O-G1 at ${M1}"; exit 1; }
if has rows && [ ! -e "${SDATA}" ]; then
  log "streaming rows -> ${SDATA}"
  run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_stream_rows --out ${SDATA} > logs/multispk/build_gstar_stream_v1.log 2>&1" || { log "rows failed"; exit 1; }
fi
if has la2; then
  for d in ${DELTAS:-1 2 3}; do infer "S-LA2_d${d}" "${M1}" "--policy la2 --window ${d} --post-key lowlat"; done
fi
if has b0 && [ ! -e "${EV}/B-LA2_d1/hyp.jsonl" ]; then   # the original Thinker, no speakers, LocalAgreement-2, same windows
  out="${EV}/B-LA2_d1"; run "mkdir -p ${out}"
  log "infer B-LA2_d1"
  run "${PYE} -c \"import json;[print(json.dumps({'id':w['id'],'audio':w['audio'],'dur_ms':w['dur_ms']})) for w in map(json.loads, open('${WIN}'))]\" > ${out}/items.jsonl"
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
            fh.write(json.dumps({'id': r['id'], 'segments': [{'spk': 1, 'start_s': 0.0, 'end_s': 0.1, 'text': r['translation']}] if r['translation'].strip() else [],
                                 'blocks': [{'write_ms': b['write_ms'], 'spk': 1, 'start_s': 0.0, 'text': b['text']} for b in bl]}, ensure_ascii=False) + '\n')
\""
  score B-LA2_d1 ""
fi
if has train && [ ! -d "eval_outputs/uniss_qwen3_omni_v1/${S1}/thinker_merged" ]; then
  N=$(wc -l < "${SDATA}/rows_train.jsonl" 2>/dev/null || echo 128000)
  ITERS=${S_ITERS:-$(( (N + 63) / 64 ))}
  log "train S-T ${S1} (${ITERS} iterations on ${N} rows, from ${M1})"
  run "BASE=${M1} INIT_MERGED=1 RUN_ID=${S1} ROWS=${SDATA}/rows_train.jsonl TRAIN_ITERS=${ITERS} bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/${S1}.log 2>&1" || { log "S-T training failed"; exit 1; }
fi
if has st; then
  MS=eval_outputs/uniss_qwen3_omni_v1/${S1}/thinker_merged
  infer S-T_d1 "${MS}" "--policy wait --window 1 --post-key lowlat"
  infer S-T-oracle_d1 "${MS}" "--policy wait --window 1 --post-key oracle_lowlat"
fi
log "GSTAR_STREAM_DONE ${EV}"
if [ -n "${HOLDER:-}" ]; then bash "${HOLDER}" 0,1,2,3,4,5,6,7; fi
