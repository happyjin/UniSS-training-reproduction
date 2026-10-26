#!/usr/bin/env bash
# A listening demo of G-STAR-SI offline output (docs/multi_speaker_si_plan_v7.zh-CN.md): wait for the O-G1 test
# hypotheses of run_gstar_offline_eval.sh, keep the reviewed windows of the first EXCERPT_S seconds of a few test
# meetings, speak every speaker slot in its own voice (gstar_talker.py prep -> si_infer on one card, sharing it with
# whatever runs there at --gpu-mem 0.3), ASR on the speech alone, then stitch: per meeting a stereo excerpt (left:
# the meeting, right: the translation laid at the source times) and its who / when / what transcript; scores of the
# demo windows (text and speech). Launch detached; DRY_RUN=1 prints the commands.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
SYS="${SYS:?the eval directory of the system, e.g. eval_outputs/uniss_qwen3_omni_v1/gstar_offline_eval_o1_.../O-G1}"
OUT="${OUT:-${SYS}/demo}"
MEETINGS="${MEETINGS:-EN2002a ES2004a IS1009a TS3003a}"
EXCERPT_S="${EXCERPT_S:-240}"
GPU="${GPU:-7}"
DATA=data/ami_multispk_si_v1/e2e_gstar_v1
WIN=${DATA}/windows_test_ref.jsonl
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
SCR=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else eval "$*"; fi; }

[ -n "${DRY_RUN:-}" ] || until [ -s "${SYS}/score.json" ]; do sleep 30; done   # hyp.jsonl is complete once scored
[ -e "${OUT}" ] && [ -z "${DRY_RUN:-}" ] && { log "拒绝覆盖已有输出: ${OUT}"; exit 1; }
run "mkdir -p ${OUT}"
log "demo windows: ${MEETINGS}, first ${EXCERPT_S} s"
run "${PYE} -c \"
import json
ms = set('${MEETINGS}'.split()); W = [json.loads(l) for l in open('${WIN}')]
t0 = {}
for w in W:
    if w['meeting'] in ms: t0[w['meeting']] = min(t0.get(w['meeting'], 1e9), w['start_s'])
keep = {w['id'] for w in W if w['meeting'] in ms and w['start_s'] < t0[w['meeting']] + ${EXCERPT_S}}
with open('${OUT}/windows.jsonl', 'w') as f: f.writelines(json.dumps(w, ensure_ascii=False) + '\n' for w in W if w['id'] in keep)
with open('${OUT}/hyp.jsonl', 'w') as f: f.writelines(l for l in open('${SYS}/hyp.jsonl') if json.loads(l)['id'] in keep)
print(len(keep), 'windows')
\""
run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker prep --hyp ${OUT}/hyp.jsonl --windows ${OUT}/windows.jsonl --voices ${DATA}/voices_offline --out ${OUT}/talker_in"
log "talker on card ${GPU}"
run "CUDA_VISIBLE_DEVICES=${GPU} VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=16 ${PYV} -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model ${T1C} --direction en2zh --window 1 --output ${OUT}/talker --text-policy file --blocks-file ${OUT}/talker_in/blocks.jsonl --items-file ${OUT}/talker_in/items.jsonl --speed-policy fixed --history-ban 8 --max-model-len 16384 --gpu-mem 0.3 > ${OUT}/talker.log 2>&1" || { log "talker failed"; exit 1; }
run "cat ${OUT}/talker/part_*.jsonl > ${OUT}/talker/results.jsonl && ${PYE} ${SCR}/p80_speechonly.py ${OUT}/talker"
run "CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=. ${PYE} -m evaluation.asr_transcribe --input ${OUT}/talker/results_speechonly.jsonl --output ${OUT}/talker/asr_speech.jsonl --device cuda:0 --batch-size 16 > ${OUT}/asr.log 2>&1"
run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker stitch --results ${OUT}/talker/results.jsonl --prep ${OUT}/talker_in --windows ${OUT}/windows.jsonl --mode offline --excerpt-s ${EXCERPT_S} --out ${OUT}/stitched"
run "PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows ${OUT}/windows.jsonl --hyp ${OUT}/hyp.jsonl --asr ${OUT}/talker/asr_speech.jsonl --out ${OUT}/score.json > ${OUT}/score.log 2>&1"
log "GSTAR_DEMO_DONE ${OUT}"
