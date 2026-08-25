#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "Usage: $0 RUN_ID ADAPTER_CHECKPOINT_OR_NONE OUTPUT_DIR GPU0 GPU1 GPU2 GPU3 [CHUNK_MS]" >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../config.env"
RUN_ID=$1
ADAPTER=$2
OUTPUT=$3
GPUS=($4 $5 $6 $7)
CHUNK=${8:-640}
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
if [[ "${ADAPTER}" != NONE ]]; then
  [[ -f "${ADAPTER}/.metadata" ]] || { echo "missing adapter checkpoint" >&2; exit 3; }
fi
mkdir -p "${OUTPUT}/parts" "${OUTPUT}/logs"

export HF_HOME=${USER_ROOT}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TMPDIR=${USER_ROOT}/tmp
export PYTHONPATH=${REPO_ROOT}/third_party/Megatron-LM:${REPO_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
adapter_args=()
[[ "${ADAPTER}" == NONE ]] || adapter_args=(--adapter-checkpoint "${ADAPTER}")
IDS=(
  long_zh_singapore_vietnam_full
  long_zh_zhangheqiao_full
  long_en_helen_keller_full
  long_en_shimon_peres_full
)

pids=()
for index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$index]} "${PYTHON_BIN}" -u \
    "${EXPERIMENT_ROOT}/evaluation/bounded_longform.py" \
    --run-id "${RUN_ID}" \
    --audio-protocol "${EXPERIMENT_ROOT}/evaluation/protocols/long_audio4_full.json" \
    --sample-id "${IDS[$index]}" \
    --decision-chunk-ms "${CHUNK}" \
    --output "${OUTPUT}/parts/${IDS[$index]}" \
    --base-hf "${REPO_ROOT}/checkpoints/exported_hf/uniss_stage_a_formal8_iter_0000381_hf" \
    "${adapter_args[@]}" \
    --v1-checkpoint "${REPO_ROOT}/checkpoints/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_formal/stage_a_formal8_20260816T224100Z/iter_0000381" \
    --whispervq-model "${WHISPERVQ_MODEL}" \
    --bicodec-model "${REPO_ROOT}/pretrained_models/UniSS/bicodec" \
    --source-snapshot "${REPO_ROOT}/data/processed/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_causal_asr/source_snapshot_v5.json" \
    --strict-runtime "${REPO_ROOT}/eval_outputs/uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/stage_a_v1_strict_streaming_train_demo_20260820T000000Z/run_strict_causal_cascade.py" \
    --device cuda:0 > "${OUTPUT}/logs/${IDS[$index]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || { echo "one or more bounded long-form workers failed" >&2; exit 1; }

"${PYTHON_BIN}" - "${OUTPUT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
parts=sorted(root.glob('parts/*/results.json'))
if len(parts)!=4: raise SystemExit(f'expected four complete parts, found {len(parts)}')
payloads=[json.loads(p.read_text()) for p in parts]
if any(p.get('status')!='complete' or len(p.get('results',[]))!=1 for p in payloads):
    raise SystemExit('bounded long-form part is incomplete')
merged={
 'schema_version':'uniss_stagea_joint_grpo_bounded_longform_merged_v1',
 'status':'complete','mode':'complete bounded-window pseudo-streaming',
 'claim_boundary':'per-window strict causality with cross-window state reset; not cached causal long-form',
 'decision_chunk_ms':payloads[0]['decision_chunk_ms'],
 'adapter_manifest':payloads[0]['adapter_manifest'],
 'results':[p['results'][0] for p in payloads],
 'part_reports':[str(p.resolve()) for p in parts],
}
(root/'results.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(merged,ensure_ascii=False,indent=2))
PY

