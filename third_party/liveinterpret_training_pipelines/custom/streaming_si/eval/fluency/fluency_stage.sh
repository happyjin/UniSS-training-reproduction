#!/bin/bash
export FLUENCY_CODE=${ROOT}/fluency_code
mkdir -p "${FLUENCY_CODE}"
for f in "${CODE}"/fluency__*.py; do
  [[ -f "${f}" ]] || continue
  b=$(basename "${f}")
  cp "${f}" "${FLUENCY_CODE}/${b#fluency__}"
done
export LITELLM_API_BASE=${LITELLM_API_BASE:-https://litellm.ai.levelinfinite.com/v1}
export LITELLM_FLUENCY_MODEL=${LITELLM_FLUENCY_MODEL:-gpt-6-sol}
export VAD_THRESHOLD=${VAD_THRESHOLD:-0.2}
export VAD_MIN_SILENCE_MS=${VAD_MIN_SILENCE_MS:-100} VAD_MIN_SPEECH_MS=${VAD_MIN_SPEECH_MS:-250}
export MIN_PAUSE_S=${MIN_PAUSE_S:-0.10} LONG_PAUSE_S=${LONG_PAUSE_S:-2.00}
export TOL_WORD_INTERNAL=${TOL_WORD_INTERNAL:-0.15} TOL_TIGHT_STRUCTURE=${TOL_TIGHT_STRUCTURE:-0.30}
export TOL_PHRASE_BOUNDARY=${TOL_PHRASE_BOUNDARY:-0.50} TOL_SENTENCE_BOUNDARY=${TOL_SENTENCE_BOUNDARY:-1.00}
export LITELLM_RPM=${LITELLM_RPM:-90}
export LITELLM_RATE_FILE=${LITELLM_RATE_FILE:-${NVME_MOUNT}/${OWNER}/fluency_rate.txt}

run_fluency() {
  local base_py=${IMG_PY:-$(command -v python)} venv=${ROOT}/venv_fluency_vad py site
  [[ -f "${FLUENCY_CODE}/streaming_eval.py" ]] || { log "[FATAL] 缺少 fluency 代码，请用更新后的 submit.sh 提交"; return 1; }
  if [[ -f "${CREDS_DIR}/LITELLM_API_KEY" ]]; then export LITELLM_API_KEY=$(cat "${CREDS_DIR}/LITELLM_API_KEY"); fi
  [[ -n "${LITELLM_API_KEY:-}" ]] || { log "[FATAL] 缺少 speech_LLM prod key"; return 1; }
  if [[ ! -x "${venv}/bin/python" ]]; then "${base_py}" -m venv --system-site-packages "${venv}"; fi
  py=${venv}/bin/python
  site=$("${py}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  "${base_py}" -c 'import site; print("\n".join(site.getsitepackages()))' > "${site}/_image_site.pth"
  "${py}" -m pip install -q --no-deps silero-vad==6.2.0 requests==2.32.5 soundfile==0.13.1 soxr==1.0.0 jieba==0.42.1
  "${py}" -m pip install -q boto3==1.43.62
  PYTHONPATH=${FLUENCY_CODE} "${py}" -c 'import audio_vad; audio_vad.get_vad_model(); print("[fluency] VAD environment OK")'
  for s in ${SETS}; do
    [[ -f "${ROOT}/runs/${s}/results.jsonl" ]] || { log "[FATAL] 缺少 ${s} 推理结果"; return 1; }
    if [[ ! -d "${ROOT}/report/translation_audio/${s}" ]]; then
      [[ "${REPORT_ONLY:-0}" != 1 ]] || { log "[FATAL] 缺少已有译音，禁止重建：${s}"; return 1; }
      render_audio "${s}"
    fi
    log "fluency: ${s} model=${LITELLM_FLUENCY_MODEL} jobs=${FLUENCY_JOBS:-4} VAD=${VAD_THRESHOLD} tolerance=${TOL_WORD_INTERNAL}/${TOL_TIGHT_STRUCTURE}/${TOL_PHRASE_BOUNDARY}/${TOL_SENTENCE_BOUNDARY}"
    local extra=()
    if [[ -n "${FLUENCY_LIMIT:-}" ]]; then extra+=(--limit "${FLUENCY_LIMIT}"); fi
    "${py}" "${FLUENCY_CODE}/streaming_eval.py" --results "${ROOT}/runs/${s}/results.jsonl" \
      --asr "${ROOT}/runs/${s}/asr_lat.jsonl" --audio-dir "${ROOT}/report/translation_audio/${s}" \
      --out "${ROOT}/runs/${s}/${FLUENCY_RESULTS_FILE:-fluency.jsonl}" --jobs "${FLUENCY_JOBS:-4}" "${extra[@]}" \
      2>&1 | tee "${ROOT}/logs/fluency.${s}.log"
  done
}

upload_fluency_docs() {
  local destination=${S3_REPORT_PREFIX:-${S3_OUT%/}/${RUN_NAME}/report} file s cli
  cli=${AWS:-}
  [[ -n "${cli}" ]] || cli=$(command -v aws)
  [[ -n "${cli}" ]] || { log "[FATAL] 找不到 aws CLI（AWS 变量与 PATH 均无）"; return 1; }
  for file in REPORT.md report.json demo.html demo.json; do
    "${cli}" s3 cp "${ROOT}/report/${file}" "${destination}/${file}" --only-show-errors
  done
  for s in ${SETS}; do
    "${cli}" s3 cp "${ROOT}/report/raw/${s}/fluency.jsonl" "${destination}/raw/${s}/fluency.jsonl" --only-show-errors
  done
  log "仅上传报告/demo/fluency，音频不动 -> ${destination}"
}
