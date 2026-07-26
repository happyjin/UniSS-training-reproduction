#!/usr/bin/env bash
set -uo pipefail

# Download the publicly accessible evaluation corpora used by the UniSS paper.
# CVSS-T/Common Voice v4 and ESD are intentionally not automated here because
# their source audio requires license acceptance or a manual data request.

ROOT="${UNISS_EVAL_DATA_ROOT:-/opt/dlami/nvme/jasonleeeli/datasets/uniss_paper_evaluation}"
HF_HOME="${HF_HOME:-/opt/dlami/nvme/jasonleeeli/cache/huggingface}"
HF_CLI="${HF_CLI:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/huggingface-cli}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-2}"
LOG_ROOT="${ROOT}/logs"
MARKER_ROOT="${ROOT}/markers"
TOOLS_ROOT="${ROOT}/tools"

mkdir -p "${ROOT}" "${LOG_ROOT}" "${MARKER_ROOT}" "${TOOLS_ROOT}" "${HF_HOME}"

timestamp() {
  date -u +%FT%TZ
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

file_has_size() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" ]] && [[ "$(stat -c %s "${path}")" == "${expected}" ]]
}

download_with_resume() {
  local url="$1"
  local output="$2"
  local expected_size="$3"
  local partial="${output}.part"

  mkdir -p "$(dirname "${output}")"
  if file_has_size "${output}" "${expected_size}"; then
    log "SKIP complete file: ${output}"
    return 0
  fi

  curl -L --fail --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
    --retry 0 -C - -o "${partial}" "${url}" || return 1

  if ! file_has_size "${partial}" "${expected_size}"; then
    log "Size check failed: ${partial}; expected=${expected_size}, actual=$(stat -c %s "${partial}" 2>/dev/null || echo 0)"
    return 1
  fi
  mv "${partial}" "${output}"
}

verify_fleurs() {
  local root="${ROOT}/FLEURS"
  file_has_size "${root}/data/en_us/audio/test.tar.gz" 289851356 &&
    file_has_size "${root}/data/en_us/test.tsv" 367864 &&
    file_has_size "${root}/data/cmn_hans_cn/audio/test.tar.gz" 525346466 &&
    file_has_size "${root}/data/cmn_hans_cn/test.tsv" 491487
}

download_fleurs() {
  local root="${ROOT}/FLEURS"
  mkdir -p "${root}"
  if verify_fleurs; then
    log "FLEURS English/Chinese test files are already complete"
    touch "${MARKER_ROOT}/FLEURS_EN_ZH_TEST.complete"
    return 0
  fi

  "${HF_CLI}" download google/fleurs \
    data/en_us/audio/test.tar.gz \
    data/en_us/test.tsv \
    data/cmn_hans_cn/audio/test.tar.gz \
    data/cmn_hans_cn/test.tsv \
    --repo-type dataset \
    --cache-dir "${HF_HOME}/hub" \
    --local-dir "${root}" \
    --max-workers "${HF_MAX_WORKERS}" || return 1

  verify_fleurs || return 1
  touch "${MARKER_ROOT}/FLEURS_EN_ZH_TEST.complete"
  log "FLEURS English/Chinese test download complete"
}

install_local_git_lfs() {
  local version="3.7.1"
  local archive="${TOOLS_ROOT}/git-lfs-linux-amd64-v${version}.tar.gz"
  local binary="${TOOLS_ROOT}/git-lfs-${version}/git-lfs"
  local url="https://github.com/git-lfs/git-lfs/releases/download/v${version}/git-lfs-linux-amd64-v${version}.tar.gz"

  if [[ ! -x "${binary}" ]]; then
    download_with_resume "${url}" "${archive}" 5524590 || return 1
    tar -xzf "${archive}" -C "${TOOLS_ROOT}" || return 1
  fi
  [[ -x "${binary}" ]] || return 1
  printf '%s\n' "${binary}"
}

verify_cremad() {
  local root="${ROOT}/CREMA-D"
  local count pointers
  [[ -d "${root}/.git" ]] || return 1
  count=$(find "${root}/AudioWAV" -maxdepth 1 -type f -iname '*.wav' 2>/dev/null | wc -l)
  [[ "${count}" == "7442" ]] || return 1
  pointers=$(find "${root}/AudioWAV" -maxdepth 1 -type f -iname '*.wav' -size -256c -print0 2>/dev/null \
    | xargs -0 -r grep -l '^version https://git-lfs.github.com/spec/v1$' 2>/dev/null \
    | wc -l)
  [[ "${pointers}" == "0" ]]
}

download_cremad() {
  local root="${ROOT}/CREMA-D"
  local lfs_bin lfs_dir

  if verify_cremad; then
    log "CREMA-D AudioWAV is already complete"
    touch "${MARKER_ROOT}/CREMA-D_AUDIOWAV.complete"
    return 0
  fi

  lfs_bin=$(install_local_git_lfs) || return 1
  lfs_dir=$(dirname "${lfs_bin}")
  export PATH="${lfs_dir}:${PATH}"

  if [[ ! -d "${root}/.git" ]]; then
    if [[ -e "${root}" ]]; then
      log "CREMA-D destination exists but is not a git repository: ${root}"
      return 1
    fi
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
      https://gitlab.com/cs-cooper-lab/crema-d-mirror.git "${root}" || return 1
  fi

  git -C "${root}" lfs install --local || return 1
  git -C "${root}" lfs pull --include='AudioWAV/**' --exclude='AudioMP3/**,VideoFlash/**' || return 1

  verify_cremad || return 1
  touch "${MARKER_ROOT}/CREMA-D_AUDIOWAV.complete"
  log "CREMA-D AudioWAV download complete"
}

retry_forever() {
  local name="$1"
  shift
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    log "START ${name}, attempt=${attempt}"
    if "$@"; then
      log "DONE ${name}"
      return 0
    fi
    log "FAILED ${name}, retrying remaining data in ${RETRY_SECONDS}s"
    sleep "${RETRY_SECONDS}"
  done
}

show_status() {
  printf 'root=%s\n' "${ROOT}"
  if verify_fleurs; then
    printf 'FLEURS_EN_ZH_TEST=complete\n'
  else
    printf 'FLEURS_EN_ZH_TEST=incomplete\n'
  fi
  if verify_cremad; then
    printf 'CREMA-D_AUDIOWAV=complete\n'
  else
    printf 'CREMA-D_AUDIOWAV=incomplete\n'
  fi
  du -sh "${ROOT}" 2>/dev/null || true
}

case "${1:-all}" in
  fleurs)
    retry_forever FLEURS_EN_ZH_TEST download_fleurs
    ;;
  cremad)
    retry_forever CREMA-D_AUDIOWAV download_cremad
    ;;
  all)
    retry_forever FLEURS_EN_ZH_TEST download_fleurs
    retry_forever CREMA-D_AUDIOWAV download_cremad
    ;;
  status|--status)
    show_status
    ;;
  *)
    echo "Usage: $0 {fleurs|cremad|all|status}" >&2
    exit 2
    ;;
esac
