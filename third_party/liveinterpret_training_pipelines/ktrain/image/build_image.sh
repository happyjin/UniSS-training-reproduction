#!/usr/bin/env bash
# =============================================================================
# 构建并推送 ktrain 训练镜像（单一 Dockerfile，context = repo 根目录）。
#
# 用法：bash ktrain/image/build_image.sh <tag>
#   TAG 必须显式提供（immutable tag，避免误覆盖既有 tag）。
#
# 可覆盖 env：REGISTRY / IMAGE。
# =============================================================================
set -euo pipefail

TAG="${1:-${TAG:-}}"
: "${REGISTRY:=124355679795.dkr.ecr.ap-southeast-3.amazonaws.com}"
: "${IMAGE:=ktrain/speech-language-model}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

[[ -n "${TAG}" ]] || { echo "[build][FATAL] 需提供 TAG（位置参数或 env TAG），immutable tag 不允许空" >&2; exit 1; }

FULL="${REGISTRY}/${IMAGE}:${TAG}"
echo "[build] image     : ${FULL}"
echo "[build] dockerfile: ${HERE}/Dockerfile"
echo "[build] context   : ${REPO_ROOT}"

aws ecr get-login-password --region ap-southeast-3 \
  | docker login --username AWS --password-stdin "${REGISTRY}"

# context = repo 根目录（Dockerfile 里 `COPY requirements.txt` 依赖它）
docker build -f "${HERE}/Dockerfile" -t "${FULL}" "${REPO_ROOT}"
echo "[build] pushing ${FULL} ..."
docker push "${FULL}"

echo "[build] DONE"
echo "[build] spec.image = ${FULL}"
