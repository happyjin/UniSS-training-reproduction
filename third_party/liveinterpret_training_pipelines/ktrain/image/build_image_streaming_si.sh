#!/usr/bin/env bash
# =============================================================================
# 构建、验证并推送 streaming_si 镜像（Dockerfile.streaming-si，基于现有 v9 训练镜像叠加一层）。
#
# 用法（在 GPU 服务器上，需要 docker、aws CLI、对 ECR 的推送权限）：
#   bash ktrain/image/build_image_streaming_si.sh <tag>
#   例：bash ktrain/image/build_image_streaming_si.sh p5-cuda12.8-torch2.8-te-v9-streaming-si-v1
#
# 可覆盖 env：
#   REGISTRY / IMAGE     推送目的地（默认与基础镜像相同的仓库）
#   BASE_IMAGE           基础镜像（默认 ...:p5-cuda12.8-torch2.8-te-v9）
#   AWS_REGION           默认 ap-southeast-3
#   PUSH=0               只构建与验证，不推送
#   GPU_VERIFY=0         跳过 GPU 验证（无 GPU 或未装 nvidia-container-toolkit 时）
#
# GPU 验证（默认开启，推送前执行，失败则不推送）：
#   * transformers 认到 fla / causal-conv1d（is_*_available 只在有 CUDA 时为真，构建阶段测不到）；
#   * fla 的 chunk_gated_delta_rule 在 GPU 上与 torch 参考实现数值一致；
#   * 用仓库代码跑 streaming_si 单测（tiny 随机权重，含 4B 变体与推理对拍）。
# =============================================================================
set -euo pipefail

TAG="${1:-${TAG:-}}"
: "${REGISTRY:=124355679795.dkr.ecr.ap-southeast-3.amazonaws.com}"
: "${IMAGE:=ktrain/speech-language-model}"
: "${BASE_IMAGE:=${REGISTRY}/${IMAGE}:p5-cuda12.8-torch2.8-te-v9}"
: "${AWS_REGION:=ap-southeast-3}"
: "${PUSH:=1}"
: "${GPU_VERIFY:=1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

[[ -n "${TAG}" ]] || { echo "[build][FATAL] 需提供 TAG（位置参数或 env TAG）；ECR 为 immutable tag，禁止 latest" >&2; exit 1; }
[[ "${TAG}" != latest ]] || { echo "[build][FATAL] 禁止使用 latest" >&2; exit 1; }
FULL="${REGISTRY}/${IMAGE}:${TAG}"
echo "[build] image      : ${FULL}"
echo "[build] base       : ${BASE_IMAGE}"
echo "[build] dockerfile : ${HERE}/Dockerfile.streaming-si（context = ${HERE}）"

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
if [[ "${PUSH}" == 1 ]] && aws ecr describe-images --region "${AWS_REGION}" --repository-name "${IMAGE}" \
        --image-ids imageTag="${TAG}" >/dev/null 2>&1; then
    echo "[build][FATAL] ${FULL} 已存在（immutable tag），请换一个 tag" >&2
    exit 1
fi

docker pull "${BASE_IMAGE}"
docker build -f "${HERE}/Dockerfile.streaming-si" --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${FULL}" "${HERE}"

if [[ "${GPU_VERIFY}" == 1 ]]; then
    echo "[build] GPU 验证 ..."
    docker run --rm --gpus 1 --shm-size 8g -v "${REPO_ROOT}:/repo:ro" -w /repo \
        -e PYTHONDONTWRITEBYTECODE=1 "${FULL}" bash -euo pipefail -c '
python - <<"PY"
import torch
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
assert torch.cuda.is_available(), "容器内没有可用 GPU"
print("GPU:", torch.cuda.get_device_name(0))
assert is_flash_linear_attention_available(), "transformers 未识别 fla（需 >=0.2.2）"
assert is_causal_conv1d_available(), "transformers 未识别 causal-conv1d"
from transformers.models.qwen3_next import modeling_qwen3_next as qn
assert qn.chunk_gated_delta_rule is not None and qn.causal_conv1d_fn is not None
torch.manual_seed(0)
B, T, H, K, V = 1, 96, 4, 32, 32
q, k = (torch.randn(B, T, H, K, device="cuda") for _ in range(2))
v = torch.randn(B, T, H, V, device="cuda")
g = -torch.rand(B, T, H, device="cuda")
beta = torch.rand(B, T, H, device="cuda")
kw = dict(initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
ref = qn.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kw)[0].float()
fast = qn.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kw)[0].float()
err = (ref - fast).abs().max().item()
print(f"fla chunk_gated_delta_rule vs torch 参考: max|Δ|={err:.2e}")
assert err < 1e-2, err
PY
python -m pytest custom/streaming_si -q -p no:cacheprovider -W ignore -x
'
fi

if [[ "${PUSH}" == 1 ]]; then
    echo "[build] pushing ${FULL} ..."
    docker push "${FULL}"
fi
echo "[build] DONE"
echo "[build] spec.image = ${FULL}"
