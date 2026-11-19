#!/usr/bin/env bash
# =============================================================================
# 把整个训练 repo 同步到 S3，使 PyTorchJob pod 启动即读（纯 kubectl 方案：
# launch_train.sh 直接经 S3 CSI 读 TRAIN_SCRIPT）。
#
# 目标 bucket 不再硬编码：从 skill 传入的训练 yaml（pytorchjob.yaml）中解析
# s3all volume 的 persistentVolumeClaim.claimName 与 namespace，再经 kubectl
# 查 PVC -> PV -> csi.volumeAttributes.bucketName 得到（与容器挂载的 /s3all
# 同源，ap-southeast-3 / Singapore）。
# S3 代码前缀取自训练 yaml 的 CODE_PREFIX env（/s3all 下的路径前缀，如
# deferzhou/liveinterpret_training_pipelines），与 launch_train.sh 拼接 TRAIN_SCRIPT
# 时使用的同源，避免 git 仓库名与 S3 前缀不一致导致训练读旧代码。
# 同步策略：先 aws s3 rm --recursive 清空目标前缀，再全量上传（保证与本地
# 严格一致，不残留已删文件）。属变更操作，执行前需确认。
#
# 用法：
#   sync_code.sh [pytorchjob.yaml 路径]   # 默认 <repo>/pytorch_job/pytorchjob.yaml
# 环境变量覆盖：
#   PYTORCHJOB_YAML  训练 yaml 路径（同位置参数，位置参数优先）
#   NAMESPACE        k8s 命名空间（默认取 yaml metadata.namespace）
#   S3_BUCKET        跳过 kubectl 解析，直接指定 bucket
#   CODE_S3          整体覆盖目的地（s3://bucket/prefix）
#   CLUSTER/REGION   eks 集群/区域（默认 speech-h200-eks / ap-southeast-3）
#   REFRESH_KUBECONFIG=0  跳过 aws eks update-kubeconfig（submit.sh 已刷新时用）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}"; while [ ! -d pytorch_job ] && [ "${PWD}" != / ]; do cd ..; done; pwd)"
YAML_PATH="${1:-${PYTORCHJOB_YAML:-${REPO_ROOT}/pytorch_job/pytorchjob.yaml}}"
REGION="${REGION:-ap-southeast-3}"
CLUSTER="${CLUSTER:-speech-h200-eks}"

[[ -f "${YAML_PATH}" ]] || { echo "[sync][FATAL] yaml not found: ${YAML_PATH}" >&2; exit 1; }

# ---- 从训练 yaml 解析 s3all 的 claimName 与 namespace（与 derive_sync_jobs.py 同款正则）----
mapfile -t _cfg <<<"$(python3 - "${YAML_PATH}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
ns = re.search(r'^\s*namespace:\s*(\S+)', text, re.M)
pvc = re.search(r'- name: s3all\s*\n\s*persistentVolumeClaim:\s*\n\s*claimName:\s*(\S+)', text)
if not pvc:
    sys.exit('[sync][FATAL] s3all persistentVolumeClaim.claimName not found in yaml')
print(ns.group(1) if ns else 'default')
print(pvc.group(1))
PY
)"
NAMESPACE="${NAMESPACE:-${_cfg[0]}}"
CLAIM_NAME="${_cfg[1]}"

# ---- 刷 kubecontext（submit.sh 调用时可置 REFRESH_KUBECONFIG=0 跳过）----
if [ "${REFRESH_KUBECONFIG:-1}" = "1" ]; then
  echo "[sync] kubecontext -> ${CLUSTER}"
  aws eks update-kubeconfig --region "${REGION}" --name "${CLUSTER}" >/dev/null
fi

# ---- 经 kubectl 解析 PVC -> PV -> bucketName ----
if [ -z "${S3_BUCKET:-}" ]; then
  PV_NAME="$(kubectl -n "${NAMESPACE}" get pvc "${CLAIM_NAME}" -o jsonpath='{.spec.volumeName}')"
  S3_BUCKET="$(kubectl get pv "${PV_NAME}" -o jsonpath='{.spec.csi.volumeAttributes.bucketName}')"
fi
[ -n "${S3_BUCKET}" ] || { echo "[sync][FATAL] bucketName empty for PVC ${NAMESPACE}/${CLAIM_NAME}" >&2; exit 1; }

# ---- 从训练 yaml 的 CODE_PREFIX env 解析代码 S3 前缀 ----
# CODE_PREFIX 即代码在 /s3all 下的路径前缀（如 deferzhou/liveinterpret_training_pipelines），
# 与 launch_train.sh 拼接 TRAIN_SCRIPT 时使用的同源，保证训练读的是本次同步的代码。
S3_CODE_PREFIX="$(python3 - "${YAML_PATH}" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r'- name:\s*CODE_PREFIX\s*\n\s*value:\s*"?([^"\n]*)"?', t)
if not m:
    sys.exit('[sync][FATAL] CODE_PREFIX not found in yaml')
print(m.group(1).strip().lstrip('/'))
PY
)"

DEST="${CODE_S3:-s3://${S3_BUCKET}/${S3_CODE_PREFIX}}"

# 防误删保护：目标必须带至少一层前缀，禁止清 bucket 根目录
DEST_PREFIX="${DEST#s3://*/}"
if [[ "${DEST}" != s3://*/* || -z "${DEST_PREFIX}" || "${DEST_PREFIX}" == "${DEST}" ]]; then
  echo "[sync][FATAL] dest ${DEST} 缺少路径前缀，拒绝执行清理" >&2
  exit 1
fi

echo "[sync] yaml=${YAML_PATH}"
echo "[sync] pvc=${NAMESPACE}/${CLAIM_NAME} -> bucket=${S3_BUCKET}"
echo "[sync] repo=${REPO_ROOT}"
echo "[sync] dest=${DEST}"

# 清理前先探测：前缀不存在（ls 退出码非 0）或无文件则跳过；删除失败直接报错退出（不静默吞错）
OBJ_COUNT="$(aws s3 ls "${DEST}/" --recursive --region "${REGION}" 2>/dev/null | wc -l || true)"
if [ "${OBJ_COUNT}" -gt 0 ]; then
  echo "[sync] cleaning dest prefix ${DEST}/ (${OBJ_COUNT} objects) ..."
  aws s3 rm "${DEST}/" --recursive --region "${REGION}" --only-show-errors
else
  echo "[sync] dest prefix ${DEST}/ 不存在或为空，跳过清理"
fi

aws s3 sync "${REPO_ROOT}/" "${DEST}/" \
  --region "${REGION}" \
  --exclude ".codebuddy/*" --exclude ".codebuddy/**" \
  --exclude ".claude/*"   --exclude ".claude/**" \
  --exclude "models/*"    --exclude "models/**" \
  --exclude "data/*"      --exclude "data/**" \
  --exclude ".git/*"      --exclude ".git/**" \
  --exclude ".cache/*"    --exclude ".cache/**" \
  --exclude "output/*"    --exclude "output/**" \
  --exclude "__pycache__/*" --exclude "*/__pycache__/*" --exclude "*/*/__pycache__/*" \
  --exclude "*.pt" --exclude "*.ckpt" --exclude "*.safetensors" \
  --exclude "dataset_maker/*.jsonl" --exclude "dataset_maker/*.local.jsonl"

echo "[sync] done -> ${DEST}"
