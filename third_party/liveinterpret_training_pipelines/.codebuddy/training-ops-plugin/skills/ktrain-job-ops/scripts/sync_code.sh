#!/usr/bin/env bash
# =============================================================================
# training-ops / ktrain-job-ops / sync_code.sh
# =============================================================================
# 把本地代码库覆盖式同步到代码 bucket 的 CODE_PREFIX 前缀（先清空再全量上传），
# 训练 Pod 经 S3 CSI 只读挂载该 bucket。
#
# 解析规则（无任何环境硬编码）：
#   namespace   任务 yaml metadata.namespace
#   CODE_PREFIX 任务 yaml spec.env.CODE_PREFIX（缺失即 FATAL）
#   代码 PVC    --pvc 指定；否则 render-extra extra_volumes 中唯一的 PVC 卷自动采用；
#               多个 PVC 卷且未指定即 FATAL
#   bucket      --bucket 指定；否则 kubectl 解析 PVC -> PV -> csi.volumeAttributes.bucketName
#   目的地      --dest 指定；否则 s3://<bucket>/<CODE_PREFIX>
#   kubecontext 仅当 --cluster 与 --region 同时给出时执行 aws eks update-kubeconfig；
#               否则直接用当前 kubectl context
#
# 用法：
#   sync_code.sh --yaml <任务yaml> --repo <本地代码库根> \
#     [--pvc <claimName>] [--bucket <bucket>] [--dest s3://bucket/prefix] \
#     [--cluster <集群> --region <区域>] [--exclude <模式>]...
# =============================================================================
set -euo pipefail

YAML=""
REPO=""
PVC_NAME=""
S3_BUCKET=""
DEST=""
CLUSTER=""
REGION=""
EXTRA_EXCLUDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml)     YAML="$2"; shift 2 ;;
    --repo)     REPO="$2"; shift 2 ;;
    --pvc)      PVC_NAME="$2"; shift 2 ;;
    --bucket)   S3_BUCKET="$2"; shift 2 ;;
    --dest)     DEST="$2"; shift 2 ;;
    --cluster)  CLUSTER="$2"; shift 2 ;;
    --region)   REGION="$2"; shift 2 ;;
    --exclude)  EXTRA_EXCLUDES+=("$2"); shift 2 ;;
    -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[sync][FATAL] unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${YAML}" && -f "${YAML}" ]] || { echo "[sync][FATAL] --yaml not found: ${YAML}" >&2; exit 1; }
[[ -n "${REPO}" && -d "${REPO}" ]] || { echo "[sync][FATAL] --repo not found: ${REPO}" >&2; exit 1; }
if [[ -n "${CLUSTER}" || -n "${REGION}" ]]; then
  [[ -n "${CLUSTER}" && -n "${REGION}" ]] || { echo "[sync][FATAL] --cluster and --region must be given together" >&2; exit 1; }
fi

# ---- 从任务 yaml 解析 namespace + CODE_PREFIX + PVC 候选 ----
_cfg=()
while IFS= read -r _line; do
  _cfg+=("${_line}")
done < <(python3 - "${YAML}" <<'PY'
import json, sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
ns = (doc.get('metadata') or {}).get('namespace')
if not ns:
    sys.exit('[sync][FATAL] metadata.namespace missing in task yaml')
env = (doc.get('spec') or {}).get('env') or {}
code_prefix = (env.get('CODE_PREFIX') or '').strip().lstrip('/')
if not code_prefix:
    sys.exit('[sync][FATAL] spec.env.CODE_PREFIX missing or empty in task yaml')
ann = (doc.get('metadata') or {}).get('annotations') or {}
raw = ann.get('ktrain.oit-svc.com/render-extra', '')
re = json.loads(raw) if raw else {}
claims = [v['persistentVolumeClaim']['claimName']
          for v in re.get('extra_volumes', [])
          if (v.get('persistentVolumeClaim') or {}).get('claimName')]
print(ns)
print(code_prefix)
print(' '.join(claims))
PY
)
NAMESPACE="${_cfg[0]}"
CODE_PREFIX="${_cfg[1]}"
read -r -a CLAIMS <<< "${_cfg[2]:-}"

if [[ -z "${PVC_NAME}" ]]; then
  if [[ ${#CLAIMS[@]} -eq 1 ]]; then
    PVC_NAME="${CLAIMS[0]}"
  elif [[ ${#CLAIMS[@]} -eq 0 ]]; then
    echo "[sync][FATAL] no PVC-backed volume found in render-extra extra_volumes; pass --pvc" >&2
    exit 1
  else
    echo "[sync][FATAL] multiple PVC volumes in task yaml (${CLAIMS[*]}); pass --pvc to choose" >&2
    exit 1
  fi
fi

# ---- 可选：刷 kubecontext ----
if [[ -n "${CLUSTER}" ]]; then
  echo "[sync] kubecontext -> ${CLUSTER} (${REGION})"
  aws eks update-kubeconfig --region "${REGION}" --name "${CLUSTER}" >/dev/null
fi

# ---- 解析 bucket ----
if [[ -z "${DEST}" && -z "${S3_BUCKET}" ]]; then
  PV_NAME="$(kubectl -n "${NAMESPACE}" get pvc "${PVC_NAME}" -o jsonpath='{.spec.volumeName}')"
  S3_BUCKET="$(kubectl get pv "${PV_NAME}" -o jsonpath='{.spec.csi.volumeAttributes.bucketName}')"
  [[ -n "${S3_BUCKET}" ]] || { echo "[sync][FATAL] bucketName empty for PVC ${NAMESPACE}/${PVC_NAME}" >&2; exit 1; }
fi
DEST="${DEST:-s3://${S3_BUCKET}/${CODE_PREFIX}}"

# ---- 防误删保护：目的地必须带路径前缀 ----
DEST_PREFIX="${DEST#s3://*/}"
if [[ "${DEST}" != s3://*/* || -z "${DEST_PREFIX}" || "${DEST_PREFIX}" == "${DEST}" ]]; then
  echo "[sync][FATAL] dest ${DEST} has no path prefix; refusing to wipe bucket root" >&2
  exit 1
fi

echo "[sync] yaml=${YAML}"
echo "[sync] repo=${REPO}"
echo "[sync] pvc=${NAMESPACE}/${PVC_NAME} bucket=${S3_BUCKET:-<from --dest>}"
echo "[sync] dest=${DEST}"

# ---- 清空目标前缀 ----
# 空数组一律用 ${A[@]+"${A[@]}"} 展开：macOS 自带 bash 3.2 在 set -u 下直接展开空数组会报 unbound variable
REGION_ARGS=()
[[ -n "${REGION}" ]] && REGION_ARGS=(--region "${REGION}")
OBJ_COUNT="$(aws s3 ls "${DEST}/" --recursive ${REGION_ARGS[@]+"${REGION_ARGS[@]}"} 2>/dev/null | wc -l || true)"
if [[ "${SKIP_CODE_CLEAN:-}" == "1" ]]; then
  echo "[sync] SKIP_CODE_CLEAN=1; incremental upload only (${OBJ_COUNT} objects already at dest)"
elif [ "${OBJ_COUNT}" -gt 0 ]; then
  echo "[sync] cleaning dest prefix ${DEST}/ (${OBJ_COUNT} objects) ..."
  aws s3 rm "${DEST}/" --recursive ${REGION_ARGS[@]+"${REGION_ARGS[@]}"} --only-show-errors
else
  echo "[sync] dest prefix ${DEST}/ empty or absent; skip cleaning"
fi

# ---- 全量上传（内置通用排除 + --exclude 追加） ----
EXCLUDE_ARGS=(
  --exclude ".git/*"        --exclude ".git/**"
  --exclude ".codebuddy/*"  --exclude ".codebuddy/**"
  --exclude ".claude/*"     --exclude ".claude/**"
  --exclude ".cache/*"      --exclude ".cache/**"
  --exclude ".venv/*"       --exclude ".venv/**"
  --exclude "*/.venv/*"     --exclude "*/.venv/**"
  --exclude "venv/*"        --exclude "venv/**"
  --exclude "node_modules/*" --exclude "*/node_modules/*" --exclude "*/node_modules/**"
  --exclude "__pycache__/*" --exclude "*/__pycache__/*"
  --exclude "*.pyc"
  --exclude "*.pt" --exclude "*.ckpt" --exclude "*.safetensors"
)
if [[ ${#EXTRA_EXCLUDES[@]} -gt 0 ]]; then
if [[ ${#EXTRA_EXCLUDES[@]} -gt 0 ]]; then
  for pat in "${EXTRA_EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude "${pat}")
  done
fi
fi

aws s3 sync "${REPO%/}/" "${DEST}/" ${REGION_ARGS[@]+"${REGION_ARGS[@]}"} "${EXCLUDE_ARGS[@]}"

echo "[sync] done -> ${DEST}"
