#!/usr/bin/env bash
# =============================================================================
# training-ops / pytorchjob-ops / inspect_nodes.sh
#
# 通用节点健康 / 资源 / 重复 job 排查脚本（纯 kubectl，不依赖特定集群/项目）。
# 命名空间为参数；节点白名单来自项目 pytorchjob.yaml 的 affinity（不在此硬编码）。
# =============================================================================
# 用法：
#   bash inspect_nodes.sh [NAMESPACE]
#     NAMESPACE  可选，k8s 命名空间；也可用环境变量 K8S_NS 传入
#
# 输出（stdout，KEY=VALUE）：
#   NODES_TOTAL / NODES_READY / GPU_HEALTHY / PENDING_PODS /
#   DUP_JOBS / OUT_BUNDLE
# =============================================================================
set -euo pipefail

NS="${1:-${K8S_NS:-}}"
if [[ -z "${NS}" ]]; then
  echo "usage: inspect_nodes.sh [NAMESPACE]" >&2
  echo "  (NAMESPACE 也可通过环境变量 K8S_NS 传入)" >&2
  exit 2
fi

OUT="/tmp/pytorchjob-inspect-$(date +%s)"
mkdir -p "${OUT}"

echo "[inspect] ns=${NS} out=${OUT}"
echo ""

echo "=== kubectl connectivity ==="
if ! kubectl version > "${OUT}/kubectl_version.txt" 2>&1; then
    echo "INSPECT_OK=no"
    echo "FAIL_REASON=kubectl-not-connected"
    exit 2
fi
echo "  connected"
echo ""

echo "=== all GPU (training) nodes ==="
# 通用做法：按 GPU 节点标签列出；不同集群的 instance-type 标签值不同，
# 这里用 GPU 可分配数 > 0 来识别训练节点，避免硬编码具体机型。
kubectl get nodes -o custom-columns='NODE:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,GPU:.status.allocatable.nvidia\.com/gpu' --no-headers 2>&1 > "${OUT}/gpu_nodes.txt" || true
TOTAL=$(awk '$3 != "" && $3 != "0" {c++} END{print c+0}' "${OUT}/gpu_nodes.txt")
echo "${OUT}/gpu_nodes.txt:"; cat "${OUT}/gpu_nodes.txt"
echo ""

READY=0
GPU_HEALTHY=0
mapfile -t NODES < <(awk '$3 != "" && $3 != "0" {print $1}' "${OUT}/gpu_nodes.txt" 2>/dev/null || true)
for n in "${NODES[@]}"; do
    [[ -z "${n}" ]] && continue
    ready=$(kubectl get node "${n}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "?")
    gpu=$(kubectl get node "${n}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "?")
    echo "  ${n}: Ready=${ready} GPU=${gpu}"
    [[ "${ready}" == "True" ]] && READY=$((READY+1))
    [[ "${gpu}" != "0" && "${gpu}" != "?" ]] && GPU_HEALTHY=$((GPU_HEALTHY+1))
done
echo ""

echo "=== Pending pods ==="
kubectl get pods -A --no-headers 2>/dev/null | awk '$4=="Pending"' > "${OUT}/pending.txt" || true
PENDING=$(wc -l < "${OUT}/pending.txt")
echo "  Pending pods: ${PENDING}"
cat "${OUT}/pending.txt"
echo ""

echo "=== duplicate PyTorchJob detection (same base name, competing for GPU) ==="
kubectl -n "${NS}" get pytorchjob > "${OUT}/pytorchjobs.txt" 2>&1 || true
# 通用检测：若存在多个同名前缀（去掉末尾 -xxx 后缀后相同）的 PyTorchJob，视为重复。
DUP=$(awk 'NR>1{ n=$1; sub(/-[A-Za-z0-9]+$/,"",n); c[n]++; names[n]=names[n]" "$1 } END{ for(k in c){ if(c[k]>1) print k" =>"names[k] } }' "${OUT}/pytorchjobs.txt" 2>/dev/null | tr '\n' ';' | sed 's/;$//' || true)
echo "  pytorchjobs:"; cat "${OUT}/pytorchjobs.txt"
echo ""

{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "NODES_TOTAL=${TOTAL}"
    echo "NODES_READY=${READY}"
    echo "GPU_HEALTHY=${GPU_HEALTHY}"
    echo "PENDING_PODS=${PENDING}"
    echo "DUP_JOBS=${DUP:-none}"
    echo "OUT_BUNDLE=${OUT}"
    echo "================================================================="
} | tee "${OUT}/summary.txt"

exit 0
