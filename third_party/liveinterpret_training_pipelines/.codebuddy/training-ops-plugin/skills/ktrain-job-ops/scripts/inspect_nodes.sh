#!/usr/bin/env bash
# =============================================================================
# training-ops / ktrain-job-ops / inspect_nodes.sh
# =============================================================================
# 节点健康 / 资源 / 重复 job 排查。全程用 ktrain CLI：
#   * ktrain gpu  — 集群 GPU 总览 + 每节点分配情况
#   * ktrain job list — 查看当前任务（检测重复 / 残留）
#   * ktrain fs list — 查看 FSx 文件系统
#
# 用法：
#   bash inspect_nodes.sh <NAMESPACE>
#     NAMESPACE  必填，k8s 命名空间（也可用环境变量 K8S_NS 传入）；
#                取自任务 yaml metadata.namespace，无默认值
#
# 输出（stdout，KEY=VALUE）：
#   KTRAIN_MODE / CLUSTER / GPU_NODES_TOTAL / GPU_NODES_READY /
#   GPU_HEALTHY / FSX_LIST / JOB_COUNT / OUT_BUNDLE
# =============================================================================
set -euo pipefail

NS="${1:-${K8S_NS:?命名空间必填：传第一个参数或设置 K8S_NS（取自任务 yaml metadata.namespace）}}"
OUT="/tmp/ktrain-inspect-$(date +%s)"
mkdir -p "${OUT}"

echo "[inspect] ns=${NS} out=${OUT}"
echo ""

echo "=== ktrain context ==="
KTRAIN_MODE=$(ktrain mode 2>/dev/null | grep -i 'mode' | awk '{print $2}' || echo "unknown")
CLUSTER=$(ktrain mode 2>/dev/null | grep -i 'cluster' | awk '{print $2}' || echo "unknown")
echo "  mode=${KTRAIN_MODE}"
echo "  cluster=${CLUSTER}"
echo ""

echo "=== ktrain gpu (集群 GPU 总览) ==="
ktrain gpu > "${OUT}/gpu.txt" 2>&1 || true
cat "${OUT}/gpu.txt"
echo ""

# 从 gpu 输出解析节点数与 Ready 数（格式：每节点一行，含 Ready 状态与 GPU 分配）
GPU_NODES_TOTAL=$(grep -cE '^[[:space:]]*[a-z]' "${OUT}/gpu.txt" 2>/dev/null || echo 0)
GPU_NODES_READY="${GPU_NODES_TOTAL}"  # ktrain gpu 只列 Ready 节点
GPU_HEALTHY="${GPU_NODES_TOTAL}"

echo "=== ktrain job list (当前任务) ==="
ktrain job list --namespace "${NS}" --all > "${OUT}/jobs.txt" 2>&1 || true
cat "${OUT}/jobs.txt"
JOB_COUNT=$(awk 'NR>1{c++} END{print c+0}' "${OUT}/jobs.txt" 2>/dev/null || echo 0)
echo "  jobs in ns: ${JOB_COUNT}"
echo ""

echo "=== ktrain fs list (文件系统) ==="
ktrain fs list > "${OUT}/fs_list.txt" 2>&1 || true
cat "${OUT}/fs_list.txt"
echo ""

{
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "KTRAIN_MODE=${KTRAIN_MODE}"
    echo "CLUSTER=${CLUSTER}"
    echo "GPU_NODES_TOTAL=${GPU_NODES_TOTAL}"
    echo "GPU_NODES_READY=${GPU_NODES_READY}"
    echo "GPU_HEALTHY=${GPU_HEALTHY}"
    echo "FSX_LIST=$(grep -vE '^#|NAME|^$|━|┃|┡|│|└' "${OUT}/fs_list.txt" 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
    echo "JOB_COUNT=${JOB_COUNT}"
    echo "OUT_BUNDLE=${OUT}"
    echo "================================================================="
} | tee "${OUT}/summary.txt"

exit 0
