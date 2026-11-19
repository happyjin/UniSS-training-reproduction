#!/usr/bin/env bash
# =============================================================================
# training-ops / anyscale-job-ops / inspect_p5_nodes.sh
# =============================================================================
# 检查 EKS 集群中 p5.48xlarge 节点的状态：Ready/Schedulable、可分配 GPU/EFA、
# SageMaker 故障标签、Pending ray pods 等。
# 所有集群信息通过 kubectl 远程获取，不读取任何本地 repo 文件。
#
# 用法：
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh              # 仅诊断
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --recover    # 诊断并自动恢复
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --whitelist node1,node2
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh --blacklist node3,node4
#   bash .codebuddy/training-ops-plugin/skills/anyscale-job-ops/scripts/inspect_p5_nodes.sh <COMPUTE_CONFIG_NAME>
#
# 输入参数（可选）：
#   --recover              — 发现 GPU 降级节点时自动重启 device-plugin 恢复
#   --whitelist <nodes>   — 逗号分隔的白名单节点列表（只检查这些节点）
#   --blacklist <nodes>   — 逗号分隔的黑名单节点列表（排除这些节点）
#   <COMPUTE_CONFIG_NAME>  — 如 compute_config_h100:abc123（用于对比期望 vs 实际节点数）
#
# 输出（stdout，含 KEY=VALUE 结构化字段）：
#   P5_NODES_TOTAL / P5_NODES_READY / P5_NODES_UNSCHEDULABLE / P5_GPU_HEALTHY /
#   P5_GPU_DEGRADED / COMPUTE_CONFIG / EXPECTED_NODES / PENDING_PODS /
#   RECOVERY_ATTEMPTED / RECOVERY_SUCCESS / RECOVERY_FAILED / OUT_BUNDLE
#
# 退出码：0 = 检查完成；2 = kubectl 无法连接集群。
# =============================================================================
set -euo pipefail

RECOVER_MODE=false
COMPUTE_CONFIG_NAME=""
WHITELIST_NODES=""
BLACKLIST_NODES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recover)
            RECOVER_MODE=true
            shift
            ;;
        --whitelist)
            WHITELIST_NODES="$2"
            shift 2
            ;;
        --blacklist)
            BLACKLIST_NODES="$2"
            shift 2
            ;;
        *)
            COMPUTE_CONFIG_NAME="$1"
            shift
            ;;
    esac
done

TS="$(date +%s)"
OUT="/tmp/anyscale-inspect-${TS}"
mkdir -p "${OUT}"

echo "[inspect] out=${OUT}"
if [[ -n "${COMPUTE_CONFIG_NAME}" ]]; then
    echo "[inspect] compute_config=${COMPUTE_CONFIG_NAME}"
fi
echo ""

# =============================================================================
# (0) 验证 kubectl 可连接集群
# =============================================================================
echo "=== kubectl connectivity check ==="
if ! kubectl version > "${OUT}/kubectl_version.txt" 2>&1; then
    echo "[inspect] FATAL: kubectl cannot connect to cluster" >&2
    echo "INSPECT_OK=no"
    echo "FAIL_REASON=kubectl-not-connected"
    exit 2
fi
echo "  kubectl connected ✅"
echo ""

# =============================================================================
# (1) 获取所有 p5.48xlarge 节点
# =============================================================================
echo "================================================================="
echo "  All p5.48xlarge nodes — instance-type, AZ, allocatable GPU/EFA"
echo "================================================================="

mapfile -t ALL_NODES < <(kubectl get nodes -L node.kubernetes.io/instance-type --no-headers 2>&1 \
    | awk '$NF ~ /p5\.48xlarge/ {print $1}')

NODES=()

if [[ -n "${WHITELIST_NODES}" ]] && [[ -z "${BLACKLIST_NODES}" ]]; then
    IFS=',' read -ra WL <<< "${WHITELIST_NODES}"
    for NODE in "${ALL_NODES[@]}"; do
        for W in "${WL[@]}"; do
            if [[ "${NODE}" == "${W}" ]]; then
                NODES+=("${NODE}")
                break
            fi
        done
    done
    echo "  [inspect] 白名单过滤：只检查 ${#NODES[@]} 个节点（白名单: ${WHITELIST_NODES}）"
elif [[ -z "${WHITELIST_NODES}" ]] && [[ -n "${BLACKLIST_NODES}" ]]; then
    IFS=',' read -ra BL <<< "${BLACKLIST_NODES}"
    for NODE in "${ALL_NODES[@]}"; do
        skip=false
        for B in "${BL[@]}"; do
            if [[ "${NODE}" == "${B}" ]]; then
                skip=true
                break
            fi
        done
        if [[ "${skip}" == "false" ]]; then
            NODES+=("${NODE}")
        fi
    done
    echo "  [inspect] 黑名单过滤：排除 ${#BL[@]} 个节点，检查 ${#NODES[@]} 个节点（黑名单: ${BLACKLIST_NODES}）"
elif [[ -n "${WHITELIST_NODES}" ]] && [[ -n "${BLACKLIST_NODES}" ]]; then
    IFS=',' read -ra WL <<< "${WHITELIST_NODES}"
    IFS=',' read -ra BL <<< "${BLACKLIST_NODES}"
    TEMP_NODES=()
    for NODE in "${ALL_NODES[@]}"; do
        for W in "${WL[@]}"; do
            if [[ "${NODE}" == "${W}" ]]; then
                TEMP_NODES+=("${NODE}")
                break
            fi
        done
    done
    for NODE in "${TEMP_NODES[@]}"; do
        skip=false
        for B in "${BL[@]}"; do
            if [[ "${NODE}" == "${B}" ]]; then
                skip=true
                break
            fi
        done
        if [[ "${skip}" == "false" ]]; then
            NODES+=("${NODE}")
        fi
    done
    echo "  [inspect] 白名单+黑名单过滤：白名单 ${#WL[@]} 个节点，排除黑名单 ${#BL[@]} 个节点，最终检查 ${#NODES[@]} 个节点"
else
    NODES=("${ALL_NODES[@]}")
    echo "  [inspect] 无过滤：检查所有 ${#NODES[@]} 个节点（未设置黑白名单）"
fi

if [[ ${#NODES[@]} -eq 0 ]]; then
    echo "  (no p5.48xlarge nodes found)"
    echo ""
    echo "================================================================="
    echo "  STRUCTURED OUTPUT"
    echo "================================================================="
    echo "INSPECT_OK=yes"
    echo "P5_NODES_TOTAL=0"
    echo "P5_NODES_READY=0"
    echo "P5_GPU_HEALTHY=0"
    echo "P5_GPU_DEGRADED=0"
    echo "PENDING_PODS=unknown"
    echo "OUT_BUNDLE=${OUT}"
    echo "================================================================="
    exit 0
fi

printf "  Found %d p5.48xlarge nodes:\n\n" "${#NODES[@]}"
printf "  %-40s %-12s %-15s %-5s %-5s %-20s %s\n" \
    "NAME" "STATUS" "AZ" "GPU" "EFA" "FAULT" "FLAGS"
echo "  ---------------------------------------- ------------ --------------- ----- ----- -------------------------"

NODES_READY=0
NODES_UNSCH=0
GPU_HEALTHY=0
GPU_DEGRADED=0

for NODE in "${NODES[@]}"; do
    READY=$(kubectl get node "${NODE}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "?")
    UNSCH=$(kubectl get node "${NODE}" -o jsonpath='{.spec.unschedulable}' 2>/dev/null || echo "")
    if [[ "${UNSCH}" == "true" ]]; then
        STATUS="${READY},UNSCH"
        NODES_UNSCH=$((NODES_UNSCH + 1))
    else
        STATUS="${READY}"
        if [[ "${READY}" == "True" ]]; then
            NODES_READY=$((NODES_READY + 1))
        fi
    fi
    AZ=$(kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null || echo "?")
    GPU=$(kubectl get node "${NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "?")
    EFA=$(kubectl get node "${NODE}" -o jsonpath='{.status.allocatable.vpc\.amazonaws\.com/efa}' 2>/dev/null || echo "?")
    FAULT=$(kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.sagemaker\.amazonaws\.com/fault-reasons}' 2>/dev/null || "")
    HEALTH=$(kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.sagemaker\.amazonaws\.com/node-health-status}' 2>/dev/null || echo "")

    if [[ "${GPU}" == "8" ]]; then
        GPU_HEALTHY=$((GPU_HEALTHY + 1))
    elif [[ "${GPU}" =~ ^[0-9]+$ ]] && [[ "${GPU}" -gt 0 ]]; then
        GPU_DEGRADED=$((GPU_DEGRADED + 1))
    fi

    flag=""
    if [[ "${GPU}" != "8" && "${GPU}" != "?" ]]; then
        flag="⚠️ GPU=${GPU}"
    fi
    if [[ "${HEALTH}" == "Unschedulable" ]]; then
        flag="${flag} UNSCHED"
    fi

    printf "  %-40s %-12s %-15s %-5s %-5s %-20s %s\n" \
        "${NODE}" "${STATUS}" "${AZ}" "${GPU}" "${EFA}" "${FAULT:--}" "${flag}"
done | tee "${OUT}/nodes.txt"

echo ""

# =============================================================================
# (2) 每个节点的 taints
# =============================================================================
echo "================================================================="
echo "  Per-node taints (NoSchedule = node is dead / unschedulable)"
echo "================================================================="
for NODE in "${NODES[@]}"; do
    TAINTS=$(kubectl get node "${NODE}" -o jsonpath='{.spec.taints}' 2>/dev/null || echo "[]")
    echo "  ${NODE}: ${TAINTS}" | tee -a "${OUT}/taints.txt"
done
echo ""

# =============================================================================
# (3) 所有 ray-* / anyscale-* pods（任意阶段）
# =============================================================================
echo "================================================================="
echo "  All ray-* / anyscale-* pods (any phase)"
echo "================================================================="
kubectl get pods -A 2>&1 | grep -iE 'ray|anyscale' | tee "${OUT}/ray_pods.txt" | head -30 || echo "  (no ray/anyscale pods)"

echo ""

# =============================================================================
# (4) Pending pods — 什么在阻塞它们
# =============================================================================
echo "================================================================="
echo "  Pending pods — what's blocking them (FailedScheduling events)"
echo "================================================================="
mapfile -t PENDING_LINES < <(kubectl get pods -A --no-headers 2>&1 | awk '$4 == "Pending"')

PENDING_COUNT=${#PENDING_LINES[@]}
if [[ ${PENDING_COUNT} -eq 0 ]]; then
    echo "  (no Pending pods — cluster scheduling is healthy)"
else
    echo "  Found ${PENDING_COUNT} Pending pod(s):"
    for line in "${PENDING_LINES[@]:0:5}"; do
        NS=$(echo "${line}" | awk '{print $1}')
        NAME=$(echo "${line}" | awk '{print $2}')
        echo "--- ${NS}/${NAME} ---" | tee -a "${OUT}/pending.txt"
        kubectl describe pod -n "${NS}" "${NAME}" 2>&1 | sed -n '/^Events:/,$p' | head -20 | tee -a "${OUT}/pending.txt"
        echo "" | tee -a "${OUT}/pending.txt"
    done
fi
echo ""

# =============================================================================
# (5) 如果提供了 compute config，从 Anyscale 获取期望节点数做对比
# =============================================================================
EXPECTED_NODES="unknown"
if [[ -n "${COMPUTE_CONFIG_NAME}" ]]; then
    echo "================================================================="
    echo "  Compute config: ${COMPUTE_CONFIG_NAME}"
    echo "================================================================="
    if anyscale compute-config get -n "${COMPUTE_CONFIG_NAME}" > "${OUT}/compute_config.txt" 2>&1; then
        echo "  compute config fetched ✅"
        EXPECTED_NODES=$(grep -iE 'min_workers|min_nodes|worker_count' "${OUT}/compute_config.txt" | head -1 | grep -oE '[0-9]+' | head -1 || echo "unknown")
        echo "  expected nodes (min): ${EXPECTED_NODES}"
        echo ""
        echo "  --- compute config content (relevant fields) ---"
        grep -iE 'worker|node|instance|gpu|min|max' "${OUT}/compute_config.txt" | head -20 || true
        echo ""
    else
        echo "  compute config fetch FAILED (name=${COMPUTE_CONFIG_NAME})"
        echo "  (check that the compute config name is correct)"
    fi
    echo ""
fi

# =============================================================================
# (6) 诊断检查清单
# =============================================================================
echo "================================================================="
echo "  Quick diagnostic checklist"
echo "================================================================="
echo "  - Any node with allocatable GPU != 8 → likely Xid 94 / device-plugin issue"
echo "    ↳ Check device-plugin logs:"
echo "      kubectl logs -n kube-system -l app=nvidia-device-plugin --tail=100"
echo "  - Any node with HEALTH=Unschedulable → SageMaker HMA tainted it"
echo "  - Any Pending pod with 'FailedScheduling: ... Insufficient nvidia.com/gpu'"
echo "    → cluster doesn't have N×8 free GPUs to satisfy compute_config"
if [[ "${EXPECTED_NODES}" =~ ^[0-9]+$ ]] && [[ "${GPU_HEALTHY}" =~ ^[0-9]+$ ]]; then
    if [[ "${EXPECTED_NODES}" -gt "${GPU_HEALTHY}" ]]; then
        echo "  ⚠️  EXPECTED_NODES (${EXPECTED_NODES}) > healthy GPU nodes (${GPU_HEALTHY})"
        echo "    → lower min/max in compute config, or add a node blacklist"
    fi
fi
echo "  - Pod phase check: kubectl get pods -A -l ray.io/cluster=<cluster-name>"
echo ""

# =============================================================================
# (7) 输出结构化结果
# =============================================================================
echo "================================================================="
echo "  STRUCTURED OUTPUT"
echo "================================================================="
echo "INSPECT_OK=yes"
echo "P5_NODES_TOTAL=${#NODES[@]}"
echo "P5_NODES_READY=${NODES_READY}"
echo "P5_NODES_UNSCHEDULABLE=${NODES_UNSCH}"
echo "P5_GPU_HEALTHY=${GPU_HEALTHY}"
echo "P5_GPU_DEGRADED=${GPU_DEGRADED}"
echo "COMPUTE_CONFIG=${COMPUTE_CONFIG_NAME:-<not-provided>}"
echo "EXPECTED_NODES=${EXPECTED_NODES}"
echo "PENDING_PODS=${PENDING_COUNT:-0}"
echo "OUT_BUNDLE=${OUT}"
echo "================================================================="

# =============================================================================
# (8) 自动恢复 GPU 降级节点（可选 --recover）
# =============================================================================
if [[ "${RECOVER_MODE}" == "true" ]] && [[ ${GPU_DEGRADED} -gt 0 ]]; then
    echo ""
    echo "================================================================="
    echo "  AUTO RECOVERY: Restarting device-plugin on degraded nodes"
    echo "================================================================="

    RECOVERY_ATTEMPTED=0
    RECOVERY_SUCCESS=0
    RECOVERY_FAILED=0

    for NODE in "${NODES[@]}"; do
        GPU=$(kubectl get node "${NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "?")
        CAP=$(kubectl get node "${NODE}" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "?")

        if [[ "${GPU}" != "${CAP}" ]] && [[ "${GPU}" =~ ^[0-9]+$ ]] && [[ "${CAP}" =~ ^[0-9]+$ ]]; then
            RECOVERY_ATTEMPTED=$((RECOVERY_ATTEMPTED + 1))
            echo ""
            echo "  🔄 Restarting device-plugin on ${NODE} (cap=${CAP}, alloc=${GPU})"

            POD=$(kubectl -n kube-system get pods -o wide 2>/dev/null | grep nvidia-device-plugin | grep "${NODE}" | awk '{print $1}')
            if [[ -n "${POD}" ]]; then
                echo "     Deleting pod: ${POD}"
                if kubectl -n kube-system delete pod "${POD}" 2>/dev/null; then
                    echo "     Pod deleted successfully"
                else
                    echo "     WARNING: Failed to delete pod"
                    RECOVERY_FAILED=$((RECOVERY_FAILED + 1))
                    continue
                fi
            else
                echo "     WARNING: No device-plugin pod found for ${NODE}"
                RECOVERY_FAILED=$((RECOVERY_FAILED + 1))
                continue
            fi
        fi
    done

    if [[ ${RECOVERY_ATTEMPTED} -gt 0 ]]; then
        echo ""
        echo "  Waiting 15 seconds for device-plugin pods to restart..."
        sleep 15

        echo ""
        echo "  Verifying recovery results..."
        echo ""
        printf "  %-40s %-5s %-5s %s\n" "NODE" "CAP" "ALLOC" "STATUS"
        echo "  ---------------------------------------- ----- ----- ------"

        for NODE in "${NODES[@]}"; do
            CAP=$(kubectl get node "${NODE}" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "?")
            GPU=$(kubectl get node "${NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "?")

            if [[ "${GPU}" == "${CAP}" ]] && [[ "${GPU}" =~ ^[0-9]+$ ]]; then
                printf "  %-40s %-5s %-5s ✅ OK\n" "${NODE}" "${CAP}" "${GPU}"
                RECOVERY_SUCCESS=$((RECOVERY_SUCCESS + 1))
            elif [[ "${GPU}" != "${CAP}" ]] && [[ "${GPU}" =~ ^[0-9]+$ ]] && [[ "${CAP}" =~ ^[0-9]+$ ]]; then
                printf "  %-40s %-5s %-5s ❌ Still degraded\n" "${NODE}" "${CAP}" "${GPU}"
                RECOVERY_FAILED=$((RECOVERY_FAILED + 1))
            else
                printf "  %-40s %-5s %-5s ⚠️ Unknown\n" "${NODE}" "${CAP}" "${GPU}"
            fi
        done

        echo ""
        echo "================================================================="
        echo "  RECOVERY SUMMARY"
        echo "================================================================="
        echo "RECOVERY_ATTEMPTED=${RECOVERY_ATTEMPTED}"
        echo "RECOVERY_SUCCESS=${RECOVERY_SUCCESS}"
        echo "RECOVERY_FAILED=${RECOVERY_FAILED}"

        if [[ ${RECOVERY_FAILED} -gt 0 ]]; then
            echo ""
            echo "  ⚠️  Some nodes failed to recover. Check:"
            echo "    - Whether the same GPU is reporting repeated XID 94 errors (hardware fault)"
            echo "    - Contact ops if the issue persists"
        fi
    else
        echo ""
        echo "  No degraded nodes found to recover."
    fi

    echo "================================================================="
fi

exit 0
