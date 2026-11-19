#!/usr/bin/env bash
# env 差异实证：PATH 劫持 swift 二进制，抓两侧训练链路在 swift sft 前的全量 env。
# 用法：bash env_diff_dump.sh <zy|merge>   （pod 内需 /s3all 两桶 + /nvme 工件）
set -u
SIDE="${1:?usage: env_diff_dump.sh <zy|merge>}"

mkdir -p /tmp/hijack
cat > /tmp/hijack/swift <<'EOF'
#!/usr/bin/env bash
env | sort > "/tmp/env_${HIJACK_SIDE}.txt"
echo "[hijack] swift intercepted, env dumped to /tmp/env_${HIJACK_SIDE}.txt" >&2
exit 0
EOF
chmod +x /tmp/hijack/swift
export PATH="/tmp/hijack:${PATH}"
export HIJACK_SIDE="${SIDE}"

if [[ "${SIDE}" == "zy" ]]; then
  REPO=/s3all/deferzhou/liveinterpret_zeyang_dev_parity
  YAML="${REPO}/ktrain/ktrainjob_sts_zeyang_custom_data_train.yaml"
  SCRIPT="pipelines/bridge_alignment/run_batch4_8_adapter_talker_unist.sh"
else
  REPO=/s3all/deferzhou/liveinterpret_merge_dprime_zydata
  YAML="${REPO}/ktrain/ktrainjob_sts_dprime_zydata_train.yaml"
  SCRIPT="pipelines/bridge_alignment/train_speech_to_speech_translation.sh"
fi

# 从任务 yaml 注入 spec.env（原样，含 fsx:// / s3:// URI——env dump 不需要真实路径）
while IFS='=' read -r k v; do
  [[ -n "${k}" ]] && export "${k}=${v}"
done < <(python3 -c "
import yaml
d = yaml.safe_load(open('${YAML}'))
for k, v in d['spec']['env'].items():
    print(f\"{k}={v}\")
")

# launch_train.sh 的 URI→路径转换中影响 env 的部分（fsx:// → FSX_MOUNT 前缀）
export BRIDGE_ALIGNMENT_DATA="${BRIDGE_ALIGNMENT_DATA/fsx:\/\//\/mnt\/fsx\/}"

# NPROC/MASTER 等 torchrun 环境（脚本_banner 用，单节点模拟）
export NPROC_PER_NODE=8 MASTER_ADDR=localhost MASTER_PORT=23456 NODE_RANK=0 NNODES=1
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=8 LOCAL_WORLD_SIZE=8

cd "${REPO}"
echo "=== dumping env for side=${SIDE} via ${SCRIPT} ==="
bash "${SCRIPT}" 2>&1 | tail -5 || true
echo "=== env file ==="
wc -l "/tmp/env_${SIDE}.txt" 2>/dev/null || echo 'NO ENV FILE'
