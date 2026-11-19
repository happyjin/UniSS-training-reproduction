#!/bin/bash
# 实验 A 的无人值守编排：a2 预热→门 A→joint，以及 a0j/a1j 跑完后的副语言评测。
#
#   bash custom/streaming_si/eval/orchestrate_expa.sh > /tmp/expa_orch.log 2>&1
# 建议用 caffeinate 包住防 Mac 休眠（否则本脚本会随睡眠停住）：
#   caffeinate -i -s setsid nohup bash custom/streaming_si/eval/orchestrate_expa.sh \
#       > /tmp/expa_orch.log 2>&1 < /dev/null &
#
# **为什么跑在本机而不是集群**：训练镜像里没有 kubectl，且 speech-h200 的 default
# ServiceAccount 对 pytorchjobs / pods 都是 `can-i create = no`，Pod 无法自行提交后继任务。
#
# 两条链并行：
#   链 1（a2）  等 A2_JOB 成功 → 建 1 卡 helper → 门 A → 过则提交 a2_joint
#               （per_device 64 / GA 1 / 关 gradient_checkpointing；OOM 则自动降档重提）
#   链 2（评测）等 a0j+a1j 成功 → 各自 master 节点上建 3 卡 pod → 三个评测集推理
#               → SIM-o / AutoPCP / ESim 打分 → 出对比报告
#
# 产物：
#   /tmp/expa_orch.log          总日志
#   /tmp/expa_orch/*.log        各步细日志
#   <节点本地>/nvme/expa_eval/  推理 wav 与 results.jsonl（不写 FSX：FSX 已 96% 满）
#   /tmp/expa_orch/REPORT.md    最终对比报告
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1
NS=speech-h200
W=/tmp/expa_orch; mkdir -p $W
SUBMIT=.codebuddy/training-ops-plugin/skills/ktrain-job-ops/scripts/submit.sh

# 任务名（如有变化用环境变量覆盖）
A2_JOB=${A2_JOB:-expa-a2-545w}
A0J_JOB=${A0J_JOB:-expa-a0j-rdb2}
A1J_JOB=${A1J_JOB:-expa-a1j-a1dw}
# 各臂 master 所在节点（checkpoint 写在 master 的本地 NVMe，非共享存储）
A2_NODE=${A2_NODE:-hyperpod-i-07d384c1277ead0cd}
A0J_NODE=${A0J_NODE:-hyperpod-i-07e7a9b74bdcbb8b0}
A1J_NODE=${A1J_NODE:-hyperpod-i-09a68c454800140d6}

E=/fsx/stevehguo/eval
P2=/nvme/speech_model_backup/stevehguo/data_fortrain/open_data_0803_fortrain
SETS="realsi_en2zh realsi_zh2en acl6060_en2zh"
IMG=124355679795.dkr.ecr.ap-southeast-3.amazonaws.com/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9-streaming-si-v1

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 等某个 pytorchjob 终态；回显 Succeeded / Failed / timeout ----
wait_job() {           # $1=job  $2=超时秒
  local j=$1 t=${2:-54000} i=0
  while [ $i -lt "$t" ]; do
    local st; st=$(kubectl -n $NS get pytorchjob "$j" -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null)
    case "$st" in
      Succeeded) echo Succeeded; return 0 ;;
      Failed)    echo Failed;    return 1 ;;
    esac
    sleep 60; i=$((i+60))
  done
  echo timeout; return 1
}

# ---- 在指定节点建一个带 FSX + NVMe 的工具 pod ----
mk_pod() {             # $1=名字 $2=节点 $3=GPU 数
  kubectl -n $NS delete pod "$1" --wait=false >/dev/null 2>&1
  sleep 5
  cat <<YAML | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Pod
metadata: {name: $1, namespace: $NS}
spec:
  restartPolicy: Never
  nodeSelector: {kubernetes.io/hostname: $2}
  containers:
    - name: main
      image: $IMG
      command: ["sleep", "86400"]
      resources:
        requests: {cpu: "32", memory: 240Gi, nvidia.com/gpu: "$3"}
        limits:   {cpu: "32", memory: 240Gi, nvidia.com/gpu: "$3"}
      envFrom: [{secretRef: {name: aws-local-creds}}]
      volumeMounts:
        - {name: nvme, mountPath: /nvme}
        - {name: fsx,  mountPath: /fsx}
        - {name: shm,  mountPath: /dev/shm}
  volumes:
    - {name: nvme, hostPath: {path: /opt/dlami/nvme, type: Directory}}
    - {name: fsx,  persistentVolumeClaim: {claimName: speech-h200-fsx-pvc}}
    - {name: shm,  emptyDir: {medium: Memory, sizeLimit: 128Gi}}
YAML
  local i=0
  while [ $i -lt 600 ]; do
    [ "$(kubectl -n $NS get pod "$1" -o jsonpath='{.status.phase}' 2>/dev/null)" = Running ] && { kubectl -n $NS cp custom "$NS/$1:/workspace/" >/dev/null 2>&1; return 0; }
    sleep 10; i=$((i+10))
  done
  return 1
}

# ---- 找某节点上最新的 checkpoint 目录 ----
find_ckpt() {          # $1=pod  $2=s4w_s2st|s4_s2st  $3=期望步数
  kubectl -n $NS exec "$1" -- sh -c \
    "ls -d /nvme/output/stevehguo/streaming_si/$2/*/checkpoint-$3 2>/dev/null | tail -1" < /dev/null 2>/dev/null | tr -d '\r'
}

# =====================================================================
# 链 1：a2 预热 → 门 A → joint
# =====================================================================
chain_a2() {
  log "链1 等 $A2_JOB 预热完成"
  local st; st=$(wait_job "$A2_JOB" 14400)
  log "链1 $A2_JOB -> $st"
  [ "$st" = Succeeded ] || { log "链1 预热未成功，终止"; return 1; }

  mk_pod si-gate-a2 "$A2_NODE" 1 || { log "链1 helper pod 起不来"; return 1; }
  local ck; ck=$(find_ckpt si-gate-a2 s4w_s2st 2500)
  [ -n "$ck" ] || { log "链1 找不到 checkpoint-2500"; return 1; }
  log "链1 checkpoint = $ck，开始门 A"

  # 门 A：每方向 full 的 speech CE 必须 < text_only
  kubectl -n $NS exec si-gate-a2 -- bash -c "
    env PYTHONPATH=/workspace/custom PYTHONWARNINGS=ignore HF_HOME=/nvme/hf CUDA_VISIBLE_DEVICES=0 \
      /opt/venv/bin/python /workspace/custom/streaming_si/scripts/s4w_bridge_gate.py \
      --model $ck --data '$P2/machine_0_shard/p2_fsq_v3/*.parquet' \
      --wav_dir $P2/machine_0_shard/p2_source_wav" < /dev/null > $W/gate_a2.log 2>&1
  grep -E 'gate A\]' $W/gate_a2.log | tail -6

  local npass; npass=$(grep -cE 'gate A\] (cmn->eng|eng->cmn): PASS' $W/gate_a2.log)
  log "链1 门 A 通过方向数 = $npass / 2"

  # 顺手把预热产物同步到 S3（本地 NVMe 非共享，节点一旦被回收就没了）
  kubectl -n $NS exec si-gate-a2 -- bash -c \
    "aws s3 sync $ck/ s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/model_checkpoint/streaming_si_swift/trained_streaming_si_4b/expA/a2_w2500/ --only-show-errors" \
    < /dev/null > $W/sync_a2.log 2>&1 &

  if [ "${npass:-0}" -lt 2 ]; then
    log "链1 门 A 未全过 —— 按预注册规则**不自动停**（预热阶段 thinker 冻结，"
    log "     换层优势用不上，门 A 略差不代表该臂失败），继续提交 joint；判据留给 joint 后的指标"
  fi

  # a2_joint：MODEL_PATH 直接用本地路径（与 joint 同节点，省掉 S3 同步等待）
  log "链1 提交 a2_joint（per_device 64 / GA 1 / 关 gradient_checkpointing）"
  submit_a2_joint "$ck" 64 "--gradient_checkpointing false" a2j1 || return 1

  # OOM 看护：头 25 分钟内若因显存失败，自动降档重提
  sleep 1500
  local ph; ph=$(kubectl -n $NS get pods --no-headers 2>/dev/null | grep "$A2J_NAME-master-0" | awk '{print $3}')
  if kubectl -n $NS logs "$A2J_NAME-master-0" --tail=200 2>/dev/null | grep -qiE 'out of memory|OutOfMemoryError'; then
    log "链1 a2_joint OOM —— 降档重提：per_device 32 + GA 2（全局仍 512）+ 仍关检查点"
    kubectl -n $NS delete pytorchjob "$A2J_NAME" --wait=false >/dev/null 2>&1; sleep 30
    submit_a2_joint "$ck" 32 "--gradient_checkpointing false" a2j2 2
    sleep 1500
    if kubectl -n $NS logs "$A2J_NAME-master-0" --tail=200 2>/dev/null | grep -qiE 'out of memory|OutOfMemoryError'; then
      log "链1 仍 OOM —— 退回与 a0/a1 完全相同的配置（per_device 32 + GA 2 + 开检查点）"
      kubectl -n $NS delete pytorchjob "$A2J_NAME" --wait=false >/dev/null 2>&1; sleep 30
      submit_a2_joint "$ck" 32 "" a2j3 2
    fi
  else
    log "链1 a2_joint 未见 OOM（phase=$ph），保持当前配置"
  fi
  log "链1 完成；a2_joint = $A2J_NAME"
}

A2J_NAME=""
submit_a2_joint() {    # $1=起点 $2=per_device $3=额外 swift 参数 $4=前缀 $5=GA(可选)
  local ck=$1 pd=$2 extra=$3 pfx=$4 ga=${5:-1}
  local y=$W/a2_joint.$pfx.yaml
  /usr/bin/python3 - "$ck" "$pd" "$extra" "$ga" "$y" <<'PY'
import re, sys
ck, pd, extra, ga, out = sys.argv[1:6]
s = open('ktrain/ktrainjob_streaming_si_expA_a2_joint.yaml', encoding='utf-8').read()
s = re.sub(r'^    MODEL_PATH: ".*"$', f'    MODEL_PATH: "{ck}"', s, count=1, flags=re.M)
s = re.sub(r'^    PER_DEVICE_TRAIN_BATCH_SIZE: ".*"$',
           f'    PER_DEVICE_TRAIN_BATCH_SIZE: "{pd}"', s, count=1, flags=re.M)
if extra.strip():
    if re.search(r'^    SI_EXTRA_ARGS: ".*"$', s, flags=re.M):
        s = re.sub(r'^    SI_EXTRA_ARGS: ".*"$', f'    SI_EXTRA_ARGS: "{extra}"', s, count=1, flags=re.M)
    else:
        s = re.sub(r'^(    SI_STAGE: ".*"\n)', rf'\1    SI_EXTRA_ARGS: "{extra}"\n', s, count=1, flags=re.M)
else:
    # 整行连同行尾注释一起删掉（yaml 里该行形如 `SI_EXTRA_ARGS: "..."   # 注释`，
    # 不带 .* 吃掉注释会留下半行导致 yaml 解析失败）
    s = re.sub(r'^    SI_EXTRA_ARGS: ".*".*\n', '', s, count=1, flags=re.M)
if ga != '1':
    if re.search(r'^    GRADIENT_ACCUMULATION_STEPS: ".*"$', s, flags=re.M):
        s = re.sub(r'^    GRADIENT_ACCUMULATION_STEPS: ".*"$',
                   f'    GRADIENT_ACCUMULATION_STEPS: "{ga}"', s, count=1, flags=re.M)
    else:
        s = re.sub(r'^(    SI_STAGE: ".*"\n)',
                   rf'\1    GRADIENT_ACCUMULATION_STEPS: "{ga}"\n', s, count=1, flags=re.M)
open(out, 'w', encoding='utf-8').write(s)
print(f'per_device={pd} ga={ga} extra={extra!r} global={int(pd)*8*int(ga)}')
PY
  bash $SUBMIT --yaml "$y" --repo . --launcher ktrain/launch_train.sh \
    --aws-creds-secret aws-local-creds --job-name-prefix "expa-$pfx" \
    --task-deadline 'MODEL_PATH=5400' > $W/submit.$pfx.log 2>&1
  A2J_NAME=$(grep -oE 'expa-'"$pfx"'-[a-z0-9]{4}' $W/submit.$pfx.log | head -1)
  [ -n "$A2J_NAME" ] && { log "  已提交 $A2J_NAME"; return 0; }
  log "  提交失败，见 $W/submit.$pfx.log"; return 1
}

# =====================================================================
# 链 2：a0j / a1j joint 跑完 → 三评测集推理 → 副语言打分 → 报告
# =====================================================================
eval_arm() {           # $1=臂名(a0j|a1j) $2=节点 $3=任务名
  local arm=$1 node=$2 job=$3
  local pod=si-eval-$arm
  mk_pod "$pod" "$node" 3 || { log "链2 $arm eval pod 起不来"; return 1; }
  local ck; ck=$(find_ckpt "$pod" s4_s2st 37470)
  [ -n "$ck" ] || { ck=$(kubectl -n $NS exec "$pod" -- sh -c \
      'ls -dt /nvme/output/stevehguo/streaming_si/s4_s2st/*/checkpoint-* 2>/dev/null | head -1' < /dev/null 2>/dev/null | tr -d '\r'); }
  [ -n "$ck" ] || { log "链2 $arm 找不到 joint checkpoint"; return 1; }
  log "链2 $arm checkpoint = $ck"

  # 推理需要 speech_decoder/（flow + HiFT + CAM++）；swift 一般会一起存，缺了就从 s4_ck 补
  kubectl -n $NS exec "$pod" -- bash -c "
    [ -d $ck/speech_decoder ] || { echo '  speech_decoder 缺失，从 s4_ck 复制'; cp -r $E/models/s4_ck/speech_decoder $ck/; }
    ls $ck/speech_decoder | head -3" < /dev/null 2>&1 | tail -4

  # 三个评测集各占一张卡并行推理。COSYVOICE_SPEECH_TOKENIZER 必须给——
  # 缺了它源语音 prompt 会整轮静默关闭（曾因此跑废 10 个消融臂）。
  local i=0
  for d in $SETS; do
    kubectl -n $NS exec "$pod" -- bash -c "
      export COSYVOICE_REPO=$E/code/CosyVoice
      export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
      export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
      mkdir -p /nvme/expa_eval/$arm
      (setsid env CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$E/code/custom /opt/venv/bin/python \
        -m streaming_si.inference.infer_s2st --model_dir $ck \
        --input_jsonl $E/work/manifests/$d.jsonl --out_dir /nvme/expa_eval/$arm/$d \
        --cosyvoice_repo $E/code/CosyVoice --device cuda:0 \
        > /nvme/expa_eval/$arm/infer.$d.log 2>&1 < /dev/null &)" < /dev/null >/dev/null 2>&1
    i=$((i+1)); sleep 3
  done
  log "链2 $arm 三集推理已启动"

  # 等推理完（1193 条合计，约 20 句/分钟/卡）
  local k=0
  while [ $k -lt 7200 ]; do
    local n; n=$(kubectl -n $NS exec "$pod" -- sh -c \
      'pgrep -cf "[i]nfer_s2st"' < /dev/null 2>/dev/null | tr -d '\r')
    [ "${n:-0}" = 0 ] && break
    sleep 60; k=$((k+60))
  done
  kubectl -n $NS exec "$pod" -- sh -c \
    'for d in /nvme/expa_eval/*/*/results.jsonl; do echo "  $(grep -c . $d) $d"; done' < /dev/null 2>&1 | tail -4

  # 副语言打分（SIM-o / ESim / AutoPCP，均带随机配对基线）
  i=0
  for d in $SETS; do
    kubectl -n $NS exec "$pod" -- bash -c "
      export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
      mkdir -p /nvme/expa_eval/$arm/_para
      (setsid env CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$E/code/site_para:$E/code/custom \
        /opt/venv/bin/python $E/code/custom/streaming_si/eval/run_para_metrics.py \
        --results /nvme/expa_eval/$arm/$d/results.jsonl --manifest $E/work/manifests/all.jsonl \
        --out /nvme/expa_eval/$arm/_para/$d.jsonl \
        --wavlm $E/models/wavlm-base-plus-sv \
        --emotion2vec /fsx/stevehguo/tts_bench/models/emotion2vec_plus_large \
        --autopcp $E/models/autopcp/model.config --w2v $E/models/wav2vec2-large-xlsr-53 \
        --device cuda:0 --batch 8 > /nvme/expa_eval/$arm/para.$d.log 2>&1 < /dev/null &)" < /dev/null >/dev/null 2>&1
    i=$((i+1)); sleep 3
  done
  log "链2 $arm 打分已启动"
  k=0
  while [ $k -lt 7200 ]; do
    local n; n=$(kubectl -n $NS exec "$pod" -- sh -c \
      'pgrep -cf "[r]un_para_metrics"' < /dev/null 2>/dev/null | tr -d '\r')
    [ "${n:-0}" = 0 ] && break
    sleep 60; k=$((k+60))
  done
  # 汇总 jsonl 拉回本机，便于离线出表
  mkdir -p $W/para/$arm
  for d in $SETS; do
    kubectl -n $NS cp "$NS/$pod:/nvme/expa_eval/$arm/_para/$d.jsonl" "$W/para/$arm/$d.jsonl" >/dev/null 2>&1
  done
  log "链2 $arm 打分完成"
}

chain_eval() {
  log "链2 等 $A0J_JOB 与 $A1J_JOB 完成"
  local s0 s1
  s0=$(wait_job "$A0J_JOB" 54000); log "链2 $A0J_JOB -> $s0"
  s1=$(wait_job "$A1J_JOB" 54000); log "链2 $A1J_JOB -> $s1"
  [ "$s0" = Succeeded ] && eval_arm a0j "$A0J_NODE" "$A0J_JOB"
  [ "$s1" = Succeeded ] && eval_arm a1j "$A1J_NODE" "$A1J_JOB"
  /usr/bin/python3 custom/streaming_si/eval/report_expa_para.py "$W/para" > $W/REPORT.md 2>&1
  log "链2 报告 -> $W/REPORT.md"
  cat $W/REPORT.md
}

log "编排开始：A2_JOB=$A2_JOB A0J=$A0J_JOB A1J=$A1J_JOB"
chain_a2  > $W/chain_a2.log 2>&1 &
P1=$!
chain_eval > $W/chain_eval.log 2>&1 &
P2=$!
wait $P1 $P2
log "全部完成。细日志在 $W/，报告 $W/REPORT.md"
