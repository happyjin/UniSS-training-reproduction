#!/bin/bash
# 实验 A 的**译文质量**评测链：ASR-BLEU / ASR-chrF / ASR-COMET / Text-BLEU / Text-COMET / UTMOS。
# 与 `orchestrate_expa.sh` 的副语言链解耦，等它产出 `<root>/<arm>/<ds>.jsonl` 后接着跑，
# 最后重出完整报告（质量 + 音色 + 副语言）。这样不必改正在运行的编排脚本
# （bash 边读边执行，运行中改它可能让后半段读错位）。
#
#   bash custom/streaming_si/eval/orchestrate_expa_quality.sh
#
# 口径与第二轮正式评测**逐项一致**（`run_post2.sh` 的 score 步骤）：同一个
# `score_eval.py`、同一个 COMET 权重 `Unbabel/wmt22-comet-da`、同一个 ASR 模型
# `whisper-large-v3`、同样的 BLEU 归一化（中文 tokenize=zh / 英文 13a）。
#
# 参比行：把第二轮的 **S4-offline**（= 起点 checkpoint-24980 的官方评测值，
# 已缓存在 $E/out2/_merged/s4.jsonl + _asr + _utmos + _para）一并塞进同一份报告，
# 这样能直接看出「joint 之后相对起点是涨了还是掉了」。该行的代价只有 COMET 重算。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1
NS=speech-h200
W=${W:-/tmp/expa_orch}
PARA_ROOT=${PARA_ROOT:-$W/para}
OUT_ROOT=${OUT_ROOT:-$W/quality}
ARMS=${ARMS:-"a0j a1j"}
SETS="realsi_en2zh realsi_zh2en acl6060_en2zh"
E=/fsx/stevehguo/eval
WH=$E/models/whisper-large-v3
IMG=124355679795.dkr.ecr.ap-southeast-3.amazonaws.com/ktrain/speech-language-model:p5-cuda12.8-torch2.8-te-v9-streaming-si-v1
# ASR / UTMOS / COMET 三张卡并行（pod 内 GPU 0/1/2）
POD=si-quality
NODE=${NODE:-hyperpod-i-07d384c1277ead0cd}

mkdir -p "$OUT_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 先等副语言链把每臂三个数据集都产出 ----
log "等 $PARA_ROOT 下各臂的副语言结果就绪（臂：$ARMS，集：$SETS）"
waited=0
while [ $waited -lt 43200 ]; do           # 最多等 12 小时
  ready=1
  for a in $ARMS; do
    for d in $SETS; do
      [ -s "$PARA_ROOT/$a/$d.jsonl" ] || ready=0
    done
  done
  [ $ready -eq 1 ] && break
  sleep 120; waited=$((waited+120))
done
[ $ready -eq 1 ] || { log "等待超时，已就绪的照常处理"; }
log "副语言结果就绪情况：$(for a in $ARMS; do printf '%s=%s ' $a "$(ls $PARA_ROOT/$a/*.jsonl 2>/dev/null | wc -l | tr -d ' ')"; done)"

# ---- 建打分 pod（3 卡：ASR / UTMOS / COMET 各一张）----
kubectl -n $NS delete pod $POD --wait=false >/dev/null 2>&1; sleep 5
cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata: {name: $POD, namespace: $NS}
spec:
  restartPolicy: Never
  nodeSelector: {kubernetes.io/hostname: $NODE}
  containers:
    - name: main
      image: $IMG
      command: ["sleep", "43200"]
      resources:
        requests: {cpu: "48", memory: 360Gi, nvidia.com/gpu: "3"}
        limits:   {cpu: "48", memory: 360Gi, nvidia.com/gpu: "3"}
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
i=0; while [ $i -lt 900 ]; do
  [ "$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null)" = Running ] && break
  sleep 10; i=$((i+10))
done
[ "$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null)" = Running ] \
  || { log "打分 pod 起不来，退出"; exit 1; }
kubectl -n $NS cp custom "$NS/$POD:/workspace/" >/dev/null 2>&1
log "打分 pod 就绪（node=$NODE）"

# 推理产物在各臂 master 的本地 NVMe（/nvme/expa_eval/<arm>/<ds>/results.jsonl），
# 本 pod 在别的节点上读不到，所以先把两个 master 的结果拷到 FSX 中转
# （每条臂约 1193 条 × 100KB ≈ 120MB，FSX 只剩 941GB，够用）。
for a in $ARMS; do
  src_pod=si-eval-$a
  kubectl -n $NS get pod $src_pod >/dev/null 2>&1 || { log "警告：$src_pod 不存在，跳过 $a"; continue; }
  kubectl -n $NS exec $src_pod -- bash -c \
    "mkdir -p $E/quality/$a && cp -r /nvme/expa_eval/$a/* $E/quality/$a/ 2>/dev/null; ls $E/quality/$a" \
    < /dev/null >/dev/null 2>&1
  log "  $a 推理产物已转到 FSX（$E/quality/$a）"
done

# ---- 每个臂 × 每个数据集：ASR + UTMOS（并行占 2 卡，COMET 稍后集中跑）----
for a in $ARMS; do
  for d in $SETS; do
    R=$E/quality/$a/$d/results.jsonl
    kubectl -n $NS exec $POD -- sh -c "[ -s $R ] && echo ok || echo missing" < /dev/null 2>/dev/null | tr -d '\r' | grep -q ok \
      || { log "  $a/$d 无 results.jsonl，跳过"; continue; }
    kubectl -n $NS exec $POD -- bash -c "
      export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
      mkdir -p $E/quality/_asr $E/quality/_utmos
      (setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$E/code/custom /opt/venv/bin/python \
        $E/code/custom/streaming_si/eval/run_asr.py --wav_jsonl $R \
        --manifest $E/work/manifests/$d.jsonl --out $E/quality/_asr/$a.$d.jsonl \
        --model $WH --device cuda:0 --batch 8 > $E/quality/asr.$a.$d.log 2>&1 < /dev/null &)
      (setsid env CUDA_VISIBLE_DEVICES=1 /opt/venv/bin/python \
        $E/code/custom/streaming_si/eval/run_utmos.py --wav_jsonl $R \
        --out $E/quality/_utmos/$a.$d.jsonl > $E/quality/utmos.$a.$d.log 2>&1 < /dev/null &)" \
      < /dev/null >/dev/null 2>&1
    log "  $a/$d ASR+UTMOS 已启动"
    # 串行排队，避免 6 个数据集同时抢两张卡
    k=0; while [ $k -lt 3600 ]; do
      n=$(kubectl -n $NS exec $POD -- sh -c 'pgrep -cf "[r]un_asr|[r]un_utmos"' < /dev/null 2>/dev/null | tr -d '\r')
      [ "${n:-0}" = 0 ] && break
      sleep 20; k=$((k+20))
    done
  done
done
log "ASR/UTMOS 全部完成"

# ---- 打分：三数据集并行，每集一张卡跑 score_eval.py（含 COMET）----
CK=$(kubectl -n $NS exec $POD -- sh -c \
  "ls $E/models/comet/models--Unbabel--wmt22-comet-da/snapshots/*/checkpoints/model.ckpt 2>/dev/null | head -1" < /dev/null 2>/dev/null | tr -d '\r')
[ -n "$CK" ] || { log "找不到 COMET 权重，改为 --no_comet"; CK=""; }
log "COMET 权重 = ${CK:-（无）}"

j() { kubectl -n $NS exec $POD -- sh -c "ls $1 2>/dev/null | paste -sd, -" < /dev/null 2>/dev/null | tr -d '\r'; }

g=0
for d in $SETS; do
  specs=""
  for a in $ARMS; do
    [ -s "$PARA_ROOT/$a/$d.jsonl" ] || continue
    # 副语言 jsonl 在本机，先推进 pod
    kubectl -n $NS exec $POD -- mkdir -p $E/quality/_para < /dev/null >/dev/null 2>&1
    kubectl -n $NS cp "$PARA_ROOT/$a/$d.jsonl" "$NS/$POD:$E/quality/_para/$a.$d.jsonl" >/dev/null 2>&1
    # score_eval 的 --sys 格式：名称=results:asr:utmos:lat:para
    specs="$specs --sys ${a}=$E/quality/$a/$d/results.jsonl:$(j "$E/quality/_asr/$a.$d.jsonl"):$(j "$E/quality/_utmos/$a.$d.jsonl")::$(j "$E/quality/_para/$a.$d.jsonl")"
  done
  [ -n "$specs" ] || { log "  $d 无可用臂，跳过"; continue; }
  # 参比：第二轮 S4-offline（= joint 起点 checkpoint-24980 的官方评测）
  specs="$specs --sys S4-offline-start=$E/out2/_merged/s4.jsonl:$(j "$E/out2/_asr/s4.*.jsonl"):$(j "$E/out2/_utmos/s4.*.jsonl")::$(j "$E/out2/_para/s4.*.jsonl")"
  comet_arg="--no_comet"; [ -n "$CK" ] && comet_arg="--comet_model $CK"
  kubectl -n $NS exec $POD -- bash -c "
    export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
    (setsid env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$E/code/site_comet:$E/code/custom \
      /opt/venv/bin/python $E/code/custom/streaming_si/eval/score_eval.py \
      --manifest $E/work/manifests/$d.jsonl $specs \
      --source_utmos \"$(j "$E/out2/_utmos/source.*.jsonl")\" \
      $comet_arg --out $E/quality/report.$d.json > $E/quality/score.$d.log 2>&1 < /dev/null &)" \
    < /dev/null >/dev/null 2>&1
  log "  $d 打分已启动（GPU$g）"
  g=$((g+1))
done
k=0; while [ $k -lt 3600 ]; do
  n=$(kubectl -n $NS exec $POD -- sh -c 'pgrep -cf "[s]core_eval"' < /dev/null 2>/dev/null | tr -d '\r')
  [ "${n:-0}" = 0 ] && break
  sleep 20; k=$((k+20))
done

mkdir -p "$OUT_ROOT"
for d in $SETS; do
  kubectl -n $NS cp "$NS/$POD:$E/quality/report.$d.json" "$OUT_ROOT/$d.json" >/dev/null 2>&1 \
    && log "  取回 report.$d.json"
  kubectl -n $NS exec $POD -- sh -c "grep -A8 '^system\\|^$(echo $d | head -c 8)' $E/quality/score.$d.log" < /dev/null 2>/dev/null | tail -8
done

log "质量评测完成，重出完整报告"
/usr/bin/python3 custom/streaming_si/eval/report_expa_para.py "$PARA_ROOT" "$OUT_ROOT" > $W/REPORT.md 2>&1
log "报告 -> $W/REPORT.md"

kubectl -n $NS delete pod $POD --wait=false >/dev/null 2>&1
log "完成"
