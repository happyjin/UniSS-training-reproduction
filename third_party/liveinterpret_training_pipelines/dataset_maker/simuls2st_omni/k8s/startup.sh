#!/usr/bin/env bash
# TTS_omni 数据准备入口（美西 i-04ed1446，6 卡 / 18 worker，3 worker/GPU）
set -euo pipefail

ROOT=/opt/dlami/nvme/stevehguo/tts_omni
RAW=/opt/raw
NW=${NW:-18}                    # 本 Pod 运行的 worker 数
NWT=${NWT:-40}                  # 全局分区模数：多个 Pod 共用同一批 items 时必须一致
WORKER_START=${WORKER_START:-0} # 本 Pod 的起始 worker id（全局唯一）
NGPU=${NGPU:-6}

# 线程上限：torch/BLAS 默认每进程起 448 线程导致严重超订（load 340+/448）。
# 限到 8 后单 worker 效率更高，余出 CPU 给更多 worker 并行。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$ROOT"/{code,cache,done,logs} "$ROOT"/weights

# ---- 代码（configMap 只能平铺，包用 tar 传递）----
cp "$RAW"/make_tts_dualcodec.py "$RAW"/tts_omni_driver.py "$RAW"/uploader.py "$ROOT"/code/
if [ ! -d "$ROOT"/code/custom/simuls2st_omni ]; then
  tar xzf "$RAW"/pkg.tar.gz -C "$ROOT"/code/
fi
export PYTHONPATH="$ROOT/code/custom:$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/hf"
export PYTHONUNBUFFERED=1

# ---- 依赖（voicebox tokenizer 链：audiotools/dac/easydict/hyperpyyaml/ruamel.yaml/json5）----
python3 -m pip install -q boto3 huggingface_hub descript-audiotools descript-audio-codec \
  easydict hyperpyyaml ruamel.yaml json5 einops six 2>&1 | tail -1 || true

# ---- DualCodec + w2v 权重（幂等，约 5GB，来自 HF 公开仓库）----
if [ ! -f "$ROOT"/weights/SimulS2ST-Omni/dualcodec/dualcodec.safetensors ]; then
  echo "[startup] 下载 tokenizer 权重 ..."
  python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('HA-SA-ki/SimulS2ST-Omni',
                  allow_patterns=['dualcodec/*', 'w2v/*'],
                  local_dir='/opt/dlami/nvme/stevehguo/tts_omni/weights/SimulS2ST-Omni')
print('[startup] 权重就绪')
PY
fi

# ---- 后台增量上传 ----
nohup python3 "$ROOT"/code/uploader.py --root "$ROOT" > "$ROOT"/logs/uploader.log 2>&1 &

# ---- workers：每 GPU 个数 = NW/NGPU，全局 id 从 WORKER_START 起 ----
echo "[startup] 启动 $NW 个 worker（全局 id $WORKER_START..$((WORKER_START + NW - 1))，模数 $NWT）..."
pids=()
# 补跑参数（可选）：RESIDUE_MOD/MIN/MAX 圈定漏掉的分区，SHARD_OFFSET_BASE 避开既有 shard
RES=""
if [ -n "${ONLY_KEYS:-}" ]; then
  RES="--only-keys $ONLY_KEYS"
elif [ -n "${RESIDUE_MOD:-}" ]; then
  RES="--residue-mod $RESIDUE_MOD --residue-min ${RESIDUE_MIN:-0} --residue-max ${RESIDUE_MAX:-$((RESIDUE_MOD - 1))}"
fi
# 带跳过集重跑（可选）：SKIP_IDS_S3 为 s3://... 的 union_ids.npy；DONE_DIR 为独立完成标记目录
if [ -n "${SKIP_IDS_S3:-}" ]; then
  SKIP_LOCAL="$ROOT/meta/$(basename "$SKIP_IDS_S3")"
  mkdir -p "$ROOT/meta"
  python3 - "$SKIP_IDS_S3" "$SKIP_LOCAL" <<'PY'
import sys, boto3
b, k = sys.argv[1][5:].split('/', 1)
boto3.client('s3', region_name='ap-southeast-3').download_file(b, k, sys.argv[2])
print('[startup] 跳过集已下载', sys.argv[2])
PY
  RES="$RES --skip-ids $SKIP_LOCAL"
fi
[ -n "${DONE_DIR:-}" ] && RES="$RES --done-dir $DONE_DIR"
LOG_TAG=${LOG_TAG:-worker}
# 可选：GPU_LIST="0,1,3,..." 自定义 worker→GPU 映射（按 Pod 内序号轮转）；默认 w % NGPU
GPUS=()
[ -n "${GPU_LIST:-}" ] && IFS=, read -ra GPUS <<< "$GPU_LIST"
for w in $(seq $WORKER_START $((WORKER_START + NW - 1))); do
  SOB=""
  if [ -n "${SHARD_OFFSET_BASE:-}" ]; then
    # 按全局 worker id 编号：不同 Pod（不同节点）共用同一 S3 前缀，局部序号会撞名
    SOB="--shard-offset-base $((SHARD_OFFSET_BASE + w * 100000))"
  fi
  WID=$w; NWK=$NWT; RESW="$RES"
  if [ -n "${ONLY_KEYS_PREFIX:-}" ]; then
    # 可选：每个 worker 一份显式清单（按耗时均衡分配好的），不再取模分区
    WID=0; NWK=1; RESW="$RES --only-keys ${ONLY_KEYS_PREFIX}${w}.txt"
  fi
  if [ ${#GPUS[@]} -gt 0 ]; then
    GPU=${GPUS[$(( (w - WORKER_START) % ${#GPUS[@]} ))]}
  else
    GPU=$((w % NGPU))
  fi
  CUDA_VISIBLE_DEVICES=$GPU python3 "$ROOT"/code/tts_omni_driver.py \
    --worker-id "$WID" --nworkers "$NWK" --root "$ROOT" --device cuda:0 \
    ${SHARED_GPU:+--shared-gpu} $RESW $SOB \
    > "$ROOT"/logs/$LOG_TAG-$w.log 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

# ---- 最终全量上传 ----
# 上传器会跳过 60 秒内写入的文件（防止传半成品）；所有 worker 已退出，等过保护期再传，
# 否则最后几秒 flush 的 shard 会漏传
sleep 61
python3 "$ROOT"/code/uploader.py --root "$ROOT" --once >> "$ROOT"/logs/uploader.log 2>&1 || true
echo "[startup] ALL DONE fail=$fail（$(date -u)）"
exit $fail
