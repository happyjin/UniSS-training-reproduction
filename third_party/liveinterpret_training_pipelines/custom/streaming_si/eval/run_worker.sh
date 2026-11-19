#!/bin/bash
# 单个 worker：在指定 GPU 上把一个分片跑完三个系统（S4 离线 / S5 流式 / SimulS2ST-Omni 离线）。
# 用法：run_worker.sh <shard_id> <num_shards> <gpu_idx> [systems，默认 "s4 s5 omni"]
set -uo pipefail
K=$1; N=$2; G=$3; SYS=${4:-"s4 s5 omni"}
E=/fsx/stevehguo/eval
M=$E/work/shards/all.$K.jsonl
LOG=$E/out/logs; mkdir -p $LOG
export CUDA_VISIBLE_DEVICES=$G
export COSYVOICE_REPO=$E/code/CosyVoice
export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
export HF_HOME=$E/hf TMPDIR=${TMPDIR:-/tmp} PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=/opt/venv/bin/python
LIMIT=${LIMIT:-}
[ -n "$LIMIT" ] && LIMIT="--limit $LIMIT"

for s in $SYS; do
  t0=$(date +%s)
  echo "[worker $K] === $s 开始 $(date +%H:%M:%S)"
  case $s in
    s4)   # 离线整句 S2ST，S4 checkpoint-24980；参数全用脚本默认（与 README 8.2 一致）
      PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s2st \
        --model_dir $E/models/s4_ck --input_jsonl $M --out_dir $E/out/s4/shard$K \
        --cosyvoice_repo $E/code/CosyVoice --device cuda:0 $LIMIT \
        > $LOG/s4.$K.log 2>&1 ;;
    s5)   # S5 流式（640ms 分块 WAIT/WRITE），S4 底座 + S5 adapter
      PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s5_streaming \
        --model_dir $E/models/s4_ck --adapter $E/models/s5_adapter --input_jsonl $M \
        --out_dir $E/out/s5/shard$K --cosyvoice_repo $E/code/CosyVoice --device cuda:0 $LIMIT \
        > $LOG/s5.$K.log 2>&1 ;;
    omni) # SimulS2ST-Omni 官方离线 e2e（同一份分片清单，--num_shards 1 表示不再二次切分）
      R=$E/code/SimulS2ST-Omni-main
      PYTHONPATH=$E/code/site_omni:$R:$R/external/SimulEval:$R/src/train:$E/code/custom \
        $PY $E/code/custom/streaming_si/eval/run_omni_offline.py \
        --manifest $M --shard $K --num_shards 1 \
        --out_dir $E/out/omni --model_root $E/models/SimulS2ST-Omni --device cuda:0 $LIMIT \
        > $LOG/omni.$K.log 2>&1 ;;
  esac
  rc=$?
  echo "[worker $K] === $s 结束 rc=$rc 用时 $(( $(date +%s) - t0 ))s"
done
echo "[worker $K] WORKER_DONE"
