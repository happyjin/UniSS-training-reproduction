#!/bin/bash
# 副语言指标（SIM-o / ESim / AutoPCP）：每张卡跑一个分片，覆盖全部 7 个系统。
# 用法：run_para.sh <shard> <num_shards> <gpu_idx>
set -uo pipefail
K=$1; N=$2; G=$3
E=/fsx/stevehguo/eval
O=${OUT:-$E/out2}
PY=/opt/venv/bin/python
C=$E/code/custom/streaming_si/eval
export CUDA_VISIBLE_DEVICES=$G
export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$E/code/site_para:$E/code/custom
mkdir -p $O/_para
for s in s4 omni_off s5 omni_L1 omni_L2 omni_L3 omni_L4; do
  $PY $C/run_para_metrics.py --results $O/_merged/$s.jsonl --manifest $E/work/manifests/all.jsonl \
    --out $O/_para/$s.$K.jsonl --wavlm $E/models/wavlm-base-plus-sv \
    --emotion2vec /fsx/stevehguo/tts_bench/models/emotion2vec_plus_large \
    --autopcp $E/models/autopcp/model.config --w2v $E/models/wav2vec2-large-xlsr-53 \
    --device cuda:0 --batch 8 --shard $K --num_shards $N 2>&1 | grep -E "FAIL|/" | tail -1
  echo "[para $K] $s 完成"
done
echo PARA_DONE
