#!/bin/bash
# 给音色消融的各臂打分：UTMOS（音质）+ SIM-o/ESim/AutoPCP（音色与副语言）+ ASR（验证译文未变）。
# 消融只改推理参数，译文应逐字不变 —— ASR-BLEU 若有变化说明改动意外影响了文本侧。
# 用法：score_timbre_ablation.sh <arm> <gpu_idx>
set -uo pipefail
ARM=$1; G=$2
E=/fsx/stevehguo/eval
A=${OUT:-$E/abl}
PY=/opt/venv/bin/python
C=$E/code/custom/streaming_si/eval
export CUDA_VISIBLE_DEVICES=$G
export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
MAN=$E/work/manifests/realsi_en2zh.jsonl
MAN_ALL=$E/work/manifests/all.jsonl       # run_para_metrics 需要 dataset 字段做随机配对分组
mkdir -p $A/_asr $A/_utmos $A/_para

PYTHONPATH=$E/code/custom $PY $C/run_asr.py --wav_jsonl $A/$ARM/results.jsonl --manifest $MAN \
  --out $A/_asr/$ARM.jsonl --model $E/models/whisper-large-v3 --device cuda:0 --batch 8 2>&1 | tail -1
$PY $C/run_utmos.py --wav_jsonl $A/$ARM/results.jsonl --out $A/_utmos/$ARM.jsonl 2>&1 | tail -1
PYTHONPATH=$E/code/site_para:$E/code/custom $PY $C/run_para_metrics.py \
  --results $A/$ARM/results.jsonl --manifest $MAN_ALL --out $A/_para/$ARM.jsonl \
  --wavlm $E/models/wavlm-base-plus-sv \
  --emotion2vec /fsx/stevehguo/tts_bench/models/emotion2vec_plus_large \
  --autopcp $E/models/autopcp/model.config --w2v $E/models/wav2vec2-large-xlsr-53 \
  --device cuda:0 --batch 8 2>&1 | grep -E "FAIL|/" | tail -1
echo "ABL_SCORE_DONE $ARM"
