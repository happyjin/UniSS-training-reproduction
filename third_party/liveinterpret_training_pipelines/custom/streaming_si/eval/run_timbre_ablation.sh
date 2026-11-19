#!/bin/bash
# 音色消融（S4 离线，RealSI en→zh 346 句）：找能提升 SIM-o / UTMOS 的推理侧改动。
#
# 全部只改推理参数，不动训练，**译文逐字不变**，只有音色 / 音质 / RTF 会变。
# 第二轮评测的基线口径：ode_steps=5、prompt_sec=10、position=head、CAM++ 开。
#
#   base      基线复现（与 out2 的 S4-offline 同参数，验证可比性）
#   ode10     flow ODE 步数 5 -> 10（官方默认值）
#   p20       prompt 从 10s 放到 20s
#   tail      prompt 取源语音结尾而非开头（开头常有静音 / 气口）
#   nospk     关掉 CAM++ 说话人向量（看它当前贡献多少，反向验证）
#   ode10_p20 两项同开
#   cfg10/15/20   flow 的 inference_cfg_rate 0.7 -> 1.0 / 1.5 / 2.0
#                 （CFG 强度，越大越贴合源语音条件；Omni 的 VoiceBox 用 2.0）
#   cfg15_ode10   cfg 1.5 + ODE 10（若 cfg 有效，再看加步数能否补回音质）
#
# 用法：run_timbre_ablation.sh <arm> <gpu_idx>
set -uo pipefail
ARM=$1; G=$2
E=/fsx/stevehguo/eval
A=${OUT:-$E/abl}
PY=/opt/venv/bin/python
export CUDA_VISIBLE_DEVICES=$G
export COSYVOICE_REPO=$E/code/CosyVoice
# 必须与 run_worker2.sh 一致：少了这行，源语音 prompt 会整轮关闭（曾因此跑废 10 个臂）
export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
[ -f "$COSYVOICE_SPEECH_TOKENIZER" ] || { echo "缺 $COSYVOICE_SPEECH_TOKENIZER"; exit 2; }
export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
M=$E/work/manifests/realsi_en2zh.jsonl
LOG=$A/logs; mkdir -p $LOG

case $ARM in
  base)      EXTRA="--ode_steps 5  --prompt_sec 10" ;;
  ode10)     EXTRA="--ode_steps 10 --prompt_sec 10" ;;
  p20)       EXTRA="--ode_steps 5  --prompt_sec 20" ;;
  tail)      EXTRA="--ode_steps 5  --prompt_sec 10 --prompt_position tail" ;;
  nospk)     EXTRA="--ode_steps 5  --prompt_sec 10 --no_speaker" ;;
  ode10_p20) EXTRA="--ode_steps 10 --prompt_sec 20" ;;
  cfg10)       EXTRA="--ode_steps 5  --prompt_sec 10 --flow_cfg_rate 1.0" ;;
  cfg15)       EXTRA="--ode_steps 5  --prompt_sec 10 --flow_cfg_rate 1.5" ;;
  cfg20)       EXTRA="--ode_steps 5  --prompt_sec 10 --flow_cfg_rate 2.0" ;;
  cfg15_ode10) EXTRA="--ode_steps 10 --prompt_sec 10 --flow_cfg_rate 1.5" ;;
  noprompt)        EXTRA="--ode_steps 5 --no_source_prompt" ;;
  nospk_noprompt)  EXTRA="--ode_steps 5 --no_source_prompt --no_speaker" ;;
  *) echo "未知臂 $ARM"; exit 2 ;;
esac

t0=$(date +%s)
echo "[abl $ARM] 开始 $(date +%H:%M:%S) gpu=$G | $EXTRA"
PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s2st \
  --model_dir $E/models/s4_ck --input_jsonl $M --out_dir $A/$ARM \
  --cosyvoice_repo $COSYVOICE_REPO --device cuda:0 $EXTRA \
  ${LIMIT:+--limit $LIMIT} > $LOG/$ARM.log 2>&1
rc=$?
# 防呆：除显式关闭 prompt 的臂外，source_prompt_tokens 必须 > 0
case $ARM in
  noprompt|nospk_noprompt) ;;
  *) if [ $rc -eq 0 ] && ! grep -q '"source_prompt_tokens": [1-9]' $A/$ARM/results.jsonl 2>/dev/null; then
       echo "[abl $ARM] 源语音 prompt 未生效（source_prompt_tokens=0），结果不可用"; rc=90
     fi ;;
esac
echo "[abl $ARM] 结束 rc=$rc 用时 $(( $(date +%s) - t0 ))s 条数 $(grep -c . $A/$ARM/results.jsonl 2>/dev/null || echo 0)"
