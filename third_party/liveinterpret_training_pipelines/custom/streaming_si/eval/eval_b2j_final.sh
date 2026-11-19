#!/bin/bash
# b2j 最终评测：与 orchestrate_expa.sh 的 eval_arm（a0j/a1j）逐步同口径。
#   三评测集各一卡推理 → SIM-o / AutoPCP / ESim 打分 → 与 a0j/a1j 出对比报告。
# 用法（本机）：
#   caffeinate -i -s nohup bash custom/streaming_si/eval/eval_b2j_final.sh > /tmp/eval_b2j.log 2>&1 < /dev/null &
# 前置：si-eval-b2j（09a68，3 卡）已 Running；b2j 终点 checkpoint 在该节点本地盘。
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
source .venv/bin/activate
NS=speech-h200
POD=${POD:-si-eval-b2j}
ARM=${ARM:-b2j}
CK=${CK:-/nvme/output/stevehguo/streaming_si/s4_s2st/v8-20261021-121201/checkpoint-37470}
E=/fsx/stevehguo/eval
SETS="realsi_en2zh realsi_zh2en acl6060_en2zh"
W=${W:-inference_outputs/eval_expA_b2j_20261022}
mkdir -p $W/para
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "ckpt = $CK"
kubectl -n $NS exec $POD -- bash -c "
  test -f $CK/config.json || { echo 'CKPT_MISSING'; exit 1; }
  grep -oE '\"talker_cond_positions\": \"[^\"]*\"' $CK/config.json
  [ -d $CK/speech_decoder ] || cp -r $E/models/s4_ck/speech_decoder $CK/
  ls $CK/speech_decoder | head -3" < /dev/null 2>&1 | tail -4

i=0
for d in $SETS; do
  kubectl -n $NS exec $POD -- bash -c "
    export COSYVOICE_REPO=$E/code/CosyVoice
    export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
    export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
    mkdir -p /nvme/expa_eval/$ARM
    (setsid env CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$E/code/custom /opt/venv/bin/python \
      -m streaming_si.inference.infer_s2st --model_dir $CK \
      --input_jsonl $E/work/manifests/$d.jsonl --out_dir /nvme/expa_eval/$ARM/$d \
      --cosyvoice_repo $E/code/CosyVoice --device cuda:0 \
      > /nvme/expa_eval/$ARM/infer.$d.log 2>&1 < /dev/null &)" < /dev/null > /dev/null 2>&1
  i=$((i+1)); sleep 3
done
log "三集推理已启动"

sleep 120
while true; do
  ph=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$ph" != Running ] && { log "pod 异常（$ph），中止"; exit 1; }
  n=$(kubectl -n $NS exec $POD -- sh -c 'pgrep -cf "[i]nfer_s2st"' < /dev/null 2>/dev/null | tr -d '\r')
  [ -n "$n" ] && [ "$n" = 0 ] && break
  sleep 60
done
kubectl -n $NS exec $POD -- sh -c "for d in $SETS; do f=/nvme/expa_eval/$ARM/\$d/results.jsonl; [ -f \$f ] || cat /nvme/expa_eval/$ARM/\$d/shard*/results.jsonl > \$f 2>/dev/null; echo \"  \$d \$(grep -c . \$f 2>/dev/null) 条\"; done" < /dev/null 2>&1 | tail -3
log "推理完成"

i=0
for d in $SETS; do
  kubectl -n $NS exec $POD -- bash -c "
    export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
    mkdir -p /nvme/expa_eval/$ARM/_para
    (setsid env CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$E/code/site_para:$E/code/custom \
      /opt/venv/bin/python $E/code/custom/streaming_si/eval/run_para_metrics.py \
      --results /nvme/expa_eval/$ARM/$d/results.jsonl --manifest $E/work/manifests/all.jsonl \
      --out /nvme/expa_eval/$ARM/_para/$d.jsonl \
      --wavlm $E/models/wavlm-base-plus-sv \
      --emotion2vec /fsx/stevehguo/tts_bench/models/emotion2vec_plus_large \
      --autopcp $E/models/autopcp/model.config --w2v $E/models/wav2vec2-large-xlsr-53 \
      --device cuda:0 --batch 8 > /nvme/expa_eval/$ARM/para.$d.log 2>&1 < /dev/null &)" < /dev/null > /dev/null 2>&1
  i=$((i+1)); sleep 3
done
log "打分已启动"
sleep 120
while true; do
  ph=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$ph" != Running ] && { log "pod 异常（$ph），中止"; exit 1; }
  n=$(kubectl -n $NS exec $POD -- sh -c 'pgrep -cf "[r]un_para_metrics"' < /dev/null 2>/dev/null | tr -d '\r')
  [ -n "$n" ] && [ "$n" = 0 ] && break
  sleep 60
done

mkdir -p $W/para/$ARM
for d in $SETS; do
  kubectl -n $NS cp "$NS/$POD:/nvme/expa_eval/$ARM/_para/$d.jsonl" "$W/para/$ARM/$d.jsonl" > /dev/null 2>&1
  kubectl -n $NS cp "$NS/$POD:/nvme/expa_eval/$ARM/$d/results.jsonl" "$W/$ARM.$d.results.jsonl" > /dev/null 2>&1
  log "  回传 $d: $(grep -c . $W/para/$ARM/$d.jsonl 2>/dev/null || echo 0) 条"
done
A=inference_outputs/eval_expA_a0j_a1j_20261020
for a in a0j a1j; do mkdir -p $W/para/$a; cp $A/$a/_para/*.jsonl $W/para/$a/ 2>/dev/null; done
/usr/bin/python3 custom/streaming_si/eval/report_expa_para.py "$W/para" > $W/REPORT.md 2>&1
log "报告 -> $W/REPORT.md"
log "EVAL_B2J_DONE"
