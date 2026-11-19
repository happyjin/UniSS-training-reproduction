#!/bin/bash
# 第二轮评测 worker：一张卡跑完一个分片的
#   s4    —— 优化后的 S4 离线（thinker 增量缓存 + talker CUDA 图）
#   s5    —— S5 流式（补了逐步计时，供计算感知 LAAL）
#   omniL —— SimulS2ST-Omni 官方流式（SimulEval，L = 1/2/3/4）
# 用法：run_worker2.sh <shard_id> <gpu_idx> [systems，默认 "s4 s5 omni1 omni2 omni3 omni4"]
set -uo pipefail
K=$1; G=$2; SYS=${3:-"s4 s5 omni1 omni2 omni3 omni4"}
E=/fsx/stevehguo/eval
O=${OUT:-$E/out2}
M=$E/work/shards/all.$K.jsonl
LOG=$O/logs; mkdir -p $LOG
export CUDA_VISIBLE_DEVICES=$G
export COSYVOICE_REPO=$E/code/CosyVoice
export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
export HF_HOME=$E/hf TMPDIR=${TMPDIR:-/tmp} PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=/opt/venv/bin/python
LIMIT=${LIMIT:-}
R=$E/code/SimulS2ST-Omni-main

# 预热：每个进程第一句会把 CUDA Graph 录制 / kernel autotune 算进耗时，抬高 RTF 与计算感知延迟。
# 在清单最前面复制一句（id 前缀 _warmup_），打分时剔除。
MW=$O/manifests/shard$K.jsonl
mkdir -p $O/manifests
$PY - "$M" "$MW" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
w = dict(rows[0]); w['id'] = '_warmup_' + w['id']
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    for r in [w] + rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
EOF
[ -n "$LIMIT" ] && LIMIT=$((LIMIT + 1))
M=$MW

omni_stream() {   # $1 = L
  local L=$1
  for d in en2zh zh2en; do
    local W=$O/omni_L$L/shard${K}_$d
    rm -rf $W; mkdir -p $W
    # 本分片里该方向的句子 -> SimulEval 的 source / target 列表（一行一条，顺序即实例序号）
    $PY - "$M" "$d" "$W" "${LIMIT:-0}" <<'EOF'
import json, sys
m, d, w, lim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(m, encoding='utf-8') if l.strip()]
rows = [r for r in rows if f"{r['src_lang']}2{r['tgt_lang']}" == d and not r['id'].startswith('_warmup_')]
rows = rows[:lim - 1] if lim else rows
if rows:                                          # 每个方向单独起进程，各自预热一句
    wu = dict(rows[0]); wu['id'] = '_warmup_' + wu['id']; rows = [wu] + rows
with open(f'{w}/ids.txt', 'w') as fi, open(f'{w}/source.txt', 'w') as fs, open(f'{w}/target.txt', 'w') as ft:
    for r in rows:
        fi.write(r['id'] + '\n'); fs.write(r['audio'] + '\n'); ft.write(r['reference'].replace('\n', ' ') + '\n')
print(len(rows))
EOF
    local n=$(wc -l < $W/ids.txt)
    [ "$n" -eq 0 ] && continue
    if [ $d = en2zh ]; then SL=English; TL=Chinese; TOK=zh; UNIT=char; else SL=Chinese; TL=English; TOK=13a; UNIT=word; fi
    # 参数与官方 README「Streaming S2ST」完全一致，只改 --latency-multiplier 与语言 / 分词口径
    ( cd $R && PYTHONPATH=$E/code/site_omni:$R:$R/external/SimulEval:$R/src/train \
      $PY $E/code/custom/streaming_si/eval/omni_simuleval_wrap.py \
        --agent src/agents/simuleval_omni_talker_s2st_agent.py \
        --source-segment-size 1000 \
        --model-name-or-path models/SimulS2ST-Omni/offline \
        --checkpoint-path models/SimulS2ST-Omni/simuls2st_adapter \
        --source $W/source.txt --target $W/target.txt --output $W \
        --source-lang $SL --target-lang $TL \
        --latency-multiplier $L \
        --sacrebleu-tokenizer $TOK --eval-latency-unit $UNIT \
        --history-window-turns 28 --history-overlap-turns 16 \
        --prompt-source-chunks 2 --prompt-generated-chunks 1 \
        --thinker-max-new-tokens 256 --talker-max-new-tokens 500 \
        --thinker-no-sample --talker-no-sample \
        --thinker-num-beams 4 \
        --thinker-repetition-penalty 1.2 --thinker-no-repeat-ngram-size 5 \
        --talker-repetition-penalty 1.4 --talker-no-repeat-ngram-size 5 \
        --computation-aware \
        --no-scoring ) > $LOG/omni_L$L.$K.$d.log 2>&1
    echo "  omni L$L $d: rc=$? 实例 $(grep -c . $W/instances.log 2>/dev/null || echo 0)/$n"
  done
}

for s in $SYS; do
  t0=$(date +%s)
  echo "[worker $K] === $s 开始 $(date +%H:%M:%S)"
  case $s in
    s4)
      PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s2st \
        --model_dir $E/models/s4_ck --input_jsonl $M --out_dir $O/s4/shard$K \
        --cosyvoice_repo $E/code/CosyVoice --device cuda:0 ${LIMIT:+--limit $LIMIT} \
        > $LOG/s4.$K.log 2>&1 ;;
    s5)
      PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s5_streaming \
        --model_dir $E/models/s4_ck --adapter $E/models/s5_adapter --input_jsonl $M \
        --out_dir $O/s5/shard$K --cosyvoice_repo $E/code/CosyVoice --device cuda:0 ${LIMIT:+--limit $LIMIT} \
        > $LOG/s5.$K.log 2>&1 ;;
    omni1|omni2|omni3|omni4)
      omni_stream ${s#omni} ;;
  esac
  echo "[worker $K] === $s 结束 rc=$? 用时 $(( $(date +%s) - t0 ))s"
done
echo "[worker $K] WORKER_DONE"
