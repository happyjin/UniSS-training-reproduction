#!/bin/bash
# 第三轮评测 worker：复现的 SimulS2ST-Omni（zeyangsong_share/Qwen2.5_3B_DualCodec）
#   repoff —— 复现离线：官方离线 e2e，基座 = 官方 offline，checkpoint = 复现 stage2_multitask_full
#   rep1～rep4 —— 复现流式：官方 SimulEval agent，基座 = 官方 offline + 合并复现 stage2，
#                 checkpoint = 复现 stage3v2_fluency_retrain，L = 1/2/3/4
# 两者的推理代码、参数与第二轮的 SimulS2ST-Omni-offline / -L1～L4 逐字相同，只换权重。
# 用法：run_worker3.sh <shard_id> <gpu_idx> <system>
set -uo pipefail
K=$1; G=$2; S=$3
E=/fsx/stevehguo/eval
O=${OUT:-$E/out3}
M=$E/work/shards/all.$K.jsonl
LOG=$O/logs; mkdir -p $LOG
export CUDA_VISIBLE_DEVICES=$G
export HF_HOME=$E/hf TMPDIR=${TMPDIR:-/tmp} PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=/opt/venv/bin/python
R=$E/code/SimulS2ST-Omni-main
REP=$E/models/repro
STAGE2=${STAGE2:-$REP/stage2_multitask_full}
# STAGE3 可覆盖：评测新版复现时只换这一项（如 1019 版 stage3v2_fluency_retrain_1019），其余逐字不变
STAGE3=${STAGE3:-$REP/stage3v2_fluency_retrain}
LIMIT=${LIMIT:-}

# 预热句（同第二轮）：清单最前复制一句，id 前缀 _warmup_，合并时剔除
MW=$O/manifests/shard$K.jsonl
mkdir -p $O/manifests
[ -f $MW ] || $PY - "$M" "$MW" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
w = dict(rows[0]); w['id'] = '_warmup_' + w['id']
with open(sys.argv[2] + '.tmp', 'w', encoding='utf-8') as f:
    for r in [w] + rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
import os; os.replace(sys.argv[2] + '.tmp', sys.argv[2])
EOF
[ -n "$LIMIT" ] && LIMIT=$((LIMIT + 1))

rep_stream() {   # $1 = L
  local L=$1
  for d in en2zh zh2en; do
    local W=$O/rep_L$L/shard${K}_$d
    rm -rf $W; mkdir -p $W
    $PY - "$MW" "$d" "$W" "${LIMIT:-0}" <<'EOF'
import json, sys
m, d, w, lim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(m, encoding='utf-8') if l.strip()]
rows = [r for r in rows if f"{r['src_lang']}2{r['tgt_lang']}" == d and not r['id'].startswith('_warmup_')]
rows = rows[:lim - 1] if lim else rows
if rows:
    wu = dict(rows[0]); wu['id'] = '_warmup_' + wu['id']; rows = [wu] + rows
with open(f'{w}/ids.txt', 'w') as fi, open(f'{w}/source.txt', 'w') as fs, open(f'{w}/target.txt', 'w') as ft:
    for r in rows:
        fi.write(r['id'] + '\n'); fs.write(r['audio'] + '\n'); ft.write(r['reference'].replace('\n', ' ') + '\n')
EOF
    local n=$(wc -l < $W/ids.txt)
    [ "$n" -eq 0 ] && continue
    if [ $d = en2zh ]; then SL=English; TL=Chinese; TOK=zh; UNIT=char; else SL=Chinese; TL=English; TOK=13a; UNIT=word; fi
    # 与第二轮 SimulS2ST-Omni-L* 逐字相同的参数；只换 --checkpoint-path，并经 omni_merge 先合 Stage 2
    ( cd $R && SIMULS2ST_MERGE_ADAPTERS_FROM=$STAGE2 \
      PYTHONPATH=$E/code/site_omni:$R:$R/external/SimulEval:$R/src/train \
      $PY $E/code/custom/streaming_si/eval/omni_simuleval_wrap.py \
        --agent src/agents/simuleval_omni_talker_s2st_agent.py \
        --source-segment-size 1000 \
        --model-name-or-path models/SimulS2ST-Omni/offline \
        --checkpoint-path $STAGE3 \
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
        --no-scoring ) > $LOG/rep_L$L.$K.$d.log 2>&1
    echo "  rep L$L $d: rc=$? 实例 $(grep -c . $W/instances.log 2>/dev/null || echo 0)/$n"
  done
}

t0=$(date +%s)
echo "[worker $K/$S] 开始 $(date +%H:%M:%S) gpu=$G host=$(hostname)"
case $S in
  repoff)
    R2=$R
    ( cd $R2 && PYTHONPATH=$E/code/site_omni:$R2:$R2/external/SimulEval:$R2/src/train:$E/code/custom \
      $PY $E/code/custom/streaming_si/eval/run_omni_offline.py \
        --manifest $MW --shard $K --num_shards 1 \
        --out_dir $O/rep_off --model_root $E/models/SimulS2ST-Omni --checkpoint_path $STAGE2 \
        --device cuda:0 ${LIMIT:+--limit $LIMIT} ) > $LOG/rep_off.$K.log 2>&1
    echo "  repoff: rc=$? 条数 $(grep -c . $O/rep_off/results.$(printf %02d $K).jsonl 2>/dev/null || echo 0)" ;;
  rep1|rep2|rep3|rep4)
    rep_stream ${S#rep} ;;
  *) echo "未知系统 $S"; exit 2 ;;
esac
echo "[worker $K/$S] 结束 用时 $(( $(date +%s) - t0 ))s"
