#!/bin/bash
# 推理跑完后的第二阶段：合并各分片结果 -> 7 卡并行 ASR -> 打分（含 COMET）。
# 用法：run_score.sh <stage: merge|asr|score> [起始分片 数量 总分片]
set -uo pipefail
E=/fsx/stevehguo/eval
PY=/opt/venv/bin/python
export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
STAGE=$1

case $STAGE in
merge)
  mkdir -p $E/out/_merged
  $PY - <<'EOF'
import glob, json, os
E = '/fsx/stevehguo/eval'
for sysname, pat in (('s4', f'{E}/out/s4/shard*/results.jsonl'),
                     ('s5', f'{E}/out/s5/shard*/results.jsonl'),
                     ('omni', f'{E}/out/omni/results.*.jsonl')):
    rows, seen = [], set()
    for p in sorted(glob.glob(pat)):
        for l in open(p, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                if r['id'] in seen:          # 分片不该重叠，重复只保留第一条并报出来
                    print(f'[warn] {sysname} 重复 id {r["id"]} @ {p}')
                    continue
                seen.add(r['id']); rows.append(r)
    out = f'{E}/out/_merged/{sysname}.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok = sum(1 for r in rows if r.get('status') == 'ok')
    print(f'{sysname}: {len(rows)} 条（ok {ok}），-> {out}')
EOF
  ;;
asr)
  START=$2; CNT=$3; N=$4
  mkdir -p $E/out/_asr $E/out/logs
  for i in $(seq 0 $((CNT - 1))); do
    K=$((START + i))
    for s in s4 s5 omni; do
      echo "CUDA_VISIBLE_DEVICES=$i asr $s shard $K"
    done
    (setsid env CUDA_VISIBLE_DEVICES=$i bash -c "
      for s in ${SYSLIST:-s4 s5 omni}; do
        PYTHONPATH=$E/code/custom $PY $E/code/custom/streaming_si/eval/run_asr.py \
          --wav_jsonl $E/out/_merged/\$s.jsonl --out $E/out/_asr/\$s.$K.jsonl \
          --manifest $E/work/manifests/all.jsonl \
          --model $E/models/whisper-large-v3 --device cuda:0 --shard $K --num_shards $N --batch 8
      done" > $E/out/logs/asr.$K.log 2>&1 < /dev/null &)
    sleep 2
  done
  sleep 10; pgrep -af "[r]un_asr.py" | wc -l
  ;;
score)
  mkdir -p $E/out/_report
  CK=$(ls $E/models/comet/models--Unbabel--wmt22-comet-da/snapshots/*/checkpoints/model.ckpt | head -1)
  echo "COMET ckpt: $CK"
  for ds in realsi_en2zh realsi_zh2en acl6060_en2zh; do
    PYTHONPATH=$E/code/site_comet:$E/code/custom $PY $E/code/custom/streaming_si/eval/score_eval.py \
      --manifest $E/work/manifests/$ds.jsonl \
      --sys "S4-offline=$E/out/_merged/s4.jsonl:$(ls $E/out/_asr/s4.*.jsonl | paste -sd,)" \
      --sys "S5-streaming=$E/out/_merged/s5.jsonl:$(ls $E/out/_asr/s5.*.jsonl | paste -sd,)" \
      --sys "SimulS2ST-Omni=$E/out/_merged/omni.jsonl:$(ls $E/out/_asr/omni.*.jsonl | paste -sd,)" \
      --comet_model "$CK" --out $E/out/_report/$ds.json 2>&1 | grep -vE "Warning|warn|Encoder model frozen|Lightning automatically"
  done
  ;;
esac
