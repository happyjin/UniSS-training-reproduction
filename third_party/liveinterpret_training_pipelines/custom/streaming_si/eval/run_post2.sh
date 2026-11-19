#!/bin/bash
# 第二轮评测的后处理。
#   merge             合并各分片、整理流式分段、剔除预热句（CPU，跑一次）
#   gpu <k> <n> <gpu> 第 k 片（共 n 片）的 ASR / LAAL 词对齐 / UTMOS
#   score             三个数据集并行打分（含 COMET），每个数据集占一张卡
set -uo pipefail
E=/fsx/stevehguo/eval
O=${OUT:-$E/out2}
PY=/opt/venv/bin/python
C=$E/code/custom/streaming_si/eval
export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
MAN=$E/work/manifests/all.jsonl
WH=$E/models/whisper-large-v3
STREAM="s5 omni_L1 omni_L2 omni_L3 omni_L4"
ALL="s4 omni_off $STREAM"

case $1 in
merge)
  mkdir -p $O/_merged
  $PY $C/stream_segments.py s5 --results "$O/s5/shard*/results.jsonl" --out $O/_seg/s5
  for L in 1 2 3 4; do
    $PY $C/stream_segments.py omni --runs "$O/omni_L$L/shard*" --out $O/_seg/omni_L$L
  done
  $PY - "$O" "$E" <<'EOF'
import glob, json, sys
O, E = sys.argv[1], sys.argv[2]
man = {json.loads(l)['id']: json.loads(l) for l in open(f'{E}/work/manifests/all.jsonl', encoding='utf-8')}
srcs = {'s4': sorted(glob.glob(f'{O}/s4/shard*/results.jsonl')),
        's5': [f'{O}/_seg/s5/results.jsonl'],
        'omni_off': [f'{E}/out/_merged/omni.jsonl']}            # 离线 Omni 代码未变，沿用第一轮结果
for L in (1, 2, 3, 4):
    srcs[f'omni_L{L}'] = [f'{O}/_seg/omni_L{L}/results.jsonl']
for name, paths in srcs.items():
    rows, seen = [], set()
    for p in paths:
        for l in open(p, encoding='utf-8'):
            if not l.strip():
                continue
            r = json.loads(l)
            if r['id'].startswith('_warmup_') or r['id'] in seen:
                continue
            seen.add(r['id'])
            r.setdefault('tgt_lang', man[r['id']]['tgt_lang'])
            r.setdefault('source_sec', man[r['id']]['source_sec'])
            rows.append(r)
    with open(f'{O}/_merged/{name}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok = sum(1 for r in rows if r.get('status') in (None, 'ok'))
    miss = len(set(man) - seen)
    print(f'{name}: {len(rows)} 条（ok {ok}，缺 {miss}）')
EOF
  ;;
gpu)
  K=$2; N=$3; export CUDA_VISIBLE_DEVICES=$4
  mkdir -p $O/_asr $O/_lat $O/_utmos
  for s in s4 $STREAM; do            # 普通转写（ASR-BLEU / COMET 用；omni_off 沿用第一轮转写）
    PYTHONPATH=$E/code/custom $PY $C/run_asr.py --wav_jsonl $O/_merged/$s.jsonl --manifest $MAN \
      --out $O/_asr/$s.$K.jsonl --model $WH --device cuda:0 --shard $K --num_shards $N --batch 8
  done
  for s in $STREAM; do               # 带词级时间戳的转写 -> LAAL
    $PY $C/latency_eval.py --results $O/_merged/$s.jsonl --manifest $MAN --model $WH \
      --out $O/_lat/$s.$K.jsonl --shard $K --num_shards $N
  done
  for s in $ALL; do
    $PY $C/run_utmos.py --wav_jsonl $O/_merged/$s.jsonl --out $O/_utmos/$s.$K.jsonl --shard $K --num_shards $N
  done
  $PY $C/run_utmos.py --wav_jsonl $MAN --wav_key audio --out $O/_utmos/source.$K.jsonl --shard $K --num_shards $N
  echo GPU_STAGE_DONE
  ;;
score)
  mkdir -p $O/_report
  CK=$(ls $E/models/comet/models--Unbabel--wmt22-comet-da/snapshots/*/checkpoints/model.ckpt | head -1)
  j() { ls $1 2>/dev/null | paste -sd, -; }
  g=0
  for ds in realsi_en2zh realsi_zh2en acl6060_en2zh; do
    (CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$E/code/site_comet:$E/code/custom $PY $C/score_eval.py \
      --manifest $E/work/manifests/$ds.jsonl \
      --sys "S4-offline=$O/_merged/s4.jsonl:$(j "$O/_asr/s4.*.jsonl"):$(j "$O/_utmos/s4.*.jsonl")::$(j "$O/_para/s4.*.jsonl")" \
      --sys "SimulS2ST-Omni-offline=$O/_merged/omni_off.jsonl:$(j "$E/out/_asr/omni.*.jsonl"):$(j "$O/_utmos/omni_off.*.jsonl")::$(j "$O/_para/omni_off.*.jsonl")" \
      --sys "S5-streaming=$O/_merged/s5.jsonl:$(j "$O/_asr/s5.*.jsonl"):$(j "$O/_utmos/s5.*.jsonl"):$(j "$O/_lat/s5.*.jsonl"):$(j "$O/_para/s5.*.jsonl")" \
      --sys "SimulS2ST-Omni-L1=$O/_merged/omni_L1.jsonl:$(j "$O/_asr/omni_L1.*.jsonl"):$(j "$O/_utmos/omni_L1.*.jsonl"):$(j "$O/_lat/omni_L1.*.jsonl"):$(j "$O/_para/omni_L1.*.jsonl")" \
      --sys "SimulS2ST-Omni-L2=$O/_merged/omni_L2.jsonl:$(j "$O/_asr/omni_L2.*.jsonl"):$(j "$O/_utmos/omni_L2.*.jsonl"):$(j "$O/_lat/omni_L2.*.jsonl"):$(j "$O/_para/omni_L2.*.jsonl")" \
      --sys "SimulS2ST-Omni-L3=$O/_merged/omni_L3.jsonl:$(j "$O/_asr/omni_L3.*.jsonl"):$(j "$O/_utmos/omni_L3.*.jsonl"):$(j "$O/_lat/omni_L3.*.jsonl"):$(j "$O/_para/omni_L3.*.jsonl")" \
      --sys "SimulS2ST-Omni-L4=$O/_merged/omni_L4.jsonl:$(j "$O/_asr/omni_L4.*.jsonl"):$(j "$O/_utmos/omni_L4.*.jsonl"):$(j "$O/_lat/omni_L4.*.jsonl"):$(j "$O/_para/omni_L4.*.jsonl")" \
      --source_utmos "$(j "$O/_utmos/source.*.jsonl")" \
      --comet_model "$CK" --out $O/_report/$ds.json > $O/logs/score.$ds.log 2>&1) &
    g=$((g + 1))
  done
  wait
  for ds in realsi_en2zh realsi_zh2en acl6060_en2zh; do grep -A12 "^$ds" $O/logs/score.$ds.log; done
  echo SCORE_DONE
  ;;
esac
