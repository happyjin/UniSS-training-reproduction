#!/bin/bash
# 对 3 条不同长度的句子跑 S4 耗时分解（单卡独占）。
set -uo pipefail
E=/fsx/stevehguo/eval
PY=/opt/venv/bin/python
export COSYVOICE_REPO=$E/code/CosyVoice
export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p $E/out/_prof

$PY - <<'EOF'
import json
rows = [json.loads(l) for l in open('/fsx/stevehguo/eval/work/manifests/all.jsonl', encoding='utf-8')]
rows = [r for r in rows if r['tgt_lang'] == 'zh']
rows.sort(key=lambda r: r['source_sec'])
pick = [min(rows, key=lambda r: abs(r['source_sec'] - t)) for t in (4, 8, 15)]
json.dump([(r['audio'], r['tgt_lang'], r['source_sec']) for r in pick],
          open('/fsx/stevehguo/eval/out/_prof/pick.json', 'w'))
for a, l, s in [(r['audio'], r['tgt_lang'], r['source_sec']) for r in pick]:
    print(f'{s:.1f}s {a}')
EOF

for i in 0 1 2; do
  read -r A L < <($PY -c "
import json; p=json.load(open('$E/out/_prof/pick.json'))[$i]; print(p[0], p[1])")
  PYTHONPATH=$E/code/custom $PY $E/code/custom/streaming_si/eval/profile_s4.py \
    $E/models/s4_ck "$A" "$L" $E/code/CosyVoice $E/out/_prof/s4_$i.json 2>&1 \
    | grep -vE "Warning|warn|INFO|input frame rate|^$"
done
echo PROF_DONE
