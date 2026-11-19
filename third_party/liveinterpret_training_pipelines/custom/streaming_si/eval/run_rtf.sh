#!/bin/bash
# 单卡独占复测 RTF：从合并清单里按时长分层抽 30 句，三个系统依次单进程跑，互不并发。
set -uo pipefail
E=/fsx/stevehguo/eval
PY=/opt/venv/bin/python
export COSYVOICE_REPO=$E/code/CosyVoice
export COSYVOICE_SPEECH_TOKENIZER=$E/models/cosyvoice3/speech_tokenizer_v3.onnx
export HF_HOME=$E/hf PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
O=$E/out/_rtf; mkdir -p $O

$PY - <<'EOF'
import json
rows = [json.loads(l) for l in open('/fsx/stevehguo/eval/work/manifests/all.jsonl', encoding='utf-8')]
rows.sort(key=lambda r: r['source_sec'])
pick = [rows[int(i * (len(rows) - 1) / 29)] for i in range(30)]   # 按时长分层取 30 句
with open('/fsx/stevehguo/eval/out/_rtf/subset.jsonl', 'w', encoding='utf-8') as f:
    for r in pick:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('抽样 %d 句，%.1f 分钟，时长 %.1f~%.1fs' % (
    len(pick), sum(r['source_sec'] for r in pick) / 60, pick[0]['source_sec'], pick[-1]['source_sec']))
EOF

M=$O/subset.jsonl
echo "=== S4 单卡 ==="
PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s2st \
  --model_dir $E/models/s4_ck --input_jsonl $M --out_dir $O/s4 \
  --cosyvoice_repo $E/code/CosyVoice --device cuda:0 > $O/s4.log 2>&1
echo "rc=$?"
echo "=== S5 单卡 ==="
PYTHONPATH=$E/code/custom $PY -m streaming_si.inference.infer_s5_streaming \
  --model_dir $E/models/s4_ck --adapter $E/models/s5_adapter --input_jsonl $M \
  --out_dir $O/s5 --cosyvoice_repo $E/code/CosyVoice --device cuda:0 > $O/s5.log 2>&1
echo "rc=$?"
echo "=== Omni 单卡 ==="
R=$E/code/SimulS2ST-Omni-main
PYTHONPATH=$E/code/site_omni:$R:$R/external/SimulEval:$R/src/train:$E/code/custom \
  $PY $E/code/custom/streaming_si/eval/run_omni_offline.py --manifest $M --shard 0 --num_shards 1 \
  --out_dir $O/omni --model_root $E/models/SimulS2ST-Omni --device cuda:0 > $O/omni.log 2>&1
echo "rc=$?"

$PY - <<'EOF'
import json, glob
O = '/fsx/stevehguo/eval/out/_rtf'
src = {json.loads(l)['id']: json.loads(l)['source_sec'] for l in open(f'{O}/subset.jsonl', encoding='utf-8')}
print('\n单卡独占 RTF（30 句分层抽样）')
print('系统'.ljust(16), 'RTF(语料)'.rjust(10), 'RTF 中位'.rjust(10), '平均耗时'.rjust(10))
for name, pat in (('S4-offline', f'{O}/s4/results.jsonl'), ('S5-streaming', f'{O}/s5/results.jsonl'),
                  ('SimulS2ST-Omni', f'{O}/omni/results.00.jsonl')):
    fs = glob.glob(pat)
    if not fs:
        print(name.ljust(16), '（无结果）'); continue
    rows = [json.loads(l) for l in open(fs[0], encoding='utf-8') if l.strip()]
    w = [(r['wall_sec'], src[r['id']]) for r in rows if r.get('wall_sec')]
    rtfs = sorted(a / b for a, b in w)
    print(name.ljust(16), f'{sum(a for a,_ in w)/sum(b for _,b in w):.3f}'.rjust(10),
          f'{rtfs[len(rtfs)//2]:.3f}'.rjust(10), f'{sum(a for a,_ in w)/len(w):.2f}s'.rjust(10))
EOF
echo RTF_DONE
