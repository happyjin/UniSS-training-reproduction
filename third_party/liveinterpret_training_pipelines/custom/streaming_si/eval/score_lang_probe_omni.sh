#!/bin/bash
# 给 Omni 声码器链路（DualCodec + VoiceBox + Vocos）的重建音频打分，口径与 score_lang_probe.sh 完全一致，
# 便于和 CosyVoice3 链路横向比较。
# 用法：score_lang_probe_omni.sh <gpu_idx>
set -uo pipefail
G=${1:-0}
L=/fsx/stevehguo/lang_probe
E=/fsx/stevehguo/eval
PY=/opt/venv/bin/python
C=$L/code/custom/streaming_si/eval
export CUDA_VISIBLE_DEVICES=$G
export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$L/code/custom
mkdir -p $L/out/score_omni

$PY - "$L" <<'EOF'
import json, sys
L = sys.argv[1]
rows = [json.loads(l) for l in open(f'{L}/out/vocoder_omni/results.jsonl', encoding='utf-8') if l.strip()]
for mode in ('clone', 'zero'):
    with open(f'{L}/out/score_omni/{mode}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            if r['mode'] != mode or r.get('status') != 'ok':
                continue
            f.write(json.dumps(dict(id=r['id'], wav=r['wav'], tgt_lang=r['lang'],
                                    output_sec=r['output_sec'], source_sec=r['source_sec']),
                               ensure_ascii=False) + '\n')
    print(mode, sum(1 for r in rows if r['mode'] == mode and r.get('status') == 'ok'))
EOF

for s in clone zero; do
  $PY $C/run_asr.py --wav_jsonl $L/out/score_omni/$s.jsonl --out $L/out/score_omni/asr_$s.jsonl \
    --model $E/models/whisper-large-v3 --device cuda:0 --batch 8 2>&1 | tail -1
  $PY $C/run_utmos.py --wav_jsonl $L/out/score_omni/$s.jsonl --out $L/out/score_omni/utmos_$s.jsonl 2>&1 | tail -1
done

# SIM-o：重建 vs 源音频（manifest 需 dataset / talk 字段，这里按语言分组）
$PY - "$L" <<'EOF'
import json, sys
L = sys.argv[1]
man = {json.loads(l)['id']: json.loads(l) for l in open(f'{L}/work/manifest.jsonl', encoding='utf-8')}
for mode in ('clone', 'zero'):
    rows = [json.loads(l) for l in open(f'{L}/out/score_omni/{mode}.jsonl', encoding='utf-8') if l.strip()]
    with open(f'{L}/out/score_omni/para_in_{mode}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(dict(id=r['id'], wav=r['wav'], output_sec=r['output_sec'],
                                    status='ok'), ensure_ascii=False) + '\n')
    with open(f'{L}/out/score_omni/man_{mode}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            m = man[r['id']]
            f.write(json.dumps(dict(id=r['id'], audio=m['audio'], dataset=m['lang'], talk=r['id'],
                                    source_sec=m['source_sec'], reference=m['text'], src_text=m['text'],
                                    tgt_lang=m['lang'], src_lang=m['lang']), ensure_ascii=False) + '\n')
EOF
for mode in clone zero; do
  PYTHONPATH=$E/code/site_para:$L/code/custom $PY $C/run_para_metrics.py \
    --results $L/out/score_omni/para_in_$mode.jsonl --manifest $L/out/score_omni/man_$mode.jsonl \
    --out $L/out/score_omni/simo_$mode.jsonl --wavlm $E/models/wavlm-base-plus-sv \
    --emotion2vec /fsx/stevehguo/tts_bench/models/emotion2vec_plus_large \
    --autopcp $E/models/autopcp/model.config --w2v $E/models/wav2vec2-large-xlsr-53 \
    --device cuda:0 --batch 8 --skip_esim 2>&1 | grep -E "FAIL|/" | tail -1
done
echo OMNI_LANG_SCORE_DONE
