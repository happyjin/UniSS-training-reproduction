"""对生成语音做 ASR（Whisper-large-v3），为 ASR-BLEU / ASR-COMET 提供转写。

三个系统（S4 离线、S5 流式、SimulS2ST-Omni 离线）用**同一个** ASR 与同一组解码参数，
语种由清单里的 `tgt_lang` 强制指定（不让模型自己猜，避免中英互判）。
长音频按 Whisper 的 30s 窗口分段：这里统一用 transformers 的 chunked long-form（chunk 30s、stride 5s）。

用法：python run_asr.py --wav_jsonl results.jsonl --out out_asr.jsonl --model <whisper 目录> \
          [--shard i --num_shards n] [--batch 8]
输入 jsonl 需含 `id`、`wav`（生成语音路径）、`tgt_lang`。
"""
import argparse
import json
import os

import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav_jsonl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--manifest', default=None,
                    help='按 id 回查 tgt_lang（结果 jsonl 自身没带该字段时用）')
    args = ap.parse_args()

    lang_of = {}
    if args.manifest:
        for l in open(args.manifest, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                lang_of[r['id']] = r['tgt_lang']

    from transformers import pipeline
    asr = pipeline('automatic-speech-recognition', model=args.model, device=args.device,
                   dtype=torch.float16, chunk_length_s=30, stride_length_s=5)

    rows = [json.loads(l) for l in open(args.wav_jsonl, encoding='utf-8') if l.strip()]
    for r in rows:
        r['tgt_lang'] = r.get('tgt_lang') or lang_of.get(r['id'])
        assert r['tgt_lang'], f'{r["id"]} 缺 tgt_lang，且清单里也查不到'
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    todo = [r for r in rows if r.get('wav') and os.path.isfile(r['wav']) and r.get('output_sec', 1) > 0.05]
    skipped = [r for r in rows if r not in todo]
    out = open(args.out, 'w', encoding='utf-8')
    for r in skipped:                                  # 空输出：转写记空串，仍计入语料级 BLEU
        out.write(json.dumps(dict(id=r['id'], asr='', note='empty_or_missing_wav'), ensure_ascii=False) + '\n')

    for lang in sorted({r['tgt_lang'] for r in todo}):
        sub = [r for r in todo if r['tgt_lang'] == lang]
        for i in range(0, len(sub), args.batch):
            chunk = sub[i:i + args.batch]
            wavs = []
            for r in chunk:
                x, sr = sf.read(r['wav'], dtype='float32', always_2d=False)
                wavs.append({'raw': x.mean(1) if x.ndim > 1 else x, 'sampling_rate': sr})
            res = asr(wavs, batch_size=len(wavs),
                      generate_kwargs={'language': lang, 'task': 'transcribe', 'num_beams': 1})
            for r, o in zip(chunk, res if isinstance(res, list) else [res]):
                out.write(json.dumps(dict(id=r['id'], asr=o['text'].strip()), ensure_ascii=False) + '\n')
            out.flush()
            print(f'[asr {lang} shard {args.shard}] {min(i + args.batch, len(sub))}/{len(sub)}', flush=True)
    out.close()


if __name__ == '__main__':
    main()
