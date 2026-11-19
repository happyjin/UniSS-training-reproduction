"""UTMOS（UTMOS22 strong，SpeechMOS 的 torch.hub 实现）逐句打分。

* 模型：`tarepan/SpeechMOS:v1.2.0` 的 `utmos22_strong`（VoiceMOS 2022 冠军系统的复现），预测 1～5 的 MOS。
* 输入统一重采样到 16kHz 单声道；长于 30s 的只取前 30s（模型训练分布为短句）。
* 可对任意 jsonl 打分：需含 `id` 与音频路径字段（`--wav_key`，默认 `wav`；源音频用 `audio`）。
* UTMOS 在英文 BVCC 数据上训练，对中文语音同样常用但绝对值的可比性弱于英文，
  建议主要做同语种的系统间比较。

用法：python run_utmos.py --wav_jsonl r.jsonl --out utmos.jsonl [--wav_key wav] [--shard i --num_shards n]
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000
MAX_SEC = 30.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wav_jsonl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wav_key', default='wav')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    args = ap.parse_args()

    model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True).to(args.device).eval()
    rows = [json.loads(l) for l in open(args.wav_jsonl, encoding='utf-8') if l.strip()]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    out = open(args.out, 'w', encoding='utf-8')
    for n, r in enumerate(rows, 1):
        p = r.get(args.wav_key)
        rec = dict(id=r['id'])
        if not p or not os.path.isfile(p):
            rec['utmos'] = None
        else:
            x, sr = sf.read(p, dtype='float32', always_2d=True)
            x = torch.from_numpy(x.mean(1))
            if sr != SR:
                x = torchaudio.functional.resample(x, sr, SR)
            x = x[:int(MAX_SEC * SR)]
            if x.numel() < SR // 4:                       # < 0.25s：太短，不打分
                rec['utmos'] = None
            else:
                with torch.inference_mode():
                    rec['utmos'] = round(float(model(x.unsqueeze(0).to(args.device), SR).detach().cpu()), 4)
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if n % 100 == 0:
            out.flush()
            print(f'[utmos shard {args.shard}] {n}/{len(rows)}', flush=True)
    out.close()


if __name__ == '__main__':
    main()
