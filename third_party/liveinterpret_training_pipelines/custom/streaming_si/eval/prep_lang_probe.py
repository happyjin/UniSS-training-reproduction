"""从 FLEURS 抽取测试音频，生成统一清单（id / ms + zh / en 对照）。

FLEURS test.tsv 列：id, file_name, raw_transcription, transcription, 音素, 时长, 性别
输出 manifest：{id, audio（16k 单声道 wav）, text, lang, source_sec}
"""
import argparse
import csv
import json
import os
import random
import subprocess
import tarfile

LANGS = {'id_id': 'id', 'ms_my': 'ms', 'cmn_hans_cn': 'zh', 'en_us': 'en'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fleurs_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_per_lang', type=int, default=40)
    ap.add_argument('--min_sec', type=float, default=3.0)
    ap.add_argument('--max_sec', type=float, default=15.0)
    args = ap.parse_args()

    import soundfile as sf
    wav_dir = os.path.join(args.out_dir, 'wav')
    os.makedirs(wav_dir, exist_ok=True)
    rows = []
    for sub, lang in LANGS.items():
        tsv = os.path.join(args.fleurs_root, 'data', sub, 'test.tsv')
        tgz = os.path.join(args.fleurs_root, 'data', sub, 'audio', 'test.tar.gz')
        if not (os.path.isfile(tsv) and os.path.isfile(tgz)):
            print(f'[skip] {sub}: 缺 tsv 或 tar')
            continue
        meta = {}
        with open(tsv, encoding='utf-8') as f:
            for r in csv.reader(f, delimiter='\t'):
                if len(r) >= 4:
                    meta[r[1]] = r[2]            # file_name -> raw_transcription
        ext = os.path.join(args.out_dir, '_ext', sub)
        if not os.path.isdir(ext):
            os.makedirs(ext, exist_ok=True)
            with tarfile.open(tgz) as t:
                t.extractall(ext)                                      # noqa: S202
        files = []
        for root, _, fs in os.walk(ext):
            for fn in fs:
                if fn.endswith('.wav') and fn in meta:
                    files.append(os.path.join(root, fn))
        files.sort()
        random.Random(0).shuffle(files)
        picked = 0
        for p in files:
            if picked >= args.n_per_lang:
                break
            try:
                info = sf.info(p)
            except Exception:
                continue
            if not (args.min_sec <= info.duration <= args.max_sec):
                continue
            uid = f'{lang}_{picked:03d}'
            dst = os.path.join(wav_dir, f'{uid}.wav')
            # 统一转 16k 单声道（flow prompt 另需 24k，推理时现场重采样）
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', p,
                            '-ac', '1', '-ar', '16000', dst], check=True)
            d = sf.info(dst).duration
            rows.append(dict(id=uid, audio=os.path.abspath(dst), text=meta[os.path.basename(p)],
                             lang=lang, source_sec=round(d, 3)))
            picked += 1
        print(f'{sub} -> {lang}: {picked} 条')

    out = os.path.join(args.out_dir, 'manifest.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'共 {len(rows)} 条 -> {out}')


if __name__ == '__main__':
    main()
