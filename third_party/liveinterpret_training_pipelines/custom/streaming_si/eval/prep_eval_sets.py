"""把 RealSI 与 ACL 60/60 eval 切成句级评测清单（供三个系统共用）。

RealSI：`data/{en2zh,zh2en}/wav/<vid>.wav` 是 5 分钟长音频，`json/<vid>.json` 带句级切分
（`start_time` / `end_time` 毫秒、`src_text` 源文、`trg_text` 参考译文）。按 segment 切出句级 wav。
ACL 60/60：`eval/segmented_wavs/gold/sent_N.wav`（N=1..416）与 `eval/text/txt/ACL.6060.eval.en-xx.{en,zh}.txt`
的第 N 行一一对应（gold 是人工切分，数量与参考行数相同）。只有 en→xx 方向。

输出：`<out>/manifests/{realsi_en2zh,realsi_zh2en,acl6060_en2zh}.jsonl`，每行
`{id, audio, tgt_lang, src_lang, src_text, reference, source_sec, talk}`，以及切好的 16kHz 单声道 wav。

用法：python prep_eval_sets.py <ds_root> <out_root>
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 16000


def load_mono16k(path):
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        import torch
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def prep_realsi(ds_root, out_root):
    rows_by_dir = {}
    for direction in ('en2zh', 'zh2en'):
        src_lang, tgt_lang = direction.split('2')
        base = os.path.join(ds_root, 'realsi', direction)
        wav_dir = os.path.join(out_root, 'audio', f'realsi_{direction}')
        os.makedirs(wav_dir, exist_ok=True)
        rows = []
        for jf in sorted(os.listdir(os.path.join(base, 'json'))):
            meta = json.load(open(os.path.join(base, 'json', jf), encoding='utf-8'))
            vid = meta['vid']
            full = load_mono16k(os.path.join(base, 'wav', f'{vid}.wav'))
            for i, seg in enumerate(meta['segment']):
                a, b = int(seg['start_time'] / 1000 * SR), int(seg['end_time'] / 1000 * SR)
                clip = full[a:b]
                if clip.size < SR // 10:                      # < 0.1s 视为无效切分
                    print(f'[skip] {vid} seg{i} 仅 {clip.size / SR:.3f}s', file=sys.stderr)
                    continue
                uid = f'{vid}_{i:04d}'
                path = os.path.join(wav_dir, f'{uid}.wav')
                sf.write(path, clip, SR)
                rows.append(dict(id=uid, audio=os.path.abspath(path), tgt_lang=tgt_lang, src_lang=src_lang,
                                 src_text=seg['src_text'].strip(), reference=seg['trg_text'].strip(),
                                 source_sec=round(clip.size / SR, 3), talk=vid))
        rows_by_dir[f'realsi_{direction}'] = rows
    return rows_by_dir


def acl_talks(base, n_wav):
    """sent_1..sent_N 按顺序归属到 5 个 talk：从 XML 的 <doc docid=...> 取每个 talk 的句数。

    XML 里 doc 的出现顺序与 `FILE_ORDER` 一致，句数合计等于 gold 切分数。
    拿不到 XML 时退回单一 talk（随机配对基线会因此偏乐观，见 eval/README）。
    """
    import re
    p = os.path.join(base, 'text', 'xml', 'ACL.6060.eval.en-xx.en.xml')
    if not os.path.isfile(p):
        print(f'[warn] 缺 {p}，talk 退化为单一值', file=sys.stderr)
        return ['acl6060_eval'] * n_wav
    s = open(p, encoding='utf-8').read()
    talks = []
    for d in re.split(r'<doc', s)[1:]:
        m = re.search(r'docid=.([^"\']+)', d)
        talks += [m.group(1) if m else f'doc{len(talks)}'] * len(re.findall(r'<seg', d))
    assert len(talks) == n_wav, f'XML 句数 {len(talks)} 与 gold 音频数 {n_wav} 不一致'
    return talks


def prep_acl(ds_root, out_root):
    base = os.path.join(ds_root, 'acl6060', '2', 'acl_6060', 'eval')
    txt = os.path.join(base, 'text', 'txt')
    en = [l.strip() for l in open(os.path.join(txt, 'ACL.6060.eval.en-xx.en.txt'), encoding='utf-8')]
    zh = [l.strip() for l in open(os.path.join(txt, 'ACL.6060.eval.en-xx.zh.txt'), encoding='utf-8')]
    gold = os.path.join(base, 'segmented_wavs', 'gold')
    n_wav = len([f for f in os.listdir(gold) if f.endswith('.wav')])
    assert len(en) == len(zh) == n_wav, f'行数/音频数不一致: en {len(en)} zh {len(zh)} wav {n_wav}'
    talks = acl_talks(base, n_wav)
    wav_dir = os.path.join(out_root, 'audio', 'acl6060_en2zh')
    os.makedirs(wav_dir, exist_ok=True)
    rows = []
    for i in range(1, n_wav + 1):
        src = os.path.join(gold, f'sent_{i}.wav')
        clip = load_mono16k(src)
        path = os.path.join(wav_dir, f'sent_{i:04d}.wav')
        sf.write(path, clip, SR)                              # 统一成 16k 单声道，三个系统同一份输入
        rows.append(dict(id=f'sent_{i:04d}', audio=os.path.abspath(path), tgt_lang='zh', src_lang='en',
                         src_text=en[i - 1], reference=zh[i - 1],
                         source_sec=round(clip.size / SR, 3), talk=talks[i - 1]))
    return {'acl6060_en2zh': rows}


def main():
    ds_root, out_root = sys.argv[1], sys.argv[2]
    man_dir = os.path.join(out_root, 'manifests')
    os.makedirs(man_dir, exist_ok=True)
    allrows = {}
    allrows.update(prep_realsi(ds_root, out_root))
    allrows.update(prep_acl(ds_root, out_root))
    for name, rows in allrows.items():
        with open(os.path.join(man_dir, f'{name}.jsonl'), 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        secs = [r['source_sec'] for r in rows]
        print(f'{name}: {len(rows)} 条，总时长 {sum(secs) / 60:.1f} 分钟，'
              f'中位 {sorted(secs)[len(secs) // 2]:.1f}s，最长 {max(secs):.1f}s')


if __name__ == '__main__':
    main()
