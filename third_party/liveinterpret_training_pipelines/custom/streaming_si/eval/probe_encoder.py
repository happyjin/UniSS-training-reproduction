"""验证二：冻结的 Voxtral 音频编码器对各语言的内容承载能力。

S1-S5 全程冻结 `audio_encoder`（Voxtral realtime，50Hz -> adapter -> 12.5Hz / 3072 维）。
如果它对印尼语/马来语的编码质量差，后面怎么补训 thinker/talker 都救不回来——这是扩语言的硬前提。

编码器输出是连续隐状态，没有标签可比。这里用「表征是否退化」间接判断：编码器听不懂某门语言时，
典型表现是输出趋于常量或塌缩到低维子空间。四个指标都不需要训练任何下游模块：

* `self_sim`  ：同一条音频前半段 / 后半段的均值表征余弦。**越低越好**（随内容变化）。塌缩时趋近 1。
* `cross_sim` ：同语言内随机 8 条其它音频的均值表征余弦。**越低越好**（句子间可区分）。
* `rank_ratio`：帧级表征矩阵保留 99% 奇异值能量所需的秩 / 总维数。**越高越好**（没退化到低维）。
* `frame_std` ：帧间标准差的均值。**越高越好**（输出不是常量）。

中英是「已验证可用」的参照：id / ms 的四项与 zh / en 同量级即说明编码器可用。
注意这四项是**必要条件而非充分条件**——通过只能排除"编码器完全听不懂"，不能证明语义质量。

用法：python probe_encoder.py --manifest m.jsonl --out out.jsonl --voxtral <Voxtral 目录> [--device cuda:0]
"""
import argparse
import collections
import json
import os
import random

import numpy as np
import soundfile as sf
import torch


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def eff_rank(m, energy=0.99):
    s = np.linalg.svd(m - m.mean(0, keepdims=True), compute_uv=False)
    c = np.cumsum(s ** 2) / max(1e-12, float((s ** 2).sum()))
    return int(np.searchsorted(c, energy) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--voxtral', help='原始 Voxtral 目录（含 config.json + 权重）')
    ap.add_argument('--si_ckpt', help='S1-S4 checkpoint 目录：按结构建壳后灌入其 audio_encoder.* 权重，'
                                      '即实际训练中使用的那一份编码器（推荐）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    assert args.voxtral or args.si_ckpt, '需要 --voxtral 或 --si_ckpt'

    from streaming_si.model.base_model import VoxtralAudioBackend
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor

    if args.si_ckpt:
        import glob

        from safetensors.torch import load_file
        backend = VoxtralAudioBackend.from_config_dir(args.si_ckpt)
        sd = {}
        for p in sorted(glob.glob(os.path.join(args.si_ckpt, '*.safetensors'))):
            for k, v in load_file(p).items():
                if k.startswith('audio_encoder.'):
                    sd[k[len('audio_encoder.'):]] = v
        missing, unexpected = backend.load_state_dict(sd, strict=False)
        assert not unexpected, f'checkpoint 里有多余张量: {list(unexpected)[:5]}'
        assert not missing, f'checkpoint 缺张量: {list(missing)[:5]}'
        backend = backend.to(device=args.device, dtype=torch.bfloat16).eval()
        processor = VoxtralFeatureExtractor.from_pretrained(args.si_ckpt)
        print(f'从 {args.si_ckpt} 载入 {len(sd)} 个 audio_encoder 张量', flush=True)
    else:
        backend, processor = VoxtralAudioBackend.load(args.voxtral, device=args.device,
                                                      dtype=torch.bfloat16)
        assert backend is not None, f'加载 Voxtral 失败: {args.voxtral}'
    print(f'voxtral: hidden={backend.hidden_size} frame_rate={backend.frame_rate}Hz', flush=True)

    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    pooled, recs, halves = collections.defaultdict(list), [], {}
    for n, r in enumerate(rows, 1):
        wav16, sr = sf.read(r['audio'], dtype='float32')
        assert sr == 16000, f'{r["id"]} 不是 16k'
        mel = processor(wav16, sampling_rate=sr)['input_features']   # (1, 128, T) log-mel
        mel = torch.as_tensor(mel, dtype=torch.bfloat16, device=args.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        h = backend.encode_audio(mel)[0].float().cpu().numpy()      # (n_frames, hidden)
        half = max(1, h.shape[0] // 2)
        rec = dict(id=r['id'], lang=r['lang'], source_sec=r['source_sec'],
                   n_frames=int(h.shape[0]), dim=int(h.shape[1]),
                   self_sim=round(cos(h[:half].mean(0), h[half:].mean(0)), 4),
                   rank_ratio=round(eff_rank(h) / h.shape[1], 4),
                   frame_std=round(float(h.std(0).mean()), 4))
        halves[r['id']] = (h[:half].mean(0), h[half:].mean(0))
        pooled[r['lang']].append((r['id'], h.mean(0)))
        recs.append(rec)
        if n % 20 == 0 or n == len(rows):
            print(f'[encoder] {n}/{len(rows)}', flush=True)

    # 去均值（centered）：音频编码器输出通常有很强的公共 DC 分量，原始余弦恒在 0.99 以上、
    # 看不出区分度。减掉全体样本（跨语言）的均值方向后再算余弦，才反映真实可分性。
    allvec = np.stack([v for items in pooled.values() for _, v in items])
    g = allvec.mean(0)
    emb_c = {i: v - g for items in pooled.values() for i, v in items}
    rnd = random.Random(0)
    for lang, items in pooled.items():
        ids = sorted(i for i, _ in items)
        for rec in recs:
            if rec['lang'] != lang:
                continue
            others = [i for i in ids if i != rec['id']]
            picks = rnd.sample(others, min(8, len(others)))
            emb = dict(items)
            rec['cross_sim'] = round(float(np.mean([cos(emb[rec['id']], emb[p]) for p in picks])), 4)
            rec['cross_sim_c'] = round(float(np.mean([cos(emb_c[rec['id']], emb_c[p]) for p in picks])), 4)
            a, b = halves[rec['id']]
            rec['self_sim_c'] = round(cos(a - g, b - g), 4)

    with open(args.out, 'w', encoding='utf-8') as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    cols = ('self_sim', 'self_sim_c', 'cross_sim', 'cross_sim_c', 'rank_ratio', 'frame_std')
    print(f'\n{"语言":<6}{"条数":>5}' + ''.join(f'{c:>12}' for c in cols))
    for lang in ('zh', 'en', 'id', 'ms'):
        grp = [r for r in recs if r['lang'] == lang]
        if not grp:
            continue
        vals = ''.join(f'{float(np.mean([r[c] for r in grp])):>12.4f}' for c in cols)
        print(f'{lang:<6}{len(grp):>5}{vals}')
    print('\n带 _c 的是去全体均值后的余弦（越低越可分）；rank_ratio / frame_std 越高越好。')
    print(f'-> {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
