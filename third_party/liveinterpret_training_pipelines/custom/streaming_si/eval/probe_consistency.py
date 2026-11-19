"""一致性诊断（行动 3）：talker 到底「听不听」源语音的韵律？比 delta 更直观。

论文（COLM 2026）用 Expresso 的同文本×7 风格算 "% perfect agreement"：
    Qwen2-Audio 30.5%（内容驱动，充耳不闻） vs Omni 0.6%（真在听）。
我们没有受控风格数据，用**扰动等价**：同一句源音频做变速/变调，看生成语音的韵律跟不跟着动。

    源(原始) ─┐
    源(1.3x 快) ├─ 各自走一遍 S4 推理 ─→ 比较输出韵律特征的位移
    源(0.75x 慢)│      Δout / Δin = 灵敏度；Δout ≈ 0 = 「无反应」
    源(+4 半音) ┘

判读（a0_w2500 = answer-only vs b2_w2500 = audio+answer 的预期对照）：
* answer-only 若大量「无反应」→ talker 没在用源音频的韵律（内容驱动），**正是实验 b 要修的失败模式**
* b2 若无反应率显著下降 → 音频位置确实让 talker 开始「听」

两个子命令：
  prep    : 从 realsi_en2zh 清单取 N 条，生成 {orig,fast,slow,pitch} 四个变体的音频 + manifest
  analyze : 对 infer_s2st 的四组输出算韵律特征，输出灵敏度与无反应率

推理由外部跑（infer_s2st 是成熟 CLI，不复刻）：
  /opt/venv/bin/python -m streaming_si.inference.infer_s2st --model_dir <arm> \
      --input_jsonl /nvme/consistency/<cond>.jsonl --out_dir /nvme/consistency/<arm>/<cond> ...
"""
import argparse
import json
import os

import numpy as np

BASE = '/nvme/consistency'
MANIFEST = '/fsx/stevehguo/eval/work/manifests/realsi_en2zh.jsonl'
KEYS = ['lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']


def prosody_feats(wav, sr=16000):
    """7 项韵律特征（与 paraling_probe.py 同族；一致性诊断看位移，绝对口径不苛求）。"""
    import librosa
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    wav, _ = librosa.effects.trim(wav, top_db=35)
    hop = 160
    f0, _, _ = librosa.pyin(wav, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=hop)
    f0 = np.asarray(f0, dtype=float)
    voiced_f = f0[~np.isnan(f0)]
    lf0 = np.log2(np.maximum(voiced_f, 1e-5)) if voiced_f.size else np.array([0.0])
    rms = librosa.feature.rms(y=wav, frame_length=400, hop_length=hop)[0]
    dur = max(len(wav) / sr, 1e-5)
    vfrac = float((~np.isnan(f0)).mean())
    return {
        'lf0_std': float(np.std(lf0)),
        'final_rise': float(lf0[-5:].mean() - lf0[:5].mean()) if len(lf0) > 10 else 0.0,
        'en_mean': float(np.mean(rms)),
        'en_std': float(np.std(rms)),
        'rate': float((~np.isnan(f0)).sum() / dur / sr * sr),   # 每秒有声帧数（近似语速）
        'pause': float((rms < rms.max() * 0.02).mean()),
        'voiced': vfrac,
    }


def prep(args):
    import soundfile as sf
    import librosa
    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()][:args.n]
    for cond in ('orig', 'fast', 'slow', 'pitch'):
        os.makedirs(f'{BASE}/{cond}', exist_ok=True)
    for i, r in enumerate(rows, 1):
        wav, sr = sf.read(r['audio'], dtype='float32', always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        rid = r['id']
        for cond, w in (('orig', wav),
                        ('fast', librosa.effects.time_stretch(wav, rate=1.3)),
                        ('slow', librosa.effects.time_stretch(wav, rate=0.75)),
                        ('pitch', librosa.effects.pitch_shift(wav, sr=sr, n_steps=4))):
            sf.write(f'{BASE}/{cond}/{rid}.wav', w, sr)
        if i % 25 == 0 or i == len(rows):
            print(f'[prep] {i}/{len(rows)}', flush=True)
    for cond in ('orig', 'fast', 'slow', 'pitch'):
        with open(f'{BASE}/{cond}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows:
                d = dict(r)
                d['audio'] = f'{BASE}/{cond}/{r["id"]}.wav'
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'-> {BASE}/{{orig,fast,slow,pitch}}.jsonl（各 {len(rows)} 条）')


def analyze(args):
    import soundfile as sf
    def feats_dir(arm, cond):
        d = f'{BASE}/{arm}/{cond}'
        return {os.path.basename(p)[:-4]: p for p in
                [os.path.join(d, f) for f in os.listdir(d)] if p.endswith('.wav')}
    def feats_dir_src(cond):
        return feats_dir('.', cond)   # prep 生成的源音频：{BASE}/<cond>/<id>.wav
    def read_feats(paths):
        out = {}
        for rid, p in paths.items():
            w, sr = sf.read(p, dtype='float32', always_2d=False)
            out[rid] = prosody_feats(w, sr)
        return out
    # 源（输入）端特征：扰动前后源音频本身的位移，作为灵敏度的分母
    src = {cond: read_feats(feats_dir_src(cond)) for cond in ('orig', 'fast', 'slow', 'pitch')}
    report = {}
    for arm in args.arms.split(','):
        fo = read_feats(feats_dir(arm, 'orig'))
        print(f'\n===== {arm}：输出韵律对源扰动的灵敏度（n={len(fo)}）=====')
        print(f'{"扰动":<6}' + ''.join(f'{k:>11}' for k in KEYS) + f'{"无反应率":>10}')
        rep = {}
        for cond in ('fast', 'slow', 'pitch'):
            fc = read_feats(feats_dir(arm, cond))
            common = [i for i in fo if i in fc and i in src['orig'] and i in src[cond]]
            # 灵敏度 = 输出位移 / 源（输入）位移。
            # 旧版 din 误用了输出端的 fc-fo，与 dout 相同，比值恒为 1.0（报告作废）。
            # pitch 只动 F0 类、变速只动时长类，其余特征 din≈0 会被跳过，读表时注意。
            ratios, dead = {k: [] for k in KEYS}, []
            for i in common:
                per = []
                for k in KEYS:
                    din = abs(src[cond][i][k] - src['orig'][i][k])
                    if din < 1e-6:
                        continue
                    dout = abs(fc[i][k] - fo[i][k])
                    r_ = min(dout / max(din, 1e-9), 10)
                    ratios[k].append(r_)
                    per.append(r_)
                if per and float(np.median(per)) < 0.1:
                    dead.append(i)
            med = {k: (float(np.median(v)) if v else float('nan')) for k, v in ratios.items()}
            dead_rate = len(dead) / max(len(common), 1)
            rep[cond] = dict(median_ratio=med, dead_rate=round(dead_rate, 4), n=len(common))
            print(f'{cond:<6}' + ''.join(f'{med[k]:>11.2f}' for k in KEYS) + f'{dead_rate:>10.1%}')
        report[arm] = rep
    if ',' in args.arms:
        print('\n===== 对照（b2 应显著低于 a0：talker 开始「听」）=====')
        for cond in ('fast', 'slow', 'pitch'):
            a0 = report.get('a0_w2500', {}).get(cond, {}).get('dead_rate')
            b2 = report.get('b2_w2500', {}).get(cond, {}).get('dead_rate')
            if a0 is not None and b2 is not None:
                print(f'  {cond:<6} 无反应率 a0={a0:.1%}  b2={b2:.1%}  '
                      f'{"b2 更敏感（正向）" if b2 < a0 else "b2 一样聋（负向信号）"}')
    dst = f'{BASE}/consistency_report.json'
    json.dump(report, open(dst, 'w'), indent=2, ensure_ascii=False)
    print(f'-> {dst}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('prep')
    p1.add_argument('--manifest', default=MANIFEST)
    p1.add_argument('--n', type=int, default=100)
    p2 = sub.add_parser('analyze')
    p2.add_argument('--arms', default='a0_w2500,b2_w2500')
    args = ap.parse_args()
    (prep if args.cmd == 'prep' else analyze)(args)


if __name__ == '__main__':
    main()
