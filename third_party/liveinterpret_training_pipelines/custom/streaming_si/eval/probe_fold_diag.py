"""探针折划分诊断：随机 KFold 是否因说话人泄漏抬高了层排序？

动机（COLM 2026 "Heard but Not Heeded" 的方法学）：该论文用 **leave-one-speaker-out**
划折，而 `paraling_probe.py` / `probe_omni_audio_tower.py` 的 `_cv_r2` 用的是
`KFold(5, shuffle)`。若同一说话人的样本同时落在训练折与测试折，探针可以靠"认出这个人"
抬高 R²；而说话人身份在**浅层更线性可读**，所以可能把 `Voxtral L4 > L32` 这个排序也抬出来。
我们对实验 c（L4 旁路）的立项依据正是这个排序。

本脚本不重新提取表示（直接复用 `out/extract_*.npz` 的 47 个表示），只做两件事：

  1. 用 CAM++ 说话人向量对探针样本聚类，**先量化泄漏风险本身**
     —— 若几乎每条一个说话人（CommonVoice 风格），泄漏在结构上不可能发生，结论可直接采信；
  2. 若存在多条同说话人的簇，用 `GroupKFold`（簇作组）重算 R²，与随机 KFold 并排对比。

用法（Pod 内，1 卡即可，CAM++ 走 CPU onnxruntime）：
  /opt/venv/bin/python probe_fold_diag.py --direction eng->cmn
"""
import argparse
import glob
import io
import json
import os
import pickle

import numpy as np

W = '/fsx/si_s3_work/paraling/out'
P2 = '/nvme/speech_model_backup/stevehguo/data_fortrain/open_data_0803_fortrain'
CAMPP = '/nvme/zeyangsong/projects/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx'
# 与 paralinguistic_preservation.md §5.1 一致：7 项不含 lf0_mean（它主要随说话人变化）
KEYS = ['lf0_mean', 'lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']
PROSODY = ['lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']


def tar_read(shard_dir, ref):
    """'src-train-00000.tar::offset:size' -> 音频字节。

    偏移直接指向 tar 内的**音频数据**（不是成员头），与 `probe_omni_audio_tower.read_tar_wav`
    / `paraling_probe.py` 同口径——按字节范围取出即可，不必解 tar。
    """
    name, loc = ref.split('::', 1)
    off, ln = (int(x) for x in loc.split(':'))
    with open(os.path.join(shard_dir, name), 'rb') as f:
        f.seek(off)
        return f.read(ln)


def spk_embeds(ids, direction):
    """CAM++ 说话人向量（80 维 fbank -> 192 维 embedding），按 ids 顺序返回。"""
    import onnxruntime as ort
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi

    # id -> source_wav 引用
    ref = {}
    for f in sorted(glob.glob(f'{P2}/machine_*_shard/p2_fsq_v3/*.parquet')):
        import pyarrow.parquet as pq
        t = pq.read_table(f, columns=['id', 'source_wav']).to_pydict()
        shard = os.path.dirname(os.path.dirname(f)) + '/p2_source_wav'
        for i, s in zip(t['id'], t['source_wav']):
            if i in ids and i not in ref:
                ref[i] = (shard, s)
        if len(ref) >= len(ids):
            break
    miss = [i for i in ids if i not in ref]
    if miss:
        raise SystemExit(f'{len(miss)} 条找不到音频，例如 {miss[:3]}')

    sess = ort.InferenceSession(CAMPP, providers=['CPUExecutionProvider'])
    out = np.zeros((len(ids), 192), dtype=np.float32)
    for n, i in enumerate(ids, 1):
        shard, s = ref[i]
        wav, sr = sf.read(io.BytesIO(tar_read(shard, s)), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        t = torch.from_numpy(wav).unsqueeze(0)
        feat = kaldi.fbank(t, num_mel_bins=80, dither=0, sample_frequency=sr)
        feat = (feat - feat.mean(0, keepdim=True)).unsqueeze(0).numpy()
        out[n - 1] = sess.run(None, {sess.get_inputs()[0].name: feat})[0].reshape(-1)
        if n % 200 == 0 or n == len(ids):
            print(f'  [spk] {n}/{len(ids)}', flush=True)
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    return out


def cluster(emb, thr=0.6):
    """余弦 >= thr 视为同一说话人（CAM++ 常用阈值 0.6~0.7），单链聚合。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    d = pdist(emb, metric='cosine')
    return fcluster(linkage(d, method='single'), t=1 - thr, criterion='distance')


def r2(X, Y, groups=None, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=np.float64)
    pred = np.zeros_like(Y)
    split = (GroupKFold(5).split(X, Y, groups) if groups is not None
             else KFold(5, shuffle=True, random_state=seed).split(X))
    for tr, te in split:
        sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
        m = RidgeCV(alphas=np.logspace(-1, 6, 15), alpha_per_target=True).fit(
            sx.transform(X[tr]), sy.transform(Y[tr]))
        pred[te] = sy.inverse_transform(m.predict(sx.transform(X[te])))
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', default='eng->cmn', choices=['eng->cmn', 'cmn->eng'])
    ap.add_argument('--thr', type=float, default=0.6)
    ap.add_argument('--cache', default='/nvme/probe_spk')
    ap.add_argument('--reps', default='vox4,vox8,vox12,vox16,vox20,vox24,vox28,vox32,'
                                      'adapter,audio_proj,aud_L4,aud_L32,tgt_L0')
    args = ap.parse_args()
    tag = args.direction.replace('->', '2')
    os.makedirs(args.cache, exist_ok=True)

    pro = pickle.load(open(f'{W}/prosody.pkl', 'rb'))[args.direction]
    za = np.load(f'{W}/extract_{tag}.npz', allow_pickle=True)
    ok = za['ok']
    ids = [r['id'] for r in pro['rows'][:len(ok)]]
    ids_ok = [i for i, k in zip(ids, ok) if k]
    # 与 paraling_probe.cmd_probe 一致：Y = 目标韵律对文本特征回归后的残差
    # （预测"可迁移"的部分；用原始 src 会把任务变成自重建，结论不可比）
    n = len(ok)
    tgt = np.asarray(pro['tgt'][:n][ok], dtype=np.float64)
    txt = np.asarray(pro['txt'][:n][ok], dtype=np.float64)
    Zc = np.column_stack([np.ones(len(txt)), txt])
    Yfull = tgt[:, [KEYS.index(k) for k in PROSODY]]
    Y = Yfull - Zc @ np.linalg.lstsq(Zc, Yfull, rcond=None)[0]

    cache = f'{args.cache}/spk_{tag}.npy'
    if os.path.isfile(cache):
        emb = np.load(cache)
        print(f'  说话人向量读缓存 {cache}  {emb.shape}')
    else:
        emb = spk_embeds(ids_ok, args.direction)
        np.save(cache, emb)
    lab = cluster(emb, args.thr)
    n, k = len(lab), len(set(lab))
    sizes = np.bincount(lab)
    sizes = sizes[sizes > 0]
    print(f'\n===== 说话人聚类（CAM++，余弦阈值 {args.thr}）=====')
    print(f'  样本 {n}  簇 {k}  平均每簇 {n / k:.2f} 条  最大簇 {sizes.max()} 条')
    print(f'  size>1 的簇: {(sizes > 1).sum()} 个，覆盖 {sizes[sizes > 1].sum()} 条'
          f'（{100 * sizes[sizes > 1].sum() / n:.1f}%）')
    if k >= n * 0.95:
        print('  -> 几乎每条一个说话人：**随机 KFold 的说话人泄漏在结构上不可能发生**')

    print(f'\n===== 7 项韵律 R²：随机 KFold vs GroupKFold（说话人分组）=====')
    print(f'{"表示":<14}{"KFold":>9}{"GroupKFold":>12}{"差":>9}')
    rows = []
    for name in args.reps.split(','):
        if name not in za.files:
            continue
        a = r2(za[name], Y).mean()
        b = r2(za[name], Y, groups=lab).mean()
        rows.append((name, a, b))
        print(f'{name:<14}{a:>9.3f}{b:>12.3f}{b - a:>+9.3f}', flush=True)
    if rows:
        ka = sorted(rows, key=lambda t: -t[1])[0][0]
        kb = sorted(rows, key=lambda t: -t[2])[0][0]
        print(f'\n  KFold 最高层 = {ka}   GroupKFold 最高层 = {kb}'
              f'   -> 排序{"**翻转**，既有结论需重估" if ka != kb else "一致，结论稳健"}')
        json.dump({'direction': args.direction, 'n': int(n), 'clusters': int(k),
                   'rows': [{'rep': r, 'kfold': float(a), 'groupkfold': float(b)} for r, a, b in rows]},
                  open(f'{args.cache}/fold_diag_{tag}.json', 'w'), ensure_ascii=False, indent=2)
        print(f'  -> {args.cache}/fold_diag_{tag}.json')


if __name__ == '__main__':
    main()
