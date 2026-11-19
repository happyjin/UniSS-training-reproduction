"""风格探针的并行版：按 (层, 折) 并行，结果与 probe_voxtral_style.py probe 逐位一致。

为什么快：原版是 32 层 for 循环内嵌 4 折 LOSO，**单个进程串行**跑 128 次
LogisticRegression(saga)，每次约 1.6 分钟 → 约 3.5 小时。而分类本身是
"层 × 折"互相独立的凸优化问题，Pod 有 192 核，可并行执行。

口径完全不变（否则不可比）：
* 逐折 StandardScaler().fit(X[tr]) 只在训练折上拟合（与原版同一句）；
* LogisticRegression(C=0.1, l2, saga, max_iter=500, tol=1e-3,
  class_weight='balanced', random_state=0) 参数逐项相同；
* LeaveOneGroupOut 切分相同（顺序由 logo.split 决定）；
* acc 取各折 mean/std。

用法：
  python probe_style_fast.py --cache /nvme/vox_style_s4 --jobs 16
"""
import argparse
import json
import os

import numpy as np
from joblib import Parallel, delayed


def _fit_one(cache, key, tr, te):
    """单层单折：子进程内自己加载该层特征，避免把 109MB 数组跨进程传输。

    口径与原版（probe_voxtral_style.probe）逐项一致：scaler 只在训练折上 fit；
    LR 参数/种子相同；返回该折准确率。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = np.load(f'{cache}/feats.npz', allow_pickle=True)
    X = z[key].astype(np.float64)
    y, spk = z['labels'], z['speakers']
    sx = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=500,
                           tol=1e-3, class_weight='balanced',
                           random_state=0).fit(sx.transform(X[tr]), y[tr])
    return float((m.predict(sx.transform(X[te])) == y[te]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--jobs', type=int, default=16,
                    help='并行进程数；每任务峰值约 1GB（float64 X 为 109MB），120G 上限下 16 很安全')
    args = ap.parse_args()

    from sklearn.model_selection import LeaveOneGroupOut
    z = np.load(f'{args.cache}/feats.npz', allow_pickle=True)
    y, spk = z['labels'], z['speakers']
    keys = sorted([x for x in z.files if x not in ('labels', 'speakers', 'ids')],
                  key=lambda t: (0, int(t[1:])) if t[0] == 'L' else (1, 0))
    logo = LeaveOneGroupOut()
    splits = list(logo.split(np.zeros(len(y)), y, spk))     # 只依赖 y/spk，切分顺序与原版一致
    print(f'\n===== 并行 LOSO 风格探针（7 类，chance=14.3%）=====', flush=True)
    print(f'n={len(y)}  splits={len(splits)}  jobs={args.jobs}  reps={keys}', flush=True)

    # 128 个任务摊平；子进程内自行加载所在层的 X（float64 109MB），避免父进程
    # 持有 32 份拷贝 + joblib 序列化传大数组。npz 是 zip 结构，子进程按 key 解压
    # 单层开销约 1-2 秒，可接受。
    tasks = [(ki, fi) for ki in range(len(keys)) for fi in range(len(splits))]
    accs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_fit_one)(args.cache, keys[ki], splits[fi][0], splits[fi][1])
        for ki, fi in tasks)

    rows = []
    for ki, k in enumerate(keys):
        a = np.array([accs[i] for i, (kk, _) in enumerate(tasks) if kk == ki])
        rows.append({'rep': k, 'acc': round(float(a.mean()), 4), 'std': round(float(a.std()), 4)})
        print(f'{k:<12}{a.mean():>10.4f}{a.std():>8.4f}', flush=True)

    lay = [r for r in rows if r['rep'].startswith('L')]
    if lay:
        peak = max(lay, key=lambda r: r['acc'])
        n_lay = len(lay)
        top_third = max(1, n_lay * 2 // 3)
        print(f'\n峰值层 = {peak["rep"]}（acc={peak["acc"]:.1%}），位于第 {peak["rep"][1:]}/{n_lay} 层',
              flush=True)
        print(f'论文对照：Whisper 系四模型峰值在 top third（82~85%）；'
              f'本模型{"在晚 1/3（与论文一致）" if int(peak["rep"][1:]) >= top_third else "不在晚 1/3"}',
              flush=True)
    dst = f'{args.cache}/style_probe.json'
    json.dump(rows, open(dst, 'w'), indent=2)
    print(f'-> {dst}', flush=True)


if __name__ == '__main__':
    main()
