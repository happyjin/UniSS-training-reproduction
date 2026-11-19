"""MLP（非线性）探针 vs 线性 Ridge 探针 —— 在关键边界层上判断「信息丢失」还是「线性不可读」。

背景：`paralinguistic_preservation.md` §5.1 观察到 R² 从 Voxtral L4 0.286 一路降到
adapter 0.244 / audio_proj 0.216，并把下降解读为「adapter 12.5Hz 下采样丢了韵律信息」。
但 COLM 2026 "Heard but Not Heeded" §5.3 的 CKA-探针分离现象说明还存在另一种解释：
**表示几何重组后线性探针读不出来，信息其实还在**（他们用 MLP 探针在 19 个边界层
验证线性探针未低估，差值 −4.1~+0.4）。

判读（决定实验 c 要不要做 120 行的 L4 旁路）：
* MLP 在 adapter / audio_proj 上恢复到 ≈ vox4 的水平（~0.28）→ 信息还在、只是线性
  不可读 → 实验 c 可以简化成「12.5Hz 表示后接非线性 MLP」，省掉 50Hz→12.5Hz 对齐；
* MLP 仍 ~0.22 → 确实丢了 → 才值得做 L4 旁路。

口径：与既有线性结果**逐位可比** —— 同一 npz 表示、同一 prosody.pkl 目标、同一
KFold(5, shuffle, seed=0) 折、同一 StandardScaler；只把 RidgeCV 换成 MLPRegressor
（hidden 256，L2 α=1e-3，early_stopping，max_iter=300 —— 对齐论文附录 C 的 MLPClassifier
超参，只是任务是回归）。

用法（CPU 即可，Pod 内）：
  /opt/venv/bin/python probe_mlp.py            # 默认 eng->cmn
"""
import argparse
import json
import os
import pickle

import numpy as np

W = '/fsx/si_s3_work/paraling/out'
KEYS = ['lf0_mean', 'lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']
PROSODY = ['lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']
# 关键边界：浅层峰值 / 深层 / 两个投影 / thinker 两侧
DEFAULT_REPS = 'vox4,vox8,vox12,vox32,adapter,audio_proj,aud_L4,aud_L32,tgt_L0,tgt_L32'


def _ols_resid(y, Z):
    """与 paraling_probe.py 逐字相同：目标韵律去掉文本可解释部分。"""
    Z = np.column_stack([np.ones(len(Z)), Z])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def oof(pred_fn, X, Y, seed=0):
    """与 paraling_probe._cv_r2 相同的折与标准化，只是回归器由调用方给出。"""
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
        pred[te] = sy.inverse_transform(pred_fn(sx.transform(X[tr]), sy.transform(Y[tr]),
                                                sx.transform(X[te])))
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', default='eng->cmn')
    ap.add_argument('--reps', default=DEFAULT_REPS)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    tag = args.direction.replace('->', '2')

    pro = pickle.load(open(f'{W}/prosody.pkl', 'rb'))[args.direction]
    za = np.load(f'{W}/extract_{tag}.npz', allow_pickle=True)
    ok = za['ok']
    n = len(ok)
    # 与 paraling_probe.cmd_probe 完全一致：Y = 目标韵律对文本特征回归后的残差
    # （预测的是「可迁移到译文语音的部分」，不是源自身的韵律——后者会高估一大截）
    tgt = np.asarray(pro['tgt'][:n][ok], dtype=np.float64)
    txt = np.asarray(pro['txt'][:n][ok], dtype=np.float64)
    Y = _ols_resid(tgt[:, [KEYS.index(k) for k in PROSODY]], txt)

    def ridge(xtr, ytr, xte):
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=np.logspace(-1, 6, 15),
                       alpha_per_target=True).fit(xtr, ytr).predict(xte)

    def mlp(xtr, ytr, xte):
        """非线性探针 = RBF 核岭回归（内层 3 折选 gamma/alpha）。

        不用 MLP：实测（2026-10-20）n=1467 下 MLP(hidden256, PCA256) 训练 R²=1.000、
        OOF≤0 —— 纯粹记住训练集（论文有 3 万+对才用得动 MLP，我们样本量不够）。
        RBF 核在小 n 下样本效率高得多，才是合格的「线性之外还有没有信息」检验。
        PCA(白化)→256 维只是给核方法控制复杂度（PCA 线性，不添删信息）。
        """
        from sklearn.decomposition import PCA
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import KFold
        p = PCA(n_components=min(256, xtr.shape[0] - 1, xtr.shape[1]), whiten=True,
                random_state=args.seed).fit(xtr)
        Xt, Xe = p.transform(xtr), p.transform(xte)
        best, best_m = -1e9, None
        for gamma in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            for alpha in (1e-3, 1e-1, 1.0, 10.0, 100.0):
                sc = []
                for a_, b_ in KFold(3, shuffle=True, random_state=args.seed).split(Xt):
                    m = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma).fit(Xt[a_], ytr[a_])
                    pred = m.predict(Xt[b_])
                    sc.append(1 - ((ytr[b_] - pred) ** 2).sum() /
                              ((ytr[b_] - ytr[b_].mean(0)) ** 2).sum())
                s = float(np.mean(sc))
                if s > best:
                    best, best_m = s, KernelRidge(kernel='rbf', alpha=alpha,
                                                  gamma=gamma).fit(Xt, ytr)
        return best_m.predict(Xe)

    print(f'方向 {args.direction}  n={len(Y)}  hidden={args.hidden}  折=KFold(5, shuffle, seed={args.seed})')
    print(f'{"表示":<10}{"Ridge(线性)":>12}{"MLP(非线性)":>13}{"Δ=MLP-Ridge":>13}')
    out = []
    for name in args.reps.split(','):
        if name not in za.files:
            print(f'{name:<10}（npz 里没有，跳过）')
            continue
        X = np.asarray(za[name][ok], dtype=np.float64)
        lin = oof(ridge, X, Y, args.seed).mean()
        nl = oof(mlp, X, Y, args.seed).mean()
        out.append({'rep': name, 'ridge': round(float(lin), 4), 'mlp': round(float(nl), 4),
                    'delta': round(float(nl - lin), 4)})
        print(f'{name:<10}{lin:>12.3f}{nl:>13.3f}{nl - lin:>+13.3f}', flush=True)
    vox4 = next((r for r in out if r['rep'] == 'vox4'), None)
    print('\n判读：')
    for name in ('adapter', 'audio_proj'):
        r = next((r for r in out if r['rep'] == name), None)
        if r and vox4:
            gap_lin = vox4['ridge'] - r['ridge']
            recovered = r['mlp'] - r['ridge']
            verdict = ('MLP 大幅恢复 → 信息还在，**线性不可读**；实验 c 可简化为非线性 MLP 接在 '
                       f'{name} 后，免掉 L4 旁路与 50Hz→12.5Hz 对齐'
                       if recovered > 0.6 * gap_lin else
                       f'MLP 未恢复（Δ={recovered:+.3f} vs 缺口 {gap_lin:.3f}）→ **确实丢了信息**，'
                       'L4 旁路（实验 c）才有必要')
            print(f'  {name}: 线性缺口 {gap_lin:.3f}，MLP 补回 {recovered:+.3f}\n    -> {verdict}')
    dst = f'/nvme/probe_mlp_{tag}.json'
    json.dump({'direction': args.direction, 'rows': out}, open(dst, 'w'), indent=2)
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
