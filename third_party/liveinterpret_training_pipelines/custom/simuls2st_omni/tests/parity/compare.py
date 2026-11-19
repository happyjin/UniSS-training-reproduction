"""比较两份训练轨迹（native_trace.py / swift_trace_plugin.py 的输出）。

    python compare.py ref.json got.json --rtol 1e-6 [--ranks 2]

逐项检查：每个 rank 每一步的任务与样本 id 完全相同；每步 loss / grad_norm、各 eval_loss 在容差内；
训练结束时每个可训参数张量的和在容差内（参数名集合必须相同）。不满足时退出码 1。
"""
import argparse
import json
import math
import sys


def _load(path, ranks):
    out = [json.load(open(path))]
    for r in range(1, ranks):
        out.append(json.load(open(f'{path}.rank{r}')))
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('got')
    ap.add_argument('--rtol', type=float, default=1e-6)
    ap.add_argument('--ranks', type=int, default=1)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    refs, gots = _load(a.ref, a.ranks), _load(a.got, a.ranks)
    problems = []

    for r, (ref, got) in enumerate(zip(refs, gots)):
        if len(ref['steps']) != len(got['steps']):
            problems.append(f'rank{r}: 步数 {len(ref["steps"])} != {len(got["steps"])}')
        for i, (x, y) in enumerate(zip(ref['steps'], got['steps'])):
            if x['ids'] != y['ids'] or x['task'] != y['task']:
                problems.append(f'rank{r} step{i + 1}: batch 不同 {x["task"]}{x["ids"][:3]} vs {y["task"]}{y["ids"][:3]}')
                break

    ref, got = refs[0], gots[0]
    tl = [l for l in ref['log'] if 'loss' in l and 'grad_norm' in l]
    gl = [l for l in got['log'] if 'loss' in l and 'grad_norm' in l]
    worst = {'loss': 0.0, 'grad_norm': 0.0}
    for i, (x, y) in enumerate(zip(tl, gl)):
        for k in worst:
            xv, yv = float(x[k]), float(y[k])
            # HF 原生 Trainer 日志里的 loss 四舍五入到 4 位小数（swift 不取整）：允许 5e-5 的舍入误差
            slack = 5.0001e-5 if k == 'loss' else 0.0
            d = max(0.0, abs(xv - yv) - slack) / max(abs(xv), abs(yv), 1e-12)
            worst[k] = max(worst[k], d)
            if d > a.rtol:
                problems.append(f'log{i + 1} {k}: {x[k]} vs {y[k]} (rel {d:.2e})')
    if len(tl) != len(gl):
        problems.append(f'日志条数 {len(tl)} != {len(gl)}')
    ev_r = [(l['step'], l['eval_loss']) for l in ref['log'] if 'eval_loss' in l]
    ev_g = [(l['step'], l['eval_loss']) for l in got['log'] if 'eval_loss' in l]
    # swift 在训练最后一步总会再评估一次（步数不是 eval_steps 整数倍时 HF 不会）：多出的这一条不算差异
    last_step = max((l.get('step', 0) for l in got['log']), default=0)
    if len(ev_g) == len(ev_r) + 1 and ev_g[-1][0] == last_step and all(s != last_step for s, _ in ev_r):
        ev_g = ev_g[:-1]
    worst['eval_loss'] = max([_rel(x[1], y[1]) for x, y in zip(ev_r, ev_g)] or [0.0])
    if [s for s, _ in ev_r] != [s for s, _ in ev_g] or worst['eval_loss'] > a.rtol:
        problems.append(f'eval_loss {ev_r} vs {ev_g}')

    worst['param'] = 0.0
    n_params = 0
    for part in ('thinker', 'talker'):
        rp, gp = ref['checksum'].get(part, {}), got['checksum'].get(part, {})
        if set(rp) != set(gp):
            problems.append(f'{part} 可训参数集合不同：仅参照有 {sorted(set(rp) - set(gp))[:3]}，仅被测有 {sorted(set(gp) - set(rp))[:3]}')
            continue
        for name in rp:
            n_params += 1
            d = _rel(rp[name], gp[name])
            worst['param'] = max(worst['param'], d)
            if d > a.rtol and not math.isclose(rp[name], gp[name], abs_tol=1e-9):
                problems.append(f'{part}.{name}: {rp[name]} vs {gp[name]} (rel {d:.2e})')

    steps = len(ref['steps'])
    summary = ', '.join(f'max rel Δ{k}={v:.1e}' for k, v in worst.items())
    tag = f'[{a.label}] ' if a.label else ''
    if problems:
        print(f'{tag}FAIL ({steps} 步 × {a.ranks} rank, {n_params} 个可训张量; {summary})')
        for p in problems[:12]:
            print('  -', p)
        sys.exit(1)
    print(f'{tag}OK ({steps} 步 × {a.ranks} rank, batch 序列一致, {n_params} 个可训张量; {summary})')


if __name__ == '__main__':
    main()
