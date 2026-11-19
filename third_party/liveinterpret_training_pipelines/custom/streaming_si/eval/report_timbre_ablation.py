"""汇总音色消融结果（S4 离线，RealSI en→zh 346 句）。

各臂只改推理参数，译文应逐字不变 —— 脚本会核对这一点，若 Text-BLEU 有变化说明改动意外影响了文本侧。

用法：python report_timbre_ablation.py <abl 目录> [基线臂名，默认 base]
"""
import glob
import json
import os
import sys

ARMS = ('base', 'noprompt', 'nospk', 'nospk_noprompt', 'p20', 'tail', 'ode10', 'ode10_p20',
        'cfg10', 'cfg15', 'cfg20', 'cfg15_ode10',
        'rs_spk', 'rs_prompt', 'rs_both', 'rs_nospk_noprompt')
DESC = {'base': '基线（ode5 / prompt 10s / head / cfg 0.7）', 'ode10': 'ODE 步数 5→10',
        'p20': 'prompt 10s→20s', 'tail': 'prompt 取结尾', 'nospk': '关掉 CAM++ 向量',
        'ode10_p20': 'ODE 10 + prompt 20s',
        'noprompt': '关掉源语音 prompt（只留 CAM++）', 'nospk_noprompt': '两路都关',
        'rs_spk': '【上界】复制合成：源自身 token + CAM++', 'rs_prompt': '【上界】复制合成：源自身 token + 仅 prompt',
        'rs_both': '【上界】复制合成：源自身 token + CAM++ + prompt',
        'rs_nospk_noprompt': '【判别】复制合成：源自身 token + 两路条件全关',
        'cfg10': 'flow CFG 0.7→1.0', 'cfg15': 'flow CFG 0.7→1.5',
        'cfg20': 'flow CFG 0.7→2.0', 'cfg15_ode10': 'CFG 1.5 + ODE 10'}


def load(p, key=None):
    out = {}
    for f in glob.glob(p):
        for l in open(f, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                out[r['id']] = r if key is None else r.get(key)
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def f(v, nd=3):
    return '-' if v is None else f'{v:.{nd}f}'


def main():
    A = sys.argv[1] if len(sys.argv) > 1 else '/fsx/stevehguo/eval/abl'
    base_arm = sys.argv[2] if len(sys.argv) > 2 else 'base'
    arms = [a for a in ARMS if os.path.isfile(f'{A}/{a}/results.jsonl')]
    data = {}
    for a in arms:
        res = load(f'{A}/{a}/results.jsonl')
        data[a] = dict(
            res=res,
            utmos=load(f'{A}/_utmos/{a}.jsonl', 'utmos'),
            para=load(f'{A}/_para/{a}.jsonl'),
            asr=load(f'{A}/_asr/{a}.jsonl', 'asr'),
        )
    # 指标还没跑完的臂只在译文核对里出现，不进指标表（否则会显示成 0）
    scored = [a for a in arms if data[a]['para'] and data[a]['utmos']]
    if base_arm not in scored:
        raise SystemExit(f'基线臂 {base_arm} 还没打完分，无法算相对变化')

    # 译文是否逐字不变
    ref_text = {i: (r.get('text') or '').strip() for i, r in data[base_arm]['res'].items()}
    print('## 译文一致性核对（消融只改推理参数，应逐字相同）\n')
    for a in arms:
        if a.startswith('rs_'):          # 复制合成无译文，不参与文本核对
            continue
        same = sum(1 for i, r in data[a]['res'].items()
                   if (r.get('text') or '').strip() == ref_text.get(i))
        print(f'- `{a}`：{same}/{len(ref_text)} 条与基线逐字相同'
              + ('' if same == len(ref_text) else '  ← **文本侧被影响，需排查**'))

    print('\n## 音色与音质\n')
    hdr = ['臂', '说明', 'SIM-o', 'SIM-o delta', 'UTMOS', 'AutoPCP delta', 'RTF', '时长比']
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    rows = {}
    for a in scored:
        d = data[a]
        ids = sorted(d['res'])
        simo = mean([d['para'].get(i, {}).get('sim_o') for i in ids])
        simo_r = mean([d['para'].get(i, {}).get('sim_o_rand') for i in ids])
        pcp = mean([d['para'].get(i, {}).get('autopcp') for i in ids])
        pcp_r = mean([d['para'].get(i, {}).get('autopcp_rand') for i in ids])
        walls = [d['res'][i].get('wall_sec') for i in ids if d['res'][i].get('wall_sec')]
        srcs = [d['res'][i].get('source_sec') or 0 for i in ids]
        outs = [d['res'][i].get('output_sec') or 0 for i in ids]
        rows[a] = dict(simo=simo, simo_d=None if None in (simo, simo_r) else simo - simo_r,
                       utmos=mean([d['utmos'].get(i) for i in ids]),
                       pcp_d=None if None in (pcp, pcp_r) else pcp - pcp_r,
                       rtf=sum(walls) / sum(srcs) if walls and sum(srcs) else None,
                       lr=sum(outs) / sum(srcs) if sum(srcs) else None)
        r = rows[a]
        print(f"| `{a}` | {DESC.get(a, '')} | {f(r['simo'])} | {f(r['simo_d'])} | {f(r['utmos'])} "
              f"| {f(r['pcp_d'])} | {f(r['rtf'])} | {f(r['lr'])} |")

    print('\n## 相对基线的变化\n')
    b = rows[base_arm]
    print('| 臂 | ΔSIM-o | ΔSIM-o delta | ΔUTMOS | ΔAutoPCP delta | ΔRTF |')
    print('|' + '---|' * 6)
    for a in scored:
        if a == base_arm:
            continue
        r = rows[a]
        d = lambda k, nd=4: ('-' if None in (r[k], b[k])
                             else f'{r[k] - b[k]:+.{nd}f}')
        print(f"| `{a}` | {d('simo')} | {d('simo_d')} | {d('utmos', 3)} | {d('pcp_d', 3)} | {d('rtf', 3)} |")


if __name__ == '__main__':
    main()
