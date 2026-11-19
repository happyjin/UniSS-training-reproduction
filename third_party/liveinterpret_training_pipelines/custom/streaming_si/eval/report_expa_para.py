"""汇总实验 A 各臂在三个评测集上的**译文质量**与**音色/副语言**指标，出 Markdown 对比报告。

    python custom/streaming_si/eval/report_expa_para.py <para 根目录> [quality 根目录]

* `<para 根>`：`<arm>/<dataset>.jsonl`（`run_para_metrics.py` 的逐句输出）
* `<quality 根>`：`<dataset>.json`（`score_eval.py` 的汇总，含 ASR-BLEU/chrF/COMET、
  Text-BLEU/COMET、UTMOS、RTF 等；每份内含多个系统）
  缺这一层时只出音色/副语言部分，并在报告里标出「质量指标未跑」。

**音色与副语言一律报「配对 − 随机配对」的 delta**，不单看配对值：
SIM-o / ESim / AutoPCP 的绝对值受说话人池、语言、录音条件影响，同一套系统换个评测集
就能差 0.1 以上；随机配对基线把这部分共同偏移抵掉。本轮曾因只看配对值，
在 8 句小样本上把噪声当成趋势误判过一次。

**质量指标不报 delta**：BLEU / COMET / UTMOS 本身已是与参考比较后的量，
再扣一层没有意义；改为与对照臂（`a0j`）逐项比差值，方便看「谁掉了多少」。
"""
import glob
import json
import os
import sys

SETS = ('realsi_en2zh', 'realsi_zh2en', 'acl6060_en2zh')
DS_LABEL = {'realsi_en2zh': 'RealSI en→zh', 'realsi_zh2en': 'RealSI zh→en',
            'acl6060_en2zh': 'ACL 60/60 en→zh'}
CTRL = 'a0j'

# 音色/副语言：(显示名, 配对字段, 随机配对字段, 小数位, 是否主 KPI)
PARA_METRICS = (('SIM-o', 'sim_o', 'sim_o_rand', 4, True),
                ('ESim', 'esim', 'esim_rand', 4, False),
                ('AutoPCP', 'autopcp', 'autopcp_rand', 3, True))

# 译文质量：(显示名, score_eval 字段, 小数位, BLEU 掉超这条线算不可接受)
QUALITY_METRICS = (('ASR-BLEU', 'asr_bleu', 2, True),
                   ('ASR-chrF', 'asr_chrf', 2, False),
                   ('ASR-COMET', 'asr_comet', 4, False),
                   ('Text-BLEU', 'text_bleu', 2, True),
                   ('Text-COMET', 'text_comet', 4, False),
                   ('UTMOS', 'utmos', 3, False))


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    rows = []
    for ln in open(path, encoding='utf-8'):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def load_json(path):
    if not os.path.isfile(path):
        return {}
    try:
        return json.load(open(path, encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def agg_para(rows, pair_key, rand_key):
    """返回 (n, 配对均值, 随机均值, delta)。两个字段都非空才算，保证配对可比。"""
    g = [r for r in rows if r.get(pair_key) is not None and r.get(rand_key) is not None]
    if not g:
        return 0, None, None, None
    p = sum(r[pair_key] for r in g) / len(g)
    q = sum(r[rand_key] for r in g) / len(g)
    return len(g), p, q, p - q


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    para_root = argv[0] if argv else '/tmp/expa_orch/para'
    qual_root = argv[1] if len(argv) > 1 else '/tmp/expa_orch/quality'

    # 臂清单以副语言结果为准（质量层是后跑的，可能只覆盖部分臂）
    arms = sorted(d for d in os.listdir(para_root)
                  if os.path.isdir(os.path.join(para_root, d))) if os.path.isdir(para_root) else []
    qual = {d: load_json(os.path.join(qual_root, f'{d}.json')) for d in SETS}
    has_qual = any(v.get('systems') for v in qual.values())

    # 质量层里出现的臂（含参比行 S4-offline-start）
    q_arms = []
    for d in SETS:
        for name in (qual[d].get('systems') or {}):
            if name not in q_arms:
                q_arms.append(name)
    if not arms and not q_arms:
        print(f'# 实验 A\n\n(`{para_root}` 与 `{qual_root}` 下都没有结果)')
        return

    desc = {'a0j': 'a0 joint：取层 −1（末层 / 探针 L32），对照臂',
            'a1j': 'a1 joint：取层 6（探针 L7，峰值区 L4~L8）',
            'a2j': 'a2 joint：取层 15（探针 L16，半深度，对标 Omni 24/48）',
            'S4-offline-start': '**参比**：S4-offline = joint 起点 checkpoint-24980 的官方评测值'}

    print('# 实验 A：talker 取层对译文质量、音色与副语言的影响\n')
    print('起点同为 S4 checkpoint-24980，各臂起点权重逐位相同，唯一差别是 `talker_bridge_layer`；')
    print('同全局 batch（512）、同 LR 调度、同数据分片。推理口径与第二轮正式评测一致'
          '（源语音 prompt 开启，`COSYVOICE_SPEECH_TOKENIZER` 已显式传入）。\n')
    for a in (arms or q_arms):
        if a in desc:
            print(f'- `{a}`：{desc[a]}')
    print()

    # ================= 一、译文质量 =================
    print('## 一、译文质量\n')
    if not has_qual:
        print(f'> 质量指标尚未产出（`{qual_root}` 为空）。跑 '
              '`custom/streaming_si/eval/orchestrate_expa_quality.sh` 生成后重出本报告。\n')
    else:
        print('口径与第二轮正式评测逐项一致：同一个 `score_eval.py`、同一 COMET 权重 '
              '`Unbabel/wmt22-comet-da`、同一 ASR 模型 `whisper-large-v3`；'
              'BLEU 前做轻归一化（中文 `tokenize=zh` / 英文 `13a`）。\n')
        for name, key, nd, _is_bleu in QUALITY_METRICS:
            print(f'### {name}\n')
            print('| 系统 | ' + ' | '.join(DS_LABEL[d] for d in SETS) + ' | 三集平均 |')
            print('|' + '---|' * (len(SETS) + 2))
            table = {}
            for a in q_arms:
                vals, cells = [], []
                for d in SETS:
                    v = (qual[d].get('systems') or {}).get(a, {}).get(key)
                    table.setdefault(a, {})[d] = v
                    cells.append('—' if v is None else f'{v:.{nd}f}')
                    if v is not None:
                        vals.append(v)
                avg = sum(vals) / len(vals) if vals else None
                label = f'`{a}`' if not a.startswith('S4-offline') else a
                print(f'| {label} | ' + ' | '.join(cells) + ' | '
                      + (f'{avg:.{nd}f}' if avg is not None else '—') + ' |')
            print()
        # 相对对照臂的变化（BLEU/COMET/UTMOS 越高越好；掉超阈值要标出）
        if CTRL in q_arms:
            print('### 相对对照臂 `a0j` 的变化（正 = 更好）\n')
            print('| 指标 | 臂 | ' + ' | '.join(DS_LABEL[d] for d in SETS) + ' | 三集平均 |')
            print('|' + '---|' * (len(SETS) + 3))
            for name, key, nd, is_bleu in QUALITY_METRICS:
                for a in q_arms:
                    if a == CTRL or a.startswith('S4-offline'):
                        continue
                    ds, ok = [], True
                    for d in SETS:
                        cv = (qual[d].get('systems') or {}).get(CTRL, {}).get(key)
                        av = (qual[d].get('systems') or {}).get(a, {}).get(key)
                        ds.append(None if (cv is None or av is None) else av - cv)
                    got = [x for x in ds if x is not None]
                    if not got:
                        continue
                    avg = sum(got) / len(got)
                    flag = ''
                    if is_bleu and avg < -0.5:
                        flag = ' ⚠️超门槛'
                    print(f'| {name} | `{a}` | '
                          + ' | '.join('—' if x is None else f'{x:+.{nd}f}' for x in ds)
                          + f' | **{avg:+.{nd}f}**{flag} |')
            print('\n> BLEU 类指标的门槛：相对对照臂**不许掉超过 0.5**。'
                  '超过则即使音色/副语言涨了也不算有效（翻译能力被侵蚀）。\n')

    # ================= 二、音色与副语言 =================
    print('## 二、音色与副语言\n')
    if not arms:
        print(f'> 副语言指标尚未产出（`{para_root}` 为空）。\n')
    else:
        print('**所有 delta = 配对 − 随机配对**，已扣掉说话人池与语言造成的共同偏移。\n')
        for name, pk, rk, nd, is_kpi in PARA_METRICS:
            tag = '' if is_kpi else '（*近乎无区分度，不作判据*）'
            print(f'### {name} delta{tag}\n')
            print('| 臂 | ' + ' | '.join(DS_LABEL[d] for d in SETS) + ' | 三集平均 |')
            print('|' + '---|' * (len(SETS) + 2))
            base = {}
            for a in arms:
                cells, deltas = [], []
                for d in SETS:
                    n, p, q, dl = agg_para(load_jsonl(os.path.join(para_root, a, f'{d}.jsonl')), pk, rk)
                    if dl is None:
                        cells.append('—')
                        continue
                    deltas.append(dl)
                    base.setdefault(a, {})[d] = dl
                    cells.append(f'{dl:+.{nd}f}（配对 {p:.{nd}f} / 随机 {q:.{nd}f}，n={n}）')
                avg = sum(deltas) / len(deltas) if deltas else None
                print(f'| `{a}` | ' + ' | '.join(cells) + ' | '
                      + (f'**{avg:+.{nd}f}**' if avg is not None else '—') + ' |')
            others = [a for a in arms if a != CTRL and base.get(a) and base.get(CTRL)]
            if others:
                print(f'\n相对 `{CTRL}` 的变化（正 = 更好）：\n')
                print('| 臂 | ' + ' | '.join(DS_LABEL[d] for d in SETS) + ' | 三集平均 |')
                print('|' + '---|' * (len(SETS) + 2))
                for a in others:
                    ds = [base[a][d] - base[CTRL][d] for d in SETS if d in base[a] and d in base[CTRL]]
                    if ds:
                        print(f'| `{a}` | ' + ' | '.join(f'{x:+.{nd}f}' for x in ds)
                              + f' | **{sum(ds)/len(ds):+.{nd}f}** |')
            print()

    # ================= 三、判读 =================
    print('## 三、判读规则\n')
    print('- **主 KPI**：AutoPCP delta（官方 v2 口径：`pick_layer=9` + `symmetrize=true`）、'
          'SIM-o delta；译文质量侧看 ASR-BLEU 与 Text-BLEU。')
    print('- **ESim 不作判据**：RealSI/ACL 上对全部 12 个系统的 delta 都落在 0.01~0.03，'
          '近乎无区分度，不要用它的小差异下结论。')
    print('- **质量门槛**：任何臂相对 `a0j` 的 BLEU 掉超 0.5 即判无效——'
          '副语言涨而翻译能力退化不是可接受的交换。')
    print('- **看 delta 不看绝对值**：音色/副语言的绝对值受说话人池与语言支配，'
          '跨集比较只能用 delta。')
    print('- **样本量**：n 明显小于评测集条数说明有塌缩或打分失败，先查 '
          '`infer.*.log` / `para.*.log` / `asr.*.log`。')
    print('- **预注册提醒**：预热阶段的门 A 略差于对照臂**不构成**该臂失败的证据'
          '（thinker 冻结时换层优势用不上，实验 4 即如此）；判据只在 joint 之后。')


if __name__ == '__main__':
    main()
