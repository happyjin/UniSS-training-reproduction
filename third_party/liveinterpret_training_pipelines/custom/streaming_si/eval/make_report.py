"""汇总三个数据集的打分报告，输出 Markdown。

每个数据集三张表：质量（ASR/Text 的 BLEU、COMET，UTMOS）、延迟（流式系统的 LAAL 与起止偏移）、
效率与稳定性（RTF、时长比、空输出）。另附 SimulS2ST-Omni 流式 L1～L4 的质量-延迟对照。

用法：python make_report.py <report_dir> <out.md>
"""
import json
import os
import sys

DS_LABEL = {'realsi_en2zh': 'RealSI en→zh', 'realsi_zh2en': 'RealSI zh→en',
            'acl6060_en2zh': 'ACL 60/60 en→zh'}


def f(v, nd=2):
    if v is None:
        return '-'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return out


def main():
    rep_dir, out_p = sys.argv[1], sys.argv[2]
    reports = {ds: json.load(open(os.path.join(rep_dir, f'{ds}.json'), encoding='utf-8'))
               for ds in DS_LABEL if os.path.isfile(os.path.join(rep_dir, f'{ds}.json'))}
    L = []
    for ds, r in reports.items():
        S = r['systems']
        L += [f'## {DS_LABEL[ds]}（{r["n_utt"]} 句，源音频 {r["total_source_sec"] / 60:.1f} 分钟）', '']
        L += ['**质量**', '']
        L += table(['系统', 'ASR-BLEU', 'ASR-chrF', 'ASR-COMET', 'Text-BLEU', 'Text-COMET', 'UTMOS'],
                   [[n, f(s.get('asr_bleu')), f(s.get('asr_chrf')), f(s.get('asr_comet'), 4),
                     f(s.get('text_bleu')), f(s.get('text_comet'), 4), f(s.get('utmos'), 3)]
                    for n, s in S.items()]
                   + [['（源音频）', '', '', '', '', '', f(r.get('source_utmos'), 3)]])
        L += ['', '**延迟（流式系统，秒）**', '']
        L += table(['系统', 'LAAL', 'LAAL (CA)', 'LAAL-EOW', 'LAAL-EOW (CA)', 'StartOffset', 'StartOffset (CA)',
                    'EndOffset', 'EndOffset (CA)', '计入句数'],
                   [[n, f(s.get('laal_nca_bow')), f(s.get('laal_ca_bow')), f(s.get('laal_nca_eow')),
                     f(s.get('laal_ca_eow')), f(s.get('start_offset_nca')), f(s.get('start_offset_ca')),
                     f(s.get('end_offset_nca')), f(s.get('end_offset_ca')), f(s.get('latency_n'))]
                    for n, s in S.items() if s.get('laal_nca_bow') is not None])
        L += ['', '**音色与副语言（配对 / 随机配对基线 / delta = 配对 − 随机）**', '',
              '随机 = 本系统生成语音 vs 另一 talk 的源语音（各系统配对对象相同），是该系统自身的基线；'
              '跨系统比较优先看 delta。', '']
        para_keys = ('sim_o', 'esim', 'autopcp')
        L += table(['系统', 'SIM-o', 'SIM-o 随机', 'SIM-o delta', 'ESim', 'ESim 随机', 'ESim delta',
                    'AutoPCP', 'AutoPCP 随机', 'AutoPCP delta', '计入句数'],
                   [[n] + [f(s.get(f'{k}{suf}'), 3) for k in para_keys for suf in ('', '_rand', '_delta')]
                    + [f(s.get('para_n'))]
                    for n, s in S.items() if any(s.get(k) is not None for k in para_keys)])
        L += ['', '**效率与稳定性**', '']
        L += table(['系统', 'RTF', 'RTF 中位', '平均单句耗时', '时长比', '空译文', '空语音', '失败'],
                   [[n, f(s.get('rtf_corpus'), 3), f(s.get('rtf_median'), 3),
                     f(s.get('mean_wall_sec')) + 's' if s.get('mean_wall_sec') else '-',
                     f(s.get('len_ratio'), 3), f(s.get('empty_text')), f(s.get('empty_wav')), f(s.get('failed'))]
                    for n, s in S.items()])
        L.append('')

    # Omni 流式 L1～L4 对照（三个数据集并排）
    L += ['## SimulS2ST-Omni 流式：L1～L4 质量-延迟对照', '']
    head = ['L']
    for ds in reports:
        head += [f'{DS_LABEL[ds]} ASR-BLEU', 'ASR-COMET', 'LAAL', 'LAAL (CA)']
    rows = []
    for l in (1, 2, 3, 4):
        row = [f'L{l}']
        for ds, r in reports.items():
            s = r['systems'].get(f'SimulS2ST-Omni-L{l}', {})
            row += [f(s.get('asr_bleu')), f(s.get('asr_comet'), 4), f(s.get('laal_nca_bow')), f(s.get('laal_ca_bow'))]
        rows.append(row)
    for ref in ('S5-streaming', 'SimulS2ST-Omni-offline', 'S4-offline'):
        row = [ref]
        for ds, r in reports.items():
            s = r['systems'].get(ref, {})
            row += [f(s.get('asr_bleu')), f(s.get('asr_comet'), 4), f(s.get('laal_nca_bow')), f(s.get('laal_ca_bow'))]
        rows.append(row)
    L += table(head, rows)
    L.append('')

    # 复现 vs 官方（有复现系统时才出）：同一 L 档两行并排
    if any('Repro-Omni-offline' in r['systems'] for r in reports.values()):
        L += ['## SimulS2ST-Omni：官方权重 vs 复现权重', '']
        head = ['档位', '权重']
        for ds in reports:
            head += [f'{DS_LABEL[ds]} ASR-BLEU', 'ASR-COMET', 'Text-BLEU', 'UTMOS', 'LAAL']
        rows = []
        for tag in ['offline'] + [f'L{l}' for l in (1, 2, 3, 4)]:
            for who, prefix in (('官方', 'SimulS2ST-Omni'), ('复现', 'Repro-Omni')):
                row = [tag, who]
                for ds, r in reports.items():
                    s = r['systems'].get(f'{prefix}-{tag}', {})
                    row += [f(s.get('asr_bleu')), f(s.get('asr_comet'), 4), f(s.get('text_bleu')),
                            f(s.get('utmos'), 3), f(s.get('laal_nca_bow'))]
                rows.append(row)
        L += table(head, rows)
        L.append('')
    open(out_p, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
