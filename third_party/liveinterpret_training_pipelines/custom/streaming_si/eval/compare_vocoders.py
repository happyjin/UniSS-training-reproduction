"""对比两套声码器链路在同一 160 条上的跨语言重建质量。

CosyVoice3 链路（S4/S5 用）：speech_tokenizer_v3 -> flow -> HiFT，25Hz / 6561 词表
Omni 链路（SimulS2ST-Omni 用）：DualCodec(w2v-bert) -> VoiceBox -> Vocos，12.5Hz / 16384 词表

zero 档（无 prompt）是主口径，反映 token 空间对该语言的内容承载能力。
"""
import importlib.util
import json
import os

L = '/fsx/stevehguo/lang_probe'
spec = importlib.util.spec_from_file_location(
    'rp', f'{L}/code/custom/streaming_si/eval/report_lang_probe.py')
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

man = {json.loads(l)['id']: json.loads(l) for l in open(f'{L}/work/manifest.jsonl')}


def ld(p, key='asr'):
    o = {}
    if os.path.isfile(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                o[r['id']] = r.get(key)
    return o


A = {'cv3': {m: ld(f'{L}/out/score/asr_{m}.jsonl') for m in ('source', 'clone', 'zero')},
     'omni': {m: ld(f'{L}/out/score_omni/asr_{m}.jsonl') for m in ('clone', 'zero')}}
U = {'cv3': {m: ld(f'{L}/out/score/utmos_{m}.jsonl', 'utmos') for m in ('source', 'zero')},
     'omni': {m: ld(f'{L}/out/score_omni/utmos_{m}.jsonl', 'utmos') for m in ('zero',)}}
S = {'cv3': ld(f'{L}/out/score/simo_clone.jsonl', 'sim_o'),
     'omni': ld(f'{L}/out/score_omni/simo_clone.jsonl', 'sim_o')}
V = {}
for tag, p in (('cv3', f'{L}/out/vocoder/results.jsonl'),
               ('omni', f'{L}/out/vocoder_omni/results.jsonl')):
    V[tag] = {}
    for l in open(p):
        r = json.loads(l)
        V[tag][(r['id'], r['mode'])] = r

def sent_rates(hyps, refs, lang):
    """逐句错误率的中位数 + 截断在 100% 的均值。语料级 WER 会被个别重复输出的离群句主导
    （Whisper 偶发复读会让单句 WER 达到 1000%+），中位数更能反映典型表现。"""
    import jiwer
    rs = []
    for h, r in zip(hyps, refs):
        if not r.strip():
            continue
        if lang == 'zh':
            h, r = ' '.join(h), ' '.join(r)
        rs.append(min(jiwer.wer(r, h), 3.0))          # 单句上限 300%，防复读句吃掉全部权重
    rs.sort()
    return (round(rs[len(rs) // 2] * 100, 2),
            round(sum(min(x, 1.0) for x in rs) / len(rs) * 100, 2),
            sum(1 for x in rs if x > 1.0))


print('## 两套声码器链路对比（同一 160 条 FLEURS，zero 档为主口径）\n')
cols = ['语言', '源底噪', 'CV3 zero', 'CV3 Δ', 'Omni zero', 'Omni Δ',
        'CV3 tok/s', 'Omni tok/s', 'CV3 SIM-o', 'Omni SIM-o', 'CV3 UTMOS', 'Omni UTMOS']
print('| ' + ' | '.join(cols) + ' |')
print('|' + '---|' * len(cols))
sent = {}
for lang, lab in (('zh', '中文'), ('en', '英文'), ('id', '印尼语'), ('ms', '马来语')):
    ids = [i for i, m in man.items() if m['lang'] == lang]
    refs = [rp.norm(man[i]['text'], lang) for i in ids]
    e_src = rp.err_rate([rp.norm(A['cv3']['source'].get(i, ''), lang) for i in ids], refs, lang)
    e_cv3 = rp.err_rate([rp.norm(A['cv3']['zero'].get(i, ''), lang) for i in ids], refs, lang)
    e_omn = rp.err_rate([rp.norm(A['omni']['zero'].get(i, ''), lang) for i in ids], refs, lang)
    sent[lang] = dict(
        src=sent_rates([rp.norm(A['cv3']['source'].get(i, ''), lang) for i in ids], refs, lang),
        cv3=sent_rates([rp.norm(A['cv3']['zero'].get(i, ''), lang) for i in ids], refs, lang),
        omni=sent_rates([rp.norm(A['omni']['zero'].get(i, ''), lang) for i in ids], refs, lang))
    tk = lambda t: rp.mean([V[t].get((i, 'zero'), {}).get('tokens_per_sec') for i in ids])
    sm = lambda t: rp.mean([S[t].get(i) for i in ids])
    um = lambda t: rp.mean([U[t]['zero'].get(i) for i in ids])
    row = [lab, f'{e_src:.2f}%', f'{e_cv3:.2f}%', f'+{e_cv3 - e_src:.2f}',
           f'{e_omn:.2f}%', f'+{e_omn - e_src:.2f}',
           f'{tk("cv3"):.2f}', f'{tk("omni"):.2f}',
           f'{sm("cv3"):.3f}', f'{sm("omni"):.3f}',
           f'{um("cv3"):.3f}', f'{um("omni"):.3f}']
    print('| ' + ' | '.join(row) + ' |')
print('\nΔ = zero 档错误率 − 源音频底噪，即该链路自身引入的损失。中文 CER，其余 WER。')
print('SIM-o 取 clone 档（带声学 prompt），反映音色克隆能力。')

print('\n### 句级错误率（中位 / 截断均值 / 崩坏句数）\n')
print('语料级 WER 会被个别复读句主导，这里给逐句口径作对照。「崩坏句」= 单句错误率 > 100%。\n')
print('| 语言 | 源底噪 中位 | CV3 中位 | Omni 中位 | 源崩坏 | CV3 崩坏 | Omni 崩坏 |')
print('|' + '---|' * 7)
for lang, lab in (('zh', '中文'), ('en', '英文'), ('id', '印尼语'), ('ms', '马来语')):
    s = sent[lang]
    print(f'| {lab} | {s["src"][0]:.2f}% | {s["cv3"][0]:.2f}% | {s["omni"][0]:.2f}% '
          f'| {s["src"][2]} | {s["cv3"][2]} | {s["omni"][2]} |')
