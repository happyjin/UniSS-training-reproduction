"""汇总跨语言验证结果：声码器重建保真度（CER/WER、UTMOS、SIM-o）+ 编码器表征质量。

CER/WER 用 jiwer 的标准口径算，中文按字（去空白后逐字）、其余语言按词。
两档对比的关键是 **zero 档的 CER/WER**：它没有任何来自源音频的声学 prompt，
反映 speech token 空间对该语言的内容承载能力。源音频自身的转写错误率作为 ASR 底噪基线。

用法：python report_lang_probe.py <lang_probe 根目录>
"""
import json
import os
import re
import string
import sys
import unicodedata

ZH_PUNCT = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·'
PUNCT = set(string.punctuation) | set(ZH_PUNCT)
LANGS = ('zh', 'en', 'id', 'ms')
LABEL = {'zh': '中文', 'en': '英文', 'id': '印尼语', 'ms': '马来语'}


def norm(t, lang):
    t = unicodedata.normalize('NFKC', t or '')
    keep = "'" if lang != 'zh' else ''
    t = ''.join(' ' if (c in PUNCT and c not in keep) else c for c in t)
    if lang == 'zh':
        return re.sub(r'\s+', '', t).strip()
    return re.sub(r'\s+', ' ', t.lower()).strip()


def err_rate(hyps, refs, lang):
    """中文算 CER（按字），其余算 WER（按词）。返回百分数。"""
    import jiwer
    if lang == 'zh':
        h = [' '.join(x) for x in hyps]
        r = [' '.join(x) for x in refs]
    else:
        h, r = hyps, refs
    pairs = [(a, b) for a, b in zip(h, r) if b.strip()]
    if not pairs:
        return None
    return round(jiwer.wer([b for _, b in pairs], [a for a, _ in pairs]) * 100, 2)


def load(p, key='asr'):
    out = {}
    if os.path.isfile(p):
        for l in open(p, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                out[r['id']] = r.get(key)
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '/fsx/stevehguo/lang_probe'
    man = {json.loads(l)['id']: json.loads(l)
           for l in open(f'{root}/work/manifest.jsonl', encoding='utf-8') if l.strip()}
    voc = {}
    for l in open(f'{root}/out/vocoder/results.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            voc[(r['id'], r['mode'])] = r
    asr = {m: load(f'{root}/out/score/asr_{m}.jsonl') for m in ('source', 'clone', 'zero')}
    utm = {m: load(f'{root}/out/score/utmos_{m}.jsonl', 'utmos') for m in ('source', 'clone', 'zero')}
    simo = {m: load(f'{root}/out/score/simo_{m}.jsonl', 'sim_o') for m in ('clone', 'zero')}
    enc = {}
    for l in open(f'{root}/out/encoder.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            enc[r['id']] = r

    print('## 验证一：CosyVoice3 声码器链路（tokenizer -> flow -> HiFT）重建保真度\n')
    hdr = ['语言', '条数', 'tok/s', '源音频错误率', 'zero 错误率', 'zero Δ', 'clone 错误率',
           '源 UTMOS', 'zero UTMOS', 'clone UTMOS', 'zero SIM-o', 'clone SIM-o']
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    for lang in LANGS:
        ids = [i for i, m in man.items() if m['lang'] == lang]
        if not ids:
            continue
        refs = [norm(man[i]['text'], lang) for i in ids]
        e = {m: err_rate([norm(asr[m].get(i, ''), lang) for i in ids], refs, lang)
             for m in ('source', 'clone', 'zero')}
        delta = None if (e['zero'] is None or e['source'] is None) else round(e['zero'] - e['source'], 2)
        row = [LABEL[lang], str(len(ids)),
               f"{mean([voc[(i, 'zero')]['tokens_per_sec'] for i in ids if (i, 'zero') in voc]):.2f}",
               f"{e['source']:.2f}%", f"{e['zero']:.2f}%", f'+{delta:.2f}' if delta else '-',
               f"{e['clone']:.2f}%",
               f"{mean([utm['source'].get(i) for i in ids]):.3f}",
               f"{mean([utm['zero'].get(i) for i in ids]):.3f}",
               f"{mean([utm['clone'].get(i) for i in ids]):.3f}",
               f"{mean([simo['zero'].get(i) for i in ids]):.3f}",
               f"{mean([simo['clone'].get(i) for i in ids]):.3f}"]
        print('| ' + ' | '.join(row) + ' |')
    print('\n中文按 CER（字），英文 / 印尼语 / 马来语按 WER（词）。「源音频错误率」是 Whisper 在原始'
          '音频上的错误率，即 ASR 底噪；「zero Δ」= zero 档减去底噪，才是声码器链路自身引入的损失。')

    print('\n## 验证二：冻结 Voxtral 编码器的表征质量\n')
    cols = ('self_sim_c', 'cross_sim_c', 'rank_ratio', 'frame_std', 'n_frames')
    print('| 语言 | 条数 | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 2))
    for lang in LANGS:
        g = [r for r in enc.values() if r['lang'] == lang]
        if not g:
            continue
        print(f'| {LABEL[lang]} | {len(g)} | '
              + ' | '.join(f'{mean([r[c] for r in g]):.4f}' for c in cols) + ' |')
    print('\n`self_sim_c` / `cross_sim_c` 是去全体均值后的余弦（越低越可分），'
          '`rank_ratio` / `frame_std` 越高越好。id / ms 落在 zh / en 的区间内即说明编码器未塌缩。')


if __name__ == '__main__':
    main()
