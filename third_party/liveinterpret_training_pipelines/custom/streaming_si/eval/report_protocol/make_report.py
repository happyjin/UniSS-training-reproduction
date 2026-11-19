"""汇总打分，生成 report.json 与 REPORT.md（口径同 open_ai_test/make_report.py，另加 NCA 延迟与 RTF）。

每个数据集输入：清单（manifest）、统一格式推理结果（results.jsonl）、ASR+LAAL（asr_lat.jsonl）。

指标：
* Text-BLEU / chrF：模型自己吐出的译文对参考。
* ASR-BLEU / chrF：ASR 转写译音对参考（目标语中文用 Qwen3-ASR-1.7B，英文用 Whisper-large-v3）。
  SacreBLEU，中文 tokenize=zh、英文 13a；先轻归一化（NFKC、去标点、英文小写、压空白；中文繁转简）。
  失败 / 空输出按空假设计入语料级 BLEU。
* 延迟两套口径（见 results.jsonl 的 segments / segments_nca）：
  CA  = computation-aware：单卡实时推流模拟，t_k = max(t_{k-1}, 已读源_k) + 第 k 步计算耗时；
  NCA = computation-unaware：只看读到多少源音频时发出（SimulEval delays）。
  LAAL（语音，BOW / EOW）、StartOffset（首包）、EndOffset（尾包）由 asr_latency.py 逐句给出，这里取均值；
  Text-LAAL / 首字 / 尾字由 text_events(_nca) 计算。
* RTF：Σ计算耗时 / Σ源时长（单卡串行）。
* 时长比：Σ译音时长 / Σ源时长。

用法：python make_report.py --root <out_root> --sets realsi_en2zh,realsi_zh2en,acl6060_en2zh \
        --model_label <名字> [--model_desc ...] [--infer_desc ...]
"""
import argparse
import json
import math
import os
import re
import string
import time
import unicodedata
import sys

sys.path.insert(0, os.environ.get('FLUENCY_CODE', os.path.join(os.path.dirname(__file__), 'fluency')))
from fluency_report import attach_fluency, markdown_section

ZH_PUNCT = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·《'
PUNCT = set(string.punctuation) | set(ZH_PUNCT)
PUNCT_RE = re.compile(r'[\s\W_]+', re.UNICODE)

try:
    from opencc import OpenCC
    _T2S = OpenCC('t2s').convert
except Exception:                                                          # noqa: BLE001
    _T2S = None

SET_NAMES = {
    'realsi_en2zh': 'RealSI en→zh',
    'realsi_zh2en': 'RealSI zh→en',
    'acl6060_en2zh': 'ACL 60/60 en→zh',
    'acl6060_en2ja': 'ACL 60/60 en→ja',
    'fleurs_en2ja': 'FLEURS en→ja',
    'fleurs_en2ko': 'FLEURS en→ko',
    'fleurs_zh2ja': 'FLEURS zh→ja',
    'fleurs_zh2ko': 'FLEURS zh→ko',
}
CHAR_LANGS = ('zh', 'ja')                 # 按字计数、去空白；en / ko 按空格分词
_BLEU_TOK = {'zh': 'zh', 'ja': 'ja-mecab', 'ko': 'ko-mecab'}


def bleu_tokenize(lang):
    """ja / ko 用 SacreBLEU 的 MeCab 分词（sacrebleu[ja,ko]）；没装上就退回 char，并在报告里写明。"""
    import sacrebleu
    tok = _BLEU_TOK.get(lang, '13a')
    try:
        sacrebleu.BLEU(tokenize=tok).corpus_score(['a'], [['a']])
        return tok
    except Exception as exc:                                               # noqa: BLE001
        print(f'[report][WARN] tokenize={tok} 不可用（{exc}），{lang} 退回 char', flush=True)
        return 'char'


def norm(text, lang):
    t = unicodedata.normalize('NFKC', text or '')
    if lang == 'zh' and _T2S:
        t = _T2S(t)
    keep = "'" if lang == 'en' else ''
    t = ''.join(' ' if (c in PUNCT and c not in keep) else c for c in t)
    if lang in CHAR_LANGS:
        return re.sub(r'\s+', '', t).strip()
    return re.sub(r'\s+', ' ', t.lower()).strip()


def bleu_chrf(hyps, refs, lang):
    import sacrebleu
    b = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tokenize(lang))
    c = sacrebleu.corpus_chrf(hyps, [refs])
    return round(b.score, 2), round(c.score, 2)


def read_jsonl(p):
    out = {}
    if p and os.path.isfile(p):
        for l in open(p, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                if r['id'] not in out or r.get('status', 'ok') == 'ok':
                    out[r['id']] = r
    return out


def stats(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not xs:
        return dict(mean=None, median=None, p90=None, n=0)
    return dict(mean=round(sum(xs) / len(xs), 3), median=round(xs[len(xs) // 2], 3),
                p90=round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 3), n=len(xs))


def laal(delays, src_len, tgt_len):
    if not delays:
        return None
    if delays[0] > src_len:
        return delays[0]
    gamma = max(len(delays), tgt_len) / src_len
    acc, tau = 0.0, 0
    for i, d in enumerate(delays):
        acc += d - i / gamma
        tau = i + 1
        if d >= src_len:
            break
    return acc / tau


def text_delays(text_events, lang):
    """文本增量 -> 每个计数单元（中文字 / 英文词）的到达时刻。"""
    chars = []
    for t, d in text_events or []:
        for c in unicodedata.normalize('NFKC', d):
            chars.append((c, float(t)))
    if lang in CHAR_LANGS:
        return [t for c, t in chars if not PUNCT_RE.fullmatch(c)]
    out, cur, last_t = [], '', None
    for c, t in chars + [(' ', None)]:
        if c.isspace():
            if PUNCT_RE.sub('', cur):
                out.append(last_t)
            cur = ''
        else:
            cur += c
            last_t = t
    return out


def ref_units(ref, lang):
    t = unicodedata.normalize('NFKC', ref or '')
    if lang in CHAR_LANGS:
        return len([c for c in t if not PUNCT_RE.fullmatch(c)])
    return len([w for w in re.split(r'\s+', t) if PUNCT_RE.sub('', w)])


def text_lat(ok, mrow, lang, key):
    first, last, tl = [], [], []
    for r in ok:
        ev = r.get(key) or []
        src = float(r['source_sec'])
        if ev:
            first.append(float(ev[0][0]))
            last.append(float(ev[-1][0]) - src)
        v = laal(text_delays(ev, lang), src, ref_units(mrow[r['id']]['reference'], lang))
        if v is not None:
            tl.append(v)
    return stats(tl), stats(first), stats(last)


def score_set(root, name):
    man = [json.loads(l) for l in open(os.path.join(root, 'manifests', f'{name}.jsonl'), encoding='utf-8')
           if l.strip()]
    res = read_jsonl(os.path.join(root, 'runs', name, 'results.jsonl'))
    lat = read_jsonl(os.path.join(root, 'runs', name, 'asr_lat.jsonl'))
    lang = man[0]['tgt_lang']
    mrow = {r['id']: r for r in man}
    ids = [r['id'] for r in man if r['id'] in res]       # LIMIT 冒烟时只统计跑过的
    rows = [res[i] for i in ids]
    refs = [mrow[i]['reference'] for i in ids]
    hyp_text = [(r.get('text') or '') if r.get('status') == 'ok' else '' for r in rows]
    hyp_asr = [(lat.get(i, {}).get('asr') or '') for i in ids]
    nr = [norm(x, lang) for x in refs]
    tb, tc = bleu_chrf([norm(h, lang) for h in hyp_text], nr, lang)
    ab, ac = bleu_chrf([norm(h, lang) for h in hyp_asr], nr, lang)

    ok = [r for r in rows if r.get('status') == 'ok']
    src_sec = sum(mrow[i]['source_sec'] for i in ids)
    lv = [lat.get(i, {}) for i in ids]
    stops = {}
    for r in ok:
        stops[str(r.get('stop'))] = stops.get(str(r.get('stop')), 0) + 1
    ok_src = sum(float(r['source_sec']) for r in ok)
    comp = sum(float(r.get('compute_sec') or 0) for r in ok)
    s = dict(name=name, label=SET_NAMES.get(name, name), tgt_lang=lang, n_manifest=len(man), n=len(ids),
             n_ok=len(ok), n_failed=len(ids) - len(ok),
             empty_text=sum(1 for h in hyp_text if not h.strip()),
             empty_wav=sum(1 for r in rows if (r.get('output_sec') or 0) <= 0.05),
             total_source_min=round(src_sec / 60, 1),
             text_bleu=tb, text_chrf=tc, asr_bleu=ab, asr_chrf=ac,
             len_ratio=round(sum((r.get('output_sec') or 0) for r in ok) / src_sec, 3) if src_sec else None,
             rtf=round(comp / ok_src, 3) if ok_src else None,
             compute_sec=stats(r.get('compute_sec') for r in ok),
             misaligned=sum(1 for r in ok if r.get('align_ok') is False),
             stop=stops, t2s=bool(_T2S), bleu_tokenize=bleu_tokenize(lang))
    for suf, tkey in (('', 'text_events'), ('_nca', 'text_events_nca')):
        for k in ('laal_bow', 'laal_eow', 'start_offset', 'end_offset'):
            s[k + suf] = stats(x.get(k + suf) for x in lv)
        s['text_laal' + suf], s['first_text' + suf], s['last_text' + suf] = text_lat(ok, mrow, lang, tkey)
    cp = os.path.join(root, 'runs', name, 'comet.json')
    cm = json.load(open(cp, encoding='utf-8')) if os.path.isfile(cp) else {}
    if cm and cm.get('n') != len(ids):
        print(f'[warn] {name} comet.json n={cm.get("n")} 与本次统计句数 {len(ids)} 不一致，COMET 置空', flush=True)
        cm = {}
    s['asr_comet'], s['text_comet'] = cm.get('asr_comet'), cm.get('text_comet')
    return s


def fmt(v):
    return '-' if v is None else (f'{v:.2f}' if isinstance(v, float) else str(v))


def fmt4(v):
    return '-' if v is None else f'{v:.4f}'


def mm(x):
    return f'{fmt(x["mean"])} ({fmt(x["median"])})'


def write_md(rep, path):
    S = rep['sets']
    L = [f'# {rep["model_label"]} 流式同传评测报告\n', f'- 生成时间：{rep["time"]}']
    for d in rep.get('model_desc') or []:
        L.append(f'- {d}')
    L.append('- ASR：目标语中文（en→zh）用 Qwen3-ASR-1.7B 转写，目标语英文（zh→en）用 Whisper-large-v3 转写；'
             '时间戳（LAAL / 首尾包）统一用 Whisper-large-v3 词级时间戳（`num_beams=1`，语种强制为目标语）')
    for d in rep.get('infer_desc') or []:
        L.append(f'- {d}')
    L.append('- 测试集：句级切分（RealSI 按 json segment、ACL 60/60 eval gold 切分），源音频统一 16kHz 单声道。')
    if any(s['n'] < s['n_manifest'] for s in S):
        L.append(f'- **注意：本次为冒烟测试，每个子集只跑了前 {max(s["n"] for s in S)} 条。**')
    L.append('')

    L.append('## 1. 质量\n')
    L.append('| 测试集 | 句数 | 成功 | 源时长(min) | ASR-BLEU | Text-BLEU | ASR-COMET | Text-COMET | ASR-chrF | Text-chrF | '
             '空文本 | 空语音 | 时长比 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in S:
        L.append(f'| {s["label"]} | {s["n"]} | {s["n_ok"]} | {s["total_source_min"]} | **{fmt(s["asr_bleu"])}** | '
                 f'**{fmt(s["text_bleu"])}** | **{fmt4(s.get("asr_comet"))}** | **{fmt4(s.get("text_comet"))}** | '
                 f'{fmt(s["asr_chrf"])} | {fmt(s["text_chrf"])} | {s["empty_text"]} | '
                 f'{s["empty_wav"]} | {fmt(s["len_ratio"])} |')
    L.append('')

    for title, suf in (('2. 延迟 · computation-aware（CA，含计算耗时；秒，逐句均值，括号内中位数）', ''),
                       ('3. 延迟 · computation-unaware（NCA，不计算力；秒，逐句均值，括号内中位数）', '_nca')):
        L.append(f'## {title}\n')
        L.append('| 测试集 | LAAL-BOW | LAAL-EOW | Text-LAAL | 首包语音 StartOffset | 尾包语音 EndOffset | 首字文本 | 尾字文本 |')
        L.append('|---|---|---|---|---|---|---|---|')
        for s in S:
            L.append(f'| {s["label"]} | **{mm(s["laal_bow" + suf])}** | {mm(s["laal_eow" + suf])} | '
                     f'{mm(s["text_laal" + suf])} | **{mm(s["start_offset" + suf])}** | **{mm(s["end_offset" + suf])}** | '
                     f'{mm(s["first_text" + suf])} | {mm(s["last_text" + suf])} |')
        L.append('')
    L.append('CA 口径 P90：\n')
    L.append('| 测试集 | LAAL-BOW | 首包语音 | 尾包语音 | 首字文本 | 尾字文本 |')
    L.append('|---|---|---|---|---|---|')
    for s in S:
        L.append(f'| {s["label"]} | {fmt(s["laal_bow"]["p90"])} | {fmt(s["start_offset"]["p90"])} | '
                 f'{fmt(s["end_offset"]["p90"])} | {fmt(s["first_text"]["p90"])} | {fmt(s["last_text"]["p90"])} |')
    L.append('')

    L.append('## 4. 计算开销\n')
    L.append('| 测试集 | RTF（Σ计算/Σ源） | 单句计算耗时 s 均值 (中位) | P90 | CA 退化为 NCA 的句数 |')
    L.append('|---|---|---|---|---|')
    for s in S:
        L.append(f'| {s["label"]} | **{fmt(s["rtf"])}** | {mm(s["compute_sec"])} | {fmt(s["compute_sec"]["p90"])} | '
                 f'{s["misaligned"]} |')
    L.append('')

    L.append('## 5. 指标口径\n')
    L.append('- **ASR-BLEU**：ASR 转写译音后与参考算 SacreBLEU（中文 `tokenize=zh`，英文 `13a`；本次实际分词：'
             + '，'.join(f'{s["name"]}=`{s["bleu_tokenize"]}`' for s in S) +
             '；ja/ko 评分文本用 Qwen3-ASR，日文按字、韩文按空格词计延迟单元）；**Text-BLEU**：模型输出的译文文本。'
             'BLEU 前做轻归一化：NFKC、去标点、英文小写、压空白' + ('、中文繁转简' if S and S[0]['t2s'] else '') +
             '。失败 / 空输出按空假设计入语料级 BLEU。')
    L.append('- **ASR-COMET / Text-COMET**：`Unbabel/wmt22-comet-da`（unbabel-comet 2.2.6，batch 32），输入 src=源文 / '
             'mt / ref=参考，三者都用**未归一化**原文；ASR-COMET 的 mt 是 ASR 转写，Text-COMET 的 mt 是模型译文；'
             '取 `system_score`（逐句均值，0–1）。口径同 liveinterpret_training_pipelines `eval/score_eval.py`；'
             '空输出照常计入。')
    L.append('- **CA 时间线**（computation-aware）：模拟单卡实时推流——第 k 步要等源音频到齐、且上一步算完才开始，'
             '`t_k = max(t_{k-1}, 已读源_k) + 第 k 步计算耗时`，该步吐出的译音 / 译文在 `t_k` 到达。'
             '**NCA 时间线**：到达时刻 = 该步已读源音频时长（SimulEval `delays`）。')
    L.append('- **LAAL**（Papi et al. 2022，SimulEval `LAAL_SpeechAlign` 口径）：把每段译音按到达时刻排到播放时间线'
             '（start_i = max(prev_end, emit_i)），用 Whisper 词级时间戳得到每个输出字（中文）/ 词（英文）的播放时刻作为 d_i；'
             'BOW 取词起点、EOW 取词终点；gamma = max(|Y|, |Y*|) / |X|。')
    L.append('- **Text-LAAL**：每个输出字 / 词的延迟取其所在文本增量的到达时刻。')
    L.append('- **首包语音（StartOffset）**：第一段译音开始播放时刻（相对源音频开始）。'
             '**尾包语音（EndOffset）**：最后一段译音播完时刻 − 源音频时长（可为负）。')
    L.append('- **首字 / 尾字（文本）**：首个 / 最后一个译文文本增量到达时刻；尾字相对源结束。')
    L.append('- **RTF**：Σ每句计算耗时 / Σ源音频时长；**时长比**：Σ译音时长 / Σ源音频时长。\n')

    L.append('## 6. 收尾方式统计\n')
    for s in S:
        L.append(f'- {s["label"]}：{s["stop"]}')
    L.append('')
    L.append('## 7. 译音音频与试听\n')
    L.append('- `translation_audio/<set>/<id>.wav`：**全部成功句子**的译音，单声道 24kHz（只有译音，不混源音频）。'
             '时间轴与源音频对齐（0 = 源开始），每段按 CA 到达时刻起播（上一段没播完则顺延，start_i = max(prev_end, emit_i)），'
             '开头和段间的静音就是实时收听时真实要等的时间。S5 整句末尾一次声码，段边界按 40ms/token 切分。')
    L.append('- `demo.html` / `demo.md`：每个子集挑若干句试听（源时长 3–30s 中按固定 seed 抽样，三个系统抽样规则相同），'
             '列出源文、参考、模型译文、ASR 转写与逐句延迟；源音频在 `demo_src/` 单独播放对照。')
    L.extend(markdown_section(rep))
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--sets', default='realsi_en2zh,realsi_zh2en,acl6060_en2zh')
    ap.add_argument('--model_label', required=True)
    ap.add_argument('--model_desc', action='append', default=[], help='报告头部的模型说明（可多次）')
    ap.add_argument('--infer_desc', action='append', default=[], help='报告头部的推理方式说明（可多次）')
    ap.add_argument('--out_dir', default=None, help='默认 <root>/report')
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.root, 'report')
    os.makedirs(out_dir, exist_ok=True)
    sets = []
    for name in args.sets.split(','):
        name = name.strip()
        if not os.path.isfile(os.path.join(args.root, 'runs', name, 'results.jsonl')):
            print(f'[skip] {name} 无结果')
            continue
        s = score_set(args.root, name)
        sets.append(s)
        print(f'{s["label"]:<18} n={s["n"]} ok={s["n_ok"]} ASR-BLEU={s["asr_bleu"]} Text-BLEU={s["text_bleu"]} '
              f'ASR-COMET={s["asr_comet"]} Text-COMET={s["text_comet"]} '
              f'LAAL-BOW(CA/NCA)={s["laal_bow"]["mean"]}/{s["laal_bow_nca"]["mean"]} '
              f'首包={s["start_offset"]["mean"]} 尾包={s["end_offset"]["mean"]} RTF={s["rtf"]}', flush=True)
    rep = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), model_label=args.model_label,
               model_desc=args.model_desc, infer_desc=args.infer_desc, sets=sets)
    attach_fluency(rep, args.root)
    json.dump(rep, open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    write_md(rep, os.path.join(out_dir, 'REPORT.md'))
    print(f'-> {out_dir}/REPORT.md')


if __name__ == '__main__':
    main()
