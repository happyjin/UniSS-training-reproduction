"""流式 S2ST 的 LAAL（按语音对齐，口径同 SimulEval 的 `LAAL_SpeechAlign_{BOW,EOW}`）。

SimulEval 原生做法：把输出语音按发出时刻排到时间轴上，用 MFA 做词级强制对齐，每个输出词的
时刻作为该词的 delay，再按 LAAL 公式算。MFA 只带英文模型，en→zh 用不了，所以这里换成
Whisper-large-v3 的词级时间戳（两种方向、两个流式系统同一套），其余步骤与 SimulEval 一致：

1. 在「只有语音」的波形（各段首尾相接）上跑 Whisper，语种强制为目标语，取每个词的起止时间；
   中文按字切（一个 Whisper 词含多个字时在其时间区间内均分），标点不计。
2. 词在拼接波形中的位置 -> 所在语音段 -> 该段在播放时间线上的开始时刻 + 段内偏移，
   得到该词在「源音频时钟」上的播放时刻。时间线规则同 SimulEval：start_i = max(prev_end, emit_i)。
   NCA（emit = 已读源音频）与 CA（emit = SimulEval elapsed）各算一套。
3. LAAL（Papi et al. 2022，SimulEval 实现）：
       gamma = max(|Y_hyp|, |Y_ref|) / |X|
       LAAL  = 1/tau * sum_{i=1..tau} (d_i - (i-1)/gamma),   tau = 第一个 d_i >= |X| 的位置
   若 d_1 > |X| 直接取 d_1。单位：中文按字、英文按词（同官方 `--eval-latency-unit char/word`）。
   BOW 用词起点、EOW 用词终点。
另报 StartOffset（第一段语音开始播放的时刻）与 EndOffset（最后一段播完时刻 - 源音频时长）。
无语音输出的句子不计入（SimulEval 同样跳过），条数单独报。

用法：python latency_eval.py --results <stream_segments 的 results.jsonl> --manifest all.jsonl \
          --model <whisper 目录> --out lat.jsonl [--shard i --num_shards n]
"""
import argparse
import json
import os
import re
import unicodedata

import numpy as np
import soundfile as sf
import torch

PUNCT_RE = re.compile(r'[\s\W_]+', re.UNICODE)


def units_zh(text):
    t = unicodedata.normalize('NFKC', text or '')
    return [c for c in t if not PUNCT_RE.fullmatch(c)]


def units_en(text):
    t = unicodedata.normalize('NFKC', text or '')
    return [w for w in re.split(r'\s+', t) if PUNCT_RE.sub('', w)]


def word_spans(chunks, lang):
    """Whisper 词级 chunks -> [(start, end)]（中文按字均分，英文按词），去掉纯标点。"""
    out = []
    for c in chunks:
        a, b = c.get('timestamp') or (None, None)
        if a is None:
            continue
        b = a if b is None else b
        txt = c.get('text', '')
        if lang == 'zh':
            chars = units_zh(txt)
            if not chars:
                continue
            step = (b - a) / len(chars)
            out += [(a + k * step, a + (k + 1) * step) for k in range(len(chars))]
        else:
            if units_en(txt):
                out.append((a, b))
    return out


def timeline(segs, key):
    """按 SimulEval 规则排播放时间线，返回 [(start, dur)]。"""
    tl, prev_end = [], None
    for s in segs:
        e = s.get(key)
        if e is None:
            return None
        st = e if prev_end is None else max(prev_end, e)
        tl.append((st, s['dur']))
        prev_end = st + s['dur']
    return tl


def to_abs(t, tl):
    """拼接波形中的时刻 t -> 源音频时钟上的播放时刻（超出总长的夹到最后一段末尾）。"""
    cum = 0.0
    for k, (st, du) in enumerate(tl):
        if t < cum + du or k == len(tl) - 1:
            return st + min(max(t - cum, 0.0), du)
        cum += du
    return 0.0


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    man = {}
    for l in open(args.manifest, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            man[r['id']] = r
    rows = [json.loads(l) for l in open(args.results, encoding='utf-8') if l.strip()]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]

    from transformers import pipeline
    asr = pipeline('automatic-speech-recognition', model=args.model, device=args.device,
                   dtype=torch.float16, chunk_length_s=30, stride_length_s=5)

    out = open(args.out, 'w', encoding='utf-8')
    todo = [r for r in rows if r.get('wav') and os.path.isfile(r['wav']) and r.get('segments')]
    for r in rows:
        if r not in todo:
            out.write(json.dumps(dict(id=r['id'], note='no_speech'), ensure_ascii=False) + '\n')
    for lang in ('zh', 'en'):
        sub = [r for r in todo if man[r['id']]['tgt_lang'] == lang]
        for i in range(0, len(sub), args.batch):
            chunk = sub[i:i + args.batch]
            wavs = []
            for r in chunk:
                x, sr = sf.read(r['wav'], dtype='float32')
                wavs.append({'raw': x.mean(1) if x.ndim > 1 else x, 'sampling_rate': sr})
            res = asr(wavs, batch_size=len(wavs), return_timestamps='word',
                      generate_kwargs={'language': lang, 'task': 'transcribe', 'num_beams': 1})
            for r, o in zip(chunk, res if isinstance(res, list) else [res]):
                m = man[r['id']]
                src = float(m['source_sec'])
                spans = word_spans(o.get('chunks', []), lang)
                ref_len = len(units_zh(m['reference']) if lang == 'zh' else units_en(m['reference']))
                rec = dict(id=r['id'], asr_ts=o.get('text', '').strip(), n_hyp_units=len(spans),
                           n_ref_units=ref_len, source_sec=src)
                for key, tag in (('emit_nca', 'nca'), ('emit_ca', 'ca')):
                    tl = timeline(r['segments'], key)
                    if tl is None:
                        continue
                    rec[f'start_offset_{tag}'] = round(tl[0][0], 4)
                    rec[f'end_offset_{tag}'] = round(tl[-1][0] + tl[-1][1] - src, 4)
                    for bnd, idx in (('bow', 0), ('eow', 1)):
                        d = [to_abs(sp[idx], tl) for sp in spans]
                        v = laal(d, src, ref_len)
                        rec[f'laal_{tag}_{bnd}'] = None if v is None else round(v, 4)
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            print(f'[lat {lang} shard {args.shard}] {min(i + args.batch, len(sub))}/{len(sub)}', flush=True)
    out.close()


if __name__ == '__main__':
    main()
