"""对译音做 ASR：(1) 转写 → ASR-BLEU；(2) Whisper 词级时间戳 → 语音 LAAL / 首尾包。

ASR 模型按目标语选择：
* 目标语中文（en→zh）：Qwen3-ASR-1.7B 转写（`--qwen_model`，language="Chinese"）。
* 目标语英文（zh→en）：Whisper-large-v3 转写（language="en"）。
* 时间戳（LAAL / 首尾包）两个方向都用 Whisper-large-v3 词级时间戳
  （Qwen3-ASR 出时间戳需要另配 Qwen3-ForcedAligner）。

口径与 open_ai_test/asr_latency.py 一致：
* `num_beams=1`、语种按 tgt_lang 强制；长音频 chunk 30s / stride 5s；音频先用 soxr 重采样到 16k（不依赖 torchaudio）。
* 词级时间戳：中文一个 Whisper 词含多字时在其区间内均分，标点不计；英文按词。
* 播放时间线：每段译音 start_i = max(prev_end, emit_i)；输出单元 d_i = 其所在段 start + 段内偏移。
* LAAL = 1/tau * Σ_{i≤tau} (d_i - (i-1)/gamma)，gamma = max(|Y|,|Y*|)/|X|，tau = 第一个 d_i ≥ |X| 的位置。

每条结果按两套时间线各算一遍（见 s5_io.py）：
  segments      → start_offset / end_offset / laal_bow / laal_eow           （computation-aware）
  segments_nca  → start_offset_nca / end_offset_nca / laal_bow_nca / laal_eow_nca（SimulEval 默认，不计算力）

用法：python asr_latency.py --results runs/x/results.jsonl --manifest m.jsonl --model <whisper> \
        --qwen_model <qwen3-asr> --out runs/x/asr_lat.jsonl
"""
import argparse
import json
import os
import re
import unicodedata

import soundfile as sf
import soxr
import torch

MODEL_SR = 16000
PUNCT_RE = re.compile(r'[\s\W_]+', re.UNICODE)
QWEN_LANG = {'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'}   # 这些目标语用 Qwen3-ASR 出评分文本
CHAR_LANGS = ('zh', 'ja')                                          # 按字计数；其余（en / ko）按空格分词计数


def units_zh(text):
    t = unicodedata.normalize('NFKC', text or '')
    return [c for c in t if not PUNCT_RE.fullmatch(c)]


def units_en(text):
    t = unicodedata.normalize('NFKC', text or '')
    return [w for w in re.split(r'\s+', t) if PUNCT_RE.sub('', w)]


def word_spans(chunks, lang):
    out = []
    for c in chunks:
        a, b = c.get('timestamp') or (None, None)
        if a is None:
            continue
        b = a if b is None else b
        txt = c.get('text', '')
        if lang in CHAR_LANGS:
            chars = units_zh(txt)
            if not chars:
                continue
            step = (b - a) / len(chars)
            out += [(a + k * step, a + (k + 1) * step) for k in range(len(chars))]
        elif units_en(txt):
            out.append((a, b))
    return out


def timeline(segs):
    tl, prev_end = [], None
    for s in segs:
        e = float(s['emit'])
        st = e if prev_end is None else max(prev_end, e)
        tl.append((st, float(s['dur'])))
        prev_end = st + float(s['dur'])
    return tl


def to_abs(t, tl):
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


def latency_fields(segs, spans, src, ref_len, suffix):
    tl = timeline(segs)
    rec = {f'start_offset{suffix}': round(tl[0][0], 4), f'end_offset{suffix}': round(tl[-1][0] + tl[-1][1] - src, 4)}
    for bnd, idx in (('bow', 0), ('eow', 1)):
        v = laal([to_abs(sp[idx], tl) for sp in spans], src, ref_len)
        rec[f'laal_{bnd}{suffix}'] = None if v is None else round(v, 4)
    return rec


def load_16k(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    if sr != MODEL_SR:
        x = soxr.resample(x, sr, MODEL_SR).astype('float32')
    return {'raw': x, 'sampling_rate': MODEL_SR}


def run_whisper(asr, wavs, gk, **kw):
    try:
        out = asr([dict(w) for w in wavs], batch_size=len(wavs), generate_kwargs=gk, **kw)
    except Exception as exc:                                                    # noqa: BLE001
        print(f'[warn] whisper batch 失败，逐条重试：{exc}', flush=True)
        out = []
        for w in wavs:
            try:
                out.append(asr(dict(w), generate_kwargs=gk, **kw))
            except Exception as e2:                                             # noqa: BLE001
                print(f'[warn] whisper 单条失败：{e2}', flush=True)
                out.append({'text': '', 'chunks': []})
    return out if isinstance(out, list) else [out]


def run_qwen(qwen, wavs, language):
    try:
        res = qwen.transcribe(audio=[(w['raw'], w['sampling_rate']) for w in wavs], language=language)
        return [{'text': q.text or ''} for q in res]
    except Exception as exc:                                                    # noqa: BLE001
        print(f'[warn] qwen3-asr batch 失败，逐条重试：{exc}', flush=True)
        out = []
        for w in wavs:
            try:
                out.append({'text': qwen.transcribe(audio=[(w['raw'], w['sampling_rate'])], language=language)[0].text or ''})
            except Exception as e2:                                             # noqa: BLE001
                print(f'[warn] qwen3-asr 单条失败：{e2}', flush=True)
                out.append({'text': ''})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--model', required=True, help='Whisper-large-v3 目录（en 转写 + 全部时间戳）')
    ap.add_argument('--qwen_model', required=True, help='Qwen3-ASR-1.7B 目录（zh 目标语转写）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--shard', type=int, default=0, help='只处理第 shard 片（按句序号取模），多卡并行用')
    ap.add_argument('--num_shards', type=int, default=1)
    args = ap.parse_args()

    man = {}
    for l in open(args.manifest, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            man[r['id']] = r
    rows = {}
    for l in open(args.results, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            if r['id'] in man and (r['id'] not in rows or r.get('status') == 'ok'):
                rows[r['id']] = r
    rows = list(rows.values())[args.shard::args.num_shards]
    if not rows:
        open(args.out, 'w').close()
        return

    from transformers import pipeline
    asr = pipeline('automatic-speech-recognition', model=args.model, device=args.device,
                   torch_dtype=torch.float16, chunk_length_s=30, stride_length_s=5)
    qwen = None
    if set(QWEN_LANG) & {man[r['id']]['tgt_lang'] for r in rows}:
        from qwen_asr import Qwen3ASRModel
        qwen = Qwen3ASRModel.from_pretrained(args.qwen_model, dtype=torch.bfloat16, device_map=args.device,
                                             max_inference_batch_size=args.batch, max_new_tokens=512)
        print(f'[asr] {"/".join(QWEN_LANG)} 目标语用 Qwen3-ASR 转写：{args.qwen_model}', flush=True)

    out = open(args.out, 'w', encoding='utf-8')
    todo = [r for r in rows if r.get('status') == 'ok' and r.get('wav') and os.path.isfile(r['wav'])
            and (r.get('output_sec') or 0) > 0.05 and r.get('segments')]
    todo_ids = {r['id'] for r in todo}
    for r in rows:
        if r['id'] not in todo_ids:
            out.write(json.dumps(dict(id=r['id'], asr='', note='empty_or_missing_wav'), ensure_ascii=False) + '\n')

    for lang in sorted({man[r['id']]['tgt_lang'] for r in todo}):
        sub = [r for r in todo if man[r['id']]['tgt_lang'] == lang]
        use_qwen = lang in QWEN_LANG
        gk = {'language': lang, 'task': 'transcribe', 'num_beams': 1}
        for i in range(0, len(sub), args.batch):
            chunk = sub[i:i + args.batch]
            wavs = [load_16k(r['wav']) for r in chunk]
            plain = run_qwen(qwen, wavs, QWEN_LANG[lang]) if use_qwen else run_whisper(asr, wavs, gk)
            tsd = run_whisper(asr, wavs, gk, return_timestamps='word')
            for r, p, o in zip(chunk, plain, tsd):
                m = man[r['id']]
                src = float(m['source_sec'])
                spans = word_spans(o.get('chunks', []), lang)
                ref_len = len(units_zh(m['reference']) if lang in CHAR_LANGS else units_en(m['reference']))
                rec = dict(id=r['id'], asr=(p.get('text') or '').strip(),
                           asr_model='qwen3-asr' if use_qwen else 'whisper',
                           asr_ts=(o.get('text') or '').strip(), n_hyp_units=len(spans), n_ref_units=ref_len,
                           source_sec=src)
                rec.update(latency_fields(r['segments'], spans, src, ref_len, ''))
                if r.get('segments_nca'):
                    rec.update(latency_fields(r['segments_nca'], spans, src, ref_len, '_nca'))
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            print(f'[asr {lang}] {min(i + args.batch, len(sub))}/{len(sub)}', flush=True)
    out.close()


if __name__ == '__main__':
    main()
