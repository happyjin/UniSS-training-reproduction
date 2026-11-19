"""SageMaker 强制对齐服务；时间戳不是人工真值，不使用均匀摊派兜底。"""
import base64
import hashlib
import io
import json
import math
import os
import time
import unicodedata
import uuid
from pathlib import Path

VERSION = 1


def settings():
    return dict(endpoint=os.getenv('MFA_ENDPOINT', 'alignment-diarization-hok-esport-dev'),
                region=os.getenv('MFA_REGION', 'ap-southeast-3'),
                version=VERSION)


def canonical(text):
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def normalize_text(text, language):
    if language not in ('zh', 'en'):
        raise ValueError(f'未验证的对齐语言：{language}')
    chars = []
    for i, c in enumerate(text):
        if unicodedata.category(c).startswith(('P', 'S')):
            if language == 'en' and c in "'’" and i > 0 and i + 1 < len(text) and text[i-1].isalnum() and text[i+1].isalnum():
                chars.append("'")
            elif language == 'en':
                chars.append(' ')
        else:
            chars.append(c)
    value = ' '.join(''.join(chars).split())
    if not canonical(value):
        raise ValueError('对齐文本规范化后为空')
    return value


def parse_response(response, text, duration):
    if not isinstance(response, dict):
        raise ValueError('对齐服务返回非对象响应')
    if response.get('error'):
        raise ValueError(f'对齐服务返回错误：{str(response["error"])[:200]}')
    result = response.get('result')
    if not isinstance(result, dict) or not isinstance(result.get('words'), list):
        raise ValueError('对齐服务未返回 result.words 时间戳')
    original, offsets = [], []
    for i, ch in enumerate(text):
        for c in canonical(ch):
            original.append(c)
            offsets.append(i)
    expected = ''.join(original)
    cursor, units = 0, []
    prev_start = -1.0
    for word in result['words']:
        token = canonical(word.get('text', ''))
        if not token:
            continue
        if expected[cursor:cursor + len(token)] != token:
            raise ValueError('对齐返回文本与评分文本不一致，拒绝近似字符匹配')
        a, b = float(word['start']), float(word['end'])
        if not math.isfinite(a) or not math.isfinite(b) or a < -0.05 or b < a or b > duration + 0.15 or a < prev_start:
            raise ValueError('对齐时间戳无效或不在音频时基内')
        score = word.get('score')
        score = float(score) if score is not None else None
        units.append(dict(text=word['text'], start_s=max(0.0, a), end_s=b,
                          char_start=offsets[cursor], char_end=offsets[cursor + len(token)-1] + 1,
                          score=score))
        cursor += len(token)
        prev_start = a
    if not units or cursor != len(expected):
        raise ValueError('对齐结果未覆盖完整评分文本')
    return units


def invoke(payload):
    import boto3
    from botocore.config import Config
    cfg = settings()
    client = boto3.client('sagemaker-runtime', region_name=cfg['region'],
                          config=Config(connect_timeout=15, read_timeout=70,
                                        retries={'mode': 'standard', 'max_attempts': 3}))
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    if len(body) > 6 * 1024 * 1024:
        raise ValueError('对齐请求超过 SageMaker 同步端点 6 MiB 上限')
    for attempt in range(3):
        try:
            response = client.invoke_endpoint(EndpointName=cfg['endpoint'],
                                              ContentType='application/json', Body=body)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as exc:
            code = getattr(exc, 'response', {}).get('Error', {}).get('Code', '')
            if code in ('AccessDeniedException', 'ValidationError') or attempt == 2:
                raise
            time.sleep(2 ** attempt)


def align(audio_path, text, language, info, cache_dir=None):
    import soundfile as sf
    import audio_vad
    cfg = settings()
    normalized = normalize_text(text, language)
    raw = Path(audio_path).read_bytes()
    fingerprint = hashlib.sha256(raw + json.dumps(dict(cfg, text=text, language=language,
                                                       crop_start=info['segments'][0].start,
                                                       crop_end=info['segments'][-1].end),
                                                  sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = Path(cache_dir) / f'{fingerprint}.json' if cache_dir else None
    if cache and cache.is_file():
        response = json.loads(cache.read_text(encoding='utf-8'))
    else:
        samples, _ = audio_vad.load_audio_16k(audio_path)
        begin = max(0, int(info['segments'][0].start * 16000))
        end = min(len(samples), int(math.ceil(info['segments'][-1].end * 16000)))
        offset = begin / 16000
        duration = (end - begin) / 16000
        buf = io.BytesIO()
        sf.write(buf, samples[begin:end].numpy(), 16000, format='WAV', subtype='PCM_16')
        payload = dict(audio=base64.b64encode(buf.getvalue()).decode('ascii'),
                       trace=str(uuid.uuid4()), language=language,
                       segments=[dict(text=normalized, start=0.0, end=duration)])
        response = invoke(payload)
        response = dict(response, crop_offset_s=offset, crop_duration_s=duration)
        parse_response(response, text, duration)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix(f'.{uuid.uuid4().hex}.tmp')
            temp.write_text(json.dumps(response, ensure_ascii=False), encoding='utf-8')
            os.replace(temp, cache)
    units = parse_response(response, text, response['crop_duration_s'])
    offset = response['crop_offset_s']
    for u in units:
        u['start_s'] += offset
        u['end_s'] += offset
    scores = [u['score'] for u in units if u['score'] is not None]
    return dict(units=units, endpoint=cfg['endpoint'], region=cfg['region'],
                language=language, model_version=response.get('model_version'),
                resolution='character' if language == 'zh' else 'word',
                mean_score=sum(scores) / len(scores) if scores else None,
                low_score_units=sum(s < 0.1 for s in scores), crop_offset_s=offset,
                cache_fingerprint=fingerprint)


NEAR_S = 0.1        # 间隙与停顿相距不超过该值，视为可能位置
ONSET_EPS_S = float(os.getenv('MFA_ONSET_EPS_S', '0.03'))      # 起点早于停顿开始至少该值，才算停顿前说出
STRADDLE_S = 0.05   # 单元与 VAD 静音重叠超过该值：MFA 把静音并进了这个字/词
SQUEEZED_S = float(os.getenv('MFA_SQUEEZED_S', '0.05'))        # 短于该值：MFA 没地方放、把它挤扁了
TIE_S = 0.05        # 被拉长单元在静音前/后的发声差不足该值时，改看起点
WINDOW_S = (0.1, 0.2)   # 起点落在 [停顿开始-0.1, 停顿结束+0.2] 内的单元，其前方间隙也列为候选
PARTICLES = set('的了吗呢吧啊呀着过么嘛哦')   # 单独成词时只能附着在前一个词上，不会出现在停顿之后的开头
_JIEBA = None


def _blocked_offsets(text):
    """中文里不可能是停顿点的字符偏移：分词词内部 + 单独成词的助词之前。非中文或无 jieba 时返回空集。"""
    global _JIEBA
    if not text or not any('\u4e00' <= c <= '\u9fff' for c in text):
        return set()
    if _JIEBA is None:
        try:
            import logging
            import jieba
            jieba.setLogLevel(logging.ERROR)
            _JIEBA = jieba
        except ImportError:
            _JIEBA = False
    if not _JIEBA:
        return set()
    blocked, pos = set(), 0
    for w in _JIEBA.lcut(text, HMM=False):
        blocked.update(range(pos + 1, pos + len(w)))
        if w in PARTICLES:
            blocked.add(pos)
        pos += len(w)
    return blocked


def locate_pauses(pauses, units, segments, speech_s, text=None):
    """把 VAD 停顿定位到相邻计时单元之间。

    v2（起点规则 + 不切词）：MFA 在停顿处常把静音并进某个字（拉长），再把相邻的字挤成几十毫秒，
    v1 的“间隙离停顿最近”因此系统性偏早一个字。v2：
      1) 默认位置：起点早于停顿开始（留 ONSET_EPS_S 余量）的单元算停顿前；英文词被拉长时
         改看它在静音前/后哪边发声更多（汉字单音节，起点更可靠）；
      2) 候选：默认位置、被拉长单元两侧、紧随其后被挤扁的单元之后、起点落在停顿窗口内的单元之前；
      3) 中文去掉不可能的候选（分词词内部、单独成词的助词之前；全被去掉则保留全部）；
         默认位置仍在则用它，否则取离默认位置最近的候选（并列取靠后）。
    v1 结果保留在 alignment_legacy_index，便于审计。
    """
    import alignment
    if len(units) < 2 and any(p.scored and p.kind == 'internal' for p in pauses):
        raise ValueError('对齐仅有一个计时单元，无法定位句中停顿')
    boundaries = []
    for left, right in zip(units, units[1:]):
        a, b = left['end_s'], right['start_s']
        boundaries.append(dict(a=min(a, b), b=max(a, b), idx=right['char_start'], left=left, right=right))
    blocked = _blocked_offsets(text)
    cjk = bool(text) and any('\u4e00' <= c <= '\u9fff' for c in text)
    last = len(units) - 1
    for p in pauses:
        if not p.scored or p.kind != 'internal':
            continue
        mid = (p.start + p.end) / 2
        for bd in boundaries:
            bd['dist'] = max(0.0, bd['a'] - p.end, p.start - bd['b'])
        legacy = min(boundaries, key=lambda x: (x['dist'], abs((x['a'] + x['b']) / 2 - mid)))
        stretched = [min(u['end_s'], p.end) - max(u['start_s'], p.start) > STRADDLE_S for u in units]

        def before_pause(i):
            u = units[i]
            if not cjk and stretched[i]:
                pre = max(0.0, min(u['end_s'], p.start) - u['start_s'])
                post = max(0.0, u['end_s'] - max(u['start_s'], p.end))
                if abs(pre - post) >= TIE_S:
                    return pre > post
            return u['start_s'] < p.start - ONSET_EPS_S

        n = min(max(sum(before_pause(i) for i in range(len(units))), 1), last)
        cands = {n}
        for i, s in enumerate(stretched):
            if s:
                cands.update(k for k in (i, i + 1) if 1 <= k <= last)
                j = i + 1
                while j < last and units[j]['end_s'] - units[j]['start_s'] < SQUEEZED_S:
                    cands.add(j + 1)
                    j += 1
        cands.update(k for k in range(1, last + 1)
                     if p.start - WINDOW_S[0] <= units[k]['start_s'] <= p.end + WINDOW_S[1])
        keep = sorted(k for k in cands if boundaries[k - 1]['idx'] not in blocked) or sorted(cands)
        if n in keep:
            pick, method = n, 'onset'
        else:
            pick, method = min(keep, key=lambda k: (abs(k - n), -k)), 'onset+word'
        best = boundaries[pick - 1]
        suspects = [u['text'] for u in units
                    if min(u['end_s'], p.end) - max(u['start_s'], p.start) > STRADDLE_S
                    or (u['end_s'] - u['start_s'] < SQUEEZED_S
                        and max(0.0, u['start_s'] - p.end, p.start - u['end_s']) <= NEAR_S)]
        p.char_index = best['idx']
        p.speech_time = alignment.audio_time_to_speech_time(p.start, segments)
        p.progress = p.speech_time / speech_s if speech_s else 0.0
        p.extra.update(align_distance_s=round(best['dist'], 4),
                       alignment_ambiguous=best['dist'] > NEAR_S
                       or (len(keep) > 1 and best['idx'] != legacy['idx'])
                       or any(min(u['end_s'], p.end) - max(u['start_s'], p.start)
                              > max(0.6 * (p.end - p.start), 0.2 * (u['end_s'] - u['start_s']))
                              for u in units),
                       alignment_candidates=[boundaries[k - 1]['idx'] for k in keep],
                       alignment_method=method,
                       alignment_legacy_index=legacy['idx'],
                       alignment_suspects=suspects,
                       aligned_left=best['left']['text'], aligned_right=best['right']['text'])
    return sum(p.extra.get('alignment_ambiguous', False) for p in pauses if p.scored)
