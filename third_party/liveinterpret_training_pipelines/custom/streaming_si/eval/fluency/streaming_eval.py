"""对 results.jsonl 的全部译音做 VAD + MFA 服务强制对齐 + LLM 四类边界评测。

输入是已经按到达时刻排布的单声道译音，不使用源音频混音。首尾静音不计入 fluency。
默认使用 MFA 服务，不把文本到达时间当发声时间，不静默回退加权摊派。
逐条缓存 VAD/边界结果；仅改容忍阈值时重新计分，无需重调 LLM。
"""
import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import config
import metrics
import pipeline
from schema import BOUNDARY_TYPES

VERSION = 3  # v2：MFA 停顿定位改为“起点规则 + 不切开词”，并输出定位方式与旧位置


def read_rows(path):
    rows = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row['id']] = row
    return rows


def rescore(row, tolerance):
    row = dict(row)
    row['pauses'] = [dict(p) for p in row.get('pauses', [])]
    row['tolerance_table'] = dict(tolerance)
    by_type = {k: {'count': 0, 'abnormal_count': 0, 'total_s': 0.0, 'excess_s': 0.0}
               for k in BOUNDARY_TYPES}
    for p in row['pauses']:
        if not p['scored'] or p['kind'] != 'internal':
            continue
        tol = tolerance[p['boundary_type']]
        p['tolerance_s'] = tol
        p['excess_s'] = round(max(0.0, p['duration_s'] - tol), 3)
        b = by_type[p['boundary_type']]
        b['count'] += 1
        b['abnormal_count'] += int(p['excess_s'] > 0)
        b['total_s'] += p['duration_s']
        b['excess_s'] += p['excess_s']
        src = row['source_sec']
        p['source_phase'] = 'before_end' if p['start_s'] < src else 'after_end'
        p['crosses_source_end'] = p['start_s'] < src < p['end_s']
        excess_start = p['end_s'] - p['excess_s']
        p['excess_before_source_end_s'] = round(max(0.0, min(p['end_s'], src) - excess_start), 3)
        p['excess_after_source_end_s'] = round(p['excess_s'] - p['excess_before_source_end_s'], 3)
    row['by_boundary_type'] = by_type
    row['total_excess_s'] = round(sum(v['excess_s'] for v in by_type.values()), 3)
    chars = row.get('transcript_chars', 0)
    row['abnormal_pause_burden'] = round(row['total_excess_s'] / chars * 100, 4) if chars else None
    row['abnormal_pause_count'] = sum(v['abnormal_count'] for v in by_type.values())
    return row


def worker(task):
    try:
        for _ in range(3):
            rep = pipeline.evaluate(task['audio'], text=task['text'], use_llm=True,
                                    model=task['model'], api_base=task['api_base'],
                                    tolerance=task['tolerance'], align_source=task['align_source'],
                                    language=task['language'], alignment_cache_dir=task['alignment_cache_dir'],
                                    whisper_words=task.get('whisper_words'))
            internal = [p for p in rep['pauses'] if p['scored'] and p['kind'] == 'internal']
            complete = all(p.get('classification_source') == 'llm' for p in internal)
            if complete:
                break
        rep.update(id=task['id'], status='ok', source_sec=task['source_sec'],
                   transcript_source=task['transcript_source'], llm_complete=complete,
                   analysis_fingerprint=task['fingerprint'], llm_model=task['model'],
                   vad_threshold=config.VAD_THRESHOLD)
        return rescore(rep, task['tolerance'])
    except Exception as exc:
        if isinstance(exc, ValueError) and 'VAD 未检测到任何语音' in str(exc):
            return dict(id=task['id'], status='skipped', reason='no_vad_speech', source_sec=task['source_sec'])
        error = f'{type(exc).__name__}: {exc}'
        print(f'[fluency][error] {task["id"]}: {error}', flush=True)
        return dict(id=task['id'], status='error', source_sec=task['source_sec'],
                    error=error, analysis_fingerprint=task['fingerprint'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', required=True)
    ap.add_argument('--asr', required=True)
    ap.add_argument('--audio-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--model', default=config.LITELLM_MODEL)
    ap.add_argument('--api-base', default=config.LITELLM_API_BASE)
    ap.add_argument('--tolerance', nargs='*', default=[])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--align-source', choices=['uniform', 'mfa', 'whisper'], default=os.getenv('FLUENCY_ALIGN_SOURCE', 'mfa'))
    ap.add_argument('--words-file', help='align-source=whisper 时：whisper_align.py extract 的词级时间戳 jsonl（按 id 索引）')
    ap.add_argument('--alignment-cache-dir')
    args = ap.parse_args()
    tolerance = dict(config.TOLERANCE_S)
    for item in args.tolerance:
        key, value = item.split('=', 1)
        if key not in tolerance or not math.isfinite(float(value)) or float(value) < 0:
            ap.error(f'无效容忍阈值：{item}')
        tolerance[key] = float(value)
    if not (os.getenv('LITELLM_API_KEY') or os.getenv('SPEECH_LLM_API_KEY')):
        ap.error('未配置 prod LITELLM_API_KEY')
    rows, asr, cached = read_rows(args.results), read_rows(args.asr), read_rows(args.out)
    tasks, output = [], {}
    params = dict(version=VERSION, model=args.model, api_base=args.api_base,
                  vad=config.VAD_THRESHOLD, min_silence=config.VAD_MIN_SILENCE_MS,
                  min_speech=config.VAD_MIN_SPEECH_MS, min_pause=config.MIN_PAUSE_S,
                  count_punctuation=config.COUNT_PUNCTUATION, align_source=args.align_source)
    if args.align_source == 'mfa':
        import mfa_alignment
        params['alignment'] = mfa_alignment.settings()
    elif args.align_source == 'whisper':
        if not args.words_file:
            ap.error('--align-source whisper 需要 --words-file')
        params['alignment'] = dict(engine='faster-whisper-large-v3-int8', resolution='word')
    words = read_rows(args.words_file) if (args.align_source == 'whisper' and args.words_file) else {}
    selected = list(rows.values())[:args.limit] if args.limit else list(rows.values())
    for r in selected:
        uid = r['id']
        audio = Path(args.audio_dir) / f'{uid}.wav'
        text = (asr.get(uid, {}).get('asr') or r.get('text') or '').strip()
        reason = None
        if r.get('status') != 'ok':
            reason = 'inference_failed'
        elif not r.get('wav') or not r.get('segments') or (r.get('output_sec') or 0) <= 0.05:
            reason = 'empty_audio'
        elif not text:
            reason = 'empty_transcript'
        if reason:
            output[uid] = dict(id=uid, status='skipped', reason=reason, source_sec=r['source_sec'])
            continue
        if not audio.is_file():
            output[uid] = dict(id=uid, status='error', error=f'缺少时间轴译音：{audio}')
            continue
        stat = audio.stat()
        language = r.get('tgt_lang')
        if args.align_source == 'mfa' and language not in ('zh', 'en'):
            output[uid] = dict(id=uid, status='error', error=f'缺少支持的 tgt_lang：{language}')
            continue
        if args.align_source == 'whisper':
            wrow = words.get(uid)
            if not wrow or wrow.get('status') != 'ok' or not wrow.get('words'):
                output[uid] = dict(id=uid, status='error',
                                   error=f'缺少 whisper 词级时间戳：{uid}')
                continue
            if wrow.get('language') != language or language not in ('ja', 'ko'):
                output[uid] = dict(id=uid, status='error', error=f'tgt_lang 不符：{language}')
                continue
            text = wrow['text'].strip()
            if not text:
                output[uid] = dict(id=uid, status='skipped', reason='empty_transcript_whisper')
                continue
        fingerprint = hashlib.sha256(json.dumps(dict(params, audio=str(audio), size=stat.st_size,
                                                      mtime=stat.st_mtime_ns, text=text, language=language),
                                                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        old = cached.get(uid, {})
        if old.get('analysis_fingerprint') == fingerprint and old.get('status') == 'ok' and old.get('llm_complete'):
            output[uid] = rescore(old, tolerance)
        else:
            tasks.append(dict(id=uid, audio=str(audio), text=text, source_sec=r['source_sec'],
                              transcript_source=('whisper' if args.align_source == 'whisper'
                                                 else ('asr' if asr.get(uid, {}).get('asr') else 'model_text')),
                              fingerprint=fingerprint, tolerance=tolerance, model=args.model, api_base=args.api_base,
                              align_source=args.align_source, language=language,
                              whisper_words=words.get(uid),
                              alignment_cache_dir=args.alignment_cache_dir or str(Path(args.out).parent / 'mfa_cache')))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in output.values():
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[fluency] total={len(selected)} cached/skipped={len(output)} pending={len(tasks)} jobs={args.jobs}', flush=True)
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool, open(args.out, 'a', encoding='utf-8') as f:
            futures = [pool.submit(worker, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                output[r['id']] = r
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                if i % 20 == 0 or i == len(tasks):
                    print('[ktrain_metrics] ' + json.dumps(dict(step=i, max_steps=len(tasks), stage='fluency')), flush=True)
    errors = [r for r in output.values() if r['status'] == 'error' or (r['status'] == 'ok' and not r['llm_complete'])]
    print(f'[fluency] ok={sum(r["status"] == "ok" for r in output.values())} skipped={sum(r["status"] == "skipped" for r in output.values())} incomplete={len(errors)} -> {args.out}', flush=True)
    if errors:
        print('[fluency] 未完成样本（可断点续跑）：' + ', '.join(r['id'] for r in errors[:10]), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
