"""Run recovered report evaluators unchanged on actual S5 PCM events.

Quality uses the original Qwen-zh/Whisper-en ASR, OpenCC normalization and COMET.
Latency is not published here: accelerated input wall times are not live latency.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def rows(path):
    data = [json.loads(line) for line in Path(path).open() if line.strip()]
    if len({r['id'] for r in data}) != len(data):
        raise ValueError('duplicate result/manifest ids: ' + str(path))
    return data


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'results', 'events', 'whisper_model', 'qwen_model', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--asr_python', default=sys.executable)
    p.add_argument('--comet_python', default=sys.executable)
    p.add_argument('--comet_model')
    p.add_argument('--site_comet', default=os.environ.get('SITE_COMET'))
    p.add_argument('--asr_batch', type=int, default=16)
    p.add_argument('--skip_comet', action='store_true')
    a = p.parse_args()
    if a.asr_batch <= 0:
        p.error('asr_batch must be positive')
    for name in ('whisper_model', 'qwen_model'):
        if not Path(getattr(a, name)).is_dir():
            p.error(name + ' must be an existing model directory; ASR fallback is not permitted')
    if not a.skip_comet and (not a.comet_model or not Path(a.comet_model).is_file()):
        p.error('--comet_model is required unless --skip_comet is explicitly selected')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    source = json.loads((tools / 'SOURCE.json').read_text())
    for name, expected in source['files'].items():
        if hashlib.sha256((tools / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('recovered evaluator changed: ' + name)
    os.environ['FLUENCY_CODE'] = str(tools.parent / 'fluency')
    spec = importlib.util.spec_from_file_location('s5_original_report_metrics', tools / 'make_report.py')
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    if library._T2S is None:
        raise RuntimeError('original protocol requires OpenCC t2s; silent normalization fallback is forbidden')
    man = rows(a.manifest)
    values = {r['id']: r for r in rows(a.results)}
    timelines = {}
    offsets = {}
    for line in Path(a.events).open():
        e = json.loads(line)
        if e.get('type') != 'pcm':
            continue
        key = e['id']
        if int(e['num_samples']) <= 0 or int(e.get('sample_rate', 24000)) != 24000:
            raise ValueError('PCM event must contain positive mono24k sample count: ' + key)
        if int(e['sample_offset']) != offsets.get(key, 0):
            raise ValueError('PCM discontinuity: ' + key)
        offsets[key] = offsets.get(key, 0) + int(e['num_samples'])
        timelines.setdefault(key, []).append(dict(emit=float(e['elapsed_sec']),
            dur=int(e['num_samples']) / int(e.get('sample_rate', 24000))))
    groups = {}
    checked_tokenizers = set()
    for m in man:
        for name in ('source_sec', 'tgt_lang', 'reference', 'src_text'):
            if name not in m:
                raise ValueError('manifest lacks ' + name + ': ' + m['id'])
        if m['tgt_lang'] not in ('zh', 'en', 'ja', 'ko'):
            raise ValueError('report protocol supports en/zh/ja/ko targets')
        required_tokenizer = {'ja': 'ja-mecab', 'ko': 'ko-mecab'}.get(m['tgt_lang'])
        if required_tokenizer and required_tokenizer not in checked_tokenizers:
            if library.bleu_tokenize(m['tgt_lang']) != required_tokenizer:
                raise RuntimeError(f'{required_tokenizer} required; install sacrebleu[ja,ko]; char fallback is forbidden')
            checked_tokenizers.add(required_tokenizer)
        key = str(m.get('dataset', m['tgt_lang'])) + '__' + m['tgt_lang']
        groups.setdefault(key, []).append(m)
    (out / 'manifests').mkdir()
    (out / 'runs').mkdir()
    meta = dict(status='preparing', protocol=source, args=vars(a), commands=[],
                quality_only=True, latency_published=False,
                environment={key: os.environ.get(key) for key in
                             ('HF_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                              'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'PYTHONPATH')},
                input_sha256={name: hashlib.sha256(Path(getattr(a, name)).read_bytes()).hexdigest()
                              for name in ('manifest', 'results', 'events')})
    started = time.perf_counter()
    def invoke(command, name, overlay=None):
        env = dict(os.environ)
        if overlay:
            if not Path(overlay).is_dir():
                raise ValueError('missing dependency overlay: ' + overlay)
            env['PYTHONPATH'] = overlay + os.pathsep + env.get('PYTHONPATH', '')
        meta['commands'].append(dict(argv=command, overlay=overlay))
        write(out / 'protocol.json', meta)
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    try:
        for key, members in groups.items():
            # A stable filesystem name, independent of untrusted dataset punctuation.
            name = 'set_' + hashlib.sha256(key.encode()).hexdigest()[:12]
            d = out / 'runs' / name
            d.mkdir()
            jsonl(out / 'manifests' / (name + '.jsonl'), members)
            prepared = []
            for m in members:
                r = dict(values.get(m['id'], dict(id=m['id'], status='failed', error='missing_result')))
                if r.get('status') != 'ok':
                    r.update(text='', wav=None, output_sec=0)
                if r.get('wav') and r.get('output_sec', 0) > .05:
                    path = Path(r['wav'])
                    r['wav'] = str(path if path.is_absolute() else Path(a.results).resolve().parent / path)
                    if not Path(r['wav']).is_file() or m['id'] not in timelines:
                        raise ValueError('nonempty output requires actual WAV and PCM events: ' + m['id'])
                    import soundfile as sf
                    info = sf.info(r['wav'])
                    if info.samplerate != 24000 or info.channels != 1 or info.frames != offsets[m['id']]:
                        raise ValueError('actual WAV does not match PCM events: ' + m['id'])
                    if abs(offsets[m['id']] / 24000 - r['output_sec']) > .011:
                        raise ValueError('PCM events/WAV duration mismatch: ' + m['id'])
                r['segments'] = timelines.get(m['id'], [])
                r['source_sec'] = m['source_sec']
                prepared.append(r)
            jsonl(d / 'results.jsonl', prepared)
            meta['status'] = 'asr'
            invoke([a.asr_python, str(tools / 'asr_latency.py'), '--manifest', str(out / 'manifests' / (name + '.jsonl')),
                '--results', str(d / 'results.jsonl'), '--model', a.whisper_model, '--qwen_model', a.qwen_model,
                '--out', str(d / 'asr_lat.jsonl'), '--batch', str(a.asr_batch), '--device', 'cuda:0'], name + '-asr')
            transcript = rows(d / 'asr_lat.jsonl')
            if {r['id'] for r in transcript} != {m['id'] for m in members}:
                raise RuntimeError('ASR output coverage mismatch: ' + name)
            # The recovered library preserves individual ASR failures as empty
            # hypotheses. Keep it unchanged but never mark those runs complete.
            asr_log = (out / (name + '-asr.log')).read_text()
            if '单条失败' in asr_log:
                raise RuntimeError('original ASR evaluator reported individual failures: ' + name)
        if not a.skip_comet:
            meta['status'] = 'comet'
            invoke([a.comet_python, str(tools / 'comet_score.py'), '--root', str(out),
                    '--sets', ','.join(p.stem for p in sorted((out / 'manifests').glob('*.jsonl'))),
                    '--model', a.comet_model, '--batch', '32'], 'comet', a.site_comet)
        quality = []
        allowed = ('name', 'tgt_lang', 'n_manifest', 'n', 'n_ok', 'n_failed', 'empty_text', 'empty_wav',
                   'text_bleu', 'text_chrf', 'asr_bleu', 'asr_chrf', 'asr_comet', 'text_comet', 't2s', 'bleu_tokenize')
        for file in sorted((out / 'manifests').glob('*.jsonl')):
            score = library.score_set(str(out), file.stem)
            quality.append({key: score.get(key) for key in allowed})
        write(out / 'quality.json', dict(sets=quality, protocol=source['protocol'],
              asr_bleu_comet_complete=not a.skip_comet, full_report_complete=False,
              fluency='not_run', voice_quality='not_run', latency_published=False))
        meta['status'] = 'ok' if not a.skip_comet else 'asr_only'
    except Exception as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta['scoring_wall_sec'] = time.perf_counter() - started
        write(out / 'protocol.json', meta)


if __name__ == '__main__':
    main()
