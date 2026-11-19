"""Reproducible S5 actual-PCM benchmark (stdlib-only orchestration/analysis).

Quality commands are explicit argv arrays in a JSON file. They must use the
same protocol for paired runs; historical report scores are not a new baseline.
No simulated vocoder allocation or legacy stream_segments.py is used here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


CHUNKS = ((640, None), (1280, None), (1920, None), (2560, None))


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * probability
    lo = math.floor(x)
    return values[lo] + (values[math.ceil(x)] - values[lo]) * (x - lo)


def distribution(values):
    return {'n': len(values), **{f'p{int(p * 100)}': quantile(values, p) for p in (.5, .95, .99)}}


def probe_rows(rows, count=40):
    """Deterministic balanced datasets, including shortest/longest per group."""
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('manifest contains duplicate ids')
    groups = {}
    for row in rows:
        groups.setdefault(row.get('dataset', row.get('tgt_lang', 'unknown')), []).append(row)
    result, keys = [], sorted(groups)
    for index, key in enumerate(keys):
        group = sorted(groups[key], key=lambda r: (r['source_sec'], r['id']))
        n = min(len(group), count // len(keys) + (index < count % len(keys)))
        positions = [round(i * (len(group) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0] * n
        result.extend(group[i] for i in positions)
    return result


def analyze(results, events):
    by_id = {}
    for event in events:
        by_id.setdefault(event['id'], []).append(event)
    first_pcm, first_text, pcm_intervals, step_compute, issues = [], [], [], [], []
    samples_total, simulated_underruns = 0, []
    for row in results:
        key = row['id']
        if row.get('status') != 'ok':
            issues.append(f'{key}: inference failed')
            continue
        seq = by_id.get(key, [])
        pcm = [e for e in seq if e['type'] == 'pcm']
        text = [e for e in seq if e['type'] == 'text' and e.get('text')]
        if not pcm:
            issues.append(f'{key}: no PCM events')
            continue
        first_pcm.append(pcm[0]['elapsed_sec'])
        if text:
            first_text.append(text[0]['elapsed_sec'])
        offset, previous_time, play_end = 0, None, None
        for event in pcm:
            if event['sample_offset'] != offset:
                issues.append(f'{key}: discontinuous sample offset {event["sample_offset"]} != {offset}')
            count = event['num_samples']
            if count <= 0:
                issues.append(f'{key}: empty PCM event')
            offset += count
            at = event['elapsed_sec']
            if previous_time is not None:
                if at < previous_time:
                    issues.append(f'{key}: nonmonotonic PCM time')
                pcm_intervals.append(at - previous_time)
            # Server playback proxy only: excludes network/browser; never label as real underrun.
            if play_end is not None and at > play_end:
                simulated_underruns.append(at - play_end)
            play_end = max(play_end or at, at) + count / event.get('sample_rate', 24000)
            previous_time = at
        samples_total += offset
        if pcm[0]['source_sec'] >= row['source_sec']:
            issues.append(f'{key}: first PCM after source end (may be a short/WAIT-only sentence)')
        step_compute.extend(e['compute_sec'] for e in seq if e['type'] == 'step' and 'compute_sec' in e)
    source_sec = sum(r.get('source_sec', 0) for r in results if r.get('status') == 'ok')
    return {
        'n_results': len(results), 'n_ok': sum(r.get('status') == 'ok' for r in results),
        'source_sec': source_sec, 'pcm_samples': samples_total,
        'first_pcm_wall_sec': distribution(first_pcm), 'first_text_wall_sec': distribution(first_text),
        'pcm_interarrival_sec': distribution(pcm_intervals), 'source_step_compute_sec': distribution(step_compute),
        'server_playback_proxy_gaps': distribution(simulated_underruns),
        'server_playback_proxy_gap_total_sec': sum(simulated_underruns),
        'issues': issues, 'browser_playback_measured': False,
    }


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(args.manifest)
    if args.stage == 'probe':
        rows = probe_rows(rows)
    else:
        if not args.configuration:
            raise ValueError('full stage requires one reviewed --configuration; do not run the entire sweep blindly')
        if len(rows) != 1193:
            raise ValueError('full stage requires exactly 1193 manifest rows')
        if not args.probe_report:
            raise ValueError('full stage requires --probe_report from reviewed 40-sentence paired run')
        probe = json.loads(Path(args.probe_report).read_text())
        if probe.get('n_results') != 40 or probe.get('n_ok') != 40 or not probe.get('pcm_samples'):
            raise ValueError('probe has failures/missing PCM; repair before full run')
    manifest = out / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary'], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, diff = args.source_revision or 'unavailable-in-source-snapshot', b''
    (out / 'working_tree.patch').write_bytes(diff)
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (root / 'custom/streaming_si').rglob('*.py')}
    write_json(out / 'provenance.json', {'commit': revision, 'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'source_sha256': source_hashes, 'args': vars(args),
        'quality_protocol': 'explicit commands only; do not compare historical scores with another ASR/normalizer'})
    score_config = json.loads(Path(args.quality_commands).read_text()) if args.quality_commands else []
    if not isinstance(score_config, list) or any(not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd) for cmd in score_config):
        raise ValueError('quality commands must be a JSON array of nonempty argv arrays')
    for chunk, after in CHUNKS:
        if args.configuration and args.configuration != f'{chunk}' + (f'-{after}' if after else ''):
            continue
        for repeat in range(args.repeats):
            name = f'{chunk}' + (f'-{after}' if after else '') + f'-r{repeat + 1}'
            dest = out / name
            dest.mkdir()
            command = [args.python, '-m', 'streaming_si.inference.infer_s5_streaming_dynchunk',
                '--model_dir', args.model_dir, '--adapter', args.adapter,
                '--input_jsonl', str(manifest), '--out_dir', str(dest),
                '--synthesis_mode', 'streaming', '--input_pacing', args.pacing,
                '--chunk_ms', str(chunk), '--retry', '0', '--max_audio_sec', '0',
                '--warmup_audio', max(rows, key=lambda row: row['source_sec'])['audio'],
                '--event_jsonl', str(dest / 'events.jsonl')]
            if after:
                command += ['--chunk_ms_after_write', str(after)]
            command += args.infer_args
            write_json(dest / 'command.json', command)
            with open(dest / 'infer.log', 'w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = analyze(read_jsonl(dest / 'results.jsonl'), read_jsonl(dest / 'events.jsonl'))
            report['runtime_summary'] = json.loads((dest / 'summary.json').read_text())
            write_json(dest / 'actual_pcm_report.json', report)
            # Commands are argv arrays, never shell strings. Protocol/model files are supplied explicitly.
            score_start = time.perf_counter()
            for i, template in enumerate(score_config):
                if not isinstance(template, list) or not all(isinstance(x, str) for x in template):
                    raise ValueError('quality commands must be a JSON array of argv arrays')
                cmd = [x.replace('{run_dir}', str(dest)).replace('{manifest}', str(manifest)) for x in template]
                with open(dest / f'quality-{i}.log', 'w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(dest / 'quality_status.json', {'ran': bool(score_config),
                'wall_sec': time.perf_counter() - score_start, 'commands': score_config})


def longrun(args):
    """One persistent session; repeat a fixture without accumulating input audio.

    `silence` feeds real zero-valued samples. `no_packets` waits on wall time and
    sends nothing; it does not advance source position or reset the session.
    """
    import os
    import numpy as np
    import soundfile as sf
    import torch
    from streaming_si.inference.infer_s5_streaming_dynchunk import load_model
    from streaming_si.inference.infer_s2st import CampplusSpeaker, build_source_prompt, load_wav16
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from streaming_si.model.speech_decoder import resolve_speech_decoder
    from streaming_si.inference.s5_session import S5Session
    from streaming_si.inference.s5_segmenter import SegmentedSession, SileroFrameVad, segmentation_options

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cold_start = time.perf_counter()
    torch.manual_seed(0)
    model, tokenizer, ids = load_model(args.model_dir, args.adapter)
    model = model.to(args.device)
    dec_dir, yaml = resolve_speech_decoder(args.model_dir, args.cosyvoice_dir, None)
    model.config.talker_config_yaml = yaml
    model.attach_speech_decoder(dec_dir, device=args.device, cosyvoice_repo=args.cosyvoice_repo, strict=True)
    speaker = CampplusSpeaker(os.path.join(dec_dir, 'campplus.onnx'))
    prompter = build_source_prompt(True, dec_dir, args.prompt_tokenizer)
    mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.model_dir))
    fixture = load_wav16(args.audio, max_sec=None)
    if fixture.size == 0:
        raise ValueError('empty fixture')
    vad = SileroFrameVad(args.vad_threshold) if args.segmentation == 'vad' else None

    def fresh(callback=None):
        if vad is not None:
            vad.reset()
        core = S5Session(model, tokenizer, ids, mel_fn, args.tgt_lang, device=args.device,
            chunk_ms=args.chunk_ms, speaker=speaker, prompter=prompter,
            context_max_tokens=args.context_max_tokens, context_window_sec=args.context_window_sec,
            acoustic_max_sec=args.acoustic_max_sec,
            reference_policy=args.reference_policy, reference_refresh_sec=args.reference_refresh_sec,
            collect_audio=False, on_event=None if vad is not None else callback)
        wrapped = (SegmentedSession(core, vad, on_event=callback, **segmentation_options(args))
                   if args.segmentation != 'none' else core)
        torch.manual_seed(0)
        return core, wrapped
    # Warm the exact streaming path on a separate session, without callbacks,
    # audio collection, events, or membership in the measured source duration.
    warmup_start = time.perf_counter()
    warm_core, warm_session = fresh()
    torch.manual_seed(0)  # Graph preparation in the constructor must not shift sampling RNG.
    # 预热只需要跑通建图/建缓存，不必把整段 fixture 跑一遍：长 fixture（如 600s）会把预热
    # 变成一次几乎等长的完整推理。这里截前 30 秒预热，短 fixture 行为不变。
    warm_fixture = fixture[:min(fixture.size, 30 * 16000)]
    with torch.inference_mode():
        for offset in range(0, warm_fixture.size, 1280):
            warm_session.feed_audio(warm_fixture[offset:offset + 1280])
        warm_session.feed_audio(np.zeros(0, np.float32), finalize=True)
    if warm_core.output_samples <= 0:
        raise RuntimeError('warmup produced no PCM; Flow/HiFT have not been validated')
    warmup_sec = time.perf_counter() - warmup_start
    del warm_session, warm_core
    print(json.dumps(dict(warmup_wall_sec=warmup_sec, warmup_source_sec=warm_fixture.size / 16000)), flush=True)
    source_samples = int(args.duration_sec * 16000)
    chunk = 1280  # Same 80ms transport packets as CLI/Web, independent of model chunk.
    pause_at = int(args.pause_at_sec * 16000)
    pause_done, sent, cursor, scheduled_wait, packet_pause = False, 0, 0, 0., 0.
    sample_offset = 0
    started = time.perf_counter()
    with open(out / 'events.jsonl', 'w') as events, sf.SoundFile(
            str(out / 'output.wav'), mode='w', samplerate=24000, channels=1, subtype='PCM_16') as output:
        def callback(event):
            nonlocal sample_offset
            data = {k: v for k, v in event.items() if k != 'pcm'}
            data['id'] = 'longrun'
            if event['type'] == 'pcm':
                if event['sample_offset'] != sample_offset:
                    raise RuntimeError('PCM gap/duplication during window rollover')
                data['num_samples'] = event['pcm'].size
                sample_offset += data['num_samples']
                output.write(event['pcm'])
            if event['type'] in ('step', 'context', 'end', 'pause_start', 'pause_end') and torch.cuda.is_available():
                data['gpu_allocated_bytes'] = torch.cuda.memory_allocated()
                data['gpu_reserved_bytes'] = torch.cuda.memory_reserved()
                state = dict(thinker_tokens=sum(int(x.size(1)) for x in core.embeds),
                    talker_tokens=int(core.talker_fed),
                    reference_samples=int(core.reference_audio.size),
                    frontend_samples=int(core.waveform_stream._buffer.size),
                    pending_audio_samples=int(core.pending_audio.size),
                    pending_mel_frames=int(core.pending_mel.size(-1)))
                if state['pending_audio_samples'] > args.chunk_ms * 16 + 1280:
                    raise RuntimeError('source chunk buffer exceeded model chunk plus transport packet')
                if state['pending_mel_frames'] > args.chunk_ms // 10 + 8:
                    raise RuntimeError('stable mel buffer exceeded one model chunk plus tail')
                cache = None if core.vcache is None else core.vcache.past_key_values
                if cache is not None:
                    lengths = [int(layer.keys.shape[-2]) for layer in cache.layers if layer.keys is not None]
                    state['voxtral_kv_tokens'] = max(lengths, default=0)
                    state['voxtral_absolute_tokens'] = int(getattr(cache, '_voxtral_seen_tokens', 0))
                    window = core.model.audio_encoder.encoder.config.sliding_window
                    if window is not None and state['voxtral_kv_tokens'] > int(window) - 1:
                        raise RuntimeError('Voxtral physical cache exceeded sliding window')
                if core.decoder is not None:
                    state.update(flow_tokens=len(core.decoder.tokens),
                        flow_mel_frames=int(core.decoder.generated_mel.size(-1)),
                        hift_prefix_mel_frames=int(core.decoder.hift_prefix.size(-1)),
                        acoustic_rollovers=core.decoder.session_rollovers)
                    hs = getattr(core.decoder, 'hift_stream', None)
                    if hs is not None:
                        state.update(hift_retained_mel_frames=int(hs.retained_frames),
                            hift_absolute_mel_start=int(hs.base_frame),
                            hift_context_crops=int(hs.crops),
                            hift_cached_source_samples=0 if hs.source_cache is None else int(hs.source_cache.size(-1)))
                        # EOF can append the final incomplete synthesis hop.
                        tail = core.decoder.max_hop * core.decoder.ratio if hs.closed else 0
                        if hs.retained_frames > hs.capacity_frames + tail:
                            raise RuntimeError('HiFT retained unbounded mel context')
                    flow_bound = core.decoder.max_tokens + core.decoder.lookahead
                    if state['flow_tokens'] > flow_bound or state['flow_mel_frames'] > flow_bound * core.decoder.ratio:
                        raise RuntimeError('acoustic session retained unbounded token/mel history')
                if state['thinker_tokens'] > core.capacity or state['talker_tokens'] > core.capacity:
                    raise RuntimeError('Thinker/Talker state exceeded configured capacity')
                if state['reference_samples'] > core.reference_samples:
                    raise RuntimeError('source reference window exceeded configured capacity')
                data['retained_state'] = state
            events.write(json.dumps(data, ensure_ascii=False) + '\n')
            events.flush()
        core, session = fresh(callback)
        torch.manual_seed(0)
        started = session.started
        with torch.inference_mode():
            while sent < source_samples:
                if not pause_done and args.pause_kind != 'none' and sent >= pause_at:
                    callback(dict(type='pause_start', elapsed_sec=time.perf_counter()-started,
                                  source_sec=(session.feeder.processed_samples if vad is not None else core.consumed_samples)/16000,
                                  received_source_sec=(session.feeder.received_samples if vad is not None else core.source_samples)/16000,
                                  pause_kind=args.pause_kind))
                    if args.pause_kind == 'no_packets':
                        before = time.perf_counter()
                        time.sleep(args.pause_sec)
                        packet_pause += time.perf_counter() - before
                    else:
                        remaining = int(args.pause_sec * 16000)
                        silence_fed = 0
                        while remaining:
                            n = min(chunk, remaining)
                            if args.pacing == 'realtime':
                                deadline = started + (sent + silence_fed + n) / 16000 + packet_pause
                                delay = deadline - time.perf_counter()
                                if delay > 0:
                                    before = time.perf_counter()
                                    time.sleep(delay)
                                    scheduled_wait += time.perf_counter() - before
                            session.feed_audio(np.zeros(n, np.float32))
                            silence_fed += n
                            remaining -= n
                    pause_done = True
                    callback(dict(type='pause_end', elapsed_sec=time.perf_counter()-started,
                                  source_sec=(session.feeder.processed_samples if vad is not None else core.consumed_samples)/16000,
                                  received_source_sec=(session.feeder.received_samples if vad is not None else core.source_samples)/16000,
                                  pause_kind=args.pause_kind))
                n = min(chunk, source_samples-sent, fixture.size-cursor)
                if args.pacing == 'realtime':
                    # Keep a source-paced deadline, allowing inference to catch up after a slow chunk.
                    extra_silence = args.pause_sec if pause_done and args.pause_kind == 'silence' else 0
                    deadline = started + (sent+n)/16000 + extra_silence + packet_pause
                    delay = deadline-time.perf_counter()
                    if delay > 0:
                        before = time.perf_counter()
                        time.sleep(delay)
                        scheduled_wait += time.perf_counter()-before
                session.feed_audio(fixture[cursor:cursor+n])
                cursor = (cursor+n) % fixture.size
                sent += n
            session.feed_audio(np.zeros(0, np.float32), finalize=True)
    result = session.result()
    result.pop('wav', None)
    result.update(id='longrun', status='ok', input_pacing=args.pacing,
                  wall_sec=time.perf_counter()-started, input_wait_sec=scheduled_wait,
                  no_packet_pause_sec=packet_pause, cold_total_wall_sec=time.perf_counter()-cold_start,
                  warmup_wall_sec=warmup_sec, warmup_source_sec=warm_fixture.size/16000,
                  segmentation=args.segmentation,
                  fixture_repeated=args.duration_sec > fixture.size / 16000,
                  quality_caveat='pressure fixture tests state/transport stability, not translation quality')
    result['io_active_sec'] = max(0., result['wall_sec'] - scheduled_wait - packet_pause)
    result['io_rtf'] = result['io_active_sec'] / result['source_sec'] if result['source_sec'] else None
    result['cold_total_rtf'] = result['cold_total_wall_sec'] / result['source_sec'] if result['source_sec'] else None
    write_json(out / 'result.json', result)
    if args.segmentation != 'none':
        write_json(out / 'segments.json', result['segments'])
    report = analyze([result], read_jsonl(out / 'events.jsonl'))
    report['config'] = vars(args)
    write_json(out / 'actual_pcm_report.json', report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='command', required=True)
    a = subs.add_parser('analyze')
    a.add_argument('--results', required=True)
    a.add_argument('--events', required=True)
    a.add_argument('--out', required=True)
    p = subs.add_parser('run')
    for name in ('manifest', 'model_dir', 'adapter', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--stage', choices=('probe', 'full'), default='probe')
    p.add_argument('--probe_report')
    p.add_argument('--source_revision', help='original git commit when running a source snapshot without .git')
    p.add_argument('--pacing', choices=('accelerated', 'realtime'), default='accelerated')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--configuration', choices=('640', '1280', '1920', '2560'))
    p.add_argument('--quality_commands', help='JSON argv arrays, e.g. existing ASR/UTMOS/para/score_eval commands')
    p.add_argument('infer_args', nargs=argparse.REMAINDER, help='extra inference arguments after --')
    long = subs.add_parser('longrun')
    for name in ('model_dir', 'adapter', 'audio', 'out', 'cosyvoice_repo'):
        long.add_argument('--' + name, required=True)
    long.add_argument('--cosyvoice_dir')
    long.add_argument('--prompt_tokenizer')
    long.add_argument('--tgt_lang', choices=('en', 'zh'), required=True)
    long.add_argument('--device', default='cuda:0')
    long.add_argument('--duration_sec', type=float, default=600., help='positive source seconds; use short runs for path validation')
    long.add_argument('--pause_kind', choices=('none', 'silence', 'no_packets'), default='none')
    long.add_argument('--pause_at_sec', type=float, default=600.)
    long.add_argument('--pause_sec', type=float, choices=(30., 60.), default=30.)
    long.add_argument('--pacing', choices=('accelerated', 'realtime'), default='accelerated')
    long.add_argument('--chunk_ms', type=int, choices=(640, 1280, 1920, 2560), default=640)
    long.add_argument('--reference_policy', choices=('rolling', 'fixed'), default='rolling')
    long.add_argument('--reference_refresh_sec', type=float, default=2)
    long.add_argument('--context_max_tokens', type=int, default=2048)
    long.add_argument('--context_window_sec', type=float, default=0)
    long.add_argument('--acoustic_max_sec', type=float, default=60.)
    from streaming_si.inference.s5_segmenter import add_segmentation_args
    add_segmentation_args(long, default='vad')
    args = ap.parse_args()
    if args.command == 'analyze':
        write_json(args.out, analyze(read_jsonl(args.results), read_jsonl(args.events)))
    elif args.command == 'longrun':
        from streaming_si.inference.s5_segmenter import segmentation_options
        try:
            segmentation_options(args)
        except ValueError as exc:
            ap.error(str(exc))
        if not math.isfinite(args.duration_sec) or args.duration_sec <= 0:
            ap.error('duration_sec must be finite and positive')
        if args.chunk_ms <= 0 or args.chunk_ms % 80:
            ap.error('chunk_ms must be a positive multiple of 80')
        if args.pause_kind != 'none' and not 0 < args.pause_at_sec < args.duration_sec:
            ap.error('pause must fall inside fixture duration')
        longrun(args)
    else:
        if args.repeats < 1:
            ap.error('--repeats must be positive')
        if args.infer_args[:1] == ['--']:
            args.infer_args = args.infer_args[1:]
        run(args)


if __name__ == '__main__':
    main()
