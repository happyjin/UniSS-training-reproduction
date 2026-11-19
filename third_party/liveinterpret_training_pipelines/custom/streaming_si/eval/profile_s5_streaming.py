"""GPU-only bounded S5 profiling after an independent exact-audio warmup.

Profiles a few source steps, not model loading or every inference step. Use the
same model/checkpoint/options as the profiler-free benchmark. Profile timings
are diagnostic and must not be reported as RTF improvements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('model_dir', 'adapter', 'audio', 'cosyvoice_repo', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--cosyvoice_dir')
    p.add_argument('--prompt_tokenizer')
    p.add_argument('--tgt_lang', choices=('en', 'zh'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    p.add_argument('--profiler', choices=('torch', 'nsys'), default='torch')
    p.add_argument('--start_step', type=int, default=4, help='zero-based model source-step index')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--chunk_ms', type=int, choices=(640, 1280, 1920, 2560), default=640)
    p.add_argument('--chunk_ms_after_write', type=int, choices=(640, 1280, 1920, 2560))
    p.add_argument('--write_threshold', type=float)
    p.add_argument('--context_max_tokens', type=int, default=2048)
    p.add_argument('--context_window_sec', type=float, default=0)
    p.add_argument('--acoustic_max_sec', type=float, default=60)
    p.add_argument('--reference_window_sec', type=float, default=5)
    p.add_argument('--reference_policy', choices=('rolling', 'fixed'), default='rolling')
    p.add_argument('--reference_refresh_sec', type=float, default=2)
    p.add_argument('--prompt_sec', type=float, default=2)
    p.add_argument('--max_text_tokens', type=int, default=256)
    p.add_argument('--max_speech_tokens', type=int, default=750)
    p.add_argument('--max_tail_steps', type=int, default=8)
    p.add_argument('--talker_top_k', type=int, default=25)
    p.add_argument('--talker_top_p', type=float, default=.8)
    p.add_argument('--ode_steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_source_prompt', action='store_true')
    p.add_argument('--no_speaker', action='store_true')
    p.add_argument('--no_thinker_cache', action='store_true')
    p.add_argument('--no_thinker_graph', action='store_true')
    p.add_argument('--no_talker_graph', action='store_true')
    from streaming_si.inference.s5_segmenter import add_segmentation_args, segmentation_options
    add_segmentation_args(p, default='vad')
    args = p.parse_args(argv)
    try:
        segmentation_options(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.start_step < 0 or args.steps < 1:
        p.error('--start_step must be nonnegative and --steps must be positive')
    if not args.device.startswith('cuda'):
        p.error('CUDA device required; CPU profiling/model fallback is disabled')
    return args


class Capture:
    def __init__(self, torch, mode, out, device):
        self.torch, self.mode, self.out, self.device = torch, mode, out, device
        self.active = False
        self.prof = None
        self.complete = False

    def start(self):
        if self.complete or self.active:
            raise RuntimeError('capture range may only start once')
        self.torch.cuda.synchronize(self.device)
        if self.mode == 'nsys':
            status = self.torch.cuda.cudart().cudaProfilerStart()
            if status != 0:
                raise RuntimeError(f'cudaProfilerStart failed: {status}')
        else:
            self.prof = self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=False)
            self.prof.start()
        self.active = True

    def stop(self):
        if not self.active:
            return
        try:
            self.torch.cuda.synchronize(self.device)
        finally:
            if self.mode == 'nsys':
                status = self.torch.cuda.cudart().cudaProfilerStop()
                if status != 0:
                    raise RuntimeError(f'cudaProfilerStop failed: {status}')
            elif self.prof is not None:
                self.prof.stop()
            self.active = False
            self.complete = True

    def export(self):
        if self.prof is not None and self.complete:
            self.prof.export_chrome_trace(str(self.out / 'trace.json'))
            events = self.prof.key_averages(group_by_input_shape=True)
            table = events.table(sort_by='self_device_time_total', row_limit=50)
            (self.out / 'cuda-kernel-summary.txt').write_text(table + '\n', encoding='utf-8')
            kernels, scopes = {}, {}
            for event in self.prof.events():
                if getattr(event.device_type, 'name', '') == 'CUDA' and not getattr(event, 'is_user_annotation', False):
                    duration = float(event.time_range.elapsed_us())
                    if duration <= 0:
                        continue
                    row = kernels.setdefault(event.name, dict(name=event.name, calls=0, total_us=0., max_us=0.))
                    row['calls'] += 1
                    row['total_us'] += duration
                    row['max_us'] = max(row['max_us'], duration)
                elif getattr(event.device_type, 'name', '') == 'CPU' and event.name.startswith('S5/'):
                    row = scopes.setdefault(event.name, dict(name=event.name, calls=0, inclusive_wall_us=0.))
                    row['calls'] += 1
                    row['inclusive_wall_us'] += float(event.time_range.elapsed_us())
            if not kernels:
                raise RuntimeError('no raw CUDA device events captured; use nsys with CUDA Graph node tracing')
            total = sum(row['total_us'] for row in kernels.values())
            for row in kernels.values():
                row.update(mean_us=row['total_us'] / row['calls'], share_cuda_percent=100 * row['total_us'] / total)
            (self.out / 'cuda-kernels.json').write_text(json.dumps(dict(total_cuda_us=total,
                kernels=sorted(kernels.values(), key=lambda row: -row['total_us'])), indent=2))
            (self.out / 'scope-wall.json').write_text(json.dumps(dict(nested_scopes_not_additive=True,
                scopes=sorted(scopes.values(), key=lambda row: -row['inclusive_wall_us'])), indent=2))

    @contextmanager
    def region(self, label):
        if not self.active:
            yield
            return
        self.torch.cuda.nvtx.range_push(label)
        try:
            with self.torch.profiler.record_function(label):
                yield
        finally:
            self.torch.cuda.nvtx.range_pop()


@contextmanager
def instrument(capture, model, thinker_runner):
    """Restore instance/class descriptors even on an inference/profiler error."""
    restore = []

    def patch(owner, name, label, generator=False):
        original = getattr(owner, name)
        own = name in vars(owner)
        previous = vars(owner).get(name)
        if generator:
            @wraps(original)
            def wrapped(*args, **kwargs):
                # Keep the context alive while executing the generator body.
                with capture.region(label):
                    yield from original(*args, **kwargs)
        else:
            @wraps(original)
            def wrapped(*args, **kwargs):
                with capture.region(label):
                    return original(*args, **kwargs)
        if isinstance(owner, type):
            setattr(owner, name, wrapped)
        else:
            object.__setattr__(owner, name, wrapped)
        restore.append((owner, name, own, previous))

    try:
        from streaming_si.inference.s5_segmenter import SileroFrameVad
        from streaming_si.inference.s5_session import S5Session
        from streaming_si.inference.s5_hift_stream import BoundedCausalHiFT
        patch(thinker_runner, '__call__', 'S5/Thinker')
        patch(model, 'encode_audio_chunk', 'S5/Voxtral')
        patch(model, '_flow_infer', 'S5/Flow')
        patch(model, '_hift_infer', 'S5/HiFT')
        patch(BoundedCausalHiFT, 'feed', 'S5/HiFT')
        patch(model.talker, 'stream_generate_s4', 'S5/Talker', generator=True)
        patch(SileroFrameVad, 'is_speech', 'S5/VAD')
        patch(S5Session, '_reset_segment', 'S5/segment_reset')
        patch(S5Session, '_compact_thinker', 'S5/thinker_window')
        patch(S5Session, '_compact_talker', 'S5/talker_window')
        yield
    finally:
        for owner, name, own, previous in reversed(restore):
            if isinstance(owner, type):
                if own:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            elif own:
                object.__setattr__(owner, name, previous)
            else:
                object.__delattr__(owner, name)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback or CPU model test')
    torch.cuda.set_device(args.device)
    from streaming_si.inference.infer_s5_streaming_dynchunk import load_model, ThinkerRunner
    from streaming_si.inference.infer_s2st import CampplusSpeaker, build_source_prompt, load_wav16
    from streaming_si.model.base_model import VoxtralAudioProcessor
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    from streaming_si.model.speech_decoder import resolve_speech_decoder
    from streaming_si.inference.s5_session import S5Session
    from streaming_si.inference.s5_segmenter import SegmentedSession, SileroFrameVad, segmentation_options

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='loading', args=vars(args), device_name=torch.cuda.get_device_name(args.device),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    capture_steps=[], profile_is_rtf_benchmark=False,
                    warmup='independent full session using exact same audio/settings; warmup outputs discarded',
                    source_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), Path(__file__).parents[1] / 'inference/s5_session.py',
                                             Path(__file__).parents[1] / 'inference/s5_pcm_decoder.py',
                                             Path(__file__).parents[1] / 'inference/s5_hift_stream.py',
                                             Path(__file__).parents[1] / 'inference/s5_segmenter.py',
                                             Path(__file__).parents[1] / 'inference/infer_s5_streaming_dynchunk.py']})
    def save():
        (out / 'profile_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    capture = Capture(torch, args.profiler, out, args.device)
    try:
        torch.manual_seed(args.seed)
        model, tokenizer, ids = load_model(args.model_dir, args.adapter, dtype=args.dtype)
        model.to(args.device)
        mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.model_dir))
        directory, yaml = resolve_speech_decoder(args.model_dir, args.cosyvoice_dir, None)
        model.config.talker_config_yaml = yaml
        model.attach_speech_decoder(directory, device=args.device, cosyvoice_repo=args.cosyvoice_repo, strict=True)
        speaker = None if args.no_speaker else CampplusSpeaker(os.path.join(directory, 'campplus.onnx'))
        prompter = build_source_prompt(not args.no_source_prompt, directory, args.prompt_tokenizer)
        audio = load_wav16(args.audio, max_sec=None)
        if not audio.size:
            raise ValueError('empty audio fixture')
        metadata.update(source_sec=audio.size / 16000,
                        audio_sha256=hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
                        resolved_decoder_dir=directory, resolved_decoder_yaml=yaml,
                        speaker_enabled=speaker is not None, source_prompt_enabled=prompter is not None)
        settings = {key: getattr(args, key) for key in ('device', 'chunk_ms', 'chunk_ms_after_write',
            'write_threshold', 'context_max_tokens', 'context_window_sec', 'acoustic_max_sec', 'reference_window_sec', 'reference_policy', 'reference_refresh_sec', 'prompt_sec',
            'max_text_tokens', 'max_speech_tokens', 'max_tail_steps', 'talker_top_k', 'talker_top_p', 'ode_steps')}
        settings.update(thinker_cache=not args.no_thinker_cache, thinker_graph=not args.no_thinker_graph,
                        talker_graph=not args.no_talker_graph)
        metadata['session_settings'] = settings
        metadata['segmentation_settings'] = segmentation_options(args)
        vad = SileroFrameVad(args.vad_threshold) if args.segmentation == 'vad' else None
        def fresh():
            torch.manual_seed(args.seed)
            session = S5Session(model, tokenizer, ids, mel_fn, args.tgt_lang, speaker=speaker,
                                prompter=prompter, collect_audio=False, **settings)
            if args.segmentation != 'none':
                if vad is not None:
                    vad.reset()
                session = SegmentedSession(session, vad, **segmentation_options(args))
            torch.manual_seed(args.seed)
            return session
        def feed(session):
            for pos in range(0, audio.size, 1280):
                session.feed_audio(audio[pos:pos + 1280])
            session.feed_audio(np.zeros(0, np.float32), finalize=True)
        metadata['status'] = 'warming'
        save()
        warm = fresh()
        feed(warm)
        torch.cuda.synchronize(args.device)
        del warm
        measured = fresh()
        core = measured.session if isinstance(measured, SegmentedSession) else measured
        original_step, original_prepare = core._step, core._prepare_mel
        index = 0
        def measured_prepare(*prepare_args, **prepare_kwargs):
            if index == args.start_step and not capture.active and not capture.complete:
                capture.start()
            with capture.region('S5/Frontend'):
                return original_prepare(*prepare_args, **prepare_kwargs)
        def measured_step(*step_args, **step_kwargs):
            nonlocal index
            current = index
            index += 1
            selected = args.start_step <= current < args.start_step + args.steps
            if current == args.start_step and not capture.active:
                capture.start()
            try:
                with capture.region(f'S5/source_step/{current}'):
                    result = original_step(*step_args, **step_kwargs)
                if selected:
                    metadata['capture_steps'].append(dict(index=current, model_source_sec=core.consumed_samples / 16000,
                        original_source_observed_sec=(measured.feeder.processed_samples / 16000
                            if isinstance(measured, SegmentedSession) else core.source_samples / 16000),
                        source_done=bool(step_kwargs.get('source_done', False)), tail=bool(step_kwargs.get('tail', False))))
                return result
            finally:
                if current == args.start_step + args.steps - 1:
                    capture.stop()
        object.__setattr__(core, '_step', measured_step)
        object.__setattr__(core, '_prepare_mel', measured_prepare)
        metadata['status'] = 'profiling'
        save()
        try:
            with instrument(capture, model, ThinkerRunner):
                feed(measured)
        finally:
            object.__delattr__(core, '_step')
            object.__delattr__(core, '_prepare_mel')
            capture.stop()
            capture.export()
        metadata['total_measured_source_steps'] = index
        metadata['profiled_session_metrics_diagnostic_only'] = measured.metrics()
        if len(metadata['capture_steps']) != args.steps:
            raise RuntimeError('audio/session did not reach all requested capture steps; use a longer representative fixture or smaller --start_step')
        metadata['status'] = 'ok'
        metadata['artifacts'] = ['trace.json', 'cuda-kernel-summary.txt', 'cuda-kernels.json', 'scope-wall.json'] if args.profiler == 'torch' else ['external nsys report (see launcher --output)']
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
