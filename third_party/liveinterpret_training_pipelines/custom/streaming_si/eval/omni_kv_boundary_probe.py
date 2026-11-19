"""Isolated L40S probe: source duration/KV growth, history resets, allocator OOM.

The final OOM experiment holds real loaded weights and adds raw KV-equivalent
buffers. It is a memory bound, not a concurrent-request serving benchmark.
"""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

import omni_shared_run as shared


def main():
    argv = sys.argv[:]
    manifest_index = argv.index('--manifest') + 1
    output_index = argv.index('--out-dir') + 1
    root = Path(argv[output_index])
    root.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from simuleval.utils import agent as agent_utils
    from simuleval.evaluator import instance as instance_module

    loaded = {}
    current_mode = {'name': 'windowed'}
    original_import = agent_utils.import_file
    pause_after = float(os.environ.get('SIMULS2ST_BOUNDARY_PAUSE_AFTER', '0'))
    pause_seconds = float(os.environ.get('SIMULS2ST_BOUNDARY_PAUSE_SECONDS', '30'))
    realtime = os.environ.get('SIMULS2ST_BOUNDARY_REALTIME') == '1'
    if pause_after < 0 or pause_seconds < 0 or pause_seconds > 60:
        raise ValueError('Pause position must be nonnegative; pause duration must be 0..60 seconds')
    original_send = instance_module.SpeechInputInstance.send_source
    (root / 'input_schedule.json').write_text(json.dumps(dict(
        realtime=realtime, pause_after_source_sec=pause_after, pause_wall_sec=pause_seconds,
        pause_behavior='no_input_no_silence_no_session_reset'), indent=2))

    def snapshot(system, instance):
        torch.cuda.synchronize()
        states = system.states
        cache = states.talker_past_key_values
        length = int(cache.get_seq_length()) if cache is not None else 0
        digest = hashlib.sha256()
        if cache is not None:
            for tensors in [cache.key_cache, cache.value_cache]:
                for tensor in tensors:
                    raw = tensor[:, :, :length].contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    digest.update(raw)
        accelerator = getattr(system.inference.talker, '_static_graph_accelerator', None)
        return dict(monotonic_s=time.monotonic(), source_step=int(instance.step),
            processed_samples=int(states.processed_samples),
            cache_id=id(cache), cache_type=type(cache).__name__, kv_length=length,
            kv_sha256=digest.hexdigest(), history_turns=len(states.turn_history),
            replays=accelerator.replays if accelerator else 0,
            fallbacks=accelerator.fallbacks if accelerator else 0,
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())

    def send_with_pause(instance, segment_size=10):
        if realtime:
            if not hasattr(instance, '_boundary_stream_started'):
                instance._boundary_stream_started = time.monotonic()
                instance._boundary_total_pause = 0.
            deadline = (instance._boundary_stream_started
                        + instance.step / instance.sample_rate + instance._boundary_total_pause)
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        if (pause_after > 0 and not getattr(instance, '_boundary_paused', False)
                and instance.step >= int(pause_after * instance.sample_rate)
                and instance.step < len(instance.samples)):
            system = loaded['agent']
            before = snapshot(system, instance)
            event_path = system.text_step_events_path.parent / 'pause_events.jsonl'
            with event_path.open('a') as f:
                f.write(json.dumps(dict(event='pause_begin', state=before)) + '\n')
            print('[pause-begin] ' + json.dumps(before), flush=True)
            started = time.monotonic()
            time.sleep(pause_seconds)
            elapsed = time.monotonic() - started
            after = snapshot(system, instance)
            invariant_fields = ['source_step', 'processed_samples', 'cache_id', 'cache_type',
                'kv_length', 'kv_sha256', 'history_turns', 'replays', 'fallbacks']
            changed = [field for field in invariant_fields if before[field] != after[field]]
            record = dict(event='pause_end', slept_sec=elapsed, state=after, changed_fields=changed)
            with event_path.open('a') as f:
                f.write(json.dumps(record) + '\n')
            print('[pause-end] ' + json.dumps(record), flush=True)
            if changed:
                raise RuntimeError(f'State changed without input during pause: {changed}')
            instance._boundary_paused = True
            if realtime:
                instance._boundary_total_pause += time.monotonic() - started
            system._boundary_resume_pending = True
            system._boundary_resume_time = time.monotonic()
        return original_send(instance, segment_size)

    instance_module.SpeechInputInstance.send_source = send_with_pause

    def import_instrumented(path):
        original_import(path)
        cls = agent_utils.EVALUATION_SYSTEM_LIST[-1]
        original_init = cls.__init__
        original_event = cls._append_text_step_event
        original_policy = cls.policy

        def init(self, args):
            if current_mode['name'] == 'unwindowed':
                args.enable_history_window = False
            original_init(self, args)
            loaded['inference'] = self.inference
            loaded['agent'] = self

        def policy(self, states=None):
            states = self.states if states is None else states
            before = states.turn_history
            before_count = len(before)
            result = original_policy(self, states)
            if (self.enable_history_window and self.history_window_turns > 0
                    and before_count >= self.history_window_turns
                    and before is not states.turn_history):
                record = dict(mode=current_mode['name'],
                    source_consumed_sec=states.processed_samples / states.source_sample_rate,
                    before_turns=before_count, after_turns=len(states.turn_history),
                    monotonic_s=time.monotonic())
                with (self.text_step_events_path.parent / 'history_resets.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print('[history-reset] ' + json.dumps(record), flush=True)
            return result

        def event(self, states, **kw):
            original_event(self, states, **kw)
            cache = states.talker_past_key_values
            accelerator = getattr(self.inference.talker, '_static_graph_accelerator', None)
            free, total = torch.cuda.mem_get_info()
            # Upstream SimulEval may leave states.current_instance_index at zero.
            # The agent's ordered sidecar counter tracks our sequential manifest.
            record_id = int(self._sidecar_record_id)
            source = self._source_entries[record_id] if 0 <= record_id < len(self._source_entries) else self._source_entry_for_state(states)
            record = dict(
                mode=current_mode['name'], source=source, record_id=record_id,
                monotonic_s=time.monotonic(),
                source_consumed_sec=kw['source_consumed_ms'] / 1000,
                output_sec=kw['output_audio_num_samples'] / 24000,
                cache_type=type(cache).__name__,
                kv_length=int(cache.get_seq_length()) if cache is not None else 0,
                history_turns=len(states.turn_history), history_window=self.history_window_turns,
                window_enabled=self.enable_history_window,
                fallbacks=accelerator.fallbacks if accelerator else 0,
                replays=accelerator.replays if accelerator else 0,
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(), free_bytes=free, total_bytes=total,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            with (self.text_step_events_path.parent / 'kv_boundary.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print('[kv-boundary] ' + json.dumps(record), flush=True)
            if getattr(self, '_boundary_resume_pending', False):
                resumed = dict(event='first_decision_after_resume', state=record,
                    elapsed_since_resume_sec=time.monotonic()-self._boundary_resume_time)
                with (self.text_step_events_path.parent / 'pause_events.jsonl').open('a') as f:
                    f.write(json.dumps(resumed) + '\n')
                print('[pause-resumed] ' + json.dumps(resumed), flush=True)
                self._boundary_resume_pending = False

        cls.__init__ = init
        cls._append_text_step_event = event
        cls.policy = policy

    agent_utils.import_file = import_instrumented
    rows = [json.loads(line) for line in Path(argv[manifest_index]).read_text().splitlines() if line.strip()]
    duration = float(os.environ.get('SIMULS2ST_BOUNDARY_SECONDS', '128'))
    clip_limit = int(os.environ.get('SIMULS2ST_BOUNDARY_SOURCE_CLIPS', '8'))
    probe_rows = []
    inputs = []
    for lang, target in [('en', 'zh'), ('zh', 'en')]:
        candidates = [r for r in rows if r['src_lang'] == lang and not r['id'].startswith('_warmup_')]
        if clip_limit > 0:
            candidates = candidates[:clip_limit]
        pieces = []
        samples = 0
        required = int(duration * 16000)
        for row in candidates:
            audio, sr = sf.read(row['audio'], dtype='float32', always_2d=True)
            audio = audio.mean(axis=1)
            if sr != 16000:
                audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
            pieces.append(audio)
            samples += len(audio)
            if samples >= required:
                break
        speech = np.concatenate(pieces)
        long_audio = np.tile(speech, (required + len(speech) - 1) // len(speech))[:required]
        inputs.append(dict(lang=lang, clips_used=len(pieces), requested_sec=duration,
                           material_sec=len(speech)/16000, repeated=len(speech)<required))
        for kind, audio in [('short', long_audio[:48000]), ('long', long_audio)]:
            path = root / f'{lang}_{kind}.wav'
            sf.write(path, audio, 16000)
            probe_rows.append(dict(id=f'boundary_{lang}_{kind}', audio=str(path),
                source_sec=len(audio) / 16000, src_lang=lang, tgt_lang=target,
                reference='boundary probe'))
    manifest = root / 'boundary_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in probe_rows))
    (root / 'input_composition.json').write_text(json.dumps(inputs, indent=2))
    results = []
    modes = os.environ.get('SIMULS2ST_BOUNDARY_MODES', 'windowed,unwindowed').split(',')
    if not modes or any(mode not in {'windowed', 'unwindowed'} for mode in modes):
        raise ValueError(f'Unsupported boundary modes: {modes}')
    for mode in modes:
        current_mode['name'] = mode
        sys.argv = argv[:]
        sys.argv[manifest_index] = str(manifest)
        sys.argv[output_index] = str(root / mode)
        started = time.perf_counter()
        shared.START = started
        try:
            shared.main()
            results.append(dict(mode=mode, status='completed', elapsed_sec=time.perf_counter()-started))
        except torch.cuda.OutOfMemoryError as exc:
            results.append(dict(mode=mode, status='cuda_oom', error=str(exc)))
            (root / 'boundary_results.json').write_text(json.dumps(results, indent=2))
            raise
        (root / 'boundary_results.json').write_text(json.dumps(results, indent=2))

    model = loaded['inference'].talker
    config = model.model.config
    itemsize = next(model.parameters()).element_size()
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    kv_bytes_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * itemsize
    before_free, total = torch.cuda.mem_get_info()
    block_size = 256 * 1024**2
    blocks = []
    memory = dict(kind='kv_equivalent_allocator_stress_not_real_concurrency',
        kv_bytes_per_token=kv_bytes_per_token, kv_4096_bytes=4096*kv_bytes_per_token,
        free_before_bytes=before_free, total_bytes=total, block_size=block_size)
    if os.environ.get('SIMULS2ST_BOUNDARY_STRESS_OOM') != '1':
        memory['status'] = 'allocation_stress_disabled'
        (root / 'memory_capacity.json').write_text(json.dumps(memory, indent=2))
        return
    try:
        for _ in range(total // block_size + 2):
            blocks.append(torch.empty(block_size, dtype=torch.uint8, device='cuda'))
            memory['successful_extra_bytes'] = len(blocks) * block_size
            (root / 'memory_capacity.json').write_text(json.dumps(memory, indent=2))
    except torch.cuda.OutOfMemoryError as exc:
        memory.update(status='allocator_oom_observed', error=str(exc),
                      allocated_at_oom=torch.cuda.memory_allocated(),
                      reserved_at_oom=torch.cuda.memory_reserved())
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
        (root / 'memory_capacity.json').write_text(json.dumps(memory, indent=2))
    print('[memory-boundary] ' + json.dumps(memory), flush=True)


if __name__ == '__main__':
    main()
