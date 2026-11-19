"""Shared, causal S5 file/microphone runtime. Times are measured, never synthesized.

A model (including graph buffers) may serve only one active session at a time.
Callbacks are synchronous: PCM is visible before the enclosing WRITE completes.
"""
from collections import deque
from dataclasses import dataclass, replace
import time

import numpy as np
import torch

from streaming_si.model.base_model import VoxtralWaveformStream
from streaming_si.model.model_si import TalkerStreamState
from streaming_si.inference.infer_s5_streaming_dynchunk import (
    ThinkerRunner, sample_delta, _pick, INSTRUCTION, LANG_NAME, NO_SPACE_LANGS,
)


def validate_chunks(chunk_ms, chunk_ms_after_write=None, write_threshold=None):
    for value in (chunk_ms, chunk_ms_after_write):
        if value is not None and (value <= 0 or value % 80):
            raise ValueError('chunk sizes must be positive multiples of 80ms')
    if write_threshold is not None and not 0 <= write_threshold <= 1:
        raise ValueError('write_threshold must be in [0, 1]')


@dataclass(frozen=True)
class ContextRecord:
    """One complete READ/WRITE, on the consumed-source clock (not wall time)."""
    start_sec: float
    end_sec: float
    embeds: tuple
    turn: int = 0

    def __iter__(self):
        return iter(self.embeds)

    @property
    def tokens(self):
        return sum(int(x.size(1)) for x in self.embeds)


class S5Session:
    @torch.inference_mode()
    def __init__(self, model, tokenizer, ids, mel_fn, tgt_lang, *, device,
                 chunk_ms=640, chunk_ms_after_write=None, write_threshold=None,
                 speaker=None, prompter=None, ode_steps=5, thinker_cache=True,
                 thinker_graph=True, talker_graph=True, max_speech_tokens=750,
                 max_text_tokens=256, max_tail_steps=8, context_max_tokens=2048,
                 context_window_sec=0, context_window_32_16=False,
                 acoustic_max_sec=60, reference_window_sec=5, prompt_sec=2,
                 reference_policy='rolling', reference_refresh_sec=2,
                 talker_top_k=25, talker_top_p=0.8, decode=True,
                 on_event=None, collect_audio=False, **unsupported):
        if unsupported:
            raise TypeError(f'unsupported streaming options: {sorted(unsupported)}')
        validate_chunks(chunk_ms, chunk_ms_after_write, write_threshold)
        if context_max_tokens < 128 or context_max_tokens > 16384:
            raise ValueError('context_max_tokens must be in [128, 16384]')
        if not np.isfinite(context_window_sec) or context_window_sec < 0:
            raise ValueError('context_window_sec must be finite and nonnegative')
        if context_window_sec and context_window_32_16:
            raise ValueError('select either context_window_sec or context_window_32_16')
        if reference_window_sec <= 0 or prompt_sec <= 0 or prompt_sec > reference_window_sec:
            raise ValueError('require 0 < prompt_sec <= reference_window_sec')
        if reference_policy not in ('rolling', 'fixed') or not np.isfinite(reference_refresh_sec) or reference_refresh_sec <= 0:
            raise ValueError('require reference_policy rolling/fixed and positive finite reference_refresh_sec')
        if max_text_tokens <= 0 or max_speech_tokens <= 0 or max_tail_steps < 0:
            raise ValueError('generation limits must be positive')
        if max_text_tokens + max_speech_tokens + 32 >= context_max_tokens:
            raise ValueError('context capacity must exceed per-WRITE text + speech reserves')
        if (getattr(model.config, 'talker_cond_positions', 'answer') or 'answer') != 'answer':
            raise ValueError('streaming requires an answer-conditioned checkpoint')
        self.model, self.tokenizer, self.ids = model, tokenizer, ids
        self.device, self.lang = device, tgt_lang.lower()
        self.chunk_ms, self.chunk_ms_after_write = chunk_ms, chunk_ms_after_write
        self.write_threshold = write_threshold
        self.speaker, self.prompter = speaker, prompter
        self.ode_steps, self.decode = ode_steps, decode
        self.thinker_cache, self.thinker_graph, self.talker_graph = thinker_cache, thinker_graph, talker_graph
        self.max_speech_tokens, self.max_text_tokens = max_speech_tokens, max_text_tokens
        self.max_tail_steps, self.capacity = max_tail_steps, context_max_tokens
        self.context_window_sec = float(context_window_sec)
        self.context_window_32_16 = context_window_32_16
        self.context_turn_floor = 0
        self.keep_context = bool(context_window_sec or context_window_32_16)
        self.last_step_source_sec = 0.
        self.active_reference = None
        self.acoustic_max_sec, self.reference_samples, self.prompt_sec = acoustic_max_sec, int(reference_window_sec * 16000), prompt_sec
        self.reference_policy = reference_policy
        self.reference_refresh_samples = max(1, int(reference_refresh_sec * 16000))
        self.reference_version = 0
        self.reference_updated_at = None
        self.reference_conditions = None
        self.top_k, self.top_p = talker_top_k, talker_top_p
        self.on_event, self.collect_audio = on_event, collect_audio
        self.mel_fn = mel_fn
        self.waveform_stream = VoxtralWaveformStream(mel_fn)
        self.pending_mel = torch.zeros(self.waveform_stream.feature_size, 0)
        self.seg_samples = 0            # 当前段已收到的样本数（外部分段用）
        self.segments = 0               # 已结束的外部分段个数
        self.pending_audio = np.zeros(0, np.float32)
        self.reference_audio = np.zeros(0, np.float32)
        self.source_samples = self.consumed_samples = self.output_samples = 0
        self.source_padding_mel_frames = 0
        self.vcache, self.decoder = None, None
        self.closed = self.wrote = self.failed = False
        self.stop = 'open'
        self.completed_talker_stats = [0, 0, 0]
        self.pending_codec = []
        self.started = time.perf_counter()
        self.compute_sec = 0.0
        self.first_pcm_sec = self.first_text_sec = None
        self.events, self.audio = [], []
        self.steps, self.text_pieces = [], []
        self.speech_count = self.text_count = self.forward_count = 0
        self.block_times = deque(maxlen=4096)
        self.step_count = 0
        self.context_replays = self.talker_replays = 0
        self._batch = None
        self._new_thinker()
        self._new_talker()

    def _tok(self, value):
        return self.model.thinker.embed_tokens(torch.tensor([[int(value)]], device=self.device))

    def _new_thinker(self, retained=()):
        instruction = self.tokenizer.encode(INSTRUCTION.format(tgt=LANG_NAME.get(self.lang, self.lang)), add_special_tokens=False)
        self.instruction = self.model.thinker.embed_tokens(torch.tensor([instruction], device=self.device))
        self.records = deque(retained)
        self.embeds = [self.instruction] + [x for record in self.records for x in record]
        self.run = ThinkerRunner(self.model, self.embeds, self.thinker_cache, self.thinker_graph)
        if retained:
            self.run()

    def _new_talker(self):
        self.state = TalkerStreamState(self.model.talker, cuda_graph=self.talker_graph, max_len=self.capacity)
        self.talker_records = deque()
        self.talker_fed = 0

    def _emit(self, kind, **data):
        event = dict(type=kind, elapsed_sec=time.perf_counter() - self.started,
                     source_sec=self.consumed_samples / 16000,
                     source_received_sec=self.source_samples / 16000)
        event.update(data)
        if self._batch is not None:
            self._batch.append(event)
        if self.on_event is not None:
            self.on_event(event)
        # Diagnostics retain metadata only; PCM/audio retention is explicitly optional.
        if self.collect_audio:
            self.events.append({k: v for k, v in event.items() if k != 'pcm'})
        return event

    def _pcm(self, pcm):
        pcm = pcm.detach().float().cpu().numpy().reshape(-1) if torch.is_tensor(pcm) else np.asarray(pcm, np.float32).reshape(-1)
        if not pcm.size:
            return
        if self.first_pcm_sec is None:
            self.first_pcm_sec = time.perf_counter() - self.started
        offset = self.output_samples
        self.output_samples += pcm.size
        if self.collect_audio:
            self.audio.append(pcm.copy())
        self._emit('pcm', pcm=pcm, sample_rate=24000, sample_offset=offset)

    def _ensure_decoder(self, final=False):
        if self.decoder is not None or not self.decode:
            return True
        from streaming_si.inference.s5_pcm_decoder import PersistentSpeechDecoder
        # Only consumed source is visible here. Never inspect future file samples.
        reference = self.reference_audio
        if (self.speaker is not None or self.prompter is not None) and reference.size < 6400:
            if final:
                raise ValueError('source shorter than 400ms: insufficient causal speaker/prompt enrollment')
            return False
        condition = self.active_reference or self._source_reference(force=True) or {}
        self.active_reference = condition
        emb = condition.get('spk_emb')
        prompt, feat = condition.get('prompt_speech_tokens'), condition.get('prompt_feat')
        self.decoder = PersistentSpeechDecoder(self.model, spk_emb=emb,
            prompt_speech_tokens=prompt, prompt_feat=feat, ode_steps=self.ode_steps,
            max_session_sec=self.acoustic_max_sec, prompt_tail_sec=self.prompt_sec,
            reference_provider=lambda: self.active_reference, reference_metadata=condition,
            bind_reference_to_tokens=True)
        self._emit('decoder_conditions', reference_sec=reference.size / 16000,
                   prompt_tokens=0 if prompt is None else len(prompt), speaker_enabled=emb is not None)
        return True

    def _source_reference(self, force=False):
        """Versioned, paired codec/mel from consumed source only, not target history.

        Polling on each token is cheap: encode only when the source clock has
        advanced enough. No packets cannot advance this clock. An all-quiet
        tail keeps the last usable voice condition instead of enrolling silence.
        """
        if self.speaker is None and self.prompter is None:
            return None
        if force and self.reference_conditions is not None:
            if (self.reference_policy == 'fixed' or
                    self.consumed_samples - self.reference_updated_at < self.reference_refresh_samples):
                return self.reference_conditions
        if not force:
            if self.reference_policy == 'fixed':
                return None
            if self.reference_updated_at is not None and self.consumed_samples - self.reference_updated_at < self.reference_refresh_samples:
                return None
        reference = self.reference_audio[-int(self.prompt_sec * 16000):].copy()
        if reference.size < 6400:
            return None
        if float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))) < 1e-4:
            return self.reference_conditions if force else None
        prompt, feat = (self.prompter(reference, max_sec=self.prompt_sec, position='tail')
                        if self.prompter is not None else (None, None))
        emb = self.speaker(reference) if self.speaker is not None else None
        self.reference_version += 1
        self.reference_updated_at = self.consumed_samples
        self.reference_conditions = dict(spk_emb=emb, prompt_speech_tokens=prompt,
            prompt_feat=feat, version=self.reference_version,
            source_start_sec=(self.consumed_samples - reference.size) / 16000,
            source_end_sec=self.consumed_samples / 16000)
        self._emit('decoder_conditions', phase='source_prepared',
            reference_version=self.reference_version,
            reference_start_sec=self.reference_conditions['source_start_sec'],
            reference_end_sec=self.reference_conditions['source_end_sec'],
            prompt_tokens=0 if prompt is None else len(prompt))
        return self.reference_conditions

    def _speech(self, tokens):
        tokens = list(tokens)
        if not tokens:
            return
        self.speech_count += len(tokens)
        self._emit('speech_tokens', tokens=tokens,
                   reference_version=(self.active_reference or {}).get('version'),
                   reference_source_end_sec=(self.active_reference or {}).get('source_end_sec'))
        if self.decode:
            self.pending_codec.extend(tokens)
            self._flush_codec()

    def _flush_codec(self, final=False):
        if not self.decode or not self.pending_codec:
            return
        if self._ensure_decoder(final=final):
            rollovers = self.decoder.session_rollovers
            refreshes = getattr(self.decoder, 'source_condition_updates', 0)
            self.decoder.feed(self.pending_codec, on_pcm_chunk=self._pcm)
            self.pending_codec.clear()
            if getattr(self.decoder, 'source_condition_updates', 0) != refreshes:
                self._emit('decoder_conditions', phase='source_applied',
                    reference_version=self.reference_version,
                    source_condition_updates=self.decoder.source_condition_updates,
                    **self.decoder.reference_stats)
            if self.decoder.session_rollovers != rollovers:
                self._emit('context', component='acoustic', replayed=False,
                           rollovers=self.decoder.session_rollovers,
                           retained_tokens=len(self.decoder.tokens),
                           retained_mel_frames=int(self.decoder.generated_mel.size(-1)))

    def _history(self, records, budget, prefix, *, max_turns=None, min_turn=0):
        """Keep whole recent interactions; capacity can shorten the time window.

        A block crossing the left time boundary is kept intact. Replaying the
        retained inputs reconstructs both full-attention and GDN state; this is
        a candidate context policy, not an exact replacement for unbounded KV.
        """
        cutoff = self.consumed_samples / 16000 - self.context_window_sec
        retained, size = deque(), prefix
        for record in reversed(records):
            if record.turn < min_turn or (max_turns is not None and len(retained) >= max_turns):
                break
            if self.context_window_sec and record.end_sec <= cutoff:
                break
            count = record.tokens
            if size + count > budget:
                break
            retained.appendleft(record)
            size += count
        return retained, size

    def _expired(self, records):
        return bool(self.context_window_sec and records and
                    records[0].end_sec <= self.consumed_samples / 16000 - self.context_window_sec)

    def _history_event(self, component, records, size, **extra):
        self._emit('context', component=component, retained_tokens=int(size),
            window_sec=self.context_window_sec,
            retained_turns=len(records),
            retained_source_start_sec=records[0].start_sec if records else None,
            retained_source_end_sec=records[-1].end_sec if records else None, **extra)

    def _compact_thinker(self, reserve, boundary=False):
        used = sum(e.size(1) for e in self.embeds)
        overflow = used + reserve > self.capacity
        turn_limit = self.context_window_32_16 and len(self.records) >= 32
        if not overflow and not boundary and not turn_limit and not self._expired(self.records):
            return
        retain_budget = min(self.capacity // 2 if overflow else self.capacity, self.capacity - reserve)
        retained, size = self._history(self.records, retain_budget, self.instruction.size(1),
                                       max_turns=16 if turn_limit else None)
        if size + reserve > self.capacity:
            raise RuntimeError('single source/WRITE block exceeds thinker capacity')
        self.forward_count += self.run.n_forward
        self._new_thinker(retained)
        if self.context_window_32_16:
            self.context_turn_floor = retained[0].turn if retained else self.step_count
        self.context_replays += 1
        self._history_event('thinker', retained, size, replayed=True, boundary=boundary,
                            turn_limit=turn_limit, capacity_limit=overflow)

    def _compact_talker(self, reserve, boundary=False):
        limit = self.capacity
        overflow = self.talker_fed + reserve > limit
        below_floor = self.context_window_32_16 and self.talker_records and (
            self.talker_records[0].turn < self.context_turn_floor)
        if not overflow and not boundary and not below_floor and not self._expired(self.talker_records):
            return
        retain_budget = max(0, min(self.capacity // 2 if overflow else limit, limit - reserve))
        retained, size = self._history(self.talker_records, retain_budget, 1,
                                       min_turn=self.context_turn_floor)
        if size + reserve > self.capacity:
            raise RuntimeError('single WRITE exceeds talker capacity')
        pending = self.state.pending
        decoded = self.state.decoded[-10:]
        counters = (self.state.n_seg, self.state.n_fill, self.state.n_segcap)
        cond_consumed = self.state.cond_consumed
        self._new_talker()
        if not retained:
            # 完全重置：与原推理「每句从干净 [sos] 开始」一致。fresh state 的 pending 已是 [sos]，
            # 不能再用旧 pending 覆盖（旧 pending 是接在旧上下文后面的 speech token）。
            self.state.n_seg, self.state.n_fill, self.state.n_segcap = counters
            self.state.cond_consumed = cond_consumed
            self.talker_replays += 1
            self._emit('context', component='talker', replayed=False, reset=True, retained_tokens=0)
            return
        # Rebuild both attention cache and recurrent state without sampling/output.
        sos = self.model.talker.sos_embed(self.state.dtype, self.state.device)
        inputs = torch.cat([sos] + [x for record in retained for x in record], dim=1)
        for x in inputs.split(256, dim=1):
            if self.state.graph is not None:
                self.state.graph.prefill(x)
            else:
                out = self.model.talker.backbone(inputs_embeds=x, past_key_values=self.state.past_key_values, use_cache=True)
                self.state.past_key_values = out.past_key_values
        # A completed utterance starts with its own SOS after the retained
        # history; an ordinary in-utterance slide restores the pending token.
        if not boundary:
            self.state.pending = pending
        self.state.decoded = list(decoded)
        self.state.n_seg, self.state.n_fill, self.state.n_segcap = counters
        self.state.cond_consumed = cond_consumed
        self.talker_records = retained
        self.talker_fed = size
        self.talker_replays += 1
        self._history_event('talker', retained, size, replayed=True, boundary=boundary)

    def _talk(self, cond=None, finalize=False):
        if finalize and not self.state.cond_consumed:
            return  # Never synthesize an unconditional SOS/task-only tail.
        cap = self.max_speech_tokens
        reserve = (0 if cond is None else cond.size(1)) + cap + 2
        self._compact_talker(reserve)
        # Freeze enrollment for this WRITE. Polling at token time must not
        # enroll whatever source happens to have arrived after this decision.
        if self.decode:
            condition = self._source_reference(force=self.reference_conditions is None)
            self.active_reference = condition or self.reference_conditions
        record = []
        def observe(x):
            # SOS is rebuilt separately; never retain it twice on a short replay.
            retained = x[:, 1:] if self.talker_fed == 0 else x
            if retained.size(1):
                record.append(retained.detach().clone())
            self.talker_fed += x.size(1)
        # Feed each new codec immediately, before the enclosing WRITE finishes.
        for token in self.model.talker.stream_generate_s4(self.state, cond_chunk=cond,
                finalize=finalize, max_speech_tokens=cap,
                top_k=self.top_k, top_p=self.top_p, input_observer=observe):
            self._speech([token])
        if record:
            self.talker_records.append(ContextRecord(self.last_step_source_sec,
                self.consumed_samples / 16000, tuple(record), turn=self.step_count))
        self.state.decoded = self.state.decoded[-10:]
        self.state.seg_lens = self.state.seg_lens[-4096:]
        if finalize and not self.state.finished:
            raise RuntimeError('Talker final flush hit segment cap without EOS; emitted audio cannot be retried')

    def _step(self, mel, source_done=False, tail=False):
        wait_id, write_id, chunk_id, src_end_id = self.ids
        before = time.perf_counter()
        dtype = self.model.thinker.embed_tokens.weight.dtype
        self._compact_thinker(int(mel.size(-1) // 8) + self.max_text_tokens + 6)
        begin = len(self.embeds)
        self.embeds.append(self._tok(chunk_id))
        frames = 0
        if mel.size(-1):
            feats, self.vcache = self.model.encode_audio_chunk(mel.unsqueeze(0).to(self.device), cache=self.vcache)
            frames = feats.size(1)
            if frames:
                pad = torch.full((1, frames), int(self.model.config.audio_ids[0]), device=self.device)
                self.embeds.append(self.model.inject_audio_features(pad, self.model.thinker.embed_tokens(pad), [feats[0]]).to(dtype))
        if source_done and not tail:
            self.embeds.append(self._tok(src_end_id))
        logits, _ = self.run()
        allow = [wait_id, write_id] + ([int(self.tokenizer.eos_token_id)] if source_done else [])
        if self.write_threshold is not None and not source_done:
            p = logits[[write_id, wait_id]].float().softmax(-1)[0].item()
            decision = write_id if p >= self.write_threshold else wait_id
        else:
            decision = allow[int(logits[allow].argmax())]
        action = 'write' if decision == write_id else ('wait' if decision == wait_id else 'eos')
        self._emit('decision', action=action, n_audio_frames=int(frames))
        term = None
        delta = []
        if action != 'eos':
            self.embeds.append(self._tok(decision))
        if action == 'write':
            self.wrote = True
            delta, out, term = sample_delta(self.embeds, self.run, self._tok, _pick,
                chunk_id, int(self.tokenizer.eos_token_id), wait_id,
                ban_above=int(self.tokenizer.vocab_size), temperature=1.0, top_p=0.0,
                max_tokens=self.max_text_tokens)
            if delta:
                text = self.tokenizer.decode(delta, skip_special_tokens=True).strip()
                visible_delta = (' ' if self.text_pieces and self.lang not in NO_SPACE_LANGS else '') + text
                if self.first_text_sec is None:
                    self.first_text_sec = time.perf_counter() - self.started
                self.text_pieces.append(text)
                self.text_count += len(delta)
                self._emit('text', text=visible_delta)
                cond_h = self.model.pick_bridge_hidden(out)[:, -len(delta):]
                cond = self.model.build_stream_talker_condition(cond_h, torch.tensor([delta], device=self.device))
                self._talk(cond)
        self.records.append(ContextRecord(self.last_step_source_sec,
            self.consumed_samples / 16000, tuple(x.detach() for x in self.embeds[begin:]),
            turn=self.step_count))
        self.last_step_source_sec = self.consumed_samples / 16000
        # Source end may need a few zero-frame steps to drain remaining translation.
        finished = action == 'eos' or (source_done and (action == 'wait' or term == self.tokenizer.eos_token_id))
        if term == self.tokenizer.eos_token_id and not source_done:
            # Translation EOS before live source ends: start another linguistic segment.
            self._talk(finalize=True)
            self._restart_linguistic_segment()
            self._emit('segment_end', reason='model_eos')
        dt = time.perf_counter() - before
        self.block_times.append(dt)
        step = dict(step=self.step_count, source_end_sec=self.consumed_samples / 16000,
                    source_end_ms=self.consumed_samples / 16, decision=action,
                    text_delta=self.tokenizer.decode(delta, skip_special_tokens=True) if delta else '',
                    n_audio_frames=int(frames), elapsed_sec=time.perf_counter() - self.started,
                    compute_sec=dt, src_end=bool(source_done and not tail), tail=tail)
        if self.collect_audio:
            self.steps.append(step)
        self.step_count += 1
        self._emit('step', **step)
        return finished

    @torch.inference_mode()
    def _prepare_mel(self, required_frames=0, finalize=False):
        """Wait for STFT right context before committing a complete model chunk."""
        stream = self.waveform_stream
        missing = max(0, required_frames - self.pending_mel.size(-1))
        if finalize:
            end = self.seg_samples
        elif not missing:
            return True
        else:
            # Frame k needs samples through k*hop+n_fft/2. At 640ms this
            # waits for 40 additional observed samples, not a future file read.
            end = (stream.emitted_frames + missing - 1) * stream.hop_length + stream.n_fft // 2
            if end > self.seg_samples:
                return False
        begin = stream.num_samples
        offset = begin - (self.seg_samples - self.pending_audio.size)
        if offset < 0 or end < begin or end > self.seg_samples:
            raise RuntimeError('frontend/source cursor mismatch')
        mel = stream.feed(self.pending_audio[offset:offset + end - begin], finalize=finalize)
        self.pending_mel = torch.cat([self.pending_mel, mel], dim=-1)
        return True

    @torch.inference_mode()
    def feed_audio(self, wav16, finalize=False, segment_end=False):
        """segment_end=True：外部（VAD）判定这一段说完了。收尾流程与 finalize 完全相同
        （补最后一块 → <|src_end|> → 收尾步 → talker 冲刷 → 声码器收尾），但不关会话，
        并重置下一段要用的状态（thinker / talker / Voxtral 缓存 / 声学条件），等价于基线「每句一个新会话」。"""
        chunk = np.asarray(wav16, dtype=np.float32)
        if chunk.ndim != 1 or not np.isfinite(chunk).all():
            raise ValueError('audio must be finite mono float samples at 16kHz')
        if self.failed:
            raise RuntimeError('session failed; emitted audio cannot be replayed or retried')
        finalize = bool(finalize or segment_end)
        if self.closed:
            if chunk.size or not finalize:
                raise RuntimeError('session already finalized')
            return []
        total = (self.seg_samples if segment_end else self.source_samples) + chunk.size
        if finalize and total == 0:
            if segment_end:
                return []                       # 当前没有进行中的段
            self.closed = True
            self.stop = 'empty'
            self._batch = []
            self._emit('end', stop='empty')
            result, self._batch = self._batch, None
            return result
        if finalize and total < 400:
            if segment_end:
                self._reset_segment()           # 短于 25ms：丢弃，不送进模型
                return []
            raise ValueError('source audio must contain at least 25ms for STFT')
        before = time.perf_counter()
        self._batch = []
        try:
            self.source_samples += chunk.size
            self.seg_samples += chunk.size
            self.pending_audio = np.concatenate([self.pending_audio, chunk])
            finished = False
            sent_source_end = False
            if finalize:
                self._prepare_mel(finalize=True)
            while self.pending_audio.size:
                ms = self.chunk_ms_after_write if self.wrote and self.chunk_ms_after_write else self.chunk_ms
                size = int(ms * 16)
                if self.pending_audio.size < size and not finalize:
                    break
                if not finalize and not self._prepare_mel(size // self.waveform_stream.hop_length):
                    break
                if not finalize and self.pending_mel.size(-1) < size // self.waveform_stream.hop_length:
                    raise RuntimeError('frontend did not produce the required stable mel frames')
                n = min(size, self.pending_audio.size)
                part, self.pending_audio = self.pending_audio[:n], self.pending_audio[n:]
                self.reference_audio = np.concatenate([self.reference_audio, part])[-self.reference_samples:].copy()
                self.consumed_samples += n
                self._flush_codec()
                last = finalize and not self.pending_audio.size
                take = self.pending_mel.size(-1) if last else n // self.waveform_stream.hop_length
                mel, self.pending_mel = self.pending_mel[:, :take], self.pending_mel[:, take:]
                if last:
                    remainder = 0 if self.vcache is None or self.vcache.mel_remainder is None else self.vcache.mel_remainder.size(-1)
                    pad = -(mel.size(-1) + remainder) % 8
                    if pad:
                        # Pad only the encoder's final adapter group; never discard source tail.
                        fill = mel[:, -1:] if mel.size(-1) else self.vcache.mel_remainder[0, :, -1:].cpu()
                        mel = torch.cat([mel, fill.expand(-1, pad)], dim=-1)
                        self.source_padding_mel_frames += pad
                finished = self._step(mel, source_done=last)
                sent_source_end = sent_source_end or last
            if finalize:
                if self.pending_mel.size(-1) and sent_source_end:
                    raise RuntimeError('unconsumed final mel frames')
                if not sent_source_end:
                    mel, self.pending_mel = self.pending_mel, self.pending_mel[:, :0]
                    remainder = 0 if self.vcache is None or self.vcache.mel_remainder is None else self.vcache.mel_remainder.size(-1)
                    pad = -(mel.size(-1) + remainder) % 8
                    if pad:
                        fill = mel[:, -1:] if mel.size(-1) else self.vcache.mel_remainder[0, :, -1:].cpu()
                        mel = torch.cat([mel, fill.expand(-1, pad)], dim=-1)
                        self.source_padding_mel_frames += pad
                    finished = self._step(mel, source_done=True)
                for _ in range(self.max_tail_steps):
                    if finished:
                        break
                    finished = self._step(torch.zeros(128, 0), source_done=True, tail=True)
                self._talk(finalize=True)
                self._flush_codec(final=True)
                if self.decoder is not None:
                    self.decoder.finish(on_pcm_chunk=self._pcm)
                if segment_end:
                    self.segments += 1
                    self._emit('segment_end', reason='external', segment=self.segments,
                               stop='eos' if finished else 'tail_cap')
                    self._reset_segment()
                else:
                    self.closed = True
                    self.stop = 'eos' if finished else 'tail_cap'
                    self._emit('end', stop=self.stop)
            return self._batch
        except Exception:
            self.failed = True
            self.stop = 'failed'
            raise
        finally:
            self.compute_sec += time.perf_counter() - before
            self._batch = None

    def _restart_linguistic_segment(self):
        self.completed_talker_stats = [a + b for a, b in zip(self.completed_talker_stats,
            (self.state.n_seg, self.state.n_fill, self.state.n_segcap))]
        if self.keep_context:
            # sample_delta does not append its EOS. Make the utterance boundary
            # explicit in retained history without emitting another text token.
            if self.records:
                last = self.records[-1]
                self.records[-1] = replace(last, embeds=last.embeds + (self._tok(self.tokenizer.eos_token_id),))
            self._compact_thinker(self.max_text_tokens + 38, boundary=True)
            self._compact_talker(self.max_speech_tokens + 2, boundary=True)
            self.state.n_seg = self.state.n_fill = self.state.n_segcap = 0
            self.state.cond_consumed = 0
        else:
            self.forward_count += self.run.n_forward
            self._new_talker()
            self._new_thinker()

    def _reset_segment(self):
        """Keep recent semantic/reference history only in the candidate policy.

        The guarded first phase still finalizes the waveform frontend and
        resets encoder state at VAD boundaries. Removing that boundary requires
        separate GPU validation; skipping an STFT finalize/reset is not valid.
        """
        self._restart_linguistic_segment()
        self.vcache = None                                  # Voxtral 编码器缓存：基线每句 vcache=None
        self.waveform_stream = VoxtralWaveformStream(self.mel_fn)
        self.pending_mel = torch.zeros(self.waveform_stream.feature_size, 0)
        self.pending_audio = np.zeros(0, np.float32)
        if not self.keep_context:
            self.reference_audio = np.zeros(0, np.float32)
            self.reference_updated_at = None
            self.reference_conditions = None
            self.active_reference = None
        self.pending_codec = []
        self.decoder = None                                 # 已 finish；下一段重新建
        self.wrote = False
        self._consec_cap = 0
        self.seg_samples = 0

    def note(self, kind, **data):
        """外部组件（分段器）往事件流里写一条记录。"""
        return self._emit(kind, **data)

    def close(self):
        """结束整个会话。有未结束的段先按外部分段收尾。"""
        if self.closed:
            return []
        out = []
        if self.seg_samples > 0 or self.pending_audio.size:
            out = self.feed_audio(np.zeros(0, np.float32), segment_end=True)
        self.closed = True
        self.stop = 'eos'
        self._emit('end', stop=self.stop)
        return out

    def metrics(self):
        source = self.consumed_samples / 16000
        blocks = np.asarray(self.block_times, dtype=float)
        n_seg = self.completed_talker_stats[0] + self.state.n_seg
        n_fill = self.completed_talker_stats[1] + self.state.n_fill
        return dict(source_sec=source, output_sec=self.output_samples / 24000,
                    fill_rate=n_fill / n_seg if n_seg else 0.,
                    compute_sec=self.compute_sec, inference_rtf=self.compute_sec / source if source else None,
                    elapsed_sec=time.perf_counter() - self.started,
                    first_text_sec=self.first_text_sec, first_pcm_sec=self.first_pcm_sec,
                    block_p50_sec=float(np.percentile(blocks, 50)) if blocks.size else None,
                    block_p95_sec=float(np.percentile(blocks, 95)) if blocks.size else None,
                    block_p99_sec=float(np.percentile(blocks, 99)) if blocks.size else None,
                    context_replays=self.context_replays, talker_replays=self.talker_replays,
                    context_window_sec=self.context_window_sec,
                    context_policy=('turns_32_16' if self.context_window_32_16 else
                                    'timed_replay' if self.context_window_sec else 'per_segment'),
                    acoustic_rollovers=0 if self.decoder is None else self.decoder.session_rollovers,
                    source_reference_versions=self.reference_version,
                    source_reference_policy=self.reference_policy,
                    source_condition_updates=0 if self.decoder is None else getattr(self.decoder, 'source_condition_updates', 0),
                    thinker_retained_tokens=sum(int(e.size(1)) for e in self.embeds),
                    talker_retained_tokens=int(self.talker_fed),
                    n_forward=self.forward_count + self.run.n_forward,
                    source_padding_mel_frames=self.source_padding_mel_frames,
                    mel_chunk_alignment='stable_frames', pending_mel_frames=int(self.pending_mel.size(-1)))

    def result(self):
        result = dict(text=('' if self.lang in NO_SPACE_LANGS else ' ').join(self.text_pieces),
                      text_pieces=self.text_pieces, text_token_total=self.text_count,
                      speech_token_total=self.speech_count, speech_sec=self.speech_count / 25,
                      steps=self.steps, events=self.events, synthesis_mode='streaming',
                      chunk_ms=self.chunk_ms, chunk_ms_after_write=self.chunk_ms_after_write,
                      stop=self.stop, **self.metrics())
        if self.collect_audio:
            result['wav'] = np.concatenate(self.audio) if self.audio else np.zeros(0, np.float32)
        return result
