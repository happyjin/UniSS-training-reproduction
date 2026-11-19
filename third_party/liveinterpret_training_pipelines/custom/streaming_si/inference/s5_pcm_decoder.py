"""Persistent S5/CosyVoice3 token-to-PCM decoding.

A WRITE and a bounded window rollover are input events, not end-of-stream.
Only ``finish`` finalizes Flow/HiFT; a rollover borrows real codec lookahead.
This requires causal HiFT; the legacy noncausal generator has other seam rules.
"""
from collections import deque
import inspect
import math
import time

import torch

from streaming_si.inference.s5_hift_stream import BoundedCausalHiFT


class PersistentSpeechDecoder:
    """Bound target history while emitting every PCM sample exactly once.

    Window rollover uses paired generated token/mel tail as the next Flow
    prompt. HiFT has a separate bounded stream carrying excitation phase and
    absolute noise position; a Flow window end never finalizes that stream.
    This bounds recomputation; acoustic equivalence at the new Flow boundary
    must still be checked on the actual checkpoint (it is not guaranteed).
    ``on_pcm_chunk`` is synchronous, invoked immediately for each output block.
    The caller owns transport timestamps and must not collect before sending.
    """

    # Internal diagnostic subclasses may isolate these two changes. Normal
    # inference always continues across a bounded window with real lookahead.
    _continue_flow_windows = True
    _align_window_noise = True

    def __init__(self, model, spk_emb=None, prompt_speech_tokens=None,
                 prompt_feat=None, token_hop_len=None, ode_steps=5,
                 max_session_sec=60.0, prompt_tail_sec=2.0, condition_provider=None,
                 reference_provider=None, reference_metadata=None,
                 bind_reference_to_tokens=False):
        if model.speech_decoder is None:
            raise RuntimeError('attach_speech_decoder must precede streaming decoding')
        self.model = model
        self.flow, self.hift = model.speech_decoder
        params = inspect.signature(self.hift.inference).parameters
        if 'finalize' not in params or 'cache_source' in params:
            raise ValueError('PersistentSpeechDecoder requires CosyVoice3 causal HiFT')
        self.device = next(self.flow.parameters()).device
        self.ratio = int(getattr(self.flow, 'token_mel_ratio', 2))
        self.lookahead = int(getattr(self.flow, 'pre_lookahead_len', 3))
        self.initial_hop = int(token_hop_len or model.STREAM_TOKEN_HOP_LEN)
        self.max_hop = max(self.initial_hop, int(model.STREAM_TOKEN_MAX_HOP_LEN))
        self.scale = int(model.STREAM_SCALE_FACTOR)
        self.samples_per_mel = int(model.STREAM_SAMPLES_PER_MEL)
        self.token_rate = 24000 / (self.ratio * self.samples_per_mel)
        if (not math.isfinite(max_session_sec) or max_session_sec <= 0 or
                not math.isfinite(prompt_tail_sec) or prompt_tail_sec < 0):
            raise ValueError('session seconds must be positive; prompt tail seconds nonnegative')
        self.max_tokens = int(max_session_sec * self.token_rate)
        self.tail_tokens = min(int(prompt_tail_sec * self.token_rate), self.max_tokens)
        if self.initial_hop <= 0 or self.max_tokens < self.initial_hop + self.lookahead:
            raise ValueError('acoustic window must fit at least one hop plus lookahead')
        self.prompt, self.prompt_feat, self.embedding = model._prep_decode_inputs(
            spk_emb, prompt_speech_tokens, prompt_feat, self.device, self.ratio)
        self._initial_prompt_frames = int(self.prompt_feat.size(1))
        self.source_prompt = self.prompt.clone()
        self.source_feat = self.prompt_feat.clone()
        self.continuity_prompt = self.prompt[:, :0].clone()
        self.continuity_feat = self.prompt_feat[:, :0].clone()
        self.reference_provider = reference_provider
        self.bind_reference_to_tokens = bool(bind_reference_to_tokens)
        self.token_references = []
        self.source_condition_updates = 0
        metadata = reference_metadata or {}
        self.reference_version = metadata.get('version')
        self.reference_start_sec = metadata.get('source_start_sec')
        self.reference_end_sec = metadata.get('source_end_sec')
        self.ode_steps = ode_steps
        self.pending_embedding = None
        self.condition_provider = condition_provider
        self.tokens = []
        self.offset = 0
        self.hop = self.initial_hop
        self.generated_mel = torch.zeros(1, 80, 0, device=self.device)
        self.hift_stream = BoundedCausalHiFT(self.hift, max_history_sec=max_session_sec,
                                            context_sec=prompt_tail_sec)
        # Kept for older diagnostic readers; HiFT context now lives in its own
        # stream rather than being replaced at each Flow rollover.
        self.hift_prefix = torch.zeros(1, 80, 0, device=self.device)
        self.hift_emitted = 0
        self.total_samples = 0
        self.total_tokens = 0
        self.session_rollovers = 0
        self.source_rebases = self.capacity_rollovers = 0
        self.closed = False
        self.failed = False
        self.block_compute_sec = deque(maxlen=4096)
        self.n_blocks = 0

    def set_next_embedding(self, spk_emb):
        """Freeze current speaker condition; apply this update at next rollover."""
        self.pending_embedding = torch.as_tensor(
            spk_emb, dtype=torch.float32, device=self.device).reshape(1, -1)

    @property
    def reference_stats(self):
        return dict(source_prompt_tokens=int(self.source_prompt.size(1)),
                    target_context_tokens=int(self.continuity_prompt.size(1)),
                    reference_start_sec=self.reference_start_sec,
                    reference_end_sec=self.reference_end_sec)

    @property
    def stats(self):
        return dict(total_samples=self.total_samples, total_tokens=self.total_tokens,
                    session_rollovers=self.session_rollovers, n_blocks=self.n_blocks,
                    source_rebases=self.source_rebases, capacity_rollovers=self.capacity_rollovers,
                    retained_tokens=len(self.tokens),
                    retained_mel_frames=int(self.generated_mel.size(-1)),
                    prompt_tokens=int(self.prompt.size(1)),
                    hift_retained_mel_frames=self.hift_stream.retained_frames,
                    hift_context_crops=self.hift_stream.crops,
                    flow_window_lookahead=self.lookahead if self._continue_flow_windows else 0,
                    block_compute_sec=list(self.block_compute_sec),
                    source_condition_updates=self.source_condition_updates,
                    **self.reference_stats)

    def _set_prompt(self):
        # New source timbre/prosody and generated-target continuity have distinct
        # roles, but Flow accepts a single paired acoustic prefix. Keep both
        # blocks token/mel aligned; HiFT never sees this reference-only prefix.
        self.prompt = torch.cat([self.source_prompt, self.continuity_prompt], 1)
        self.prompt_feat = torch.cat([self.source_feat, self.continuity_feat], 1)

    def _refresh_reference(self, condition=None, *, poll=True):
        if self.reference_provider is None:
            return
        if poll:
            condition = self.reference_provider()
        if not condition:
            return
        version = condition.get('version')
        if version is not None and version == self.reference_version:
            return
        prompt, feat, embedding = self.model._prep_decode_inputs(
            condition.get('spk_emb'), condition.get('prompt_speech_tokens'),
            condition.get('prompt_feat'), self.device, self.ratio)
        # Rebase at the last committed synthesis boundary. Pending lookahead is
        # kept, the old target tail becomes reference only, and no PCM is replayed.
        if self.offset:
            self._rollover(reason='reference')
        self.source_prompt, self.source_feat = prompt.clone(), feat.clone()
        self.embedding = embedding
        self.reference_version = version
        self.reference_start_sec = condition.get('source_start_sec')
        self.reference_end_sec = condition.get('source_end_sec')
        self._set_prompt()
        self.source_condition_updates += 1

    def _refresh_bound_reference(self):
        if self.bind_reference_to_tokens and self.offset < len(self.token_references):
            # The first target token in this hop owns the condition. New WRITE
            # lookahead may have a later voice, but cannot change old audio.
            self._refresh_reference(self.token_references[self.offset], poll=False)

    def _flow_mel(self, tok, n_input, finalize):
        def run():
            return self.model._flow_infer(
                self.flow, tok, torch.tensor([n_input], device=self.device),
                self.prompt, torch.tensor([self.prompt.size(1)], device=self.device),
                self.prompt_feat, torch.tensor([self.prompt_feat.size(1)], device=self.device),
                self.embedding, streaming=not finalize, finalize=finalize)
        if not self._align_window_noise or not self.session_rollovers:
            return run()
        cfm = self.flow.decoder
        noise = cfm.rand_noise
        window_start = self.total_tokens - len(self.tokens)
        absolute = self._initial_prompt_frames + window_start * self.ratio - self.prompt_feat.size(1)
        needed = self.prompt_feat.size(1) + n_input * self.ratio
        period = noise.size(-1)
        start = int(absolute) % period
        if start + needed <= period:
            aligned = noise[..., start:start + needed]
        else:
            indices = torch.arange(start, start + needed, device=noise.device) % period
            aligned = noise.index_select(-1, indices)
        # This model serves one active S5 session. Restore the shared constant
        # even after a failed callback/forward; no permanent module patching.
        cfm.rand_noise = aligned
        try:
            return run()
        finally:
            cfm.rand_noise = noise

    def _infer(self, finalize, finish_stream=False, end_token=None):
        start = time.perf_counter()
        self.model._set_ode_steps(self.flow, self.ode_steps)
        pad = -int(self.prompt.size(1)) % self.initial_hop
        this_hop = self.hop + pad if self.offset == 0 else self.hop
        end = end_token if end_token is not None else (len(self.tokens) if finalize else self.offset + this_hop)
        n_input = end if finalize else end + self.lookahead
        tok = torch.as_tensor(self.tokens[:n_input], dtype=torch.long,
                              device=self.device).reshape(1, -1)
        mel_full = self._flow_mel(tok, n_input, finalize)
        expected = end * self.ratio
        if mel_full.size(-1) < expected:
            raise RuntimeError(f'Flow returned {mel_full.size(-1)} frames; need {expected}')
        if finalize and mel_full.size(-1) != expected:
            raise RuntimeError('final Flow token/mel length mismatch')
        # Flow implementations may include the lookahead in their returned mel.
        # Commit only the completed hop, never speculative future mel.
        new_mel = mel_full[:, :, self.offset * self.ratio:expected]
        self.generated_mel = torch.cat([self.generated_mel, new_mel], dim=-1)
        out = self.hift_stream.feed(new_mel, finalize=finish_stream)
        self.hift_emitted = self.hift_stream.total_samples
        self.offset = end
        self.hop = min(self.max_hop, self.hop * self.scale)
        # GPU materialization is caller-owned. Synchronize before the timing end
        # so profiler-free block timings still measure actual completed work.
        if out.device.type == 'cuda':
            torch.cuda.synchronize(out.device)
        self.block_compute_sec.append(time.perf_counter() - start)
        self.n_blocks += 1
        return out

    def _emit(self, finalize, output, callback, finish_stream=False, end_token=None):
        chunk = self._infer(finalize, finish_stream=finish_stream, end_token=end_token)
        if chunk.size(-1):
            self.total_samples += int(chunk.size(-1))
            output.append(chunk)
            if callback is not None:
                callback(chunk)

    def _rollover(self, reason='capacity'):
        committed = self.tokens[:self.offset]
        pending = self.tokens[self.offset:]
        keep = min(self.tail_tokens, len(committed))
        tail_prompt = torch.as_tensor(committed[-keep:] if keep else [],
                                     dtype=torch.long, device=self.device).reshape(1, -1)
        tail = (self.generated_mel[..., -keep * self.ratio:].clone() if keep
                else self.generated_mel[..., :0].clone())
        if self.reference_provider is not None:
            history = torch.cat([self.continuity_prompt,
                                 torch.as_tensor(committed, dtype=torch.long, device=self.device).reshape(1, -1)], 1)
            features = torch.cat([self.continuity_feat, self.generated_mel.transpose(1, 2)], 1)
            keep = min(self.tail_tokens, history.size(1))
            self.continuity_prompt = history[:, -keep:].clone() if keep else history[:, :0].clone()
            self.continuity_feat = features[:, -keep * self.ratio:].clone() if keep else features[:, :0].clone()
            self._set_prompt()
        else:
            # Preserve the old fixed-reference A/B path exactly.
            self.prompt = tail_prompt
            self.prompt_feat = tail.transpose(1, 2).contiguous()
        self.generated_mel = self.generated_mel[..., :0].clone()
        self.tokens = pending
        if self.bind_reference_to_tokens:
            self.token_references = self.token_references[self.offset:]
        self.offset = 0
        self.hop = self.initial_hop
        if self.condition_provider is not None:
            condition = self.condition_provider() or {}
            if condition.get('spk_emb') is not None:
                self.set_next_embedding(condition['spk_emb'])
        if self.pending_embedding is not None:
            self.embedding = self.pending_embedding
            self.pending_embedding = None
        self.session_rollovers += 1
        if reason == 'reference':
            self.source_rebases += 1
        else:
            self.capacity_rollovers += 1

    @torch.inference_mode()
    def feed(self, tokens, finalize=False, on_pcm_chunk=None):
        """Accept new tokens without flushing an incomplete hop at WRITE end.

        Empty feeds are idle calls, not silence or end-of-stream. A finished or
        failed session cannot be reused: emitted audio cannot be rolled back.
        """
        if self.closed or self.failed:
            raise RuntimeError('speech decoder is already closed or failed')
        output = []
        try:
            bound_condition = (self.reference_provider() if self.bind_reference_to_tokens
                               and self.reference_provider is not None else None)
            reference_checked = False
            for token in tokens:
                if not self.bind_reference_to_tokens and not reference_checked and not finalize:
                    self._refresh_reference()
                    reference_checked = True
                # Delay rollover until there really is another token. A stream
                # ending exactly at the cap should not create an empty session.
                if not self._continue_flow_windows and len(self.tokens) == self.max_tokens:
                    self._emit(True, output, on_pcm_chunk)
                    self._rollover()
                self.tokens.append(int(token))
                if self.bind_reference_to_tokens:
                    self.token_references.append(bound_condition)
                self.total_tokens += 1
                if self._continue_flow_windows and len(self.tokens) >= self.max_tokens + self.lookahead:
                    # A capacity boundary is not an utterance end. Complete
                    # the old window with real right context, then carry those
                    # unconsumed lookahead tokens into the new window once.
                    self._refresh_bound_reference()
                    if len(self.tokens) >= self.max_tokens + self.lookahead:
                        self._emit(False, output, on_pcm_chunk, end_token=self.max_tokens)
                        self._rollover()
                pad = -int(self.prompt.size(1)) % self.initial_hop
                this_hop = self.hop + pad if self.offset == 0 else self.hop
                if (len(self.tokens) - self.offset >= this_hop + self.lookahead and
                        (not self._continue_flow_windows or self.offset + this_hop <= self.max_tokens)):
                    self._refresh_bound_reference()
                    pad = -int(self.prompt.size(1)) % self.initial_hop
                    this_hop = self.hop + pad if self.offset == 0 else self.hop
                    if (len(self.tokens) - self.offset >= this_hop + self.lookahead and
                            (not self._continue_flow_windows or self.offset + this_hop <= self.max_tokens)):
                        self._emit(False, output, on_pcm_chunk)
            if finalize:
                if self.tokens:
                    self._refresh_bound_reference()
                    self._emit(True, output, on_pcm_chunk, finish_stream=True)
                self.closed = True
                expected = self.total_tokens * self.ratio * self.samples_per_mel
                if self.total_samples != expected:
                    raise RuntimeError(f'PCM samples {self.total_samples} != token duration {expected}')
        except Exception:
            self.failed = True
            raise
        return output

    def finish(self, on_pcm_chunk=None):
        return self.feed((), finalize=True, on_pcm_chunk=on_pcm_chunk)
