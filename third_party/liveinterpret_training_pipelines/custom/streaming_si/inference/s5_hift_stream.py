"""Bounded CosyVoice3 causal HiFT state, independent of Flow window ends.

The upstream API recomputes an entire prefix. Cropping that prefix without
carrying F0 phase and the absolute excitation-noise position changes the voice
excitation at the join. Keep those states and only emit newly stable samples.
"""
import math

import torch
import torch.nn.functional as F


class BoundedCausalHiFT:
    def __init__(self, hift, max_history_sec=60., context_sec=2.):
        self.hift = hift
        self.sine = hift.m_source.l_sin_gen
        if (not getattr(self.sine, 'causal', False) or
                getattr(self.sine, 'flag_for_pulse', True) or
                not hasattr(self.sine, 'sine_waves') or
                hift.sampling_rate != 24000):
            raise ValueError('bounded HiFT requires the 24kHz causal SineGen2 checkpoint')
        self.scale = int(self.sine.upsample_scale)
        if self.scale != 480:
            raise ValueError('bounded HiFT requires 480 samples per mel frame')
        # Left context covers the causal convolutions; retain at least 2s even
        # when the Flow source prompt is shorter or deliberately disabled.
        self.context_frames = max(100, math.ceil(context_sec * 50))
        self.capacity_frames = math.ceil(max_history_sec * 50) + self.context_frames
        self.mel = self.f0_cache = self.source_cache = self.phase_cache = None
        self.phase_base = None
        self.base_frame = self.emitted = self.total_frames = self.total_samples = 0
        self.crops = 0
        self.closed = False

    @property
    def retained_frames(self):
        return 0 if self.mel is None else self.mel.size(-1)

    def _noise(self, start, count, device):
        table = self.sine.sine_waves
        period = table.size(1)
        start %= period
        if start + count <= period:
            return table[:, start:start + count].to(device)
        # The checkpoint owns a fixed 300s noise table. Index it cyclically;
        # memory and absolute phase remain bounded beyond that duration.
        parts = []
        while count:
            n = min(count, period - start)
            parts.append(table[:, start:start + n])
            count -= n
            start = 0
        return torch.cat(parts, 1).to(device)

    def _source(self, f0):
        g = self.sine
        x = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        harmonics = torch.arange(1, g.dim + 1, dtype=torch.float32, device=x.device)
        rad = (x * harmonics / g.sampling_rate) % 1
        if self.base_frame == 0:
            rad[:, 0, :] += g.rand_ini.to(rad.device)
        rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.scale,
                            mode='linear').transpose(1, 2)
        # Preserve upstream operation order until the first crop. At crops,
        # retain the accumulated harmonic phase rather than starting at zero.
        phase = torch.cumsum(rad, dim=1) * 2 * math.pi
        phase = phase * self.scale
        if self.phase_base is not None:
            phase = phase + self.phase_base[:, None, :]
        self.phase_cache = phase
        if self.base_frame == 0:
            # Preserve the checkpoint's exact source implementation until a
            # physical crop actually requires an explicit phase offset.
            source, _, _ = self.hift.m_source(x)
            self.source_cache = source.transpose(1, 2).detach()
            return self.source_cache
        wave_phase = F.interpolate(phase.transpose(1, 2), scale_factor=self.scale,
                                   mode='nearest').transpose(1, 2)
        uv = (x > g.voiced_threshold).float()
        amp = uv * g.noise_std + (1 - uv) * g.sine_amp / 3
        noise = self._noise(self.base_frame * self.scale, x.size(1), x.device)
        waves = torch.sin(wave_phase) * g.sine_amp * uv + amp * noise
        source = self.hift.m_source.l_tanh(self.hift.m_source.l_linear(waves)).transpose(1, 2)
        if self.source_cache is not None:
            n = self.source_cache.size(-1)
            if n > source.size(-1):
                raise RuntimeError('HiFT stable source prefix moved backwards')
            source[..., :n] = self.source_cache
        self.source_cache = source.detach()
        return source

    @torch.inference_mode()
    def feed(self, new_mel, finalize=False):
        if self.closed:
            raise RuntimeError('HiFT stream is closed')
        if self.mel is None:
            self.mel = new_mel.clone()
        elif new_mel.size(-1):
            self.mel = torch.cat([self.mel, new_mel], -1)
        self.total_frames += new_mel.size(-1)
        if not new_mel.size(-1) and not finalize:
            return new_mel.new_zeros((new_mel.size(0), 0))
        if not self.mel.size(-1):
            self.closed = finalize
            return new_mel.new_zeros((new_mel.size(0), 0))
        self.hift.f0_predictor.to(torch.float64)
        f0 = self.hift.f0_predictor(self.mel.to(torch.float64), finalize=finalize).to(self.mel)
        if self.f0_cache is not None and self.base_frame > 0:
            n = self.f0_cache.size(-1)
            if n > f0.size(-1):
                raise RuntimeError('HiFT stable F0 prefix moved backwards')
            f0[..., :n] = self.f0_cache
        self.f0_cache = f0.detach()
        source = self._source(f0)
        x = self.mel if finalize else self.mel[..., :-self.hift.f0_predictor.condnet[0].causal_padding]
        wav = self.hift.decode(x=x, s=source, finalize=finalize)
        if wav.size(-1) < self.emitted:
            raise RuntimeError('HiFT output moved behind emitted samples')
        out = wav[..., self.emitted:]
        self.emitted = wav.size(-1)
        self.total_samples += out.size(-1)
        if finalize:
            if self.total_samples != self.total_frames * self.scale:
                raise RuntimeError('HiFT final output length does not match committed mel')
            self.closed = True
        elif self.retained_frames > self.capacity_frames:
            # Crop only complete, already emitted mel frames. A withheld right
            # context is still needed by f0/conv_pre/ISTFT on the next call.
            cut = self.emitted // self.scale - self.context_frames
            if cut > 0:
                self.phase_base = self.phase_cache[:, cut - 1, :].remainder(2 * math.pi).clone()
                self.base_frame += cut
                self.emitted -= cut * self.scale
                self.mel = self.mel[..., cut:].clone()
                self.f0_cache = self.f0_cache[..., cut:].clone()
                self.source_cache = self.source_cache[..., cut * self.scale:].clone()
                self.phase_cache = None
                self.crops += 1
        return out
