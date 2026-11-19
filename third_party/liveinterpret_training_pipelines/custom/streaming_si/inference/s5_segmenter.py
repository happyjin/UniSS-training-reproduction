"""VAD 驱动的外部分段：听到有人说话才开始一段，停顿/软窗口/硬上限到了就结束一段。

背景：基线推理每句话都是一个全新的会话（句子边界由测试清单给出，文件结束即送 <|src_end|>）；
模型训练时看到的也是「单句、从头开始、句末有 <|src_end|>」。真流式如果一路不断地喂，thinker 永远等不到
<|src_end|>，talker 的上下文就一直累积，约 100~260 秒后塌缩（见 docs/S5_STREAMING_INFERENCE_OPTIMIZATION.md）。
这里用 VAD 在音频流里自己找边界，再调用 `S5Session.feed_audio(..., segment_end=True)` 走与基线相同的收尾和重置。

结束一段的三档条件（从上到下优先）：
  pause       停顿 >= pause_end_sec（默认 0.7s）。自然句末。
  soft_pause  段已超过 soft_sec（默认 20s）且停顿 >= soft_pause_sec（默认 0.3s）。尽量在软窗口后的第一个小停顿切。
  hard        段已达 hard_sec（默认 30s）。无论如何都切（会切在句子中间，译文可能被截）。

没有语音时不喂模型；VAD 开销与原始音频时间仍需计入端到端指标。新一段开头带 preroll_sec（默认 0.3s）的语音前音频，避免吞字头。
"""
from collections import deque
import math
import time

import numpy as np

FRAME = 512          # 16kHz 下 32ms，Silero VAD 的帧长
SR = 16000


class SileroFrameVad:
    """逐帧 Silero VAD（有内部状态，按流式顺序调用）。pip 包自带权重，不联网。"""

    def __init__(self, threshold=0.5):
        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        self.model = load_silero_vad()
        self.threshold = threshold
        self.model.reset_states()

    def reset(self):
        self.model.reset_states()

    def is_speech(self, frame):
        with self._torch.no_grad():
            p = self.model(self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)), SR)
        return float(p) >= self.threshold


def validate_segmentation_options(*, pause_end_sec, soft_sec, soft_pause_sec,
                                  hard_sec, preroll_sec, min_speech_sec):
    values = (pause_end_sec, soft_sec, soft_pause_sec, hard_sec, preroll_sec, min_speech_sec)
    if not all(math.isfinite(v) for v in values) or not (
            0 < soft_pause_sec <= pause_end_sec and 0 < soft_sec < hard_sec
            and preroll_sec >= 0 and min_speech_sec > 0):
        raise ValueError('require 0 < soft_pause <= pause_end and 0 < soft_sec < hard_sec; '
                         'preroll >= 0 and min_speech > 0')


class SegmentedFeeder:
    def __init__(self, session, vad, *, pause_end_sec=0.7, soft_sec=20.0, soft_pause_sec=0.3,
                 hard_sec=30.0, preroll_sec=0.3, min_speech_sec=0.25, input_observer=None):
        validate_segmentation_options(pause_end_sec=pause_end_sec, soft_sec=soft_sec,
            soft_pause_sec=soft_pause_sec, hard_sec=hard_sec,
            preroll_sec=preroll_sec, min_speech_sec=min_speech_sec)
        self.session, self.vad = session, vad
        self.input_observer = input_observer
        self.received_samples = self.processed_samples = 0
        self.pause_end = int(round(pause_end_sec * SR / FRAME))
        self.soft_pause = int(round(soft_pause_sec * SR / FRAME))
        self.soft_frames = int(round(soft_sec * SR / FRAME))
        self.hard_frames = int(round(hard_sec * SR / FRAME))
        self.min_speech = max(1, int(round(min_speech_sec * SR / FRAME)))
        self.preroll_frames = int(round(preroll_sec * SR / FRAME))
        self.recent = deque(maxlen=self.preroll_frames + self.min_speech)   # idle 时滚动保留的最近帧
        self.buf = np.zeros(0, np.float32)
        self.active = False
        self.run_speech = self.run_sil = self.seg_frames = 0
        self.total_frames = 0
        self.last_speech_frame = -1          # 全局帧号，用来判断「在哪里切」
        self.segments = []                   # 每段的记录（开始/结束的全局时间、原因、时长）
        self._cur = None

    # ---- 对外 ----
    def feed(self, wav16):
        chunk = np.asarray(wav16, dtype=np.float32)
        if chunk.ndim != 1 or not np.isfinite(chunk).all():
            raise ValueError('expected finite mono 16kHz audio')
        self.received_samples += chunk.size
        self.buf = np.concatenate([self.buf, chunk])
        while self.buf.size >= FRAME:
            frame, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self._frame(frame)

    def close(self):
        """EOF：有效段的尾部原样送入，不制造额外源采样。"""
        self.processed_samples = self.received_samples
        if self.buf.size and self.active:
            self._send(self.buf, self.total_frames * FRAME)
        self.buf = np.zeros(0, np.float32)
        if self.active:
            self._end('eof')
        return self.session.close()

    # ---- 内部 ----
    def _t(self, frame_index):
        return frame_index * FRAME / SR

    def _frame(self, f):
        speech = self.vad.is_speech(f)
        idx = self.total_frames
        self.total_frames += 1
        self.processed_samples = self.total_frames * FRAME
        if speech:
            self.last_speech_frame = idx
        if not self.active:
            self.recent.append(f)
            self.run_speech = self.run_speech + 1 if speech else 0
            if self.run_speech >= self.min_speech:
                self._start(idx)
            return
        self._send(f, idx * FRAME)
        self.seg_frames += 1
        self.run_sil = 0 if speech else self.run_sil + 1
        if self.run_sil >= self.pause_end:
            self._end('pause')
        elif self.seg_frames >= self.soft_frames and self.run_sil >= self.soft_pause:
            self._end('soft_pause')
        elif self.seg_frames >= self.hard_frames:
            self._end('hard')

    def _start(self, idx):
        frames = list(self.recent)
        self.recent.clear()
        self.active, self.run_sil, self.run_speech = True, 0, 0
        self.seg_frames = len(frames)
        start_frame = idx - len(frames) + 1
        self._cur = dict(index=len(self.segments) + 1, start_sec=self._t(start_frame),
                         source_start_sample=start_frame * FRAME,
                         output_start_sample=int(getattr(self.session, 'output_samples', 0)))
        self.session.note('seg_gate', action='start', segment=self._cur['index'], at_sec=self._cur['start_sec'])
        self._send(np.concatenate(frames), start_frame * FRAME)

    def _end(self, reason):
        cur = self._cur
        cur.update(reason=reason, end_sec=self.processed_samples / SR,
                   source_end_sample=self.processed_samples,
                   dur_sec=(self.processed_samples - cur['source_start_sample']) / SR,
                   last_speech_sec=self._t(self.last_speech_frame + 1))
        self.session.note('seg_gate', action='end', segment=cur['index'], reason=reason,
                          dur_sec=cur['dur_sec'], at_sec=cur['end_sec'], last_speech_sec=cur['last_speech_sec'])
        self.segments.append(cur)
        self.active = False
        self.seg_frames = self.run_sil = 0
        self._cur = None
        self.session.feed_audio(np.zeros(0, np.float32), segment_end=True)
        cur['output_end_sample'] = int(getattr(self.session, 'output_samples', 0))
        cur['output_sec'] = (cur['output_end_sample'] - cur['output_start_sample']) / 24000


    def _send(self, samples, source_start_sample):
        if self.input_observer is not None:
            self.input_observer(int(source_start_sample), int(samples.size))
        self.session.feed_audio(samples)


class IntervalFeeder:
    """Temporary duration guard with no VAD, silence removal, or preroll.

    Every input sample is forwarded once. The guard is independent of the
    context-retention policy and is not a substitute for a semantic window.
    """
    def __init__(self, session, interval_sec, input_observer=None):
        if not math.isfinite(interval_sec) or interval_sec <= 0:
            raise ValueError('interval_sec must be finite and positive')
        self.session, self.input_observer = session, input_observer
        self.limit = int(interval_sec * SR)
        if self.limit < 400:
            raise ValueError('interval must contain at least 25ms')
        self.received_samples = self.processed_samples = 0
        self.segments = []
        self._cur = None
        self.buf = np.zeros(0, np.float32)

    def feed(self, wav16):
        chunk = np.asarray(wav16, dtype=np.float32)
        if chunk.ndim != 1 or not np.isfinite(chunk).all():
            raise ValueError('expected finite mono 16kHz audio')
        self.received_samples += chunk.size
        self.buf = np.concatenate([self.buf, chunk])
        while self.buf.size:
            if self._cur is None:
                self._cur = dict(index=len(self.segments) + 1,
                    start_sec=self.processed_samples / SR,
                    source_start_sample=self.processed_samples,
                    output_start_sample=int(self.session.output_samples))
                self.session.note('seg_gate', action='start', segment=self._cur['index'],
                                  at_sec=self._cur['start_sec'])
            n = min(self.buf.size, self.limit -
                    (self.processed_samples - self._cur['source_start_sample']))
            if n == 0:
                # Do not strand a <25ms final segment: wait for enough source
                # to start the next frontend, or merge this tiny tail at EOF.
                if self.buf.size < 400:
                    break
                self._end('hard')
                continue
            start = self.processed_samples
            self.processed_samples += n
            if self.input_observer is not None:
                self.input_observer(start, n)
            self.session.feed_audio(self.buf[:n])
            self.buf = self.buf[n:]

    def _end(self, reason):
        cur = self._cur
        cur.update(reason=reason, end_sec=self.processed_samples / SR,
            source_end_sample=self.processed_samples,
            dur_sec=(self.processed_samples - cur['source_start_sample']) / SR)
        self.session.note('seg_gate', action='end', segment=cur['index'], reason=reason,
                          dur_sec=cur['dur_sec'], at_sec=cur['end_sec'])
        self.session.feed_audio(np.zeros(0, np.float32), segment_end=True)
        cur['output_end_sample'] = int(self.session.output_samples)
        cur['output_sec'] = (cur['output_end_sample'] - cur['output_start_sample']) / 24000
        self.segments.append(cur)
        self._cur = None

    def close(self):
        if self.buf.size:
            start = self.processed_samples
            self.processed_samples += self.buf.size
            if self.input_observer is not None:
                self.input_observer(start, self.buf.size)
            self.session.feed_audio(self.buf)
            self.buf = np.zeros(0, np.float32)
        if self._cur is not None:
            duration = self.processed_samples - self._cur['source_start_sample']
            if duration < 400:
                raise ValueError('source audio must contain at least 25ms for STFT')
            self._end('hard' if duration >= self.limit else 'eof')
        return self.session.close()


class SegmentedSession:
    """CLI/Web adapter with independent original-source and model-input clocks.

    Core S5Session counters stay unchanged. Events expose observed original
    source position in source_sec, compressed model input in model_source_sec,
    and the corresponding original consumed position in model_source_origin_sec.
    """

    def __init__(self, session, vad=None, *, on_event=None, interval_sec=None, **segment_options):
        self.session = session
        self.started = session.started
        self.on_event = on_event if on_event is not None else session.on_event
        self.session.on_event = self._event
        self._maps = deque()
        self._batch = None
        self.pipeline_compute_sec = 0.0
        self.first_input_at = self.first_pcm_at = self.first_text_at = None
        self.acoustic_rollovers = 0
        self.events = [] if getattr(session, "collect_audio", False) else None
        self.steps = [] if self.events is not None else None
        self.closed = self.failed = False
        self.feeder = (IntervalFeeder(session, interval_sec, input_observer=self._register)
                      if interval_sec is not None else
                      SegmentedFeeder(session, vad if vad is not None else SileroFrameVad(),
                                      input_observer=self._register, **segment_options))

    def _register(self, source_start, count):
        model_start = int(self.session.source_samples)
        if self._maps and (self._maps[-1][0] + self._maps[-1][2] == model_start
                           and self._maps[-1][1] + self._maps[-1][2] == source_start):
            self._maps[-1][2] += count
        else:
            self._maps.append([model_start, source_start, count])

    def _origin(self, model_sample):
        while len(self._maps) > 1 and model_sample > self._maps[0][0] + self._maps[0][2]:
            self._maps.popleft()
        if not self._maps:
            return 0.0
        start, origin, count = self._maps[0]
        if model_sample < start or model_sample > start + count:
            raise RuntimeError('model source position is outside the original-source mapping')
        return (origin + model_sample - start) / SR

    def _event(self, event):
        data = dict(event)
        model_sec = float(event.get('source_sec', self.session.consumed_samples / SR))
        origin = self._origin(int(round(model_sec * SR)))
        data.update(model_source_sec=model_sec, model_source_origin_sec=origin,
                    source_sec=self.feeder.processed_samples / SR,
                    source_received_sec=self.feeder.received_samples / SR)
        if 'source_end_sec' in data:
            data['model_source_end_sec'] = data['source_end_sec']
            data['source_end_sec'] = origin
            data['source_end_ms'] = origin * 1000
        now = time.perf_counter()
        if data['type'] == 'pcm' and self.first_pcm_at is None:
            self.first_pcm_at = now
        if data['type'] == 'text' and self.first_text_at is None:
            self.first_text_at = now
        if data['type'] == 'segment_end' and data.get('reason') == 'external':
            decoder = getattr(self.session, 'decoder', None)
            if decoder is not None:
                self.acoustic_rollovers += decoder.session_rollovers
        if self.events is not None:
            self.events.append({key: value for key, value in data.items() if key != "pcm"})
            if data["type"] == "step":
                self.steps.append(dict(data))
        if self._batch is not None:
            self._batch.append(data)
        if self.on_event is not None:
            self.on_event(data)

    def feed_audio(self, wav16, finalize=False):
        if self.failed:
            raise RuntimeError('segmented session failed; emitted PCM cannot be retried')
        if self.closed:
            if finalize and np.asarray(wav16).size == 0:
                return []
            raise RuntimeError('segmented session is closed')
        started = time.perf_counter()
        self._batch = []
        try:
            if np.asarray(wav16).size and self.first_input_at is None:
                self.first_input_at = started
            self.feeder.feed(wav16)
            if finalize:
                self.feeder.close()
                self.closed = True
            return self._batch
        except Exception:
            self.failed = True
            raise
        finally:
            self.pipeline_compute_sec += time.perf_counter() - started
            self._batch = None

    def close(self):
        return self.feed_audio(np.zeros(0, np.float32), finalize=True)

    def metrics(self):
        data = dict(self.session.metrics())
        original_sec = self.feeder.received_samples / SR
        model_sec = data['source_sec']
        data.update(model_input_sec=model_sec, model_compute_sec=data.get('compute_sec'),
                    model_inference_rtf=data.get('inference_rtf'), compute_sec=self.pipeline_compute_sec,
                    source_sec=original_sec, source_observed_sec=self.feeder.processed_samples / SR,
                    pipeline_compute_sec=self.pipeline_compute_sec,
                    inference_rtf=self.pipeline_compute_sec / original_sec if original_sec else None,
                    n_segments=len(self.feeder.segments), vad_segmented=True,
                    acoustic_rollovers=self.acoustic_rollovers,
                    source_mapping_intervals=len(self._maps))
        for kind, at in [('pcm', self.first_pcm_at), ('text', self.first_text_at)]:
            data[f'first_{kind}_after_first_input_sec'] = (
                at - self.first_input_at if at is not None and self.first_input_at is not None else None)
        return data

    def result(self):
        data = self.session.result()
        data.update(self.metrics())
        data['segments'] = self.feeder.segments
        if self.events is not None:
            data['events'], data['steps'] = self.events, self.steps
        return data


def add_segmentation_args(parser, *, default='vad'):
    """Shared CLI/Web policy; thresholds are not retuned by the transport."""
    parser.add_argument('--segmentation', choices=('vad', 'interval', 'none'), default=default,
                        help='vad: pause boundaries; interval: duration guard without VAD; none: no source segmentation')
    parser.add_argument('--vad_threshold', type=float, default=0.5)
    for key, value in [('pause_end_sec', 1.2), ('soft_sec', 20.), ('soft_pause_sec', .3),
                       ('hard_sec', 30.), ('preroll_sec', .3), ('min_speech_sec', .25)]:
        parser.add_argument('--' + key, type=float, default=value)


def segmentation_options(args):
    if args.segmentation == 'interval':
        if not math.isfinite(args.hard_sec) or args.hard_sec <= 0:
            raise ValueError('hard_sec must be finite and positive')
        return dict(interval_sec=args.hard_sec)
    values = {key: getattr(args, key) for key in ('pause_end_sec', 'soft_sec', 'soft_pause_sec',
              'hard_sec', 'preroll_sec', 'min_speech_sec')}
    validate_segmentation_options(**values)
    return values
