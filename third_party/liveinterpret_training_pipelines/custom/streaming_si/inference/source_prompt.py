"""源语音 → flow 声学 prompt（CosyVoice3 speech token + 24kHz mel），推理时用，不需要重训。

背景（`docs/paralinguistic_preservation.md` 第 7 节）：只给 flow 一个 CAM++ 说话人向量时，带到输出里的
只有音色；SimulS2ST-Omni 每次合成都把最近的源语音当声学前缀交给 VoiceBox 续写，情感、发声方式、
能量、录音质感都会跟过去。CosyVoice flow 本身就支持同样的用法（官方 zero-shot / cross-lingual
克隆：`prompt_token` + `prompt_feat`，flow 只输出 prompt 之后的 mel），这里把源语音编成这对 prompt。

提取方式与官方 `cosyvoice/cli/frontend.py` 相同：
  * speech token：16kHz 波形 → whisper 128 维 log-mel → `speech_tokenizer_v3.onnx`（25Hz）；
  * mel：24kHz 波形 → `matcha.utils.audio.mel_spectrogram`（参数取自 cosyvoice3.yaml 的 feat_extractor）；
  * 两者按 `token_mel_ratio=2` 截齐（`frontend_zero_shot` 里 `token_len = min(feat_len // 2, token_len)`）。
容器里没有 openai-whisper，log-mel 按 whisper 的公式自己算（mel 滤波器 = librosa slaney，与 whisper 的
`mel_filters.npz` 同源）。

用法：
    sp = SourcePrompt(tokenizer_onnx)
    tokens, feat = sp(wav16, max_sec=10, position='head')
    wav = model.decode_speech(speech_tokens, spk_emb=emb, prompt_speech_tokens=tokens, prompt_feat=feat)
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch

SR16, SR24 = 16000, 24000
TOKEN_MEL_RATIO = 2
TOKENIZER_NAME = 'speech_tokenizer_v3.onnx'
# cosyvoice3.yaml: feat_extractor（与 mel_spec_transform1 相同）
FEAT_KW = dict(n_fft=1920, num_mels=80, sampling_rate=SR24, hop_size=480, win_size=1920,
               fmin=0, fmax=None, center=False)


def find_tokenizer(decoder_dir: Optional[str] = None, explicit: Optional[str] = None) -> Optional[str]:
    """`--prompt_tokenizer` > `<speech_decoder>/speech_tokenizer_v3.onnx` > 环境变量 COSYVOICE_SPEECH_TOKENIZER。"""
    for p in (explicit, os.path.join(decoder_dir, TOKENIZER_NAME) if decoder_dir else None,
              os.environ.get('COSYVOICE_SPEECH_TOKENIZER')):
        if p and os.path.isfile(p):
            return p
    return None


class SourcePrompt:
    def __init__(self, tokenizer_onnx: str, threads: int = 4):
        import librosa
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(tokenizer_onnx, sess_options=so, providers=['CPUExecutionProvider'])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.whisper_filters = torch.from_numpy(
            librosa.filters.mel(sr=SR16, n_fft=400, n_mels=128)).float()          # (128, 201)
        self._mel24 = None

    # ---------------------------------------------------------------- token
    def _whisper_log_mel(self, wav16: np.ndarray) -> torch.Tensor:
        """whisper.log_mel_spectrogram(audio, n_mels=128) 的等价实现 -> (1, 128, T)。"""
        audio = torch.from_numpy(np.ascontiguousarray(wav16, dtype=np.float32))
        stft = torch.stft(audio, 400, 160, window=torch.hann_window(400), return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self.whisper_filters @ mag, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return ((log_spec + 4.0) / 4.0).unsqueeze(0)

    def tokens(self, wav16: np.ndarray) -> np.ndarray:
        if wav16.size / SR16 > 30:
            raise ValueError('speech tokenizer 只支持 <= 30s 的音频')
        feat = self._whisper_log_mel(wav16)
        out = self.sess.run(None, {self.inputs[0]: feat.numpy(),
                                   self.inputs[1]: np.array([feat.shape[2]], dtype=np.int32)})[0]
        return np.asarray(out, dtype=np.int64).reshape(-1)

    # ---------------------------------------------------------------- mel
    def feat(self, wav24: np.ndarray) -> torch.Tensor:
        """24kHz 波形 -> 80 维 mel (1, T, 80)。需要 CosyVoice 仓库的 third_party/Matcha-TTS 已在 sys.path
        （`attach_speech_decoder` 会挂上）。"""
        if self._mel24 is None:
            from matcha.utils.audio import mel_spectrogram
            self._mel24 = mel_spectrogram
        wav = torch.from_numpy(np.ascontiguousarray(wav24, dtype=np.float32)).unsqueeze(0)
        return self._mel24(wav, **FEAT_KW).squeeze(0).transpose(0, 1).unsqueeze(0).float()

    # ---------------------------------------------------------------- 组合
    def __call__(self, wav16: np.ndarray, max_sec: float = 10.0, position: str = 'head',
                 wav24: Optional[np.ndarray] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """取源语音的开头（head）或结尾（tail）最多 max_sec 秒，返回成对的 (tokens, feat)。

        `wav24`：同一段源语音的 24kHz 版本（从原文件直接重采样，保留 8kHz 以上频段）；
        缺省时由 16kHz 上采样（高频为空，flow 会把输出也做「闷」）。
        """
        import torchaudio
        if wav24 is None:
            wav24 = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(wav16, dtype=np.float32)), SR16, SR24).numpy()
        sec = min(float(max_sec), wav16.size / SR16) if max_sec and max_sec > 0 else wav16.size / SR16
        n16, n24 = int(sec * SR16), int(sec * SR24)
        c16, c24 = (wav16[:n16], wav24[:n24]) if position == 'head' else (wav16[-n16:], wav24[-n24:])
        tok, feat = self.tokens(c16), self.feat(c24)
        m = min(feat.size(1) // TOKEN_MEL_RATIO, tok.size)
        return tok[:m], feat[:, :m * TOKEN_MEL_RATIO]


def load_wav24(path: str, max_sec: Optional[float] = None) -> np.ndarray:
    """原文件 -> 24kHz 单声道（给 prompt mel 用）。"""
    import soundfile as sf
    import torchaudio
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != SR24:
        wav = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, SR24).numpy()
    if max_sec is not None:
        wav = wav[:int(max_sec * SR24)]
    return np.ascontiguousarray(wav, dtype=np.float32)
