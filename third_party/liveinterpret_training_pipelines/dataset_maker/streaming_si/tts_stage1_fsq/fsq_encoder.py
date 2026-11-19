"""Fun-CosyVoice3 v3 FSQ 编码：16k 波形 -> 25Hz speech token。

mel 前端必须与官方 `whisper.log_mel_spectrogram` 逐步一致（log10 + 逐样本动态范围
裁剪 + 归一化）。历史教训（源项目 2026-08-04）：用 torchaudio 的自然对数 mel 时
loss/acc 完全正常（val CE 3.14 / acc@1 22%），但码字语义错乱——解码后 ASR 只能
转写出 "."。故这里不复用任何"看起来也行"的 mel 实现。

本文件是 `streaming-si-0.6b/data_prep/_fsq_mel.py` + `_common.build_fsq_session`
的等价实现，改为可独立拷走（不依赖源仓库目录结构）。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 128

_FILT = {}
_WIN = {}


def _filters(n_mels, device, dtype):
    key = (n_mels, str(device), dtype)
    if key not in _FILT:
        import librosa
        f = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=n_mels)
        _FILT[key] = torch.from_numpy(f).to(device=device, dtype=dtype)
    return _FILT[key]


def _window(device, dtype):
    key = (str(device), dtype)
    if key not in _WIN:
        _WIN[key] = torch.hann_window(N_FFT, device=device, dtype=dtype)
    return _WIN[key]


def log_mel_spectrogram(wav, n_mels=N_MELS, lengths=None):
    """(B, T) 或 (T,) 16k 波形 -> (B, n_mels, T_frame) log-mel。

    `lengths` 必须传每条的真实采样点数：`max(x, x.max()-8)` 的 max 是逐样本的，
    padding 参与会与单条推理不一致。
    """
    single = wav.dim() == 1
    if single:
        wav = wav.unsqueeze(0)
    dtype = torch.float32 if wav.dtype not in (torch.float32, torch.float64) else wav.dtype
    wav = wav.to(dtype)
    stft = torch.stft(wav, N_FFT, HOP, window=_window(wav.device, dtype), return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = _filters(n_mels, wav.device, dtype) @ magnitudes
    log_spec = mel_spec.clamp(min=1e-10).log10()

    if lengths is None:
        peak = log_spec.amax(dim=(1, 2), keepdim=True)
    else:
        n_frame = log_spec.size(-1)
        idx = torch.arange(n_frame, device=log_spec.device)
        valid = idx.unsqueeze(0) < (torch.as_tensor(lengths, device=log_spec.device).view(-1, 1) // HOP)
        masked = log_spec.masked_fill(~valid.unsqueeze(1), -float("inf"))
        peak = masked.amax(dim=(1, 2), keepdim=True)
        peak = torch.where(torch.isfinite(peak), peak, log_spec.amax(dim=(1, 2), keepdim=True))
    log_spec = torch.maximum(log_spec, peak - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.squeeze(0) if single else log_spec


def n_tokens_for(n_sample: int) -> int:
    """采样点数 -> token 数：mel 100Hz，再 4:1 降到 25Hz。"""
    return max(1, (n_sample // HOP) // 4)


def wavs_to_fsq_tokens(sess, wavs: Sequence, device: str = "cpu", n_mels: int = N_MELS) -> List[List[int]]:
    """一批 16k 波形 -> list[list[int]]（1-D 单码本，码本 6561=3^8，25Hz）。"""
    lens = [int(len(w)) for w in wavs]
    max_len = max(lens)
    batch = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, w in enumerate(wavs):
        w = w.detach().float().cpu().numpy() if torch.is_tensor(w) else np.asarray(w)
        batch[i, :len(w)] = w
    t = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        m = log_mel_spectrogram(t, n_mels=n_mels, lengths=lens)
    mnp = m.cpu().numpy().astype(np.float32)
    del t, m
    if str(device).startswith("cuda") and os.environ.get("FSQ_GPU_MEM_GB"):
        # 多进程共卡时把 torch 缓存还给驱动，避免与 onnx 显存池互相挤占
        torch.cuda.empty_cache()
    frame_lens = np.array([min(l // HOP, mnp.shape[-1]) for l in lens], dtype=np.int32)
    idx = sess.run(None, {"feats": mnp, "feats_length": frame_lens})[0]
    out = []
    for i, l in enumerate(lens):
        n_tok = min(n_tokens_for(l), idx.shape[1])
        out.append([int(x) for x in idx[i, :n_tok]])
    return out


def build_session(tokenizer_path: str, device: str, require_cuda: bool = True):
    """onnxruntime session，绑到 `device` 对应的 GPU。

    device_id 必须从 device 串解析：曾硬编码 0 导致多卡进程全挤在 GPU0 上 OOM。
    """
    import onnxruntime as ort
    if not os.path.exists(tokenizer_path):
        raise RuntimeError(f"speech tokenizer 不存在: {tokenizer_path}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    dev = str(device)
    dev_id = (int(dev.split(":")[1]) if dev.startswith("cuda") and ":" in dev
              else (0 if dev.startswith("cuda") else None))
    cuda_opts = {"device_id": dev_id}
    mem_gb = os.environ.get("FSQ_GPU_MEM_GB")
    if mem_gb:
        # 多进程共卡：默认 BFC 显存池按 2 的幂扩张且从不归还，5 进程/卡时会把 141GB 顶满。
        # 设上限并改为按需扩张；超限时 run 抛 Failed to allocate，由调用方拆批重试。
        cuda_opts.update(gpu_mem_limit=int(float(mem_gb) * (1 << 30)),
                         arena_extend_strategy="kSameAsRequested")
    providers = ([("CUDAExecutionProvider", cuda_opts)] if dev_id is not None else []) \
        + ["CPUExecutionProvider"]
    sess = ort.InferenceSession(tokenizer_path, sess_options=so, providers=providers)
    if require_cuda and dev_id is not None and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(
            "onnx 未启用 CUDA EP（会退化到 CPU，极慢）。集群镜像常只装了 onnxruntime："
            "pip install onnxruntime-gpu==$(python -c 'import onnxruntime;print(onnxruntime.__version__)')")
    return sess


def _is_oom(exc) -> bool:
    return any(k in str(exc) for k in ("Failed to allocate", "out of memory", "CUDA out of memory"))


def run_with_oom_split(fn, chunk, log=print, _depth=0) -> int:
    """`fn(chunk)` 显存不足时拆半重试，返回丢弃条数。

    长音频组下 tokenizer 的 attention 中间张量可达 ~8.6GB；整批被 except 吞掉会造成
    数据静默缺失（断点续跑按"输出存在"判断，重启也不补）。
    """
    try:
        fn(chunk)
        return 0
    except Exception as exc:
        if not _is_oom(exc) or len(chunk) == 1:
            log(f"    [drop] {len(chunk)} 条: {type(exc).__name__} {str(exc)[:120]}")
            return len(chunk)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _depth == 0:
            log(f"    [oom] 批 {len(chunk)} -> 拆半重试")
        mid = len(chunk) // 2
        return (run_with_oom_split(fn, chunk[:mid], log, _depth + 1)
                + run_with_oom_split(fn, chunk[mid:], log, _depth + 1))
