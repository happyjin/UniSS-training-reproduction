#!/usr/bin/env python3
"""TTS evaluation metrics for t2s translation eval (seed-tts-eval aligned).

指标（简洁、无重复功能）：
* **BLEU**   — thinker 译文 vs GT 文本翻译质量（sacrebleu；zh 用 zh 分词）。
* **WER/CER**— TTS 生成准确率：Whisper ASR 转写 AR 音频，与译文比对
               （en → WER，zh → CER，jiwer 计算）。
* **SIM**    — 说话人音色相似度：WavLM-SV x-vector 余弦相似度
               （AR 音频 vs GT 音频，与 seed-tts-eval 同思路）。
* **UTMOS/MOS** — 自然度：torchaudio ``SQUIM_SUBJECTIVE`` 无参考 MOS 预测。

所有模型 **懒加载 + 单例**，只在首次用到时下载/载入，评测大批量样本时复用。
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------- #
# 文本规整
# --------------------------------------------------------------------------- #
def _norm_zh(text: str) -> str:
    """繁→简 + 去标点空格，用于 CER。"""
    try:
        import zhconv
        text = zhconv.convert(text, "zh-cn")
    except Exception:
        pass
    text = re.sub(r"[\s，。！？、；：""''（）《》…—·.,!?;:\"'()\[\]]", "", text)
    return text.strip()


def _norm_en(text: str) -> str:
    """小写 + 去标点，用于 WER。"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# BLEU（翻译质量）
# --------------------------------------------------------------------------- #
def compute_bleu(hyp: str, ref: str, target_lang: str):
    """句级 BLEU；中文用 zh tokenizer，其它用 13a。"""
    import sacrebleu
    if not hyp or not ref:
        return None
    tok = "zh" if str(target_lang).lower().startswith("zh") else "13a"
    try:
        return round(sacrebleu.sentence_bleu(hyp, [ref], tokenize=tok).score, 2)
    except Exception:
        return None


def corpus_bleu(hyps, refs, target_lang: str):
    import sacrebleu
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return None
    hs = [h for h, _ in pairs]
    rs = [[r for _, r in pairs]]
    tok = "zh" if str(target_lang).lower().startswith("zh") else "13a"
    try:
        return round(sacrebleu.corpus_bleu(hs, rs, tokenize=tok).score, 2)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# ASR (Whisper) → WER/CER（TTS 生成准确率）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_asr(model_id: str, device: str):
    from transformers import pipeline
    dev = 0 if device.startswith("cuda") else -1
    if device.startswith("cuda") and ":" in device:
        dev = int(device.split(":")[1])
    return pipeline("automatic-speech-recognition", model=model_id, device=dev,
                    chunk_length_s=30)


def asr_transcribe(wav: np.ndarray, sr: int, lang: str, model_id: str, device: str):
    """Whisper 转写。lang: 'zh'/'en' 指定解码语种。"""
    asr = _get_asr(model_id, device)
    lang_code = "chinese" if str(lang).lower().startswith("zh") else "english"
    out = asr({"array": np.asarray(wav, dtype=np.float32), "sampling_rate": sr},
              generate_kwargs={"language": lang_code, "task": "transcribe"})
    return (out.get("text") or "").strip()


def compute_wer_cer(asr_text: str, ref_text: str, target_lang: str):
    """en → WER，zh → CER（用 jiwer）。返回 (metric_name, value)。"""
    import jiwer
    if not ref_text:
        return (None, None)
    if str(target_lang).lower().startswith("zh"):
        h, r = _norm_zh(asr_text), _norm_zh(ref_text)
        if not r:
            return ("CER", None)
        val = jiwer.cer(r, h)
        return ("CER", round(float(val), 4))
    else:
        h, r = _norm_en(asr_text), _norm_en(ref_text)
        if not r:
            return ("WER", None)
        val = jiwer.wer(r, h)
        return ("WER", round(float(val), 4))


# --------------------------------------------------------------------------- #
# SIM（说话人相似度，WavLM-SV，seed-tts-eval 思路）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_sv(device: str):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    return fe, model


def _resample(wav: np.ndarray, sr: int, target_sr: int = 16000):
    if sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    import librosa
    return librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=target_sr)


def compute_sim(gen_wav: np.ndarray, gen_sr: int, ref_wav: np.ndarray, ref_sr: int, device: str):
    """AR 音频与 GT 音频的说话人 x-vector 余弦相似度 ∈ [-1,1]。"""
    import torch
    fe, model = _get_sv(device)
    a = _resample(gen_wav, gen_sr, 16000)
    b = _resample(ref_wav, ref_sr, 16000)
    with torch.no_grad():
        inp = fe([a, b], sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        emb = model(**inp).embeddings
        emb = torch.nn.functional.normalize(emb, dim=-1)
        sim = torch.matmul(emb[0], emb[1]).item()
    return round(float(sim), 4)


# --------------------------------------------------------------------------- #
# MOS（自然度，torchaudio SQUIM_SUBJECTIVE，无参考）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_squim(device: str):
    import torch
    from torchaudio.pipelines import SQUIM_SUBJECTIVE
    model = SQUIM_SUBJECTIVE.get_model().to(device).eval()
    return model


def compute_mos(gen_wav: np.ndarray, gen_sr: int, ref_wav: np.ndarray, ref_sr: int, device: str):
    """SQUIM 主观 MOS（1-5）。SQUIM_SUBJECTIVE 需要一个非匹配参考(NMR)，用 GT 音频。"""
    import torch
    model = _get_squim(device)
    a = _resample(gen_wav, gen_sr, 16000)
    b = _resample(ref_wav, ref_sr, 16000)
    with torch.no_grad():
        ta = torch.from_numpy(a).float().unsqueeze(0).to(device)
        tb = torch.from_numpy(b).float().unsqueeze(0).to(device)
        # 长度对齐（NMR 只需任意干净参考）
        n = min(ta.shape[-1], tb.shape[-1])
        mos = model(ta[:, :n], tb[:, :n])
    return round(float(mos.item()), 3)


# --------------------------------------------------------------------------- #
# 聚合工具
# --------------------------------------------------------------------------- #
def safe_mean(vals):
    xs = [v for v in vals if v is not None]
    return round(float(np.mean(xs)), 4) if xs else None
