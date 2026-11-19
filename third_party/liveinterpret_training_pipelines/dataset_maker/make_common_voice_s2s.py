#!/usr/bin/env python3
"""Convert Common Voice scripted-speech S2S (en↔zh) into the project's
precomputed-codec parquet format for large-scale training.

数据源
------
* curated 元数据：``curated/part-*.parquet``（支持 en→zh / zh→en），字段含
  ``source_transcript`` / ``target_text`` / ``target_audio_utmos_score`` /
  ``target_audio_ecapa_similarity`` 以及音频定位
  ``{source,target}_audio_shard_uri`` + ``_byte_offset`` + ``_byte_length``。
* 目标音频 shards：``shards/`` 或 ``target_shards/``（TTS 合成译文语音）。
* 源音频 shards：``source_shards/``（Common Voice 源语音）。
  音频用 (offset, length) 从 tar 里 seek 读取，无需解压整个 tar。

产物（真 S2S + 离线预编码）
--------------------------
``qwen3_tts_codec/speech_to_speech_translation.parquet``，每行：
* ``messages``          : system=按 target_lang 生成的翻译指令，assistant=译文
* ``audios``            : [源 wav 路径] → thinker 多模态输入 + S2S x-vector 源
* ``target_text``       : 目标语言译文
* ``target_codec_codes``: 目标语音的 Qwen3-TTS codec ``[T,16]``（离线编码）
* ``codec_len``         : 目标 codec 帧数（供 group_by_length 零成本分桶）
* ``source_codec_codes``: 源英文语音 codec ``[T,16]``（**ICL 用**，可选）
* ``source_text_ids``   : 源英文文本 token（**ICL 用**，thinker tokenizer）
* ``source_lang``/``target_lang``/``id`` 等

源音频抽成独立 wav（``source_wavs/<id>.wav``），parquet 只存路径。

质量过滤：``UTMOS >= --min-utmos`` 且 ``ECAPA >= --min-ecapa``（默认 2.5 / 0.4）。

并行大 batch 预编码：``--rank/--world-size`` 做数据分片，多进程各占一卡，
每张卡内用帧预算打包 batch 调用 ``qwen.encode``。
"""
from __future__ import annotations

import argparse
import io
import re
import sys

# --- Suppress qwen_tts waveform-range WARNING prints (harmless, just noisy) ---
class _WaveformWarningFilter:
    _pattern = re.compile(r"\[WARNING\] (Min|Max) value of input waveform signal")
    def __init__(self, stream):
        self._stream = stream
    def write(self, buf):
        if self._pattern.search(buf):
            return
        self._stream.write(buf)
    def flush(self):
        self._stream.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)
sys.stdout = _WaveformWarningFilter(sys.stdout)
import json
import os
import tarfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from task_prompts import s2s_task_prompt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK = "speech_to_speech_translation"

# 输出 schema：large_list 让 codec 列用 int64 offset，可用大 row group 不溢出。
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("task", pa.string()),
    pa.field("task_prompt", pa.string()),
    pa.field("messages", pa.list_(pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ]))),
    pa.field("audios", pa.list_(pa.string())),
    pa.field("target_text", pa.string()),
    pa.field("target_codec_codes", pa.large_list(pa.large_list(pa.int16()))),
    pa.field("codec_len", pa.int32()),
    pa.field("source_codec_codes", pa.large_list(pa.large_list(pa.int16()))),
    pa.field("source_codec_len", pa.int32()),
    pa.field("source_text_ids", pa.large_list(pa.int32())),
    pa.field("source_xvector", pa.large_list(pa.float32())),
    pa.field("source_lang", pa.string()),
    pa.field("target_lang", pa.string()),
    pa.field("source_transcript", pa.string()),
])


# --------------------------------------------------------------------------- #
# tar 内音频按 (offset, length) seek 读取
# --------------------------------------------------------------------------- #
def read_audio_from_tar(tar_path: str, offset: int, length: int):
    """Seek into a WebDataset tar and decode the embedded wav."""
    with open(tar_path, "rb") as fh:
        fh.seek(int(offset))
        raw = fh.read(int(length))
    wav, sr = sf.read(io.BytesIO(raw))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def local_tar(shard_root: str, shard_uri: str):
    return os.path.join(shard_root, os.path.basename(shard_uri))


# --------------------------------------------------------------------------- #
# 质量过滤 + 分片迭代 curated
# --------------------------------------------------------------------------- #
def iter_curated_rows(curated_dir: str, min_utmos: float, min_ecapa: float,
                      rank: int, world_size: int, limit: int = 0):
    """Yield curated rows passing the quality filter, sharded by rank."""
    import glob
    files = sorted(glob.glob(os.path.join(curated_dir, "part-*.parquet")))
    cols = ["id", "source_transcript", "target_text", "src_lang", "tgt_lang",
            "source_audio_shard_uri", "source_audio_byte_offset", "source_audio_byte_length",
            "target_audio_shard_uri", "target_audio_byte_offset", "target_audio_byte_length",
            "target_audio_utmos_score", "target_audio_ecapa_similarity"]
    seen = 0
    yielded = 0
    for f in files:
        t = pq.read_table(f, columns=cols)
        for row in t.to_pylist():
            u = row.get("target_audio_utmos_score")
            e = row.get("target_audio_ecapa_similarity")
            if u is None or e is None or u < min_utmos or e < min_ecapa:
                continue
            if (seen % world_size) == rank:
                yield row
                yielded += 1
                if limit and yielded >= limit:
                    return
            seen += 1


# --------------------------------------------------------------------------- #
# 帧预算打包（大 batch 编码，控制显存）
# --------------------------------------------------------------------------- #
def batch_by_frame_budget(items, max_rows: int, max_total_seconds: float):
    """Greedily group items so a batch stays under a duration budget."""
    batch, acc = [], 0.0
    for it in items:
        dur = it["_dur"]
        if batch and (len(batch) >= max_rows or acc + dur > max_total_seconds):
            yield batch
            batch, acc = [], 0.0
        batch.append(it)
        acc += dur
    if batch:
        yield batch


def build_messages(target_text: str, target_lang: str):
    return [
        {"role": "system", "content": s2s_task_prompt(target_lang)},
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": target_text},
    ]


def main():
    ap = argparse.ArgumentParser()
    base = "/opt/dlami/nvme/zeyangsong/data/common_voice_scripted_speech_en_zh_v2"
    ap.add_argument("--curated-dir", default=os.path.join(base, "curated"))
    ap.add_argument("--target-shards", default=os.path.join(base, "shards"))
    ap.add_argument("--source-shards", default=os.path.join(base, "source_shards"))
    ap.add_argument("--out-root", default=os.path.join(base, "converted"))
    ap.add_argument("--source-wav-dir", default=os.path.join(base, "source_wavs"))
    ap.add_argument("--qwen-tokenizer",
                    default=os.path.join(REPO, "models", "Qwen3-TTS-Tokenizer-12Hz"))
    ap.add_argument("--text-tokenizer",
                    default=os.path.join(REPO, "models", "qwen2_5_3b_omnispeak_a"),
                    help="thinker tokenizer for ICL source_text_ids")
    ap.add_argument("--min-utmos", type=float, default=2.5)
    ap.add_argument("--min-ecapa", type=float, default=0.4)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-batch-rows", type=int, default=48)
    ap.add_argument("--max-batch-seconds", type=float, default=240.0)
    ap.add_argument("--row-group-size", type=int, default=2000)
    ap.add_argument("--flush-every", type=int, default=2000,
                    help="write a parquet chunk every N rows")
    ap.add_argument("--no-icl", action="store_true",
                    help="skip source codec + source_text_ids (no ICL support in output)")
    ap.add_argument("--no-xvector", action="store_true",
                    help="skip source_xvector pre-computation")
    ap.add_argument("--min-duration", type=float, default=0.5,
                    help="skip clips whose source OR target audio is shorter than "
                         "this many seconds (no training value + crashes codec/ECAPA "
                         "conv layers). Default 0.5s.")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows per rank")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.source_wav_dir, exist_ok=True)
    out_path = os.path.join(args.out_root, f"{TASK}.rank{args.rank}.parquet")

    # 载入 Qwen3-TTS codec tokenizer（24kHz encode）
    from qwen_tts import Qwen3TTSTokenizer
    print(f"[rank{args.rank}] loading Qwen3-TTS tokenizer on {args.device} ...", flush=True)
    qwen = Qwen3TTSTokenizer.from_pretrained(args.qwen_tokenizer, device_map=args.device)
    # thinker tokenizer for source_text_ids (ICL)
    text_tok = None
    if not args.no_icl:
        from transformers import AutoTokenizer
        text_tok = AutoTokenizer.from_pretrained(
            args.text_tokenizer, trust_remote_code=True)

    # Speaker encoder (ECAPA x-vector) for source_xvector pre-computation
    spk_embedder = None
    if not args.no_xvector:
        from speaker_encoder import get_speaker_embedder
        spk_embedder = get_speaker_embedder(enc_dim=2048)
        print(f"[rank{args.rank}] speaker encoder loaded on {args.device}", flush=True)

    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    buf = []
    n_ok = n_err = n_seen = 0
    t0 = time.time()

    def flush(rows):
        if not rows:
            return
        tbl = pa.Table.from_pylist(rows, schema=SCHEMA)
        writer.write_table(tbl, row_group_size=args.row_group_size)

    # 预取一批 curated 行，读音频、算时长，供帧预算打包
    def prepared_items():
        for row in iter_curated_rows(args.curated_dir, args.min_utmos, args.min_ecapa,
                                     args.rank, args.world_size, args.limit):
            try:
                # 目标音频（编码用）
                tgt_tar = local_tar(args.target_shards, row["target_audio_shard_uri"])
                if not os.path.exists(tgt_tar):
                    continue
                tgt_wav, tgt_sr = read_audio_from_tar(
                    tgt_tar, row["target_audio_byte_offset"], row["target_audio_byte_length"])
                # 源音频（抽 wav 存路径 + ICL 编码用）
                src_tar = local_tar(args.source_shards, row["source_audio_shard_uri"])
                if not os.path.exists(src_tar):
                    continue
                src_wav, src_sr = read_audio_from_tar(
                    src_tar, row["source_audio_byte_offset"], row["source_audio_byte_length"])
                # Skip ultra-short clips: no training value + they crash the
                # codec/ECAPA conv layers (need a minimum receptive field).
                src_dur = len(src_wav) / max(src_sr, 1)
                tgt_dur = len(tgt_wav) / max(tgt_sr, 1)
                if src_dur < args.min_duration or tgt_dur < args.min_duration:
                    continue
                yield {"row": row, "tgt_wav": tgt_wav, "tgt_sr": tgt_sr,
                       "src_wav": src_wav, "src_sr": src_sr,
                       "_dur": tgt_dur}
            except Exception:
                continue

    for batch in batch_by_frame_budget(prepared_items(), args.max_batch_rows, args.max_batch_seconds):
        n_seen += len(batch)
        # 1) 抽取源 wav 存独立文件（16k）
        for it in batch:
            sid = _safe_id(it["row"]["id"])
            wav_path = os.path.join(args.source_wav_dir, f"{sid}.wav")
            if not os.path.exists(wav_path):
                sf.write(wav_path, it["src_wav"], it["src_sr"])
            it["_src_wav_path"] = wav_path

        # 2) 批量编码目标音频 → codec [T,16]（写临时 wav 交给 qwen.encode）
        try:
            tgt_codes = _encode_batch(qwen, [it["tgt_wav"] for it in batch],
                                      [it["tgt_sr"] for it in batch], args)
            src_codes = None
            if not args.no_icl:
                src_codes = _encode_batch(qwen, [it["src_wav"] for it in batch],
                                          [it["src_sr"] for it in batch], args)
        except Exception as ex:
            n_err += len(batch)
            print(f"[rank{args.rank}] encode batch failed: {ex}", flush=True)
            continue

        # 3) 组装行
        for i, it in enumerate(batch):
            row = it["row"]
            tc = tgt_codes[i]                       # [T,16] int
            # source x-vector: guard against extraction failures (e.g. audio
            # too short for the ECAPA conv/pad). On failure store None so the
            # model falls back to per-step extraction at train time.
            xvec = None
            if spk_embedder is not None:
                try:
                    xvec = spk_embedder.extract(
                        it["src_wav"], it["src_sr"],
                        device=args.device, dtype=torch.float32,
                    ).detach().cpu().numpy().astype(np.float32).tolist()
                except Exception as ex:
                    print(f"[rank{args.rank}] xvector failed for {row['id']}: {ex}",
                          flush=True)
                    xvec = None
            rec = {
                # ``messages[0]`` is the single canonical prompt location.
                # Keep the legacy column empty so the dataset loader cannot
                # append the same prompt a second time.
                "id": row["id"], "task": TASK, "task_prompt": "",
                "messages": build_messages(
                    row.get("target_text") or "", row.get("tgt_lang")),
                "audios": [it["_src_wav_path"]],
                "target_text": row.get("target_text") or "",
                "target_codec_codes": tc.astype(np.int16).tolist(),
                "codec_len": int(tc.shape[0]),
                "source_codec_codes": (src_codes[i].astype(np.int16).tolist()
                                       if src_codes is not None else None),
                "source_codec_len": (int(src_codes[i].shape[0]) if src_codes is not None else None),
                "source_text_ids": (list(map(int, text_tok.encode(
                    row.get("source_transcript") or "", add_special_tokens=False)))
                    if text_tok is not None else None),
                "source_xvector": xvec,
                "source_lang": row.get("src_lang"), "target_lang": row.get("tgt_lang"),
                "source_transcript": row.get("source_transcript") or "",
            }
            buf.append(rec)
            n_ok += 1

        if len(buf) >= args.flush_every:
            flush(buf); buf = []
            el = time.time() - t0
            print(f"[rank{args.rank}] ok={n_ok} err={n_err} seen={n_seen} "
                  f"({n_ok/max(el,1):.1f} rows/s)", flush=True)

    flush(buf)
    writer.close()
    print(f"[rank{args.rank}] DONE ok={n_ok} err={n_err} -> {out_path} "
          f"({time.time()-t0:.0f}s)", flush=True)


def _safe_id(raw_id: str):
    return str(raw_id).replace(":", "_").replace("/", "_")


def _encode_batch(qwen, wavs, srs, args):
    """Encode a batch of waveforms → list of [T,16] int arrays.

    Qwen3TTSTokenizer.encode 接受文件路径列表；这里把内存波形先写临时 wav
    再批量编码（与 convert_minimind_mimi_to_qwen3 的做法一致，保证 24kHz）。
    """
    import tempfile
    tmp_paths = []
    try:
        for w, sr in zip(wavs, srs):
            fd, p = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(p, w, sr)
            tmp_paths.append(p)
        encoded = qwen.encode(tmp_paths)
        out = []
        for codes in encoded.audio_codes:            # codes: [16, T] (可能是 cuda tensor)
            if hasattr(codes, "detach"):
                codes = codes.detach().cpu()
            arr = np.asarray(codes)
            if arr.ndim == 2 and arr.shape[0] == 16:
                arr = arr.T                          # → [T,16]
            out.append(arr.astype(np.int64))
        return out
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
