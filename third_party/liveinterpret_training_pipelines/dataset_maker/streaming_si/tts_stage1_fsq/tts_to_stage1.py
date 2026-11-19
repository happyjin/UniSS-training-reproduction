#!/usr/bin/env python3
"""TTS 语料 -> streaming_si stage1（S1）训练数据：Fun-CosyVoice3 v3 FSQ parquet。

stage1 = S1 = Talker 发声预训练（文本 -> 25Hz FSQ speech token）。落盘格式必须是
parquet（swift 按扩展名选 reader），必需列只有两列：

    conversations          JSON 字符串 [{"role":"user",...},{"role":"assistant",...}]
    answer_speech_tokens   list<list<int>>，单轮时外层长度 1

其余列（id/text/language/speaker/duration/dnsmos/source）训练侧不读，但保留便于溯源与
后续过滤。`lengths` 列只有 --text-tokenizer 给定时才写（供 --group_by_length 分桶），
**同一批数据必须统一开关**，否则 schema 跨文件不一致、datasets 无法合并。

=========================== 两个数据源 ===========================
1) Curion curated Lance 表（ja / ko，音频在 Blob v2 列里）
     s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/data/audio/curated/task=tts/data.lance/
2) 原始下载件（s3://stevehguo-ap-southeast-3/.../data_fortrain_1031version/TTS/）
     jp  yodas3            tar: {id}.json + {id}.webm
     zh  emilia            tar: {id}.json + {id}.mp3
     zh  wenetspeech4tts   tar.gz: wavs/*.wav + txts/*.txt
     zh/en iscslp-cot-tts  tar: {segment_id}.flac + movie-xx/movie-xx.json（文本）
     ko  patterned-utter.  parquet: audio(bytes) + sentence
     en  hifitts-2         manifest_22khz.json(text) + LibriVox 章节 mp3（按 offset 切片）

=========================== 用法 ===========================
单机多卡（NPROC/卡 x NGPU 卡）：
    bash run_tts_stage1.sh 3 8            # 24 个进程，见同目录 sh
单进程调试：
    python3 tts_to_stage1.py --source emilia --lang zh \
        --s3-in  speech_model_backup/.../TTS \
        --s3-out speech_model_backup/.../TTS_FSQ \
        --fsq-tokenizer /path/speech_tokenizer_v3.batch.onnx \
        --device cuda:0 --shard 0 --nshard 24 --limit-units 1

集群多机多卡：见 k8s_stage1_fsq_job.yaml（Indexed Job，每 Pod 一个 --shard）。

生成后必须验收（loss/acc 无法证明数据正确）：
    python3 data_prep/verify_fsq_data.py --kind s1 --data <dir> --device cuda:0
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from typing import Dict, Iterator, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsq_encoder  # noqa: E402

SR = 16000
MAX_SPEECH_TOKENS = 750       # 25Hz -> 30s，与 talker 的 max_speech_len 一致
MIN_SPEECH_TOKENS = 5
MAX_DURATION = MAX_SPEECH_TOKENS / 25.0

DEFAULT_TTS_BUCKET = "stevehguo-ap-southeast-3"
DEFAULT_OUT_BUCKET = "stevehguo-ap-southeast-3"
DEFAULT_LANCE_URI = ("s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/"
                     "data/audio/curated/task=tts/data.lance/")
DEFAULT_FSQ_TOKENIZER_S3 = ("speech_model_backup/stevehguo/model_checkpoint/"
                            "cosyvoice3/speech_tokenizer_v3.batch.onnx")

PROMPTS = {
    "zh": ["请用自然的语气朗读下面这段文字。", "把这段文本读出来。", "朗读："],
    "en": ["Please read the following text aloud naturally.", "Read this text out loud.", "Speak:"],
    "ja": ["以下の文章を自然な语调で読み上げてください。", "この文章を読んでください。", "読み上げ："],
    "ko": ["다음 문장을 자연스러운 어조로 읽어 주세요.", "이 문장을 읽어 주세요.", "읽기:"],
}


def build_conversations(text: str, lang: str, rng: random.Random,
                        prompt_style: str = "en") -> str:
    """thinker 侧输入。user = 指令 + 原文（S1 的 cond 恒零，不受原文重复影响）。"""
    if prompt_style == "en":
        prompts = PROMPTS["zh"] if str(lang).lower().startswith("zh") else PROMPTS["en"]
    else:
        prompts = PROMPTS.get(str(lang).lower(), PROMPTS["en"])
    return json.dumps([
        {"role": "user", "content": rng.choice(prompts) + text},
        {"role": "assistant", "content": text},
    ], ensure_ascii=False)


# ---------------------------------------------------------------------------
# 音频解码
# ---------------------------------------------------------------------------
def decode_audio(data: bytes, fmt: str = "", sr_hint: Optional[int] = None,
                 ch_hint: Optional[int] = None) -> Optional[np.ndarray]:
    """任意容器字节 -> 16k 单声道 float32。soundfile 优先，失败退回 ffmpeg。

    webm（yodas3）libsndfile 常不认，必须走 ffmpeg；Lance 里的裸 pcm 无文件头，
    按 raw_pcm_format/sample_rate/channels 手工解释。
    """
    import soundfile as sf
    if fmt in ("pcm", "raw_pcm", "s16le") and sr_hint:
        w = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if ch_hint and ch_hint > 1:
            w = w.reshape(-1, ch_hint).mean(axis=1)
        if int(sr_hint) != SR:
            w = resample(w, int(sr_hint))
        return w.astype(np.float32)
    try:
        w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != SR:
            w = resample(w, sr)
        return np.ascontiguousarray(w, dtype=np.float32)
    except Exception:
        pass
    return _ffmpeg_decode(data)


def _ffmpeg_decode(data: bytes) -> Optional[np.ndarray]:
    try:
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
             "-f", "wav", "-ar", str(SR), "-ac", "1", "pipe:1"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if p.returncode != 0 or not p.stdout:
            return None
        import soundfile as sf
        w, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=True)
        if w.ndim > 1:
            w = w.mean(axis=1)
        return np.ascontiguousarray(w, dtype=np.float32)
    except Exception:
        return None


def resample(w: np.ndarray, orig_sr: int) -> np.ndarray:
    import librosa
    return librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=orig_sr, target_sr=SR)


def parse_transcript_segments(raw) -> List[tuple]:
    """YODAS3 transcript -> [(start_ms, duration_ms, text), ...]。"""
    try:
        items = raw if isinstance(raw, (list, tuple)) else ast.literal_eval(str(raw))
    except Exception:
        return []
    if isinstance(items, dict):
        items = [items]
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        t = str(it.get("text", "")).strip()
        if not t:
            continue
        try:
            out.append((float(it.get("start", 0) or 0), float(it.get("duration", 0) or 0), t))
        except (TypeError, ValueError):
            continue
    return out


def parse_transcript(raw) -> str:
    """yodas3 的 transcript 是 Python list 的字符串形式：[{'text': ...}, ...]。"""
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        try:
            items = ast.literal_eval(str(raw))
        except Exception:
            return str(raw or "").strip()
    if isinstance(items, dict):
        items = [items]
    return "".join(str(it.get("text", "")) for it in items if isinstance(it, dict)).strip()


# ---------------------------------------------------------------------------
# 数据源：工作单元枚举 + 样本迭代
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.args = args
        import boto3
        self.s3 = boto3.client("s3", region_name=args.region)

    def object_body(self, key: str, bucket: Optional[str] = None):
        return self.s3.get_object(Bucket=bucket or self.args.s3_in_bucket, Key=key)["Body"]

    def list_keys(self, prefix: str, suffix: str = "", bucket: Optional[str] = None) -> List[str]:
        bucket = bucket or self.args.s3_in_bucket
        out, token = [], None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                if o["Key"].endswith(suffix) if suffix else True:
                    out.append(o["Key"])
            if not r.get("IsTruncated"):
                return sorted(out)
            token = r["NextContinuationToken"]

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.args.s3_out_bucket, Key=key)
            return True
        except Exception:
            return False

    def put(self, key: str, body: bytes, ctype: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.args.s3_out_bucket, Key=key, Body=body, ContentType=ctype)


def iter_lance(ctx: Ctx, shard: int, nshard: int) -> Iterator[Dict]:
    """Curion curated TTS 表（Blob v2 音频）。按元数据批次轮转分片。"""
    import lance
    a = ctx.args
    ds = lance.dataset(a.lance_uri, version=a.lance_version,
                       storage_options={"region": a.region})
    langs = ",".join(f"'{l}'" for l in a.lang.split(","))
    cols = ["text", "language", "duration_ms", "speaker_ref", "dnsmos_ovrl",
            "audio_payload_format", "audio_sample_rate", "audio_channels", "raw_pcm_format"]
    scanner = ds.scanner(columns=cols, filter=f"language in ({langs})",
                         with_row_id=True, batch_size=a.meta_batch)
    for bi, batch in enumerate(scanner.to_batches()):
        if bi % nshard != shard:
            continue
        row_ids = batch.column("_rowid").to_pylist()
        blobs = ds.read_blobs("audio_payload", ids=row_ids, preserve_order=True)
        for row, (_rid, data) in zip(batch.to_pylist(), blobs):
            if not data:
                continue
            dur = (row.get("duration_ms") or 0) / 1000.0
            if dur <= 0 or dur > MAX_DURATION:
                continue
            w = decode_audio(data, row.get("audio_payload_format") or "",
                             row.get("audio_sample_rate"), row.get("audio_channels"))
            if w is None or len(w) < SR // 10:
                continue
            yield {"id": str(_rid), "text": (row.get("text") or "").strip(),
                   "language": row.get("language") or "",
                   "speaker": row.get("speaker_ref") or "",
                   "duration": float(len(w)) / SR,
                   "dnsmos": float(row.get("dnsmos_ovrl") or 0.0), "wav": w,
                   "source": "lance"}


def iter_webdataset_tar(ctx: Ctx, key: str, text_from: str = "json",
                        audio_exts=(".mp3", ".flac", ".wav", ".opus", ".ogg", ".webm"),
                        lang: str = "", text_map: Optional[Dict[str, str]] = None,
                        text_lookup=None, source: str = "",
                        segment_split: bool = False) -> Iterator[Dict]:
    """流式读 tar：{stem}.json(+{stem}.<audio>)，或 {id}.<audio> + 外部 text_map。"""
    body = ctx.object_body(key)
    tf = tarfile.open(fileobj=body, mode="r|")
    pending_meta: Dict[str, dict] = {}
    for m in tf:
        if not m.isfile():
            continue
        stem, ext = os.path.splitext(m.name)
        ext = ext.lower()
        if ext == ".json":
            try:
                pending_meta[stem] = json.loads(tf.extractfile(m).read().decode("utf-8"))
            except Exception:
                pass
            continue
        if ext in (".txt",):
            try:
                pending_meta[stem + "__txt"] = tf.extractfile(m).read().decode("utf-8").strip()
            except Exception:
                pass
            continue
        if ext not in audio_exts:
            continue
        meta = pending_meta.pop(stem, None)
        data = tf.extractfile(m).read()
        if text_lookup is not None:
            text = text_lookup(os.path.basename(stem)) or ""
            meta = meta or {}
        elif text_map is not None:
            text = text_map.get(os.path.basename(stem), "")
            meta = meta or {}
        elif text_from == "txt":
            text = str(pending_meta.pop(stem + "__txt", "") or "")
            meta = meta or {}
        else:
            if not meta:
                continue
            raw_t = meta.get("transcript", meta.get("text", ""))
            text = parse_transcript(raw_t) if meta.get("transcript") else str(raw_t or "").strip()
        if not text:
            continue
        dur = float(meta.get("duration", meta.get("length", 0) or 0) or 0)
        if dur <= 0:
            dur = float(len(data)) / 32000.0
        if dur > MAX_DURATION and not segment_split:
            continue
        w = decode_audio(data, fmt=ext.lstrip("."))
        if w is None or len(w) < SR // 10:
            continue
        # YODAS3：一条音频是十几分钟的长录音，transcript 里带逐句 start/duration。
        # 整条用会全部超过 30s 被丢弃（实测 108 个 tar 只留 4 千条），必须按句切开。
        if segment_split and meta:
            segs = parse_transcript_segments(meta.get("transcript"))
            if segs:
                for i, (s_ms, d_ms, stext) in enumerate(segs):
                    if not stext:
                        continue
                    dur_s = (d_ms or 0) / 1000.0
                    if dur_s < 0.2 or dur_s > MAX_DURATION:
                        continue
                    a = int((s_ms or 0) / 1000.0 * SR)
                    b = min(a + int(dur_s * SR), len(w))
                    if b - a < SR // 10:
                        continue
                    yield {"id": f"{os.path.basename(stem)}_{i:04d}", "text": stext,
                           "language": (meta.get("language") or lang),
                           "speaker": meta.get("speaker") or "",
                           "duration": (b - a) / SR,
                           "dnsmos": float(meta.get("dnsmos", 0.0) or 0.0),
                           "wav": w[a:b], "source": source}
                continue
        yield {"id": os.path.basename(stem), "text": text,
               "language": (meta.get("language") if meta else "") or lang,
               "speaker": (meta.get("speaker") if meta else "") or "",
               "duration": float(len(w)) / SR,
               "dnsmos": float((meta or {}).get("dnsmos", 0.0) or 0.0),
               "wav": w, "source": source}
    tf.close()


def iter_targz_wav_txt(ctx: Ctx, key: str, lang: str = "zh", source: str = "") -> Iterator[Dict]:
    """WenetSpeech4TTS：tar.gz 内 wavs/*.wav 与 txts/*.txt 同名配对（流式解压）。"""
    import gzip
    body = ctx.object_body(key)
    gz = gzip.GzipFile(fileobj=body)
    tf = tarfile.open(fileobj=gz, mode="r|")
    wavs: Dict[str, bytes] = {}
    txts: Dict[str, str] = {}
    for m in tf:
        if not m.isfile():
            continue
        stem, ext = os.path.splitext(os.path.basename(m.name))
        try:
            payload = tf.extractfile(m).read()
        except Exception:
            continue
        if ext.lower() == ".wav":
            wavs[stem] = payload
        elif ext.lower() == ".txt":
            txts[stem] = payload.decode("utf-8", "ignore").strip()
        if stem in wavs and stem in txts:
            w = decode_audio(wavs.pop(stem))
            text = txts.pop(stem)
            # txts/*.txt 内容形如 "<uttid>\t<文本>\n\t[[对齐时间戳]]"，只要文本部分
            if "\t" in text:
                text = text.split("\t", 1)[1]
            text = text.split("\n", 1)[0].strip()
            if w is None or not text or len(w) < SR // 10:
                continue
            if len(w) / SR > MAX_DURATION:
                continue
            yield {"id": stem, "text": text, "language": lang, "speaker": "",
                   "duration": len(w) / SR, "dnsmos": 0.0, "wav": w, "source": source}
    tf.close()


def iter_hf_audio_parquet(ctx: Ctx, key: str, lang: str = "ko",
                          source: str = "") -> Iterator[Dict]:
    """HF Audio parquet（audio{bytes,path} + sentence）。落盘到临时文件后按 row group 读。"""
    import pyarrow.parquet as pq
    tmp = os.path.join(ctx.args.tmp_dir, os.path.basename(key))
    os.makedirs(ctx.args.tmp_dir, exist_ok=True)
    try:
        with ctx.object_body(key) as body, open(tmp, "wb") as f:
            for chunk in iter(lambda: body.read(8 << 20), b""):
                f.write(chunk)
        pf = pq.ParquetFile(tmp)
        for rg in range(pf.num_row_groups):
            tb = pf.read_row_group(rg, columns=["id", "audio", "sentence"])
            ids = tb.column("id").to_pylist()
            aud = tb.column("audio").to_pylist()
            sents = tb.column("sentence").to_pylist()
            for i, a, s in zip(ids, aud, sents):
                data = (a or {}).get("bytes") if isinstance(a, dict) else None
                if not data or not s:
                    continue
                w = decode_audio(data)
                if w is None or len(w) < SR // 10 or len(w) / SR > MAX_DURATION:
                    continue
                yield {"id": str(i), "text": str(s).strip(), "language": lang,
                       "speaker": "", "duration": len(w) / SR, "dnsmos": 0.0,
                       "wav": w, "source": source}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


_ISCSLP_CACHE: Dict[str, Dict[str, str]] = {}


def load_iscslp_text_map(ctx: Ctx, prefix: str) -> Dict[str, str]:
    """movie-xx/movie-xx.json：scene[].target_segment|dialog_segments[] -> segment_id:text。

    按 movie 前缀缓存：同一 movie 下几十个 tar 共用一份文本表，逐单元重下会把
    几百 MB 的 json 重复拉几十遍。
    """
    if prefix in _ISCSLP_CACHE:
        return _ISCSLP_CACHE[prefix]
    out: Dict[str, str] = {}
    for k in ctx.list_keys(prefix, ".json"):
        if k.endswith("_cot_mos.json"):
            continue
        try:
            body = ctx.object_body(k).read()
            doc = json.loads(body.decode("utf-8"))
        except Exception:
            continue
        scenes = doc if isinstance(doc, list) else doc.get("scenes", [])
        for sc in scenes:
            for key in ("target_segment",):
                seg = sc.get(key) or {}
                if seg.get("segment_id") and seg.get("text"):
                    out[str(seg["segment_id"])] = str(seg["text"]).strip()
            for seg in sc.get("dialog_segments", []) or []:
                if seg.get("segment_id") and seg.get("text"):
                    out[str(seg["segment_id"])] = str(seg["text"]).strip()
    _ISCSLP_CACHE[prefix] = out
    return out


def iscslp_text_index(ctx: Ctx, prefix: str) -> str:
    """movie-xx/movie-xx.json（单文件 1.2GB）-> sqlite，节点内只建一次（带锁）。

    直接 json.load 会瞬开数 GB 内存，24 进程/节点会打爆；落 sqlite 后查询走索引。
    """
    import sqlite3
    movie = prefix.rstrip("/").rsplit("/", 1)[-1]
    db = os.path.join(ctx.args.tmp_dir, f"iscslp_{movie}.db")
    os.makedirs(ctx.args.tmp_dir, exist_ok=True)
    if os.path.exists(db) and os.path.getsize(db) > 0:
        return db
    lock = db + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        for _ in range(900):
            if os.path.exists(db) and os.path.getsize(db) > 0:
                return db
            time.sleep(2)
        return db
    try:
        tmp = db + f".{os.getpid()}"
        con = sqlite3.connect(tmp)
        con.execute("CREATE TABLE s (k TEXT PRIMARY KEY, t TEXT)")
        buf, n = [], 0
        for k in ctx.list_keys(prefix, ".json"):
            if k.endswith("_cot_mos.json"):
                continue
            local = os.path.join(ctx.args.tmp_dir, os.path.basename(k))
            with ctx.object_body(k) as body, open(local, "wb") as f:
                for chunk in iter(lambda: body.read(8 << 20), b""):
                    f.write(chunk)
            with open(local, "r", encoding="utf-8") as f:
                doc = json.load(f)
            os.remove(local)
            for sc in doc if isinstance(doc, list) else doc.get("scenes", []):
                seg = sc.get("target_segment") or {}
                if seg.get("segment_id") and seg.get("text"):
                    buf.append((str(seg["segment_id"]), str(seg["text"]).strip()))
                for s2 in sc.get("dialog_segments", []) or []:
                    if s2.get("segment_id") and s2.get("text"):
                        buf.append((str(s2["segment_id"]), str(s2["text"]).strip()))
                if len(buf) >= 20000:
                    con.executemany("INSERT OR REPLACE INTO s VALUES (?,?)", buf)
                    n += len(buf)
                    buf = []
        if buf:
            con.executemany("INSERT OR REPLACE INTO s VALUES (?,?)", buf)
            n += len(buf)
        con.commit()
        con.close()
        os.replace(tmp, db)
        print(f"[iscslp] {movie} text index: {n} 条 -> {db}", flush=True)
    finally:
        os.close(fd)
        try:
            os.remove(lock)
        except OSError:
            pass
    return db


def hifitts_text_index(ctx: Ctx) -> str:
    """manifest_22khz.json（JSONL：audio_filepath/text）-> sqlite，避免 7.6GB 常驻内存。"""
    import sqlite3
    db = os.path.join(ctx.args.tmp_dir, "hifitts_manifest.db")
    os.makedirs(ctx.args.tmp_dir, exist_ok=True)
    if os.path.exists(db) and os.path.getsize(db) > 0:
        return db
    tmp = db + f".{os.getpid()}"
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE m (k TEXT PRIMARY KEY, t TEXT)")
    n = 0
    buf = []
    body = ctx.object_body(ctx.args.hifitts_manifest)
    for line in body.iter_lines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        k, t = o.get("audio_filepath"), o.get("text")
        if k and t:
            buf.append((k, t))
        if len(buf) >= 20000:
            con.executemany("INSERT OR REPLACE INTO m VALUES (?,?)", buf)
            n += len(buf)
            buf = []
    if buf:
        con.executemany("INSERT OR REPLACE INTO m VALUES (?,?)", buf)
        n += len(buf)
    con.commit()
    con.close()
    os.replace(tmp, db)
    print(f"[hifitts] manifest index: {n} 条 -> {db}", flush=True)
    return db


def iter_hifitts(ctx: Ctx, shard: int, nshard: int) -> Iterator[Dict]:
    """LibriVox 章节 mp3 + chapters_22khz.json 的 utterance 偏移 -> 逐条切片。

    一个章节 mp3 只下载一次，切片后得到多条 utterance（text 来自 manifest）。
    """
    import sqlite3
    import urllib.request
    db = hifitts_text_index(ctx)
    con = sqlite3.connect(db)
    # 断点续跑：hifitts 只有一个超大单元，重启一次就前功尽弃；按"已处理到第几章"记状态
    state_dir = os.path.join(ctx.args.tmp_dir, "hifitts_state")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, f"shard-{shard}.json")
    resume_from = -1
    if os.path.exists(state_path):
        try:
            st = json.load(open(state_path))
            if st.get("nshard") == nshard:
                resume_from = int(st.get("last_ci", -1))
                print(f"[hifitts] 断点续跑：从第 {resume_from + 1} 章继续", flush=True)
        except Exception:
            pass

    body = ctx.object_body(ctx.args.hifitts_chapters)
    ci = -1
    for line in body.iter_lines():
        try:
            ch = json.loads(line)
        except Exception:
            continue
        ci += 1
        if ci % nshard != shard or ci <= resume_from:
            continue
        url, utts = ch.get("url"), ch.get("utterances") or []
        if not url or not utts:
            continue
        url = url.replace("http://", "https://")
        # 章节音频已经在下载阶段镜像到 S3 了（TTS/en/hifitts-2/librivox/<item>/<file>.mp3），
        # 直接读桶里的副本；只有镜像阶段失败的章节才回源 archive.org。
        tail = url.split("/download/", 1)[-1].split("?")[0]
        s3_key = f"{ctx.args.s3_in_prefix}/en/hifitts-2/librivox/{tail}"
        data = None
        try:
            data = ctx.object_body(s3_key).read()
        except Exception:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                with urllib.request.urlopen(req, timeout=900) as r:
                    data = r.read()
            except Exception:
                continue
        if not data:
            continue
        chapter = decode_audio(data)
        if chapter is None:
            continue
        for u in utts:
            key = u.get("audio_filepath")
            if not key:
                continue
            row = con.execute("SELECT t FROM m WHERE k=?", (key,)).fetchone()
            if not row or not row[0]:
                continue
            off = int(float(u.get("offset", 0)) * SR)
            dur = float(u.get("duration", 0))
            if dur <= 0 or dur > MAX_DURATION:
                continue
            seg = chapter[off:off + int(dur * SR)]
            if len(seg) < SR // 10:
                continue
            yield {"id": os.path.basename(key).rsplit(".", 1)[0], "text": row[0].strip(),
                   "language": "en", "speaker": str(ch.get("chapter_filepath", "")).split("/")[0],
                   "duration": len(seg) / SR, "dnsmos": 0.0, "wav": seg, "source": "hifitts2"}
        if ci % 25 == 0:
            json.dump({"nshard": nshard, "last_ci": ci}, open(state_path, "w"))
    con.close()


def enumerate_units(ctx: Ctx, source: str, shard: int, nshard: int) -> List[tuple]:
    """返回 [(unit_id, payload)]，按 index % nshard 分片（文件级，无需 rank/world_size）。"""
    a = ctx.args
    base = a.s3_in_prefix.rstrip("/")
    items: List[tuple] = []
    if source == "lance":
        keys = [f"chunk{i}" for i in range(nshard)]  # 由 iter_lance 内部按批次轮转
    elif source == "emilia":
        keys = ctx.list_keys(f"{base}/zh/emilia/", ".tar")
    elif source == "yodas":
        keys = ctx.list_keys(f"{base}/jp/audio/", ".tar")
    elif source == "wenet":
        keys = ctx.list_keys(f"{base}/zh/wenetspeech4tts/", ".tar.gz")
    elif source == "iscslp":
        lang_dir = "zh" if a.lang == "zh" else "en"
        pairs: List[tuple] = []
        for mv in a.iscslp_movies.split(","):
            if not mv:
                continue
            pre = f"{base}/{lang_dir}/iscslp2026_cot_tts/{mv}/"
            for k in ctx.list_keys(pre, ".tar"):
                pairs.append((k, pre))
        # 先分片、再按需要的 movie 建文本索引（避免为不属于本分片的 tar 拉 1.2GB json）
        return [(k, iscslp_text_index(ctx, pre))
                for i, (k, pre) in enumerate(pairs) if i % nshard == shard]
    elif source == "ko":
        keys = ctx.list_keys(f"{base}/ko/patterned_utterance_with_number/data/", ".parquet")
    elif source == "hifitts":
        return [("chapters", None)] if shard == shard else []
    else:
        raise SystemExit(f"未知 source: {source}")
    return [(k, None) for i, k in enumerate(keys) if i % nshard == shard]


def iter_unit(ctx: Ctx, source: str, unit: tuple, shard: int, nshard: int) -> Iterator[Dict]:
    key, payload = unit
    a = ctx.args
    if source == "lance":
        yield from iter_lance(ctx, shard, nshard)
    elif source == "emilia":
        yield from iter_webdataset_tar(ctx, key, text_from="json", lang="zh", source="emilia")
    elif source == "yodas":
        yield from iter_webdataset_tar(ctx, key, text_from="json", lang="ja",
                                       source="yodas3", segment_split=True)
    elif source == "wenet":
        yield from iter_targz_wav_txt(ctx, key, lang="zh", source="wenetspeech4tts")
    elif source == "iscslp":
        import sqlite3
        con = sqlite3.connect(payload)

        def lookup(seg_id: str) -> str:
            row = con.execute("SELECT t FROM s WHERE k=?", (seg_id,)).fetchone()
            return row[0] if row else ""

        try:
            yield from iter_webdataset_tar(ctx, key, lang=a.lang, text_lookup=lookup,
                                           audio_exts=(".flac", ".wav", ".mp3"),
                                           source="iscslp2026_cot_tts")
        finally:
            con.close()
    elif source == "ko":
        yield from iter_hf_audio_parquet(ctx, key, lang="ko", source="patterned_utterance")
    elif source == "hifitts":
        yield from iter_hifitts(ctx, shard, nshard)


# ---------------------------------------------------------------------------
# 主流程：攒批 -> FSQ -> parquet -> S3
# ---------------------------------------------------------------------------
def write_parquet(rows: Dict[str, list], with_lengths: bool, lengths: Optional[List[int]],
                  with_source: bool = False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {
        "id": pa.array(rows["id"], pa.string()),
        "text": pa.array(rows["text"], pa.string()),
        "language": pa.array(rows["language"], pa.string()),
        "speaker": pa.array(rows["speaker"], pa.string()),
        "duration": pa.array(rows["duration"], pa.float32()),
        "dnsmos": pa.array(rows["dnsmos"], pa.float32()),
        "answer_speech_tokens": pa.array(rows["answer_speech_tokens"],
                                         pa.list_(pa.list_(pa.int32()))),
        "conversations": pa.array(rows["conversations"], pa.string()),
    }
    if with_source:
        # 现网 s1_emilia_fsq_v3 没有这一列；默认不加，保持逐列一致
        cols["source"] = pa.array(rows["source"], pa.string())
    if with_lengths:
        cols["lengths"] = pa.array(lengths or [], pa.int32())
    buf = io.BytesIO()
    pq.write_table(pa.table(cols), buf)
    return buf.getvalue()


TOKENIZER_MIN_BYTES = 200 * 1024 * 1024   # 969MB；小于此值视为没下完


def resolve_tokenizer(ctx: Ctx, local: str) -> str:
    """取 FSQ tokenizer 到本地。

    同一节点上多个进程共享 `--tmp-dir`（hostPath NVMe），并发写同一路径会让别的进程
    读到半截文件（onnxruntime 报 InvalidProtobuf / Protobuf parsing failed）。
    故先写 <path>.<pid>.tmp 再原子 rename。
    """
    if local and os.path.exists(local) and os.path.getsize(local) > TOKENIZER_MIN_BYTES:
        return local
    os.makedirs(ctx.args.tmp_dir, exist_ok=True)
    dst = os.path.join(ctx.args.tmp_dir, "speech_tokenizer_v3.batch.onnx")

    bucket, key = ctx.args.s3_in_bucket, DEFAULT_FSQ_TOKENIZER_S3
    if local.startswith("s3://"):
        bucket, key = local[5:].split("/", 1)
    expected = ctx.s3.head_object(Bucket=bucket, Key=key)["ContentLength"]

    # 只有字节数与 S3 完全一致才复用（半截文件会让 onnxruntime 报 InvalidProtobuf）
    if os.path.exists(dst) and os.path.getsize(dst) == expected:
        return dst
    # 临时名必须全局唯一：不同 Pod 的 PID 会撞号，同名 tmp 会被对方 rename 走
    for _ in range(3):
        tmp = f"{dst}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with ctx.s3.get_object(Bucket=bucket, Key=key)["Body"] as body, open(tmp, "wb") as f:
                shutil.copyfileobj(body, f)
            if os.path.getsize(tmp) != expected:
                os.remove(tmp)
                raise RuntimeError(f"tokenizer 下载不完整: {os.path.getsize(tmp)} != {expected}")
            os.replace(tmp, dst)
            return dst
        except FileNotFoundError:
            time.sleep(3)
    raise RuntimeError(f"tokenizer 下载失败（并发冲突）: {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description="TTS 语料 -> streaming_si stage1 FSQ parquet")
    ap.add_argument("--source", required=True,
                    choices=["lance", "emilia", "yodas", "wenet", "iscslp", "ko", "hifitts"])
    ap.add_argument("--lang", default="", help="输出 language 字段（iscslp 需显式 zh/en）")
    ap.add_argument("--s3-in-bucket", default=DEFAULT_TTS_BUCKET)
    ap.add_argument("--s3-out-bucket", default=DEFAULT_OUT_BUCKET)
    ap.add_argument("--s3-in-prefix",
                    default="speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS")
    ap.add_argument("--s3-out-prefix",
                    default="speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS_FSQ")
    ap.add_argument("--lance-uri", default=DEFAULT_LANCE_URI)
    ap.add_argument("--lance-version", type=int, default=74)
    ap.add_argument("--hifitts-manifest", default="")
    ap.add_argument("--hifitts-chapters", default="")
    ap.add_argument("--iscslp-movies", default="movie-zh1,movie-zh2,movie-zh3,movie-zh4")
    ap.add_argument("--fsq-tokenizer", default="", help="本地路径或 s3:// 路径")
    ap.add_argument("--text-tokenizer", default="",
                    help="给定时额外写 lengths 列（Qwen 词表目录）。现网 s1_emilia_fsq_v3 没有这一列，"
                         "训练入口 setup_s1 会用 s1_parquet_prep.py 自己加，故默认不开")
    ap.add_argument("--keep-source-column", action="store_true",
                    help="额外写 source 列（默认不加，与现网 S1 parquet 逐列一致）")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--meta-batch", type=int, default=512, help="Lance 元数据批次大小")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit-units", type=int, default=0, help="只处理前 N 个单元（调试）")
    ap.add_argument("--limit-items", type=int, default=0, help="每个单元只处理前 N 条（冒烟）")
    ap.add_argument("--rows-per-file", type=int, default=0,
                    help="单元内累计到 N 行就落一个 parquet（hifitts 这种单单元超大数据源必须开）")
    ap.add_argument("--min-dnsmos", type=float, default=0.0,
                    help="默认 0=不过滤（多数源没有 dnsmos，非 0 会静默丢光）")
    ap.add_argument("--prompt-style", choices=["en", "native"], default="en")
    ap.add_argument("--tmp-dir", default="/tmp/tts_stage1")
    ap.add_argument("--region", default="ap-southeast-3")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_in = args.s3_in_prefix.rstrip("/")
    if not args.hifitts_manifest:
        args.hifitts_manifest = f"{base_in}/en/hifitts-2/22khz/manifest_22khz.json"
    if not args.hifitts_chapters:
        args.hifitts_chapters = f"{base_in}/en/hifitts-2/22khz/chapters_22khz.json"

    ctx = Ctx(args)
    rng = random.Random(0 + args.shard)
    tok_path = resolve_tokenizer(ctx, args.fsq_tokenizer)
    try:
        sess = fsq_encoder.build_session(tok_path, args.device)
    except Exception as exc:
        # 历史遗留：早期版本多进程并发写同一路径，会留下"字节数正确但内容交错"的坏文件。
        # 删掉重下一次（只按大小判断是发现不了的）。
        if "Protobuf" not in str(exc):
            raise
        print(f"[warn] tokenizer 损坏，删除重下: {tok_path}", flush=True)
        try:
            os.remove(tok_path)
        except OSError:
            pass
        tok_path = resolve_tokenizer(ctx, args.fsq_tokenizer)
        sess = fsq_encoder.build_session(tok_path, args.device)
    print(f"[shard {args.shard}/{args.nshard}] source={args.source} lang={args.lang or 'auto'} "
          f"device={args.device} EP={sess.get_providers()[0]} tokenizer={tok_path}", flush=True)

    text_tok = None
    if args.text_tokenizer:
        from transformers import AutoTokenizer
        text_tok = AutoTokenizer.from_pretrained(args.text_tokenizer)

    # 代际戳：不同代 tokenizer 的 parquet 混放无法从内容分辨
    stamp = f"{args.s3_out_prefix}/.cosy_version"
    if not ctx.exists(stamp):
        ctx.put(stamp, b"v3\n", "text/plain")

    units = enumerate_units(ctx, args.source, args.shard, args.nshard)
    if args.limit_units:
        units = units[:args.limit_units]
    print(f"[shard {args.shard}] {len(units)} 个工作单元", flush=True)

    t0 = time.time()
    total_rows = 0
    for ui, unit in enumerate(units, 1):
        unit_name = os.path.basename(str(unit[0])).replace(".tar.gz", "").replace(".tar", "")
        lang_dir = (args.lang or args.source).replace(",", "-")
        out_key = (f"{args.s3_out_prefix}/{lang_dir}/"
                   f"{args.source}-s{args.shard:04d}-{ui:05d}.parquet")
        if ctx.exists(out_key) and not args.overwrite:
            print(f"[shard {args.shard}] {ui}/{len(units)} {unit_name} -> skip(已存在)", flush=True)
            continue

        rows = {k: [] for k in ("id", "text", "language", "speaker", "duration",
                                "dnsmos", "answer_speech_tokens", "conversations", "source")}
        lens: List[int] = []
        buf_wav: List[np.ndarray] = []
        buf_meta: List[Dict] = []
        n_item = n_skip = n_len = n_drop = 0
        part = 0

        def dump(idx: int) -> int:
            """把当前 rows 落盘成一个 parquet（idx=0 用原 key，>0 加 -pN 后缀）。"""
            if not rows["id"]:
                return 0
            key = out_key if idx == 0 else out_key.replace(".parquet", f"-p{idx}.parquet")
            blob = write_parquet(rows, text_tok is not None, lens, args.keep_source_column)
            ctx.put(key, blob)
            n = len(rows["id"])
            print(f"[shard {args.shard}]   -> 落盘 {os.path.basename(key)} "
                  f"rows={n} {len(blob)/1e6:.1f} MB", flush=True)
            for k in rows:
                rows[k].clear()
            lens.clear()
            return n

        def flush():
            nonlocal buf_wav, buf_meta
            if not buf_wav:
                return 0
            chunk = list(zip(buf_meta, buf_wav))
            dropped = [0]

            def _run(chunk):
                metas = [c[0] for c in chunk]
                wavs = [c[1] for c in chunk]
                toks = fsq_encoder.wavs_to_fsq_tokens(sess, wavs, device=args.device)
                for meta, tk in zip(metas, toks):
                    if not (MIN_SPEECH_TOKENS <= len(tk) <= MAX_SPEECH_TOKENS):
                        continue
                    rows["id"].append(meta["id"])
                    rows["text"].append(meta["text"])
                    rows["language"].append(meta["language"])
                    rows["speaker"].append(meta.get("speaker", ""))
                    rows["duration"].append(float(meta["duration"]))
                    rows["dnsmos"].append(float(meta.get("dnsmos", 0.0)))
                    rows["answer_speech_tokens"].append([tk])   # 单轮 -> 外层长度 1
                    rows["conversations"].append(
                        build_conversations(meta["text"], meta["language"], rng, args.prompt_style))
                    rows["source"].append(meta.get("source", args.source))
                    if text_tok is not None:
                        n_t = min(len(text_tok.encode(meta["text"], add_special_tokens=False)) + 1,
                                  args.max_length - 1)
                        lens.append(1 + n_t + 1 + min(len(tk), MAX_SPEECH_TOKENS))

            dropped[0] = fsq_encoder.run_with_oom_split(_run, chunk)
            buf_wav, buf_meta = [], []
            return dropped[0]

        try:
            for s in iter_unit(ctx, args.source, unit, args.shard, args.nshard):
                n_item += 1
                if args.limit_items and n_item > args.limit_items:
                    break
                if not s["text"] or len(s["wav"]) < SR // 10:
                    n_skip += 1
                    continue
                if len(s["wav"]) / SR > MAX_DURATION:
                    n_len += 1
                    continue
                if args.min_dnsmos > 0 and s.get("dnsmos", 0.0) < args.min_dnsmos:
                    n_skip += 1
                    continue
                buf_wav.append(s["wav"])
                buf_meta.append(s)
                if len(buf_wav) >= args.bs:
                    n_drop += flush()
                    if args.rows_per_file and len(rows["id"]) >= args.rows_per_file:
                        part += 1
                        dump(part)
        except Exception as e:
            print(f"[shard {args.shard}] {ui}/{len(units)} {unit_name} -> 单元失败 "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        n_drop += flush()

        written = dump(part)
        if not written:
            print(f"[shard {args.shard}] {ui}/{len(units)} {unit_name} -> empty("
                  f"item={n_item}, skip={n_skip}, overlong={n_len}, drop={n_drop})", flush=True)
            continue
        total_rows += written
        print(f"[shard {args.shard}] {ui}/{len(units)} {unit_name} -> ok(rows={written}, "
              f"skip={n_skip}, overlong={n_len}, drop={n_drop}) "
              f"| 累计 {total_rows} 行 {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"[shard {args.shard}] 完成：{total_rows} 行，{(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
