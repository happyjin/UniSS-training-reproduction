#!/usr/bin/env python3
"""把 TTS/ 各源转成 simuls2st_omni stage1 的 DualCodec codes + manifest。

目标布局（stage1 TTSCodesDataset 消费）:

    <out>/manifest/<LANG>/shard-XXX.jsonl
        {"id","text","language"<小写>,"codes":"<LANG>/shard-XXX.npy:<start>:<end>",
         "dnsmos"<可选>,"duration"}
    <out>/codes/<LANG>/shard-XXX.npy   # uint16, 值域[0,16384), 12.5Hz

源类型（--source）:
  tar_pair   Emilia / Emilia-YODAS / yodas: tar 内 <id>.json + <id>.{mp3,webm}
             （json 需含 text 字段；不含 text 的记录计入 skip_text）
  parquet    ko patterned_utterance: 列 id / audio / sentence
  nemo       hifitts: NeMo manifest jsonl + 音频根（audio_filepath 相对根）
  iscslp     movie json（target_segment.text）+ segments tar（<segment_id>.<ext>）

用法（集群上每 GPU 一进程，按 --inputs 分片）:

  python make_tts_dualcodec.py --source tar_pair \
      --inputs /data/ZH-B000000.tar /data/ZH-B000001.tar \
      --lang ZH --out /out/TTS_omni \
      --weights /hf/SimulS2ST-Omni --device cuda:0

依赖: ffmpeg（解码 mp3/webm）、simuls2st_omni 包（PYTHONPATH 指向 custom/ 的父目录）、
      HF 权重目录（dualcodec/ + w2v/，来自 HA-SA-ki/SimulS2ST-Omni）。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from typing import Iterator, Optional

import numpy as np

# stage1 的过滤口径（smoke fixture STAGE1: min_codes 8, max_codes 375 = 30s）
MIN_CODES = 8
MAX_CODES = 375
MIN_DURATION = 0.5
MAX_DURATION = 30.0
SR = 16000
AUDIO_EXTS = ('.mp3', '.webm', '.flac', '.wav', '.m4a', '.ogg', '.opus', '.aac')
N_CODES = 16384


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 解码: 任意格式 bytes -> 16kHz mono float32
# ---------------------------------------------------------------------------
def decode_audio(data: bytes, timeout: int = 60) -> Optional[np.ndarray]:
    try:
        p = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', 'pipe:0',
             '-f', 'f32le', '-ac', '1', '-ar', str(SR), 'pipe:1'],
            input=data, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    wav = np.frombuffer(p.stdout, dtype=np.float32)
    if wav.size < SR // 2:
        return None
    return np.ascontiguousarray(wav)


# ---------------------------------------------------------------------------
# Tokenizer: 复用 simuls2st_omni.voicebox.tokenizer.VoiceBoxAudioTokenizer
# ---------------------------------------------------------------------------
def build_tokenizer(weights_root: str, device: str):
    from types import SimpleNamespace as NS
    dc = NS(n_codebooks=7, codebook_size=4096, semantic_codebook_size=N_CODES,
            is_causal=True, semantic_downsample_factor=4,
            pretrained_path=os.path.join(weights_root, 'dualcodec', 'dualcodec.safetensors'))
    km = NS(stat_mean_var_path=os.path.join(weights_root, 'dualcodec', 'w2v_bert_stats.pt'))
    cfg = NS(model=NS(dual_codec=dc, kmeans=km))
    os.environ.setdefault('SIMULS2ST_W2V_BERT_PATH', os.path.join(weights_root, 'w2v'))
    from simuls2st_omni.voicebox.tokenizer import VoiceBoxAudioTokenizer
    return VoiceBoxAudioTokenizer(cfg, device)


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_root: str, lang: str, shard_records: int = 5000,
                 shard_offset: int = 0, block: int = 100000):
        self.manifest_dir = os.path.join(out_root, 'manifest', lang)
        self.codes_dir = os.path.join(out_root, 'codes', lang)
        os.makedirs(self.manifest_dir, exist_ok=True)
        os.makedirs(self.codes_dir, exist_ok=True)
        self.lang = lang
        self.shard_records = shard_records
        # 多 worker 共写同一 out 时用 offset 错开 shard 编号，避免同名覆盖。
        # 重启续写：本 worker 编号块 [offset, offset+block) 里已有 shard 时从最大号+1 开始，
        # 否则重启会从 offset 重新编号，覆盖之前写好的 shard（而对应 item 已标记完成、不会重做）。
        self.shard_idx = self._resume_idx(shard_offset, block)
        if self.shard_idx != shard_offset:
            _log(f'[{lang}] 续写：块 {shard_offset} 已有 shard，从 shard-{self.shard_idx:04d} 开始')
        self.rows: list[dict] = []
        self.codes_parts: list[np.ndarray] = []
        self.offset = 0
        self.total_codes = 0

    def _resume_idx(self, offset: int, block: int) -> int:
        nxt = offset
        for d, ext in ((self.manifest_dir, '.jsonl'), (self.codes_dir, '.npy')):
            for fn in os.listdir(d):
                if not (fn.startswith('shard-') and fn.endswith(ext)):
                    continue
                try:
                    n = int(fn[len('shard-'):-len(ext)])
                except ValueError:
                    continue
                if offset <= n < offset + block:
                    nxt = max(nxt, n + 1)
        return nxt

    def add(self, uid: str, text: str, codes: np.ndarray, duration: float,
            dnsmos: Optional[float] = None) -> None:
        n = int(codes.size)
        rec = {'id': uid, 'text': text, 'language': self.lang.lower(),
               'codes': f'{self.lang}/shard-{self.shard_idx:04d}.npy:{self.offset}:{self.offset + n}',
               'duration': round(duration, 3)}
        if dnsmos is not None:
            rec['dnsmos'] = round(float(dnsmos), 3)
        self.rows.append(rec)
        self.codes_parts.append(codes)
        self.offset += n
        self.total_codes += n
        if len(self.rows) >= self.shard_records:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        shard = f'shard-{self.shard_idx:04d}'
        np.save(os.path.join(self.codes_dir, shard + '.npy'),
                np.concatenate(self.codes_parts).astype(np.uint16))
        with open(os.path.join(self.manifest_dir, shard + '.jsonl'), 'w', encoding='utf-8') as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        _log(f'shard {shard}: {len(self.rows)} records, {self.offset} codes')
        self.shard_idx += 1
        self.rows = []
        self.codes_parts = []
        self.offset = 0


# ---------------------------------------------------------------------------
# Readers: 统一产出 (uid, text, audio_bytes, dnsmos|None)
# ---------------------------------------------------------------------------
def iter_tar_pair_yield(tar_paths: list[str]) -> Iterator[tuple]:
    """生成器版（用这个）：逐 tar 流式，json+audio 成对后 yield。"""
    for tp in tar_paths:
        pending: dict[str, dict] = {}
        tname = os.path.basename(tp)
        try:
            tf = tarfile.open(tp)
        except tarfile.TarError as e:
            _log(f'[skip] 打不开 {tname}: {e}')
            continue
        with tf:
            for m in tf:
                if not m.isfile():
                    continue
                stem, ext = os.path.splitext(m.name)
                if ext == '.json':
                    try:
                        j = json.load(tf.extractfile(m))
                    except Exception:
                        continue
                    e2 = pending.setdefault(stem, {})
                    e2['json'] = j
                    e2.setdefault('uid', stem)
                elif ext.lower() in AUDIO_EXTS:
                    e2 = pending.setdefault(stem, {})
                    e2['audio'] = tf.extractfile(m).read()
                    e2.setdefault('uid', stem)
                ready = [k for k, v in pending.items() if 'audio' in v and 'json' in v]
                for k in ready:
                    v = pending.pop(k)
                    j = v['json']
                    raw = j.get('text') or j.get('transcript_text') or ''
                    if not raw:
                        # yodas3: transcript 是分段的 JSON list，每段 {"start","duration","text",...}
                        tr = j.get('transcript')
                        if isinstance(tr, list):
                            raw = ' '.join(str(seg.get('text') or '').strip()
                                           for seg in tr if isinstance(seg, dict))
                        elif isinstance(tr, str):
                            raw = tr
                    text = normalize_text(raw)
                    if text and v.get('audio'):
                        yield (v['uid'], text, v['audio'], j.get('dnsmos'))
                if len(pending) > 512:
                    # 防御：极端无配对时丢最旧的 json-only 项
                    for k in [k for k, v in pending.items() if 'audio' not in v][:256]:
                        pending.pop(k)
        _log(f'[tar done] {tname}, pending unresolved={len(pending)}')


def iter_yodas_segments(tar_paths: list[str]) -> Iterator[tuple]:
    """yodas3：tar 内 json + webm 成对，json.transcript 是带时戳分段（ms）。

    文件整长中位 ~4 分钟（不能整条进 30s 过滤），须整文件解码一次 16k PCM，
    再按 (start, duration) 切段；段间有重叠，贪心保留不重叠段。
    产出 (uid, text, wav 片段 np.ndarray, None)——注意直接给 wav 而非字节。
    """
    for tp in tar_paths:
        pending: dict[str, dict] = {}
        tname = os.path.basename(tp)
        try:
            tf = tarfile.open(tp)
        except tarfile.TarError as e:
            _log(f'[skip] 打不开 {tname}: {e}')
            continue
        with tf:
            for m in tf:
                if not m.isfile():
                    continue
                stem, ext = os.path.splitext(m.name)
                if ext == '.json':
                    try:
                        j = json.load(tf.extractfile(m))
                    except Exception:
                        continue
                    e2 = pending.setdefault(stem, {})
                    e2['json'] = j
                    e2.setdefault('uid', stem)
                elif ext.lower() in AUDIO_EXTS:
                    e2 = pending.setdefault(stem, {})
                    e2['audio'] = tf.extractfile(m).read()
                    e2.setdefault('uid', stem)
                ready = [k for k, v in pending.items() if 'audio' in v and 'json' in v]
                for k in ready:
                    v = pending.pop(k)
                    j = v['json']
                    segs = j.get('transcript')
                    if not isinstance(segs, list) or not v.get('audio'):
                        continue
                    # 长文件（可达数小时）放宽解码超时
                    wav = decode_audio(v['audio'], timeout=600)
                    if wav is None:
                        continue
                    last_end = 0.0
                    for si, seg in enumerate(segs):
                        if not isinstance(seg, dict):
                            continue
                        st = float(seg.get('start') or 0)
                        du = float(seg.get('duration') or 0)
                        text = str(seg.get('text') or '').strip()
                        if not text or du <= 0:
                            continue
                        if st < last_end:  # 与上一保留段重叠 → 跳过
                            continue
                        s0 = int(st / 1000.0 * SR)
                        s1 = min(int((st + du) / 1000.0 * SR), wav.size)
                        if s1 - s0 < int(MIN_DURATION * SR):
                            continue
                        last_end = st + du
                        yield (f'{v["uid"]}_s{si:03d}', text,
                               np.ascontiguousarray(wav[s0:s1]), None)
        _log(f'[tar done] {tname}, pending unresolved={len(pending)}')


def iter_parquet(paths: list[str]) -> Iterator[tuple]:
    import pyarrow.parquet as pq
    for p in paths:
        try:
            t = pq.read_table(p, columns=['id', 'audio', 'sentence'])
        except Exception as e:
            _log(f'[skip] parquet {os.path.basename(p)}: {e}')
            continue
        ids = t['id'].to_pylist()
        sents = t['sentence'].to_pylist()
        audios = t['audio'].to_pylist()
        for uid, sent, audio in zip(ids, sents, audios):
            text = (sent or '').strip()
            data = None
            if isinstance(audio, dict):
                data = audio.get('bytes')
            elif isinstance(audio, (bytes, bytearray)):
                data = bytes(audio)
            if text and data:
                yield (str(uid), text, data, None)
        _log(f'[parquet done] {os.path.basename(p)} rows={t.num_rows}')


def iter_nemo(manifest_path: str, audio_root: str) -> Iterator[tuple]:
    """hifitts: 每行 jsonl, audio_filepath 相对 audio_root。"""
    miss = hit = 0
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get('normalized_text') or r.get('text') or '').strip()
            p = os.path.join(audio_root, r.get('audio_filepath') or '')
            if not text or not os.path.isfile(p):
                miss += 1
                continue
            with open(p, 'rb') as af:
                data = af.read()
            hit += 1
            yield (os.path.basename(p), text, data, None)
    _log(f'[nemo done] hit={hit} miss={miss} (flac 缺失会全部 miss)')


def iter_iscslp(json_path: str, tar_dirs: list[str]) -> Iterator[tuple]:
    """movie json: [{movie_id,...,target_segment:{segment_id,text,...}}] + tar 内音频。
    音频文件按 segment_id 在 tar_dirs 的 tar 里找（首个命中即用）。"""
    index: dict[str, tuple[str, str]] = {}  # segment_id -> (tar_path, member_name)
    for d in tar_dirs:
        for tp in sorted(glob.glob(os.path.join(d, '*.tar'))):
            try:
                with tarfile.open(tp) as tf:
                    for m in tf:
                        if m.isfile() and os.path.splitext(m.name)[1].lower() in AUDIO_EXTS:
                            stem = os.path.basename(os.path.splitext(m.name)[0])
                            index.setdefault(stem, (tp, m.name))
            except tarfile.TarError as e:
                _log(f'[skip] tar {tp}: {e}')
    _log(f'[iscslp] 索引了 {len(index)} 个音频段')
    with open(json_path, encoding='utf-8') as f:
        rows = json.load(f)
    hit = 0
    for r in rows:
        seg = r.get('target_segment') or {}
        sid = str(seg.get('segment_id') or '')
        text = (seg.get('text') or '').strip()
        if not sid or not text or sid not in index:
            continue
        tp, member = index[sid]
        with tarfile.open(tp) as tf:
            data = tf.extractfile(member).read()
        hit += 1
        yield (sid, text, data, None)
    _log(f'[iscslp done] hit={hit}/{len(rows)}')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True,
                    choices=['tar_pair', 'parquet', 'nemo', 'iscslp'])
    ap.add_argument('--inputs', nargs='*', default=[], help='tar/parquet 文件列表（可 glob）')
    ap.add_argument('--manifest', default=None, help='nemo/iscslp 的 manifest/json 路径')
    ap.add_argument('--audio-root', default=None, help='nemo 的音频根')
    ap.add_argument('--tar-dirs', nargs='*', default=[], help='iscslp 的音频 tar 目录')
    ap.add_argument('--lang', required=True, help='大写语言码：EN/ZH/JA/KO')
    ap.add_argument('--out', required=True, help='TTS_omni 输出根')
    ap.add_argument('--weights', required=True, help='HF 权重根（含 dualcodec/ w2v/）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard-records', type=int, default=5000)
    ap.add_argument('--shard-offset', type=int, default=0,
                    help='多 worker 共写同一 out 时的 shard 编号起点（如 worker_id*100000）')
    ap.add_argument('--limit', type=int, default=0, help='试点限量')
    args = ap.parse_args()

    inputs = []
    for pat in args.inputs:
        inputs.extend(sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat])

    if args.source == 'tar_pair':
        it = iter_tar_pair_yield(inputs)
    elif args.source == 'parquet':
        it = iter_parquet(inputs)
    elif args.source == 'nemo':
        assert args.manifest and args.audio_root, 'nemo 需要 --manifest 和 --audio-root'
        it = iter_nemo(args.manifest, args.audio_root)
    else:
        assert args.manifest and args.tar_dirs, 'iscslp 需要 --manifest 和 --tar-dirs'
        it = iter_iscslp(args.manifest, args.tar_dirs)

    _log(f'初始化 tokenizer (device={args.device}) ...')
    tok = build_tokenizer(args.weights, args.device)
    writer = ShardWriter(args.out, args.lang, args.shard_records, args.shard_offset)

    stats = {'in': 0, 'skip_decode': 0, 'skip_dur': 0, 'skip_codes': 0, 'ok': 0}
    t0 = time.time()
    for uid, text, audio, dnsmos in it:
        stats['in'] += 1
        if args.limit and stats['ok'] >= args.limit:
            break
        wav = decode_audio(audio)
        if wav is None:
            stats['skip_decode'] += 1
            continue
        dur = wav.size / SR
        if dur < MIN_DURATION or dur > MAX_DURATION:
            stats['skip_dur'] += 1
            continue
        codes = torch_uint16_safe(tok(wav).squeeze(0))
        n = int(codes.size)
        if n < MIN_CODES or n > MAX_CODES:
            stats['skip_codes'] += 1
            continue
        writer.add(uid or f'u{stats["in"]}', text, codes, dur, dnsmos)
        stats['ok'] += 1
        if stats['ok'] % 500 == 0:
            rtf = (time.time() - t0) / max(stats['ok'] * 0.0, 1)
            _log(f'ok={stats["ok"]} in={stats["in"]} skip={stats["skip_decode"]+stats["skip_dur"]+stats["skip_codes"]} '
                 f'({stats["ok"]/max(stats["in"],1)*100:.1f}%) elapsed={time.time()-t0:.0f}s')
    writer.flush()
    stats['shards'] = writer.shard_idx
    stats['total_codes'] = writer.total_codes
    stats['elapsed_sec'] = round(time.time() - t0, 1)
    summary = os.path.join(args.out, f'_summary_{args.lang}_{os.getpid()}.json')
    with open(summary, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    _log(f'DONE {json.dumps(stats)}')


def torch_uint16_safe(x):
    """tensor -> np.uint16（值域校验）。"""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1).astype(np.int64)
    assert x.min() >= 0 and x.max() < N_CODES, f'codes 超出词表: [{x.min()},{x.max()}]'
    return x.astype(np.uint16)


def normalize_text(text: str) -> str:
    """yodas3 的 text 字段是字符串化的 list：[{'text': '...', ...}, ...]。"""
    t = (text or '').strip()
    if t.startswith('[') and t.endswith(']'):
        try:
            import ast
            v = ast.literal_eval(t)
            if isinstance(v, list) and v:
                parts = []
                for e in v:
                    if isinstance(e, dict) and e.get('text'):
                        parts.append(str(e['text']).strip())
                    elif isinstance(e, str):
                        parts.append(e.strip())
                return ' '.join(p for p in parts if p)
        except (ValueError, SyntaxError):
            pass
    return t


if __name__ == '__main__':
    main()
