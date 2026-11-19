#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""curated S2ST Lance -> streaming_si stage2（P2）训练数据。

产出与参考目录 `open_data_0803_fortrain/machine_*_shard/` 同构，只写训练真正用到的两样：
    <OUT>/p2_fsq_v3/fsq-train-%05d.parquet    每行 = 一条 S2ST 样本
    <OUT>/p2_source_wav/train-%05d.tar       源语音，parquet 里以 `train-000NN.tar::offset:length` 引用

parquet 列（与 machine_1_shard/p2_fsq_v3 逐列一致）：
    id, dataset_name, source_transcript, translation, src_lang, tgt_lang,
    duration_ratio, target_fsq(list<int32>), source_wav, target_wav

target_fsq 必须是 **cosy3 FSQ：25Hz、单码本、词表 6561**（si_sampler 强制 codec_spec=cosy3_fsq_25hz）。
curated 表里只有 broadcastspeech 贡献的 target_codec_codes 是这个口径（实测 25.1Hz / 1 codebook /
max_code 6558），其余贡献是 12.5Hz × 16 codebook 的另一套 codec，必须用本仓库的
`speech_tokenizer_v3.batch.onnx` 从 target_audio_payload 重新编码。

两阶段：
    manifest  扫轻列，按过滤条件产出全局 rowid 清单（每分片一个 npy）
    convert   按分片取 blob、编码、写 parquet + tar，上传 S3

用法：
    python s2st_lance_to_stage2.py manifest --out /fsx/.../manifest --nshards 96
    python s2st_lance_to_stage2.py convert --shard 0 --manifest /fsx/.../manifest \
        --out-prefix s3://stevehguo-ap-southeast-3/.../Speech2Speech --tmp-dir /nvme/s2st
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LANCE_URI = ("s3://iegg-speech-model-data-124355679795-ap-southeast-3-an/"
             "data/audio/curated/task=s2st/data.lance/")
DEFAULT_FSQ_TOKENIZER_S3 = ("s3://stevehguo-ap-southeast-3/speech_model_backup/stevehguo/"
                            "model_checkpoint/cosyvoice3/speech_tokenizer_v3.batch.onnx")

LANG4 = {"zh", "en", "ja", "ko", "cmn"}
# stage2  instruction 用的语言码（si_encode.LANG_NAME）
LANG_CODE = {"zh": "cmn", "cmn": "cmn", "en": "eng", "ja": "jpn", "ko": "kor"}

MAX_SPEECH_TOKENS = 750      # 25Hz -> 30s，与 talker 的 max_speech_len 一致
MIN_SPEECH_TOKENS = 5
SR = 16000

META_COLS = ["id", "dataset", "_curion_contribution_id", "src_lang", "tgt_lang",
             "source_transcript", "target_text", "duration_ms", "target_duration_ms"]
LIGHT_COLS = ["src_lang", "tgt_lang", "dataset", "_curion_contribution_id",
              "target_audio_whisper_wer", "target_audio_whisper_cer"]
# --no-err-filter（S2 宽筛）模式需要额外读 target_text 做显式文本体检
LIGHT_COLS_TEXT = LIGHT_COLS + ["target_text"]
# 中日文本无空格，按空白分词的 WER 会退化（reference 只有 1~2 个 token，值变成 0/1.0/1.5…），
# 必须改用字符级 CER；英韩有空格，WER 正常。实测见 verify_wer*.log。
UNSPACED_TGT = ("zh", "cmn", "ja", "jpn")
# target_text 显式体检：空/纯占位符的行对任何阶段都不可用（S2 的 translation 就是它）。
# 实测 ko2zh 池 CER>0.3 的 2.43M 行里空文本 18.7K + 占位符(如 "(())") 9.3K。
PLACEHOLDER_CHARS = set("()[]{}（） ._-~*#")
# 字幕标注词（去掉括号后只剩这些 = 无实际译文）
CAPTION_TAGS = {"", "噪音", "噪声", "笑声", "掌声", "音乐", "无字幕", "静音", "unknown"}

# 语言对目录名（zh2en 等）-> (源语言取值集合, 目标语言取值集合)
# 表里同一语言的写法不统一（zh/cmn、ja/jpn、ko/kor、en/eng），这里全部接受。
PAIR_LANGS = {
    "zh": ("zh", "cmn", "zho"),
    "en": ("en", "eng"),
    "ja": ("ja", "jpn"),
    "ko": ("ko", "kor"),
}


def pair_keys(pair: str):
    """'zh2en' -> ('zh', 'en')；非法返回 (None, None)。"""
    if not pair or "2" not in pair:
        return None, None
    s, t = pair.split("2", 1)
    if s not in PAIR_LANGS or t not in PAIR_LANGS:
        return None, None
    return s, t


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------- manifest
def text_ok(t):
    """S2 宽筛模式的 target_text 体检：非空、非纯占位符/字幕标注。极短正常文本保留。"""
    if t is None:
        return False
    s = str(t).strip()
    if len(s) < 1:
        return False
    if "(())" in s:
        return False
    if set(s) <= PLACEHOLDER_CHARS:
        return False
    if s.strip("()[]{}（） ._-~*#") in CAPTION_TAGS:
        return False
    return True


def cmd_manifest(a):
    import lance
    ds = lance.dataset(a.lance_uri, version=a.version,
                       storage_options={"region": a.region})
    langs = pa.array(sorted(LANG4))
    devtest_pat = ("test", "validation", "dev")
    pos, addr = [], []
    n_scan = 0
    n_allow = 0
    log("扫描轻列，version=%d  langs=%s  err_metric=%s  err_max=%s  drop_err_null=%s  no_err_filter=%s"
        % (a.version, ",".join(sorted(LANG4)), a.err_metric, a.err_max, a.drop_err_null, a.no_err_filter))
    base = 0
    src_key, tgt_key = pair_keys(a.pair)
    if src_key:
        langs_s = pa.array(list(PAIR_LANGS[src_key]))
        langs_t = pa.array(list(PAIR_LANGS[tgt_key]))
    use_cols = LIGHT_COLS_TEXT if a.no_err_filter else LIGHT_COLS
    for b in ds.scanner(columns=use_cols, with_row_id=True, batch_size=1 << 16).to_batches():
        tb = pa.Table.from_batches([b])
        n_scan += tb.num_rows
        sl, tl = tb["src_lang"], tb["tgt_lang"]
        if src_key:
            m = pc.and_(pc.is_in(sl, value_set=langs_s), pc.is_in(tl, value_set=langs_t))
        else:
            m = pc.and_(pc.is_in(sl, value_set=langs), pc.is_in(tl, value_set=langs))
        if a.exclude_devtest:
            cti = pc.utf8_lower(tb["_curion_contribution_id"].combine_chunks())
            bad = pc.match_substring(cti, devtest_pat[0])
            for p in devtest_pat[1:]:
                bad = pc.or_(bad, pc.match_substring(cti, p))
            bad = pc.or_(bad, pc.equal(tb["dataset"].combine_chunks(), "mini_librispeech"))
            m = pc.and_(m, pc.invert(pc.fill_null(bad, False)))
        if a.no_err_filter:
            # S2 宽筛：跳过 target_audio 的 WER/CER（S2 不消费目标音频），改做
            # target_text 显式体检——空/占位符文本对 S2 的 translation 监督同样不可用。
            texts = tb["target_text"].combine_chunks().to_pylist() \
                if isinstance(tb["target_text"], pa.ChunkedArray) else tb["target_text"].to_pylist()
            ok_txt = np.array([text_ok(t) for t in texts], dtype=bool)
            sel_np = m.combine_chunks().to_numpy(zero_copy_only=False).astype(bool) \
                if isinstance(m, pa.ChunkedArray) else m.to_numpy(zero_copy_only=False).astype(bool)
            sel_np = sel_np & ok_txt
            if sel_np.any():
                pos.append(np.arange(base, base + tb.num_rows, dtype=np.int64)[sel_np])
                addr.append(tb["_rowid"].combine_chunks().to_numpy(zero_copy_only=False)[sel_np])
            base += tb.num_rows
            continue
        if a.err_max is not None:
            if a.err_metric == "wer":
                err = tb["target_audio_whisper_wer"]
            elif a.err_metric == "cer":
                err = tb["target_audio_whisper_cer"]
            else:  # auto：中日目标用 CER，其余用 WER
                tgt_n = pc.if_else(pc.equal(tb["tgt_lang"], "cmn"),
                                   pa.scalar("zh"), tb["tgt_lang"])
                unspaced = pa.array(UNSPACED_TGT)
                err = pc.if_else(pc.is_in(tgt_n, value_set=unspaced),
                                 tb["target_audio_whisper_cer"],
                                 tb["target_audio_whisper_wer"])
            # 关键：先把缺失值填成常数，让后续布尔运算全程无 null。
            # 直接 and_(is_valid(err), less_equal(err,..)) 时 null 行得到的是 null 而非 False，
            # 后面的 or_ 会被它污染，导致白名单整支静默失效（实测踩过）。
            if a.drop_err_null:
                ok_e = pc.less_equal(pc.fill_null(err, 9.0), a.err_max)   # 缺失 -> 当作很差
            else:
                ok_e = pc.less_equal(pc.fill_null(err, 0.0), a.err_max)   # 缺失 -> 当作很好
            # 白名单：指标整支缺失、但已抽样实测合格的来源，允许保留
            for item in (a.allow_null or []):
                if ":" in item:
                    dname, lang = item.split(":", 1)
                else:
                    dname, lang = item, None
                cond = pc.equal(tb["dataset"].combine_chunks(), dname)
                if lang:
                    cond = pc.and_(cond, pc.is_in(
                        pc.if_else(pc.equal(tb["tgt_lang"].combine_chunks(), "cmn"),
                                   pa.scalar("zh"), tb["tgt_lang"].combine_chunks()),
                        value_set=pa.array([lang])))
                allow = pc.and_(pc.is_null(err), pc.fill_null(cond, False))
                ok_e = pc.or_(ok_e, allow)
                n_allow += int(pc.sum(pc.cast(allow, pa.int64())).as_py())
            m = pc.and_(m, pc.fill_null(ok_e, False))
        # ds.take() 要「版本内位置下标」；read_blobs() 要「_rowid 行地址」。两个都要存。
        sel = m.combine_chunks().to_numpy(zero_copy_only=False).astype(bool)
        if sel.any():
            pos.append(np.arange(base, base + tb.num_rows, dtype=np.int64)[sel])
            addr.append(tb["_rowid"].combine_chunks().to_numpy(zero_copy_only=False)[sel])
        base += tb.num_rows
    allpos = np.concatenate(pos) if pos else np.zeros(0, dtype=np.int64)
    alladdr = np.concatenate(addr) if addr else np.zeros(0, dtype=np.uint64)
    outdir = os.path.join(a.out, a.pair) if a.pair else a.out
    os.makedirs(outdir, exist_ok=True)
    for i in range(a.nshards):
        np.save(os.path.join(outdir, "shard_%04d.pos.npy" % i), allpos[i::a.nshards])
        np.save(os.path.join(outdir, "shard_%04d.addr.npy" % i), alladdr[i::a.nshards])
    meta = dict(pair=a.pair, lance_uri=a.lance_uri, version=a.version, nshards=a.nshards,
                scanned=int(n_scan), selected=int(allpos.size),
                langs=sorted(LANG4), err_metric=a.err_metric, err_max=a.err_max,
                drop_err_null=a.drop_err_null, exclude_devtest=a.exclude_devtest,
                no_err_filter=a.no_err_filter, suffix=a.suffix,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("扫描 %d 行 -> 选中 %d 行，写入 %d 个分片清单：%s"
        % (n_scan, allpos.size, a.nshards, a.out))


# ----------------------------------------------------------------------------- convert
def _decode(payload: bytes, target_sr: int = SR):
    """音频字节 -> (float32 ndarray, sr)，重采样到 target_sr。"""
    import soundfile as sf
    try:
        wav, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except Exception:
        # 容器格式 libsndfile 不认时退回 ffmpeg
        p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0",
                            "-f", "wav", "-ar", str(target_sr), "-ac", "1", "pipe:1"],
                           input=payload, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError("ffmpeg 解码失败")
        wav, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=True)
    if wav.shape[1] > 1:
        wav = wav.mean(axis=1)
    else:
        wav = wav[:, 0]
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _flac_bytes(wav: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _fetch_tokenizer(dst: str) -> str:
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    log("下载 FSQ tokenizer -> %s" % dst)
    subprocess.check_call(["aws", "s3", "cp", DEFAULT_FSQ_TOKENIZER_S3, dst,
                           "--only-show-errors"])
    return dst


def cmd_convert(a):
    import lance
    # 多进程共卡的 onnx 显存上限（GB）：参数 > 环境变量 > <tmp-dir 的上两级>/FSQ_GPU_MEM_GB 文件
    # （文件可在 worker 运行期间修改，下一个分片的进程即生效）
    if a.fsq_gpu_mem_gb:
        os.environ["FSQ_GPU_MEM_GB"] = str(a.fsq_gpu_mem_gb)
    elif not os.environ.get("FSQ_GPU_MEM_GB"):
        f = os.path.join(os.path.dirname(os.path.dirname(a.tmp_dir.rstrip("/"))), "FSQ_GPU_MEM_GB")
        if os.path.isfile(f):
            v = open(f).read().strip()
            if v:
                os.environ["FSQ_GPU_MEM_GB"] = v
    if os.environ.get("FSQ_GPU_MEM_GB"):
        log("onnx 显存上限 %s GB" % os.environ["FSQ_GPU_MEM_GB"])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # fsq_encoder.py 由 ConfigMap tts-stage1-fsq 提供（S1 管线同一份：16k -> 25Hz cosy3 FSQ）
    if os.path.isdir("/opt/tts_stage1_fsq"):
        sys.path.insert(0, "/opt/tts_stage1_fsq")
    for p in (a.fsq_dir,):
        if p and os.path.isdir(p):
            sys.path.insert(0, p)
    import fsq_encoder

    mdir = os.path.join(a.manifest, a.pair) if a.pair else a.manifest
    pos_all = np.load(os.path.join(mdir, "shard_%04d.pos.npy" % a.shard))
    addr_all = np.load(os.path.join(mdir, "shard_%04d.addr.npy" % a.shard))
    tag = (a.pair or "") + (a.suffix or "")
    if a.limit:
        pos_all, addr_all = pos_all[:a.limit], addr_all[:a.limit]
    log("分片 %d/%d：%d 行" % (a.shard, a.nshards, len(pos_all)))
    if len(pos_all) == 0:
        log("空分片，退出")
        return

    ds = lance.dataset(a.lance_uri, version=a.version,
                       storage_options={"region": a.region})

    # 复用旧版（严筛）已编码的 target_fsq：build_reuse.py 按新 manifest 分片重排到
    # <reuse_dir>/<pair>/shard=<s>/*.parquet（列 id, target_fsq，已校验码值/长度）。
    # 命中的行不读目标音频 blob、不解码、不过 GPU。
    reuse = {}
    if a.reuse_dir:
        rdir = os.path.join(a.reuse_dir, a.pair, "shard=%d" % a.shard)
        if os.path.isdir(rdir):
            for fn in sorted(os.listdir(rdir)):
                if not fn.endswith(".parquet"):
                    continue
                rt = pq.read_table(os.path.join(rdir, fn), columns=["id", "target_fsq"])
                col = rt["target_fsq"].combine_chunks()
                vals = pc.list_flatten(col).to_numpy(zero_copy_only=False)
                lens = pc.list_value_length(col).to_numpy(zero_copy_only=False)
                starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else []
                for idv, s0, ln in zip(rt["id"].to_pylist(), starts, lens):
                    reuse[idv] = vals[s0:s0 + ln]
        log("复用旧 FSQ：本分片可用 %d 条" % len(reuse))
    n_reuse = 0

    tok = _fetch_tokenizer(os.path.join(a.tmp_dir, "speech_tokenizer_v3.batch.onnx"))
    # 一个 Pod 多卡时，由调用方用 CUDA_VISIBLE_DEVICES 把每个进程绑到不同 GPU，这里统一写 cuda:0
    sess = fsq_encoder.build_session(tok, a.device, require_cuda=True)

    os.makedirs(a.tmp_dir, exist_ok=True)
    tar_path = os.path.join(a.tmp_dir, ("train-%s-%05d.tar" % (tag, a.shard)) if tag
                            else ("train-%05d.tar" % a.shard))
    rows, n_ok, n_fail = [], 0, 0
    t0 = time.time()

    with tarfile.open(tar_path, "w") as tf:
        for start in range(0, len(pos_all), a.chunk):
            pchunk = pos_all[start:start + a.chunk].tolist()
            achunk = addr_all[start:start + a.chunk].tolist()
            tb = ds.take(pchunk, columns=META_COLS)
            recs = tb.to_pylist()
            # blob 列必须走 read_blobs：take/scanner 只给描述符
            src_blobs = ds.read_blobs("source_audio_payload", ids=achunk, preserve_order=True)
            # 复用命中的行不读目标音频
            reuse_hit = {}
            for p, r in zip(pchunk, recs):
                c = reuse.get(r["id"])
                if c is not None:
                    reuse_hit[p] = [int(x) for x in c]
            need_addr = [ad for p, ad in zip(pchunk, achunk) if p not in reuse_hit]
            got = ds.read_blobs("target_audio_payload", ids=need_addr, preserve_order=True) \
                if need_addr else []
            tgt_by_pos, j = {}, 0
            for p in pchunk:
                if p not in reuse_hit:
                    tgt_by_pos[p] = got[j]
                    j += 1
            # broadcastspeech 的 codes 已是 cosy3 25Hz，直接取，不必重编码
            # 2026-10-28：lance v239 实测 broadcastspeech 的 target_codec_codes 也已不是
            # cosy3 口径（ko2zh 重生成出现 16383 越界），改为全部行强制 GPU 重编码。
            bs_pos = [] if os.environ.get('SI_FORCE_REENCODE') else \
                [p for p, r in zip(pchunk, recs) if r["dataset"] == "broadcastspeech"]
            codes_map = {}
            if bs_pos:
                ctb = ds.take(bs_pos, columns=["target_codec_codes"])
                for p, c in zip(bs_pos, ctb["target_codec_codes"].to_pylist()):
                    if c and c[0]:
                        codes_map[p] = [int(x) for fr in c
                                        for x in (fr if isinstance(fr, list) else [fr])]
            # 先解源语音（CPU 串行），再批量编码目标语音（GPU）
            src_wavs, meta = [], []
            for i, (r, rid) in enumerate(zip(recs, pchunk)):
                try:
                    w = _decode(src_blobs[i][1])
                    src_wavs.append(w)
                    meta.append((r, rid, i))
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 3:
                        log("  源音频解码失败 idx=%d: %s" % (rid, str(e)[:80]))
                    continue
            tgt_wavs, use_codes = [], []
            for (r, rid, i) in meta:
                if rid in reuse_hit:
                    use_codes.append(reuse_hit[rid])
                    tgt_wavs.append(None)
                    n_reuse += 1
                elif rid in codes_map:
                    use_codes.append(codes_map[rid])
                    tgt_wavs.append(None)
                else:
                    use_codes.append(None)
                    try:
                        tgt_wavs.append(_decode(tgt_by_pos[rid][1]))
                    except Exception:
                        tgt_wavs.append(None)   # 目标解码失败：codes 留 None，下面计入失败
            need = [i for i, x in enumerate(tgt_wavs) if x is not None]
            if need:
                for s in range(0, len(need), a.bs):
                    idxs = need[s:s + a.bs]
                    batch = [tgt_wavs[i] for i in idxs]
                    try:
                        toks = fsq_encoder.wavs_to_fsq_tokens(sess, batch, device=a.device)
                    except Exception as e:
                        # 整批失败时逐条重试，只丢真正有问题的那条
                        log("  FSQ 编码失败一批(%d)，逐条重试: %s" % (len(idxs), str(e)[:80]))
                        if fsq_encoder._is_oom(e):
                            import torch
                            torch.cuda.empty_cache()
                        toks = []
                        for w1 in batch:
                            try:
                                toks.append(fsq_encoder.wavs_to_fsq_tokens(sess, [w1], device=a.device)[0])
                            except Exception:
                                toks.append(None)
                    for i, t in zip(idxs, toks):
                        use_codes[i] = t
            for (r, rid, _i), w, codes in zip(meta, src_wavs, use_codes):
                if codes is None or len(codes) < MIN_SPEECH_TOKENS or len(codes) > MAX_SPEECH_TOKENS:
                    n_fail += 1
                    continue
                try:
                    data = _flac_bytes(w)
                    # tarfile 每次 addfile 先写 512 字节头，数据从 tf.offset+512 开始
                    off = tf.offset + tarfile.BLOCKSIZE
                    name = "%d.flac" % rid
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                except Exception as e:
                    n_fail += 1
                    continue
                sd = r["duration_ms"] or 0
                td = r["target_duration_ms"] or 0
                rows.append(dict(
                    id=r["id"],
                    dataset_name=r["dataset"],
                    source_transcript=r["source_transcript"] or "",
                    translation=r["target_text"] or "",
                    src_lang=LANG_CODE.get(str(r["src_lang"]), str(r["src_lang"])),
                    tgt_lang=LANG_CODE.get(str(r["tgt_lang"]), str(r["tgt_lang"])),
                    duration_ratio=(td / sd) if sd else 0.0,
                    target_fsq=[int(x) for x in codes],
                    source_wav="%s::%d:%d" % (os.path.basename(tar_path), off, len(data)),
                    target_wav="",
                ))
                n_ok += 1
            if (start // a.chunk) % 10 == 0:
                log("  %d/%d  成功 %d（复用 %d）  失败 %d  %.0fs"
                    % (start + len(pchunk), len(pos_all), n_ok, n_reuse, n_fail, time.time() - t0))

    schema = pa.schema([("id", pa.string()), ("dataset_name", pa.string()),
                        ("source_transcript", pa.string()), ("translation", pa.string()),
                        ("src_lang", pa.string()), ("tgt_lang", pa.string()),
                        ("duration_ratio", pa.float32()),
                        ("target_fsq", pa.list_(pa.int32())),
                        ("source_wav", pa.string()), ("target_wav", pa.string())])
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq_path = os.path.join(a.tmp_dir, ("fsq-train-%s-%05d.parquet" % (tag, a.shard)) if tag
                           else ("fsq-train-%05d.parquet" % a.shard))
    pq.write_table(tbl, pq_path)
    log("写出 %d 行 -> %s（%.1f MB），tar %.1f MB，用时 %.0fs"
        % (tbl.num_rows, pq_path, os.path.getsize(pq_path) / 1e6,
           os.path.getsize(tar_path) / 1e6, time.time() - t0))

    if a.out_prefix:
        base = a.out_prefix.rstrip("/")
        if a.pair:                      # 按语言对分目录：<base>/<pair>/p2_fsq_v3|p2_source_wav
            base = "%s/%s" % (base, a.pair)
        for local, sub in ((pq_path, "p2_fsq_v3"), (tar_path, "p2_source_wav")):
            dst = "%s/%s/%s" % (base, sub, os.path.basename(local))
            subprocess.check_call(["aws", "s3", "cp", local, dst, "--only-show-errors"])
            log("已上传 -> %s" % dst)
        for local in (pq_path, tar_path):   # 上传完立刻删本地，控制 /nvme 占用
            try:
                os.remove(local)
            except OSError:
                pass
    log("DONE shard=%d ok=%d reuse=%d fail=%d" % (a.shard, n_ok, n_reuse, n_fail))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["manifest", "convert"])
    p.add_argument("--lance-uri", default=LANCE_URI)
    p.add_argument("--version", type=int, default=239)
    p.add_argument("--region", default="ap-southeast-3")
    p.add_argument("--out")                     # manifest 输出目录
    p.add_argument("--manifest")                # convert 读的清单目录
    p.add_argument("--pair", default="",
                   help="语言对目录名，如 zh2en；manifest 按对存放，产物写到 <out-prefix>/<pair>/")
    p.add_argument("--nshards", type=int, default=96)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--err-max", type=float, default=0.3)
    p.add_argument("--no-err-filter", action="store_true", default=False,
                   help="S2 宽筛：跳过 target_audio 的 WER/CER 筛选（S2 不消费目标音频），"
                        "改做 target_text 显式体检（丢空/占位符文本）。"
                        "产物文件名需配 --suffix 与严筛版区分，避免 ktrain 合并目录时重名。")
    p.add_argument("--suffix", default="",
                   help="产物文件名后缀（如 s2w -> fsq-train-ko2zh-s2w-00013.parquet），"
                        "仅影响文件名，输出目录仍由 --pair 与 --out-prefix 决定")
    p.add_argument("--err-metric", choices=["auto", "wer", "cer"], default="auto",
                   help="auto=中日目标用 CER、英韩用 WER（推荐）")
    p.add_argument("--wer-max", type=float, default=None, help="同 --err-max，旧参数名")
    p.add_argument("--drop-err-null", action="store_true", default=True)
    p.add_argument("--keep-err-null", dest="drop_err_null", action="store_false")
    p.add_argument("--allow-null", action="append", default=None,
                   help="指标缺失但已实测合格的来源白名单，格式 dataset[:tgt_lang]，可重复")
    p.add_argument("--exclude-devtest", action="store_true", default=True)
    p.add_argument("--keep-devtest", dest="exclude_devtest", action="store_false")
    p.add_argument("--out-prefix", default="")
    p.add_argument("--tmp-dir", default="/nvme/s2st_stage2")
    p.add_argument("--fsq-dir", default="/opt/tts_stage1_fsq",
                   help="fsq_encoder.py 所在目录（ConfigMap 挂载点）")
    p.add_argument("--fsq-gpu-mem-gb", type=float, default=0,
                   help="onnx CUDA 显存池上限（GB），多进程共卡时设；0=读环境变量/文件，都没有则不限")
    p.add_argument("--reuse-dir", default="",
                   help="build_reuse.py 产出的旧 FSQ 复用目录（<dir>/<pair>/shard=<s>/）；空=不复用")
    p.add_argument("--device", default="cuda:0",
                   help="onnxruntime 设备；多卡 Pod 里配合 CUDA_VISIBLE_DEVICES 使用")
    a = p.parse_args()
    if a.wer_max is not None:      # 兼容旧参数名
        a.err_max = a.wer_max
    (cmd_manifest if a.cmd == "manifest" else cmd_convert)(a)


if __name__ == "__main__":
    main()
