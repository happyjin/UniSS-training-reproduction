#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同语言对（en2en/zh2zh/ja2ja/ko2ko）P2 训练数据构造。

从 s2st lance 表的翻译行采样**源音频**构造同语言对样本：
    输入 = 源音频；目标文本 = 源转写（source_transcript）；目标音频 = 源音频本身
    （target_fsq = 源音频的 FSQ 编码，tgt_lang = src_lang，duration_ratio = 1.0）。

产出目录结构与 s2st_lance_to_stage2.py 完全一致（训练侧零改动即可混用）：
    <OUT>/<L2L>/p2_fsq_v3/fsq-train-<L2L>-%05d.parquet
    <OUT>/<L2L>/p2_source_wav/train-<L2L>-%05d.tar

采样口径（避免「每方向独立抽 10%」的两个坑）：
    - 按**语言池**采样：src_lang = L 的全部翻译行（tgt 为四语其他语言）构成池；
      en/zh 是 3 个方向的源、ja/ko 只有 2 个，按方向抽会让 en2en/zh2zh 天然多 1.5 倍。
    - 按 **source_sample_id 去重**：同一源音频在表里平均被翻译 2~3 次
      （实测 en 池 9.04M 行 / 3.98M 源），不去重会导致同语言桶内同一音频反复出现。
    - 目标量 = frac × 池行数（= 「每个语言对抽 10%」的原始量级），从去重池里
      按 md5(sample_id) 排序确定性取前 target 个（均匀且可复现）。

两阶段（与 s2st_lance_to_stage2.py 同构）：
    manifest  扫轻列，产出采样后的 pos/addr 分片清单（每语言一个子目录）
    convert   按分片取源音频 blob、FSQ 编码、写 parquet + tar、上传 S3

用法：
    python same_lang_to_stage2.py manifest --lang en --frac 0.1 \
        --out /nvme/samelang_manifest --nshards 24
    python same_lang_to_stage2.py convert --lang en --shard 0 --nshards 24 \
        --manifest /nvme/samelang_manifest \
        --out-prefix s3://.../Speech2Speech --tmp-dir /nvme/samelang/w0
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time

import numpy as np

# 复用 s2st 造数脚本的常量与工具函数（同目录或 /opt/samelang 下）
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, "/opt/samelang"):
    if os.path.isdir(_p) and os.path.exists(os.path.join(_p, "s2st_lance_to_stage2.py")):
        sys.path.insert(0, _p)
        break
import s2st_lance_to_stage2 as base  # noqa: E402

LANCE_URI = base.LANCE_URI
SR = base.SR
MIN_SPEECH_TOKENS = base.MIN_SPEECH_TOKENS
MAX_SPEECH_TOKENS = base.MAX_SPEECH_TOKENS
LANG_CODE = base.LANG_CODE
PAIR_LANGS = base.PAIR_LANGS

# 四语全部变体（目标侧只要属于四语即可）
FOUR_LANGS = tuple(v for vs in PAIR_LANGS.values() for v in vs)

# 采样时对源音频的轻量预筛（convert 侧仍按 token 数 [5,750] 硬筛）
MIN_DUR_MS = 200
MAX_DUR_MS = 30_000


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def pair_name(lang: str) -> str:
    return "%s2%s" % (lang, lang)


# ---------------------------------------------------------------------------- manifest
def cmd_manifest(a):
    import lance
    import pyarrow as pa

    ds = lance.dataset(a.lance_uri, version=a.version,
                       storage_options={"region": a.region})
    pools = {}          # lang -> {sample_id: (pos, addr)}
    rows_cnt = {}       # lang -> 池行数（去重前）
    n_scan = 0
    base_pos = 0
    log("扫描轻列 version=%d，同语言池=%s，frac=%.3f" % (a.version, a.lang, a.frac))
    cols = ["src_lang", "tgt_lang", "source_sample_id", "duration_ms",
            "source_transcript", "_curion_contribution_id", "dataset"]
    # 一次扫描同时构建全部 4 个语言池（IO 只花一遍）
    want = [a.lang] if a.lang != "all" else list(PAIR_LANGS)
    for w in want:
        pools[w] = {}
        rows_cnt[w] = 0
    for b in ds.scanner(columns=cols, with_row_id=True, batch_size=1 << 16).to_batches():
        tb = pa.Table.from_batches([b])
        n_scan += tb.num_rows
        sl = tb["src_lang"].combine_chunks().to_pylist()
        tl = tb["tgt_lang"].combine_chunks().to_pylist()
        sid = tb["source_sample_id"].combine_chunks().to_pylist()
        dm = tb["duration_ms"].combine_chunks().to_pylist()
        st = tb["source_transcript"].combine_chunks().to_pylist()
        cti = tb["_curion_contribution_id"].combine_chunks().to_pylist()
        dsn = tb["dataset"].combine_chunks().to_pylist()
        addrs = tb["_rowid"].combine_chunks().to_numpy(zero_copy_only=False)
        for i in range(tb.num_rows):
            s = sl[i]
            key = None
            if s == "zh":
                key = "zh"
            elif s == "ja":
                key = "ja"
            elif s == "ko":
                key = "ko"
            elif s == "en":
                key = "en"
            if key is None or key not in pools:
                continue
            t = tl[i]
            if t == s or t not in FOUR_LANGS:
                continue
            d = dm[i]
            if d is None or d < MIN_DUR_MS or d > MAX_DUR_MS:
                continue
            txt = st[i]
            if not txt or len(str(txt).strip()) < 2:
                continue
            c = str(cti[i] or "").lower()
            if ("test" in c or "validation" in c or "dev" in c
                    or dsn[i] == "mini_librispeech"):
                continue
            rows_cnt[key] += 1
            sid_i = sid[i]
            k = str(sid_i) if sid_i is not None else ("__null__%d" % (base_pos + i))
            pool = pools[key]
            if k not in pool:                     # 同源多目标只留首条
                pool[k] = (base_pos + i, int(addrs[i]))
        base_pos += tb.num_rows
        if n_scan % 10_000_000 < (1 << 16):
            log("  scanned %dM" % (n_scan // 1_000_000))

    outroot = a.out
    os.makedirs(outroot, exist_ok=True)
    summary = {}
    for lang in want:
        pool = pools[lang]
        rows = rows_cnt[lang]
        target = int(round(rows * a.frac))
        # md5 排序取前 target 个：确定性 + 均匀（与行序/扫描顺序无关）
        ranked = sorted(pool.items(), key=lambda kv: hashlib.md5(kv[0].encode()).digest())
        picked = ranked[:target]
        picked.sort(key=lambda kv: kv[1][0])      # 恢复 pos 顺序，保 take 局部性
        allpos = np.array([v[0] for _, v in picked], dtype=np.int64)
        alladdr = np.array([v[1] for _, v in picked], dtype=np.uint64)
        d = os.path.join(outroot, pair_name(lang))
        os.makedirs(d, exist_ok=True)
        for i in range(a.nshards):
            np.save(os.path.join(d, "shard_%04d.pos.npy" % i), allpos[i::a.nshards])
            np.save(os.path.join(d, "shard_%04d.addr.npy" % i), alladdr[i::a.nshards])
        meta = dict(lang=lang, frac=a.frac, lance_uri=a.lance_uri, version=a.version,
                    pool_rows=rows, dedup=len(pool), sampled=len(picked),
                    nshards=a.nshards, min_dur_ms=MIN_DUR_MS, max_dur_ms=MAX_DUR_MS,
                    exclude_devtest=True, created=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        summary[lang] = meta
        log("%s: 池 %s 行 / 去重 %s 源 -> 采样 %d（%d 分片）"
            % (pair_name(lang), f"{rows:,}", f"{len(pool):,}", len(picked), a.nshards))
    with open(os.path.join(outroot, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log("DONE manifest，扫描 %d 行" % n_scan)


# ----------------------------------------------------------------------------- convert
def cmd_convert(a):
    import lance
    import pyarrow as pa

    for p in ("/opt/tts_stage1_fsq", a.fsq_dir):
        if p and os.path.isdir(p):
            sys.path.insert(0, p)
    import fsq_encoder

    tag = pair_name(a.lang)
    mdir = os.path.join(a.manifest, tag)
    pos_all = np.load(os.path.join(mdir, "shard_%04d.pos.npy" % a.shard))
    addr_all = np.load(os.path.join(mdir, "shard_%04d.addr.npy" % a.shard))
    if a.limit:
        pos_all, addr_all = pos_all[:a.limit], addr_all[:a.limit]
    log("%s 分片 %d/%d：%d 行" % (tag, a.shard, a.nshards, len(pos_all)))
    if len(pos_all) == 0:
        log("空分片，退出")
        return

    ds = lance.dataset(a.lance_uri, version=a.version,
                       storage_options={"region": a.region})
    tok = base._fetch_tokenizer(os.path.join(a.tmp_dir, "speech_tokenizer_v3.batch.onnx"))
    sess = fsq_encoder.build_session(tok, a.device, require_cuda=True)

    os.makedirs(a.tmp_dir, exist_ok=True)
    tar_path = os.path.join(a.tmp_dir, "train-%s-%05d.tar" % (tag, a.shard))
    rows, n_ok, n_fail = [], 0, 0
    t0 = time.time()

    with tarfile.open(tar_path, "w") as tf:
        for start in range(0, len(pos_all), a.chunk):
            pchunk = pos_all[start:start + a.chunk].tolist()
            achunk = addr_all[start:start + a.chunk].tolist()
            tb = ds.take(pchunk, columns=base.META_COLS)
            # 同语言对读**源**音频：目标音频就是它本身
            blobs = ds.read_blobs("source_audio_payload", ids=achunk, preserve_order=True)
            recs = tb.to_pylist()
            # 先解源语音（CPU 串行）
            wavs, meta = [], []
            for i, (r, rid) in enumerate(zip(recs, pchunk)):
                try:
                    w = base._decode(blobs[i][1])
                    wavs.append(w)
                    meta.append((r, rid))
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 3:
                        log("  源音频解码失败 idx=%d: %s" % (rid, str(e)[:80]))
                    continue
            # 全量 GPU FSQ 编码（源音频没有现成 codes，broadcastspeech 的
            # target_codec_codes 是目标侧的，不能复用）
            codes_all = [None] * len(wavs)
            for s in range(0, len(wavs), a.bs):
                idxs = list(range(s, min(s + a.bs, len(wavs))))
                batch = [wavs[i] for i in idxs]
                try:
                    toks = fsq_encoder.wavs_to_fsq_tokens(sess, batch, device=a.device)
                except Exception as e:
                    log("  FSQ 编码失败一批(%d): %s" % (len(idxs), str(e)[:80]))
                    toks = [None] * len(idxs)
                for i, t in zip(idxs, toks):
                    codes_all[i] = t
            for (r, rid), w, codes in zip(meta, wavs, codes_all):
                if not codes or len(codes) < MIN_SPEECH_TOKENS or len(codes) > MAX_SPEECH_TOKENS:
                    n_fail += 1
                    continue
                try:
                    data = base._flac_bytes(w)
                    off = tf.offset + tarfile.BLOCKSIZE
                    name = "%d.flac" % rid          # 分片内唯一即可（同 base 口径）
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                except Exception:
                    n_fail += 1
                    continue
                rows.append(dict(
                    id=r["id"],
                    dataset_name=r["dataset"],
                    source_transcript=r["source_transcript"] or "",
                    # 同语言对：目标文本 = 源转写
                    translation=r["source_transcript"] or "",
                    src_lang=LANG_CODE.get(str(r["src_lang"]), str(r["src_lang"])),
                    tgt_lang=LANG_CODE.get(str(r["src_lang"]), str(r["src_lang"])),
                    duration_ratio=1.0,
                    # 目标语音 = 源语音的 FSQ
                    target_fsq=[int(x) for x in codes],
                    source_wav="%s::%d:%d" % (os.path.basename(tar_path), off, len(data)),
                    target_wav="",
                ))
                n_ok += 1
            if (start // a.chunk) % 10 == 0:
                log("  %d/%d  成功 %d  失败 %d  %.0fs"
                    % (start + len(pchunk), len(pos_all), n_ok, n_fail, time.time() - t0))

    schema = pa.schema([("id", pa.string()), ("dataset_name", pa.string()),
                        ("source_transcript", pa.string()), ("translation", pa.string()),
                        ("src_lang", pa.string()), ("tgt_lang", pa.string()),
                        ("duration_ratio", pa.float32()),
                        ("target_fsq", pa.list_(pa.int32())),
                        ("source_wav", pa.string()), ("target_wav", pa.string())])
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq_path = os.path.join(a.tmp_dir, "fsq-train-%s-%05d.parquet" % (tag, a.shard))
    import pyarrow.parquet as pq
    pq.write_table(tbl, pq_path)
    log("写出 %d 行 -> %s（%.1f MB），tar %.1f MB，用时 %.0fs"
        % (tbl.num_rows, pq_path, os.path.getsize(pq_path) / 1e6,
           os.path.getsize(tar_path) / 1e6, time.time() - t0))

    if a.out_prefix:
        b = "%s/%s" % (a.out_prefix.rstrip("/"), tag)
        for local, sub in ((pq_path, "p2_fsq_v3"), (tar_path, "p2_source_wav")):
            dst = "%s/%s/%s" % (b, sub, os.path.basename(local))
            subprocess.check_call(["aws", "s3", "cp", local, dst, "--only-show-errors"])
            log("已上传 -> %s" % dst)
        for local in (pq_path, tar_path):
            try:
                os.remove(local)
            except OSError:
                pass
    log("DONE shard=%d ok=%d fail=%d" % (a.shard, n_ok, n_fail))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["manifest", "convert"])
    p.add_argument("--lance-uri", default=LANCE_URI)
    p.add_argument("--version", type=int, default=239)
    p.add_argument("--region", default="ap-southeast-3")
    p.add_argument("--lang", default="all",
                   help="en/zh/ja/ko 或 all（manifest 一次构建全部 4 池）")
    p.add_argument("--frac", type=float, default=0.1)
    p.add_argument("--out")                     # manifest 输出目录
    p.add_argument("--manifest")                # convert 读的清单目录
    p.add_argument("--nshards", type=int, default=24)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--out-prefix", default="")
    p.add_argument("--tmp-dir", default="/nvme/samelang_stage2")
    p.add_argument("--fsq-dir", default="/opt/tts_stage1_fsq")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    (cmd_manifest if a.cmd == "manifest" else cmd_convert)(a)


if __name__ == "__main__":
    main()
