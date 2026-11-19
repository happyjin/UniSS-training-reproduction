#!/usr/bin/env python3
"""按 QA 口径过滤 stage1 parquet：丢掉异常样本，写同 schema 的干净副本。

过滤规则（任一命中即丢弃）：
  * 空文本 / <2 字符 / 含控制字符或替换符
  * 目标语种字符占比 <30%
  * 字符速率不在 [1,30] 字/秒，或 token 速率明显偏离 25Hz
  * 最长同值连续 token 占比 >=60%（疑似静音/恒定段）
  * token 数 <5 或达到 750 上限、时长 <0.2s 或 >=29.5s
  * 同一文件内重复文本

用法：
    python3 filter_stage1.py --src-prefix .../TTS_FSQ --dst-prefix .../TTS_FSQ_FILTERED
    python3 filter_stage1.py --shard 0 --nshard 8      # 多 Pod 并行
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unicodedata
from multiprocessing import Pool

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_stage1 import CHAR_RATE_MAX, CHAR_RATE_MIN, LONG_SEC, MAX_SPEECH_TOKENS, script_ratio  # noqa: E402

BUCKET = "stevehguo-ap-southeast-3"
REGION = "ap-southeast-3"
SILENCE_RUN_RATIO = 0.6
SCHEMA = pa.schema([
    ("id", pa.string()), ("text", pa.string()), ("language", pa.string()),
    ("speaker", pa.string()), ("duration", pa.float32()), ("dnsmos", pa.float32()),
    ("answer_speech_tokens", pa.list_(pa.list_(pa.int32()))),
    ("conversations", pa.string()),
])


def longest_run_ratio(tokens) -> float:
    if len(tokens) < 10:
        return 0.0
    best = run = 1
    prev = tokens[0]
    for v in tokens[1:]:
        run = run + 1 if v == prev else 1
        prev = v
        best = max(best, run)
    return best / len(tokens)


def keep(text, lang, dur, tokens) -> bool:
    if not text or not str(text).strip():
        return False
    s = str(text)
    if len(s.strip()) < 2:
        return False
    if any(unicodedata.category(c) == "Cc" for c in s) or "�" in s:
        return False
    if script_ratio(s, str(lang).lower()) < 0.3:
        return False
    d = max(float(dur or 0), 1e-6)
    if not (CHAR_RATE_MIN <= len(s.strip()) / d <= CHAR_RATE_MAX):
        return False
    n = len(tokens)
    if n < 5 or n >= MAX_SPEECH_TOKENS:
        return False
    if d < 0.2 or d >= LONG_SEC:
        return False
    if not (10 <= n / d <= 60):
        return False
    if longest_run_ratio(tokens) >= SILENCE_RUN_RATIO:
        return False
    return True


def process(key: str, src_prefix: str, dst_prefix: str) -> dict:
    s3 = boto3.client("s3", region_name=REGION)
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    t = pq.read_table(io.BytesIO(body))
    texts = t.column("text").to_pylist()
    langs = t.column("language").to_pylist()
    durs = t.column("duration").to_pylist()
    toks = t.column("answer_speech_tokens").to_pylist()
    ids = t.column("id").to_pylist()
    spk = t.column("speaker").to_pylist() if "speaker" in t.column_names else [""] * len(ids)
    dn = t.column("dnsmos").to_pylist() if "dnsmos" in t.column_names else [0.0] * len(ids)
    conv = t.column("conversations").to_pylist()

    sel, seen = [], set()
    for i, (tx, lg, du, tk) in enumerate(zip(texts, langs, durs, toks)):
        tokens = tk[0] if tk else []
        if not keep(tx, lg, du, tokens):
            continue
        h = hash(str(tx)[:200])
        if h in seen:
            continue
        seen.add(h)
        sel.append(i)
    if not sel:
        return {"key": key, "kept": 0, "total": len(ids), "skip": True}

    out = pa.table({
        "id": [ids[i] for i in sel],
        "text": [texts[i] for i in sel],
        "language": [langs[i] for i in sel],
        "speaker": [str(spk[i] or "") for i in sel],
        "duration": pa.array([float(durs[i]) for i in sel], pa.float32()),
        "dnsmos": pa.array([float(dn[i] or 0.0) for i in sel], pa.float32()),
        "answer_speech_tokens": pa.array([[toks[i][0]] for i in sel],
                                         pa.list_(pa.list_(pa.int32()))),
        "conversations": [conv[i] for i in sel],
    }, schema=SCHEMA)
    buf = io.BytesIO()
    pq.write_table(out, buf)
    dst = key.replace(src_prefix.rstrip("/"), dst_prefix.rstrip("/"), 1)
    s3.put_object(Bucket=BUCKET, Key=dst, Body=buf.getvalue(),
                  ContentType="application/octet-stream")
    return {"key": key, "kept": len(sel), "total": len(ids), "skip": False}


_SRC = _DST = ""


def _init_pool(src: str, dst: str) -> None:
    global _SRC, _DST
    _SRC, _DST = src, dst


def _work(key: str) -> dict:
    return process(key, _SRC, _DST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-prefix", default="speech_model_backup/stevehguo/data_fortrain/"
                                            "data_fortrain_1031version/TTS_FSQ")
    ap.add_argument("--dst-prefix", default="speech_model_backup/stevehguo/data_fortrain/"
                                            "data_fortrain_1031version/TTS_FSQ_FILTERED")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--name-filter", default="",
                    help="只处理名字里含该子串的文件（如 fsq-EN- ，用于合并已有的 emilia_fsq_v3 英文）")
    args = ap.parse_args()

    # ---- 目标路径防呆：路径段重复（如 TTS_FSQ_FILTERED_FILTERED）直接拒绝启动 ----
    # 历史事故：dst 写成双重 _FILTERED 前缀，脚本不报错照写，错误副本与正确副本
    # 长期共存，清理时又删错了副本，导致同一问题反复出现。这里硬校验杜绝复发。
    src, dst = args.src_prefix.rstrip("/"), args.dst_prefix.rstrip("/")
    for p in (src, dst):
        segs = [s for s in p.split("/") if s]
        if len(segs) != len(set(segs)):
            dup = [s for i, s in enumerate(segs) if s in segs[:i]]
            raise SystemExit(f"[abort] 路径存在重复段 {dup}: {p}\n"
                             f"  src={src}\n  dst={dst}")
        for seg in segs:  # 段内 token 重复，如 TTS_FSQ_FILTERED_FILTERED
            toks = [t for t in seg.split("_") if t]
            if len(toks) != len(set(toks)):
                raise SystemExit(f"[abort] 路径段内疑似拼接事故 '{seg}': {p}\n"
                                 f"  src={src}\n  dst={dst}")
        if "FILTERED_FILTERED" in p:
            raise SystemExit(f"[abort] 双重 FILTERED 前缀: {p}")
    if dst == src or dst.startswith(src + "/") or src.startswith(dst + "/"):
        raise SystemExit(f"[abort] src/dst 前缀重叠会读写同一处:\n  src={src}\n  dst={dst}")
    print(f"[filter] dst = s3://{BUCKET}/{dst}/", flush=True)

    s3 = boto3.client("s3", region_name=REGION)
    keys, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": args.src_prefix.rstrip("/") + "/"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])
             if o["Key"].endswith(".parquet")
             and (not args.name_filter or args.name_filter in o["Key"])]
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    keys = sorted(keys)
    keys = [k for i, k in enumerate(keys) if i % args.nshard == args.shard]
    if args.max_files:
        keys = keys[:args.max_files]
    print(f"[filter shard {args.shard}/{args.nshard}] {len(keys)} 个文件", flush=True)

    kept = total = 0
    t0 = time.time()
    with Pool(processes=args.workers, initializer=_init_pool,
              initargs=(args.src_prefix, args.dst_prefix)) as pool:
        for res in pool.imap_unordered(_work, keys, chunksize=1):
            kept += res["kept"]
            total += res["total"]
    dropped = total - kept
    stat = {"shard": args.shard, "files": len(keys), "rows_in": total, "rows_out": kept,
            "dropped": dropped, "drop_ratio": round(dropped / max(total, 1), 4),
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    s3.put_object(Bucket=BUCKET,
                  Key=f"{args.dst_prefix}/_FILTER_STAT/shard-{args.shard:03d}.json",
                  Body=json.dumps(stat).encode(), ContentType="application/json")
    print(f"[filter shard {args.shard}] 保留 {kept}/{total}（丢弃 {dropped}，"
          f"{stat['drop_ratio']*100:.2f}%）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
