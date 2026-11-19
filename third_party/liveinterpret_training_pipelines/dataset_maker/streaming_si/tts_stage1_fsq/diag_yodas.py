#!/usr/bin/env python3
"""yodas 解码诊断：抽样若干条 .webm，分别用 soundfile / ffmpeg 试解，统计失败原因。

用法：
    python3 diag_yodas.py --items 12
输出：每条的 声明时长 / soundfile 结果 / ffmpeg 结果 / 失败原因（ffmpeg stderr 尾部）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_to_stage1 as T  # noqa: E402

BUCKET = "stevehguo-ap-southeast-3"
REGION = "ap-southeast-3"


def ffmpeg_try(data: bytes, timeout: int = 300):
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--tar", default="")
    args = ap.parse_args()

    ctx = T.Ctx(argparse.Namespace(
        s3_in_bucket=BUCKET, s3_out_bucket=BUCKET,
        s3_in_prefix="speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS",
        region=REGION, tmp_dir="/tmp/tts_stage1", lang="jp",
        iscslp_movies="", lance_uri=T.DEFAULT_LANCE_URI, lance_version=74,
        hifitts_manifest="", hifitts_chapters="", meta_batch=512))

    keys = ctx.list_keys(f"{ctx.args.s3_in_prefix}/jp/audio/", ".tar")
    key = args.tar or keys[0]
    print(f"tar = {key}", flush=True)

    body = ctx.object_body(key)
    tf = tarfile.open(fileobj=body, mode="r|")
    stats = {"total": 0, "sf_ok": 0, "ff_ok": 0, "both_fail": 0}
    pend = {}
    n = 0
    for m in tf:
        if not m.isfile():
            continue
        stem, ext = os.path.splitext(m.name)
        if ext == ".json":
            try:
                pend[stem] = json.loads(tf.extractfile(m).read().decode("utf-8"))
            except Exception:
                pass
            continue
        if ext != ".webm":
            continue
        meta = pend.pop(stem, {}) or {}
        data = tf.extractfile(m).read()
        declared = float(meta.get("length", 0) or 0)
        n += 1
        stats["total"] += 1

        # 1) soundfile
        sf_ok, sf_dur, sf_err = False, 0.0, ""
        try:
            w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            sf_ok, sf_dur = True, len(w) / sr
        except Exception as e:
            sf_err = f"{type(e).__name__}: {str(e)[:80]}"
        if sf_ok:
            stats["sf_ok"] += 1

        # 2) ffmpeg（只在 soundfile 失败时跑，省时间）
        ff_ok, ff_dur, ff_err = False, 0.0, ""
        if not sf_ok:
            try:
                rc, out, err = ffmpeg_try(data)
                if rc == 0 and out:
                    w, sr = sf.read(io.BytesIO(out), dtype="float32", always_2d=True)
                    ff_ok, ff_dur = True, len(w) / sr
                else:
                    ff_err = err.strip().splitlines()[-1][:160] if err.strip() else f"rc={rc}"
            except subprocess.TimeoutExpired:
                ff_err = "ffmpeg timeout(300s)"
            except Exception as e:
                ff_err = f"{type(e).__name__}: {str(e)[:80]}"
            if ff_ok:
                stats["ff_ok"] += 1
        if not sf_ok and not ff_ok:
            stats["both_fail"] += 1

        segs = T.parse_transcript_segments(meta.get("transcript"))
        print(f"[{n:02d}] {os.path.basename(stem)[:12]} "
              f"size={len(data)/1e6:.1f}MB declared={declared:.0f}s segs={len(segs)} | "
              f"sf={'OK ' + format(sf_dur, '.0f') + 's' if sf_ok else 'FAIL ' + sf_err[:40]} | "
              f"ff={'OK ' + format(ff_dur, '.0f') + 's' if ff_ok else ('FAIL ' + ff_err[:60] if ff_err else '-')}",
              flush=True)
        if n >= args.items:
            break
    tf.close()
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
