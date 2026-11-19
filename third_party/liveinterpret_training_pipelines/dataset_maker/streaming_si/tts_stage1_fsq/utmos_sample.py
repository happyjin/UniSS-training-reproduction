#!/usr/bin/env python3
"""UTMOS 抽样评分：从各数据源重新抽 N 条 16k 音频 -> UTMOS22 分布。

parquet 里只有 FSQ token、没有波形，听感类指标（UTMOS/DNSMOS）必须回到音频。
这里复用 tts_to_stage1 的源适配器，按数据源各抽 `--per-source` 条（默认 200），
用 tarepan/SpeechMOS 的 utmos22_strong 打分，结果写 `_QA/utmos.json`，
由 qa_stage1.py 合并进报告。

用法：
    python3 utmos_sample.py --per-source 200 --device cuda:0
    python3 utmos_sample.py --per-source 20 --device cpu --sources emilia,yodas
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_to_stage1 as T  # noqa: E402

BUCKET = "stevehguo-ap-southeast-3"
UTMOS_REPO = os.environ.get("TTS_QUALITY_UTMOS_REPO", "tarepan/SpeechMOS:v1.2.0")

# (source, lang, s3 输入里的数据根, 抽取单元数)
SOURCES = [
    ("emilia", "zh"),
    ("yodas", "jp"),
    ("wenet", "zh"),
    ("iscslp", "zh"),
    ("iscslp", "en"),
    ("ko", "ko"),
    ("lance", "ja,ko"),
    ("hifitts", "en"),
]


def load_utmos(device: str):
    import torch
    model = torch.hub.load(UTMOS_REPO, "utmos22_strong", trust_repo=True, source="github")
    return model.to(device).eval(), torch


def score(model, torch, wavs, device, bs=16):
    """按长度排序后分批再打分。

    UTMOS 对尾部补零敏感：一批里长短混杂时，短样本被大量静音填充会显著拉低 MOS。
    先按长度排序、相邻长度凑批，可把 padding 降到最小。
    """
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    out = [None] * len(wavs)
    for i in range(0, len(order), bs):
        idxs = order[i:i + bs]
        chunk = [wavs[j] for j in idxs]
        max_len = max(len(w) for w in chunk)
        batch = np.zeros((len(chunk), max(max_len, 16000)), dtype=np.float32)
        for j, w in enumerate(chunk):
            batch[j, :len(w)] = w
        t = torch.from_numpy(batch).to(device)
        with torch.inference_mode():
            s = model(t, 16000).float().flatten().detach().cpu().tolist()
        for j, v in zip(idxs, s):
            out[j] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--units", type=int, default=2, help="每个源取前几个工作单元")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sources", default="", help="逗号分隔，默认全部")
    ap.add_argument("--out-s3-key",
                    default="speech_model_backup/stevehguo/data_fortrain/"
                            "data_fortrain_1031version/TTS_FSQ/_QA/utmos.json")
    args = ap.parse_args()

    try:
        model, torch = load_utmos(args.device)
    except Exception as e:
        print(f"[utmos] 模型加载失败，跳过 UTMOS: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return 0

    ctx = T.Ctx(argparse.Namespace(
        s3_in_bucket=BUCKET, s3_out_bucket=BUCKET,
        s3_in_prefix="speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS",
        region="ap-southeast-3", tmp_dir="/tmp/tts_stage1", lang="",
        iscslp_movies="movie-zh1,movie-zh2,movie-zh3,movie-zh4", lance_uri=T.DEFAULT_LANCE_URI,
        lance_version=74, hifitts_manifest="", hifitts_chapters="", meta_batch=512))

    wanted = [s for s in SOURCES if not args.sources or s[0] in args.sources.split(",")]
    result = {}
    for source, lang in wanted:
        ctx.args.lang = lang
        t0 = time.time()
        try:
            units = T.enumerate_units(ctx, source, 0, 1)[:args.units]
        except Exception as e:
            print(f"[{source}] 枚举单元失败: {e}", flush=True)
            continue
        wavs, texts = [], []
        for unit in units:
            try:
                for s in T.iter_unit(ctx, source, unit, 0, 1):
                    if len(wavs) >= args.per_source:
                        break
                    if s.get("wav") is not None and len(s["wav"]) >= 16000:
                        wavs.append(s["wav"])
                        texts.append(s.get("text", ""))
            except Exception as e:
                print(f"[{source}] 抽样中断: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if len(wavs) >= args.per_source:
                break
        if not wavs:
            print(f"[{source}] 没抽到音频", flush=True)
            continue
        try:
            scores = score(model, torch, wavs, args.device)
        except Exception as e:
            print(f"[{source}] 打分失败: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        scores = [float(x) for x in scores if x is not None]
        qs = statistics.quantiles(scores, n=10) if len(scores) >= 10 else []
        result[f"{source}:{lang}"] = {
            "n": len(scores), "mean": sum(scores) / len(scores),
            "min": min(scores), "max": max(scores),
            "p10": qs[0] if qs else None, "p50": qs[4] if qs else None,
            "p90": qs[8] if qs else None,
            "hist": {str(round(b / 2, 1)): sum(1 for x in scores if b <= x < b + 0.5)
                     for b in [i / 2 for i in range(0, 11)]},
        }
        print(f"[{source}:{lang}] n={len(scores)} mean={result[f'{source}:{lang}']['mean']:.2f} "
              f"p10={result[f'{source}:{lang}']['p10']} p50={result[f'{source}:{lang}']['p50']} "
              f"p90={result[f'{source}:{lang}']['p90']} ({(time.time()-t0)/60:.1f} min)", flush=True)

    ctx.s3.put_object(Bucket=BUCKET, Key=args.out_s3_key,
                      Body=json.dumps(result, indent=2).encode(),
                      ContentType="application/json")
    print(f"UTMOS 结果 -> s3://{BUCKET}/{args.out_s3_key}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
