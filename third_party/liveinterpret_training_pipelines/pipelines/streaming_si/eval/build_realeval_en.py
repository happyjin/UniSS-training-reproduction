#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实场景英文小评测集（realeval_en_v1）构建。

用途：验证 KIC 电竞解说类失败（快语速 / 长段 / 即兴 / 带背景声）是个例还是稳定退化。
来源全部取公开数据集的 **test 划分**（不与训练集 train 划分重叠）：

    gigaspeech   播客 / YouTube 即兴语音        hf-audio/esb-datasets-test-only-sorted
    earnings22   财报电话会，口音、电话信道      同上
    ami          会议（近讲麦），多人即兴        同上
    voxpopuli    欧洲议会即兴发言，语速快        同上
    urgent2024   合成带噪语音（自带 snr_db）     同上
    peoples      People's Speech，多样真实录音   MLCommons/peoples_speech (clean/test)
    kic          KIC 电竞解说（本地 test_realdata，1 条锚点）

选样规则：时长 10–30s；参考转写有效词数 >= 20；按语速（词/秒）降序；
同一录音（会议 / 节目 / 电话 / 议会会场）最多取 1 条，保证说话人和场景多样。

输出：<out>/audio/<id>.wav（16k 单声道）+ <out>/manifest.jsonl
manifest 字段：id, category, source, source_id, duration, ref_en, n_words, wps, [snr_db]

用法：
    python build_realeval_en.py --out /fsx/.../realeval_en_v1 --kic /fsx/.../KIC_背景音.wav \
        --kic-ref "..."
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from fsspec.implementations.http import HTTPFileSystem

ESB = "hf-audio/esb-datasets-test-only-sorted"
PEOPLE = "MLCommons/peoples_speech"
SR = 16000
PLAN = {          # 来源 -> (条数, 场景说明)
    "gigaspeech": (9, "播客/YouTube 即兴"),
    "earnings22": (8, "财报电话会（口音/电话信道）"),
    "ami": (6, "会议近讲麦（多人即兴）"),
    "voxpopuli": (6, "议会即兴发言（快语速）"),
    "urgent2024": (6, "合成带噪（背景声）"),
    "peoples": (6, "多样真实录音"),
}
MIN_SEC, MAX_SEC, MIN_WORDS = 10.0, 30.0, 20
fs = HTTPFileSystem(block_size=1 << 22)


def hf_files(repo: str, prefix: str):
    j = requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=30).json()
    return sorted(s["rfilename"] for s in j["siblings"]
                  if s["rfilename"].startswith(prefix) and s["rfilename"].endswith(".parquet"))


def url(repo, fn):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{fn}"


_TAG = re.compile(r"<[^>]+>|\[[^\]]+\]|\{[^}]+\}")


def words(text: str):
    t = _TAG.sub(" ", text or "")
    return [w for w in re.findall(r"[A-Za-z0-9']+", t)]


def recording_key(source: str, sid: str) -> str:
    """同一录音只取 1 条。"""
    if source == "gigaspeech":
        return sid.split("_")[0]                      # POD0000000xxx / YOU... 节目
    if source == "earnings22":
        return sid.split("/")[0]                      # 电话会编号
    if source == "ami":
        return sid.split("_")[1]                      # 会议号 EN2002c
    if source == "voxpopuli":
        return sid.split("_")[0]                      # 会场
    if source == "peoples":
        return sid.split("/")[0]
    return sid


def candidates(source: str):
    """返回 [(fn, row_idx, sid, dur, text, extra)]，只读轻列。"""
    if source == "peoples":
        repo, fl = PEOPLE, hf_files(PEOPLE, "clean/test")
    else:
        repo, fl = ESB, hf_files(ESB, source + "/")
    out = []
    for fn in fl:
        with fs.open(url(repo, fn), "rb") as f:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            t = pf.read(columns=[c for c in names if c != "audio"]).to_pydict()
        n = len(t["id"])
        durs = t["audio_length_s"] if "audio_length_s" in t else [x / 1000 for x in t["duration_ms"]]
        for i in range(n):
            d = float(durs[i])
            if not MIN_SEC <= d <= MAX_SEC:
                continue
            w = words(t["text"][i])
            if len(w) < MIN_WORDS:
                continue
            extra = {"snr_db": round(float(t["snr_db"][i]), 1)} if "snr_db" in t else {}
            out.append((repo, fn, i, str(t["id"][i]), d, t["text"][i], len(w), extra))
    return out


def pick(source: str, k: int):
    cand = candidates(source)
    cand.sort(key=lambda c: -(c[6] / c[4]))           # 语速降序
    seen, sel = set(), []
    for c in cand:
        key = recording_key(source, c[3])
        if key in seen:
            continue
        seen.add(key)
        sel.append(c)
        if len(sel) == k:
            break
    print(f"[{source}] 候选 {len(cand)}，取 {len(sel)}；语速 "
          f"{sel[0][6]/sel[0][4]:.2f}~{sel[-1][6]/sel[-1][4]:.2f} 词/秒", flush=True)
    return sel


def load_audio(repo, fn, rows):
    with fs.open(url(repo, fn), "rb") as f:
        col = pq.ParquetFile(f).read(columns=["audio"]).column("audio").to_pylist()
    out = {}
    for i in rows:
        a = col[i]
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
        out[i] = (wav.mean(axis=1), sr)
    return out


def to16k(wav, sr):
    if sr == SR:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=sr, target_sr=SR)


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kic", help="KIC 电竞解说 wav（锚点样本）")
    ap.add_argument("--kic-ref", default="", help="KIC 参考转写")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    rows = []
    for source, (k, desc) in PLAN.items():
        sel = pick(source, k)
        by_file = {}
        for c in sel:
            by_file.setdefault((c[0], c[1]), []).append(c)
        for (repo, fn), cs in by_file.items():
            audio = load_audio(repo, fn, [c[2] for c in cs])
            for c in cs:
                wav, sr = audio[c[2]]
                wav = to16k(wav, sr)
                rid = f"{source}__{safe(c[3])}"
                sf.write(os.path.join(a.out, "audio", rid + ".wav"), wav, SR, subtype="PCM_16")
                rows.append(dict(id=rid, category=desc, source=source, source_id=c[3],
                                 duration=round(len(wav) / SR, 2), ref_en=c[5].strip(), n_words=c[6],
                                 wps=round(c[6] / (len(wav) / SR), 2), **c[7]))
    if a.kic:
        wav, sr = sf.read(a.kic, dtype="float32", always_2d=True)
        wav = to16k(wav.mean(axis=1), sr)
        sf.write(os.path.join(a.out, "audio", "kic__KIC_2.wav"), wav, SR, subtype="PCM_16")
        n = len(words(a.kic_ref))
        rows.append(dict(id="kic__KIC_2", category="电竞解说（游戏音效背景）", source="kic",
                         source_id="test_realdata/KIC_背景音.wav", duration=round(len(wav) / SR, 2),
                         ref_en=a.kic_ref, n_words=n, wps=round(n / (len(wav) / SR), 2)))
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {len(rows)} 条 -> {a.out}/manifest.jsonl；总时长 {sum(r['duration'] for r in rows)/60:.1f} 分钟")


if __name__ == "__main__":
    sys.exit(main())
