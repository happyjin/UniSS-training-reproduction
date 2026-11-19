"""输入输出辅助：tsv / jsonl / 目录扫描 -> (key, audio, text) 样本列表。"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterator, List, Tuple


def read_tsv_map(tsv_path: str) -> Dict[str, str]:
    """读取 tsv：第一列 key，第二列文本。返回 {key: text}。"""
    out: Dict[str, str] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                out[parts[0]] = "\t".join(parts[1:])
    return out


def read_tsv_transcript(tsv_path: str, key: str) -> str:
    m = read_tsv_map(tsv_path)
    if key not in m:
        raise SystemExit(f"[错误] 在 {tsv_path} 中找不到 key={key}")
    return m[key]


def key_from_audio(path: str) -> str:
    """由音频文件名推导 key：去扩展名，并去掉常见的 _dual 后缀。"""
    k = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_dual", "_pred", "_gen", "_out"):
        if k.endswith(suf):
            k = k[: -len(suf)]
            break
    return k


def iter_list_file(path: str) -> Iterator[dict]:
    """支持 jsonl（含 audio 字段）或 tsv（audio\\ttext）。"""
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                audio = obj.get("audio") or obj.get("wav") or obj.get("path")
                text = obj.get("transcript") or obj.get("text") or obj.get("tgt") or ""
                key = obj.get("key") or key_from_audio(audio)
                yield {"key": key, "audio": audio, "text": text}
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                audio = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                yield {"key": key_from_audio(audio), "audio": audio, "text": text}


def pairs_from_dir(audio_dir: str, tsv_map: Dict[str, str],
                   pattern: str = "*.wav") -> List[dict]:
    """扫描目录下所有音频，用 key 到 tsv_map 里找对应译文。"""
    out = []
    for path in sorted(glob.glob(os.path.join(audio_dir, pattern))):
        key = key_from_audio(path)
        text = tsv_map.get(key, "")
        out.append({"key": key, "audio": path, "text": text})
    return out


def build_pairs(audio_dir: str = None, list_file: str = None,
                tsv: str = None, limit: int = None) -> List[dict]:
    """统一构造样本列表，三选一：
        1) list_file          : jsonl / tsv，每行自带 audio 与 text
        2) audio_dir + tsv    : 扫目录，用文件名 key 到 tsv 取译文
        3) audio_dir          : 扫目录，无译文（只出 VAD 统计）
    """
    if list_file:
        pairs = list(iter_list_file(list_file))
    elif audio_dir:
        tsv_map = read_tsv_map(tsv) if tsv else {}
        pairs = pairs_from_dir(audio_dir, tsv_map)
    else:
        raise ValueError("需要 --list 或 --audio-dir")

    pairs = [p for p in pairs if p.get("audio")]
    if limit:
        pairs = pairs[:limit]
    return pairs
