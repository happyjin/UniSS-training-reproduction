#!/usr/bin/env python3
"""stage1（S1）TTS 数据集质量评估报告。

扫描 TTS_FSQ 下全部 parquet，按「语言 / 数据源」给出规模、时长、token 分布、
疑似空音频、超长音频、文本异常等指标，产出 JSON + Markdown 报告并回传 S3。

只依赖 parquet 里已有的列（不重新读音频），因此可以对全量数据做统计；
需要听感的指标（UTMOS/DNSMOS）属于另一类，见 --utmos（可选，需音频）。

用法：
    python3 qa_stage1.py --s3-prefix .../TTS_FSQ --out-s3-key .../TTS_FSQ/_QA/report.md
    python3 qa_stage1.py --wait-stable 20    # 先等产出稳定（单位：分钟）再跑
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import boto3
import pyarrow.parquet as pq

BUCKET = "stevehguo-ap-southeast-3"
REGION = "ap-southeast-3"

MAX_SPEECH_TOKENS = 750       # 与造数口径一致
TOKEN_RATE = 25.0             # FSQ 帧率
LONG_SEC = 29.5               # >= 视为临界 30s
SILENCE_RUN_RATIO = 0.6       # 最长同值连续段占比 >= 该值 -> 疑似静音/恒定
CHAR_RATE_MIN, CHAR_RATE_MAX = 1.0, 30.0   # 字符/秒 合理区间

SCRIPT_RANGES = {
    "zh": lambda ch: "一" <= ch <= "鿿",
    "ja": lambda ch: ("぀" <= ch <= "ヿ") or ("一" <= ch <= "鿿"),
    "ko": lambda ch: ("가" <= ch <= "힯") or ("ᄀ" <= ch <= "ᇿ"),
    "en": lambda ch: ("a" <= ch <= "z") or ("A" <= ch <= "Z"),
}


def hist_pct(hist: Counter, p: float, scale: float) -> float:
    """从直方图算分位（全量数据不落内存，千万级行数也能跑）。"""
    total = sum(hist.values())
    if not total:
        return 0.0
    target, cum = p / 100 * total, 0
    for bucket in sorted(hist):
        cum += hist[bucket]
        if cum >= target:
            return bucket * scale
    return max(hist) * scale


def script_ratio(text: str, lang: str) -> float:
    fn = SCRIPT_RANGES.get(lang)
    if not fn:
        return 1.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if fn(c)) / len(chars)


def analyse_key(key: str) -> dict:
    """统计单个 parquet（子进程执行）。"""
    s3 = boto3.client("s3", region_name=REGION)
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    t = pq.read_table(io.BytesIO(body))
    cols = set(t.column_names)
    texts = t.column("text").to_pylist()
    langs = t.column("language").to_pylist()
    durs = t.column("duration").to_pylist()
    toks = t.column("answer_speech_tokens").to_pylist()
    convs = t.column("conversations").to_pylist() if "conversations" in cols else [""] * len(texts)

    out = defaultdict(lambda: {
        "files": 0, "rows": 0, "seconds": 0.0, "tokens": 0,
        "dur_hist": Counter(), "tok_hist": Counter(),
        "empty_text": 0, "short_text": 0, "long_text": 0,
        "script_mismatch": 0, "ctrl_char": 0, "replacement_char": 0,
        "bad_char_rate": 0, "bad_token_rate": 0, "silent_like": 0,
        "long_audio": 0, "capped": 0, "min_tokens": 0, "bad_conv": 0,
        "dup_text": 0, "text_hashes": [],
    })
    # 按行级语言 + 来源（文件名前缀）分组：lance 的 parquet 是 ja/ko 混装的，
    # 按文件多数语言打标签会把 9,512h 日语藏进 ko 行，必须逐行用 language 列。
    src_key = os.path.basename(key).split("-s")[0]
    seen = set()
    touched = set()
    for text, lang, dur, tk, conv in zip(texts, langs, durs, toks, convs):
        gkey = f"{str(lang)}|{src_key}"
        g = out[gkey]
        if gkey not in touched:
            touched.add(gkey)
            g["files"] += 1
        n = len(tk[0]) if tk else 0
        g["rows"] += 1
        g["seconds"] += float(dur or 0)
        g["tokens"] += n
        g["dur_hist"][min(int(float(dur or 0) * 2), 120)] += 1   # 0.5s 分桶
        g["tok_hist"][min(n // 10, 75)] += 1                     # 10 token 分桶

        if not text or not str(text).strip():
            g["empty_text"] += 1
            continue
        s = str(text)
        ln = len(s.strip())
        if ln < 2:
            g["short_text"] += 1
        if ln > 200:
            g["long_text"] += 1
        if script_ratio(s, str(lang).lower()) < 0.3:
            g["script_mismatch"] += 1
        if any(unicodedata.category(c) == "Cc" for c in s):
            g["ctrl_char"] += 1
        if "�" in s:
            g["replacement_char"] += 1
        rate = ln / max(float(dur or 0), 1e-6)
        if rate < CHAR_RATE_MIN or rate > CHAR_RATE_MAX:
            g["bad_char_rate"] += 1
        tr = n / max(float(dur or 0), 1e-6)
        if tr < 10 or tr > 60:
            g["bad_token_rate"] += 1
        if n < 5:
            g["min_tokens"] += 1
        if float(dur or 0) >= LONG_SEC:
            g["long_audio"] += 1
        if n >= MAX_SPEECH_TOKENS:
            g["capped"] += 1
        try:
            c = json.loads(conv)
            if not c or c[-1].get("role") != "assistant":
                g["bad_conv"] += 1
        except Exception:
            g["bad_conv"] += 1
        # 静音代理：最长同值连续 token 占比过高
        if n >= 10:
            best = run = 1
            prev = tk[0][0]
            for v in tk[0][1:]:
                run = run + 1 if v == prev else 1
                prev = v
                best = max(best, run)
            if best / n >= SILENCE_RUN_RATIO:
                g["silent_like"] += 1
        h = hash(s[:200])
        if h in seen:
            g["dup_text"] += 1
        else:
            seen.add(h)
    return {"key": key, "groups": {k: {kk: vv for kk, vv in v.items() if kk != "text_hashes"}
                                   for k, v in out.items()}}


def merge(acc, res):
    for lang, g in res["groups"].items():
        a = acc.setdefault(lang, {
            "files": 0, "rows": 0, "seconds": 0.0, "tokens": 0,
            "dur_hist": Counter(), "tok_hist": Counter(),
            "empty_text": 0, "short_text": 0, "long_text": 0,
            "script_mismatch": 0, "ctrl_char": 0, "replacement_char": 0,
            "bad_char_rate": 0, "bad_token_rate": 0, "silent_like": 0,
            "long_audio": 0, "capped": 0, "min_tokens": 0, "bad_conv": 0,
            "dup_text": 0,
        })
        a["files"] += g["files"]
        a["rows"] += g["rows"]
        a["seconds"] += g["seconds"]
        a["tokens"] += g["tokens"]
        a["dur_hist"].update(g["dur_hist"])
        a["tok_hist"].update(g["tok_hist"])
        for k in ("empty_text", "short_text", "long_text", "script_mismatch", "ctrl_char",
                  "replacement_char", "bad_char_rate", "bad_token_rate", "silent_like",
                  "long_audio", "capped", "min_tokens", "bad_conv", "dup_text"):
            a[k] += g[k]
    return acc


def render_utmos(s3, out_prefix):
    """合并 utmos_sample.py 产出的抽样评分（不存在则跳过）。"""
    try:
        body = s3.get_object(Bucket=BUCKET, Key=f"{out_prefix}/utmos.json")["Body"].read()
        data = json.loads(body)
    except Exception:
        return []
    if not data:
        return []
    lines = ["", "## UTMOS 分布（每源抽样，UTMOS22 strong）", "",
             "| 源:语言 | 样本数 | 均值 | p10 | p50 | p90 | min | max | <2.5 占比 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in sorted(data.items()):
        low = sum(c for b, c in v.get("hist", {}).items() if float(b) < 2.5)
        n = v.get("n") or 1
        lines.append(f"| {k} | {v['n']} | {v['mean']:.2f} | {v['p10']} | {v['p50']} | "
                     f"{v['p90']} | {v['min']:.2f} | {v['max']:.2f} | {low/n*100:.1f}% |")
    lines += ["", "- UTMOS 为听感预测 MOS（1–5），一般 TTS 训练数据建议均值 ≥3.0、"
                  "过滤 <2.5 的低质样本；抽样统计仅用于判断分布，不等于全量。", ""]
    return lines


def render_md(acc, prefix, n_files, s3=None, out_prefix=""):
    dh_all = Counter()
    for g in acc.values():
        dh_all.update(g["dur_hist"])
    lines = [
        "# stage1（S1）TTS 数据集质量评估报告",
        "",
        f"- 数据源：`s3://{BUCKET}/{prefix}/`",
        f"- parquet 文件数：**{n_files}**",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- 说明：造数阶段已过滤 token<5 与 >30s 的样本，故「空音频/超长」应接近 0；"
        f"非 0 说明过滤逻辑未生效或有边界样本",
        "",
        "## 总览",
        "",
        "| 语言/来源 | 文件 | 行数 | 时长(h) | token 数 | 平均时长(s) | 时长p50/p90/p99 | token p50/p90/p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang_src, g in sorted(acc.items(), key=lambda x: -x[1]["rows"]):
        lang, src = lang_src.split("|", 1)
        lines.append(
            f"| {lang} / {src} | {g['files']} | {g['rows']:,} | {g['seconds']/3600:,.1f} | "
            f"{g['tokens']:,} | {g['seconds']/max(g['rows'],1):.2f} | "
            f"{hist_pct(g['dur_hist'],50,0.5):.1f}/{hist_pct(g['dur_hist'],90,0.5):.1f}"
            f"/{hist_pct(g['dur_hist'],99,0.5):.1f} | "
            f"{hist_pct(g['tok_hist'],50,10):.0f}/{hist_pct(g['tok_hist'],90,10):.0f}"
            f"/{hist_pct(g['tok_hist'],99,10):.0f} |")
    lines += ["", "## 异常指标（占该语言行数比例）", "",
              "| 语言/来源 | 空/极短文本 | 超长文本(>200字) | 语种字符不符 | 控制字符 | 替换符 | "
              "字符速率异常 | token速率异常 | 疑似静音 | ≥29.5s | 达750上限 | 重复文本 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lang_src, g in sorted(acc.items(), key=lambda x: -x[1]["rows"]):
        lang, src = lang_src.split("|", 1)
        r = g["rows"] or 1
        p = lambda k: f"{g[k]/r*100:.2f}%"
        lines.append(f"| {lang} / {src} | {p('empty_text')} | {p('long_text')} | {p('script_mismatch')} | "
                      f"{p('ctrl_char')} | {p('replacement_char')} | {p('bad_char_rate')} | "
                      f"{p('bad_token_rate')} | {p('silent_like')} | {p('long_audio')} | "
                      f"{p('capped')} | {p('dup_text')} |")
    lines += ["", "## 判读要点", "",
              f"- **疑似静音**：单条里最长连续相同 token 占比 ≥{SILENCE_RUN_RATIO:.0%}"
              "（近似静音/恒定段，无声学内容）",
              f"- **字符速率异常**：文本字数/秒 不在 [{CHAR_RATE_MIN}, {CHAR_RATE_MAX}] 区间"
              "（文本与音频不匹配、或转写缺失）",
              f"- **token 速率异常**：token/秒 明显偏离 {TOKEN_RATE}Hz 帧率",
              "- **语种字符不符**：目标语种字符占比 <30%（错语种或转写错乱）",
              "- **重复文本**：同一文本在同一文件内重复出现（去重不彻底）",
              "",
              f"全量时长分位：p50={hist_pct(dh_all,50,0.5):.1f}s "
              f"p90={hist_pct(dh_all,90,0.5):.1f}s p99={hist_pct(dh_all,99,0.5):.1f}s",
              ""]
    if s3 is not None:
        lines += render_utmos(s3, out_prefix)
    return "\n".join(lines)


def wait_stable(s3, prefix, minutes, max_wait_min=60):
    prefix = prefix.rstrip("/") + "/"
    """等产出稳定：连续 minutes 分钟文件数不再变化。

    必须带上限：hifitts 这类超长任务会持续写文件，若无限等"稳定"会把整条链路卡死。
    超过 max_wait_min 就按当前快照出报告（报告可以之后再重跑）。
    """
    last, stable_since, t0 = -1, time.time(), time.time()
    while True:
        n = sum(1 for _ in paginate(s3, prefix))
        now = time.time()
        if n != last:
            last, stable_since = n, now
        print(f"[wait] 文件数={n} 稳定 {(now-stable_since)/60:.1f} min "
              f"（最多再等 {(max_wait_min-(now-t0)/60):.0f} min）", flush=True)
        if n > 0 and (now - stable_since) / 60 >= minutes:
            return n
        if (now - t0) / 60 >= max_wait_min:
            print(f"[wait] 已达上限 {max_wait_min} min，按当前快照出报告", flush=True)
            return n
        time.sleep(60)


def paginate(s3, prefix):
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            if o["Key"].endswith(".parquet"):
                yield o["Key"]
        if not r.get("IsTruncated"):
            return
        token = r["NextContinuationToken"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3-prefix",
                    default="speech_model_backup/stevehguo/data_fortrain/"
                            "data_fortrain_1031version/TTS_FSQ")
    ap.add_argument("--out-s3-prefix", default="", help="默认 <s3-prefix>/_QA")
    ap.add_argument("--wait-stable", type=int, default=0, help="先等文件数稳定的分钟数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    n_files = 0
    if args.wait_stable:
        n_files = wait_stable(s3, args.s3_prefix, args.wait_stable)

    keys = sorted(paginate(s3, args.s3_prefix.rstrip("/") + "/"))
    if args.max_files:
        keys = keys[:args.max_files]
    print(f"待分析 {len(keys)} 个 parquet", flush=True)

    acc = {}
    done = 0
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(analyse_key, keys, chunksize=1):
            acc = merge(acc, res)
            done += 1
            if done % 100 == 0:
                print(f"  已分析 {done}/{len(keys)}", flush=True)

    out_prefix = args.out_s3_prefix or f"{args.s3_prefix}/_QA"
    md = render_md(acc, args.s3_prefix, len(keys), s3=s3, out_prefix=out_prefix)
    summary = {
        lang: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in g.items()}
        for lang, g in acc.items()
    }
    s3.put_object(Bucket=BUCKET, Key=f"{out_prefix}/report.md",
                  Body=md.encode(), ContentType="text/markdown")
    s3.put_object(Bucket=BUCKET, Key=f"{out_prefix}/report.json",
                  Body=json.dumps(summary, indent=2).encode(), ContentType="application/json")
    print(md)
    print(f"\n报告已写入 s3://{BUCKET}/{out_prefix}/report.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
