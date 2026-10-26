"""The Thinker side alone: every si_thinker_stream output scored as text, to choose what to speak.

Per file (one direction, read window, commit policy):
  BLEU       the committed translation against RealSI's reference (SacreBLEU, normalised as everywhere here)
  first      median time the first block was committed
  LAAL-text  length-adaptive average lagging with each target unit (English word / Chinese
             character) delayed to its block's commit time -- the text's latency, before speech
  blocks     median blocks a sentence

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_text_eval [--dir eval_outputs/.../realsi_thinker_stream_v1]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CJK = re.compile(r"[一-鿿]|[A-Za-z0-9']+")
EN = re.compile(r"[A-Za-z0-9']+")


def laal(delays, src_ms, ref_len):
    rate = src_ms / max(len(delays), ref_len, 1)
    total, n = 0.0, 0
    for i, d in enumerate(delays):
        total += d - i * rate
        n += 1
        if d >= src_ms:
            break
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="eval_outputs/uniss_qwen3_omni_v1/realsi_thinker_stream_v1")
    ap.add_argument("--match", default="", help="only files whose name contains this")
    args = ap.parse_args()
    from evaluation.text_metrics import corpus_bleu
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items

    items = {d: {it["id"]: it for it in realsi_items(d)} for d in ("zh2en", "en2zh")}
    rows = []
    for f in sorted(Path(args.dir).glob("blocks_*.jsonl")):
        if args.match not in f.name:
            continue
        d = f.name.split("_")[1]
        lang = "cmn" if d == "en2zh" else "eng"
        unit = CJK if lang == "cmn" else EN
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        if len(recs) < 300:
            continue
        b = corpus_bleu([r["translation"] for r in recs], [items[d][r["id"]]["trg_text"] for r in recs], language=lang)["score"]
        lags, firsts = [], []
        for r in recs:
            it = items[d][r["id"]]
            delays = [bl["write_ms"] for bl in r["blocks"] for _ in unit.findall(bl["text"])]
            if delays:
                lags.append(laal(delays, it["end"] - it["start"], len(unit.findall(it["trg_text"]))))
                firsts.append(r["blocks"][0]["write_ms"])
        rows.append((d, f.stem.split("_", 2)[2], round(b, 2), round(float(np.median(firsts)) / 1000, 2),
                     round(float(np.mean(lags)) / 1000, 2), int(np.median([len(r["blocks"]) for r in recs]))))
    for d in ("en2zh", "zh2en"):
        print(f"\n{d}\n| 配置 | 文字 BLEU | 首次定稿 | 文字 LAAL | 段数 |\n|---|---|---|---|---|")
        for r in sorted((r for r in rows if r[0] == d), key=lambda r: r[4]):
            print(f"| {r[1]} | {r[2]:.2f} | {r[3]:.2f}s | {r[4]:.2f}s | {r[5]} |")


if __name__ == "__main__":
    main()
