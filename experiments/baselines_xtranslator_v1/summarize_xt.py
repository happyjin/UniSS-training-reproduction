"""summary.md + tiers.json for the X-Translator RealSI grid (R4): reads paper/baselines_results/xtranslator/<point>_<dir>.json.
Tier rule (paper/baselines_run_plan_v1.md §1, the paper's rule): within each budget, the point with the highest
ASR-BLEU whose D1 fits; "--" where none fits; the same by D0w as reference only. Refuses existing outputs.

    PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python experiments/baselines_xtranslator_v1/summarize_xt.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "paper/baselines_results/xtranslator"
POINTS = [("xt-default", "release default (main)"), ("xt-c1280", "paraformer_chunk_size 1280 (0.08 s requests, 0.64 s endpoint)"),
          ("xt-c5120", "paraformer_chunk_size 5120 (0.32 s requests, 2.56 s endpoint)"),
          ("xt-a1.6", "ASR server chunk 1.6 s (paper App. C; secondary)")]
BUDGETS = {"en2zh": [("m1", 1.18), ("m2", 1.98), ("m3", 2.46), ("m4", 2.98)],
           "zh2en": [("m1", 1.18), ("m2", 1.92), ("m3", 2.40), ("m4", 2.98)]}
OMNI = {"en2zh": {"m1": 19.80, "m2": 24.21, "m3": 25.01, "m4": 25.54},
        "zh2en": {"m1": 14.89, "m2": 19.55, "m3": 21.07, "m4": 22.47}}
NAME = {"en2zh": "En→Zh", "zh2en": "Zh→En"}


def tier(pts, key, budget):
    ok = [p for p in pts if p.get(key) is not None and p[key] <= budget and p.get("asr_bleu") is not None
          and not p.get("asr_bleu_invalid")]
    return max(ok, key=lambda p: p["asr_bleu"]) if ok else None


def cell(p, key):
    return f"{p['asr_bleu']:.2f} ({p[key]:.2f})，`{p['point']}`" if p else "--"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "summary.md"))
    ap.add_argument("--tiers", default=str(RES / "tiers.json"))
    a = ap.parse_args()
    pts = {d: [json.load(open(RES / f"{p}_{d}.json", encoding="utf-8")) for p, _ in POINTS if (RES / f"{p}_{d}.json").exists()]
           for d in BUDGETS}
    tiers = {"by_D1": {}, "by_D0w_reference_only": {}}
    for key, name in (("laal_d1", "by_D1"), ("laal_d0w", "by_D0w_reference_only")):
        for d, bs in BUDGETS.items():
            for m, b in bs:
                t = tier(pts[d], key, b)
                tiers[name][f"{d}|{m}"] = {"budget": b, "omni_reported": OMNI[d][m], "point": t["point"] if t else None,
                                           "asr_bleu": t["asr_bleu"] if t else None, "laal_d1": t["laal_d1"] if t else None,
                                           "laal_d0w": t["laal_d0w"] if t else None}
    desc = dict(POINTS)
    L = ["# X-Translator（Qwen3-ASR 1.7B + LMT-60-8B + X-Voice 0.4B），RealSI 句级（自动生成，`experiments/baselines_xtranslator_v1/summarize_xt.py`）", "",
         f"生成于 {time.strftime('%Y-%m-%d %H:%M', time.localtime())}（布拉格时间）。登记：`paper/baselines_run_plan_xtranslator_v1.md`（含修订）。",
         "官方权重、官方代码，只做推理。X-Translator 用自己的 TTS（X-Voice）出语音；源时钟是模拟的，不计算耗时。",
         "speech-only ASR-BLEU 用 Whisper-large-v3（英）/ Paraformer（中）。D1 是段 LAAL，按块开始播放时刻计，含排队；D0w 是 SimulEval 逐词 LAAL，按块写出时刻计。",
         "打分代码就是 R1 的 `score_seamless.score()`。BLEU 分母为全部句子（En→Zh 346，Zh→En 431）。每个点每个方向只跑一次。", "",
         "## 全部点", "",
         "| 方向 | 点 | 设置 | ASR-BLEU | 文本 BLEU | D1 (s) | D0w (s) | StartOffset p50 (s) | 块/句 | 句末才写出的单位占比 | 无输出句 | TTS 失败段 | ASR 覆盖 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in BUDGETS:
        for p in pts[d]:
            L.append(f"| {NAME[d]} | `{p['point']}` | {desc[p['point']]} | {p.get('asr_bleu')} | {p['text_bleu']} | {p['laal_d1']:.2f} | "
                     f"{p['laal_d0w']:.2f} | {p['start_offset_s_p50']} | {p['blocks_per_sentence']} | {p['share_units_written_at_end']:.1%} | "
                     f"{p['n_no_output']} | {p['n_tts_fail_segments']} | {p.get('asr_coverage')} |")
    for key, name, title in (("laal_d1", "by_D1", "按 D1，论文规则"), ("laal_d0w", "by_D0w_reference_only", "按 D0w，仅作参考")):
        L += ["", f"## 分档（{title}）", "", "| 方向 | 档 | 预算 (s) | SimulS2ST-Omni（报告值） | X-Translator |", "|---|---|---|---|---|"]
        for d, bs in BUDGETS.items():
            for m, b in bs:
                t = tier(pts[d], key, b)
                L.append(f"| {NAME[d]} | {m} | {b:.2f} | {OMNI[d][m]:.2f} | {cell(t, key)} |")
    with open(a.out, "x", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(a.tiers, "x", encoding="utf-8") as fh:
        json.dump(tiers, fh, ensure_ascii=False, indent=1)
    print(a.out, a.tiers)


if __name__ == "__main__":
    main()
