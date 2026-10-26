"""The whole system on RealSI against SimulS2ST-Omni's Table 3 (sentence level, ASR-BLEU by tier).

Reads si_infer runs named <prefix>_m<k> (one read window each), their metrics.json
and metrics_extra.json, and the Thinker's committed translations for the text BLEU,
and prints one markdown table per direction. SimulS2ST-Omni's numbers are quoted
from the paper (Table 3, Whisper/Paraformer ASR; LiveInterpret 2.0 as they report it).
Directions are matched by output language: our en2zh (Chinese speech out) against
their En->Zh. Their stated sizes (En->Zh 431 segments, 51:12) are those of RealSI's
Chinese-source folder here, so their labels and this copy of the data may disagree.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_compare --runs <dir>/system100_hold --blocks-suffix _hold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER = {"en2zh": {2: 24.21, 3: 25.01, 4: 25.54, 5: 25.92, 6: 26.32, "live": 29.09},
         "zh2en": {2: 19.55, 3: 21.07, 4: 22.47, 5: 22.94, 6: 23.47, "live": 22.19}}
LABEL = {"en2zh": "英→中（输出中文）", "zh2en": "中→英（输出英文）"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="run prefix: <prefix>_m2 ... _m6")
    ap.add_argument("--tiers", default="2,3,4,5,6")
    ap.add_argument("--blocks", default="eval_outputs/uniss_qwen3_omni_v1/realsi_thinker_stream_v1")
    ap.add_argument("--blocks-suffix", default="")
    args = ap.parse_args()
    from evaluation.text_metrics import corpus_bleu
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items

    for d in ("en2zh", "zh2en"):
        lang = "cmn" if d == "en2zh" else "eng"
        refs = {it["id"]: it["trg_text"] for it in realsi_items(d)}
        print(f"\n{LABEL[d]}\n")
        print("| 档位 | 我们 ASR-BLEU | SimulS2ST-Omni ASR-BLEU | 思考模型文字 BLEU | 首次发音 | LAAL（对齐） | 结尾落后 | 静音比例 | A.PCP | 源音色相似度 | 拖长段 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in (int(x) for x in args.tiers.split(",")):
            run = Path(f"{args.runs}_m{k}") / f"realsi_{d}"
            if not (run / "metrics.json").exists():
                print(f"| m{k} | （未完成） |||||||||| ")
                continue
            m = json.load(open(run / "metrics.json"))
            x = json.load(open(run / "metrics_extra.json")) if (run / "metrics_extra.json").exists() else {}
            blocks = [json.loads(l) for l in open(Path(args.blocks) / f"blocks_{d}_w{k}{args.blocks_suffix}.jsonl")]
            tb = corpus_bleu([b["translation"] for b in blocks], [refs[b["id"]] for b in blocks], language=lang)["score"]
            f = lambda v, u="": "—" if v is None else f"{v}{u}"
            print(f"| m{k} | **{m['asr_bleu']:.2f}** | {PAPER[d].get(k, '—')} | {tb:.2f} | "
                  f"{f(x.get('start_offset_aligned_s_p50'), 's')} | {f(x.get('laal_s'), 's')} | {m['end_offset_s_p50']}s | "
                  f"{m['silence_ratio_mean']} | {f(x.get('apcp'))} | {f(x.get('sim_src'))} | {m.get('stretched_blocks')} |")
        print(f"\nLiveInterpret 2.0（论文引用）：{PAPER[d]['live']}")


if __name__ == "__main__":
    main()
