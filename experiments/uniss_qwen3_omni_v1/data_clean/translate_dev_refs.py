"""Silver references for the spontaneous dev sets (build_spont_items.py): each human transcript translated by
Hy-MT2-30B-A3B (FP8, greedy, no reasoning) with the model's own prompt for the target language, as
translate_zh2en_refs.py does for AliMeeting -- a model unrelated to the evaluated Thinker. Writes the items with
"translation" added. Runs in venvs/vllm_omni_030.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.translate_dev_refs --items data/si_spont_v1/dev_en2zh.jsonl \
        --tgt zh --out data/si_spont_v1/dev_en2zh_ref.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODEL = "/mnt/speech-h200-fsx/data/data/models/tencent-Hy-MT2-30B-A3B-FP8"
PROMPT = {"en": "Translate the following segment into English, without additional explanation.\n\n{text}",
          "zh": "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--tgt", choices=("en", "zh"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    llm = LLM(model=MODEL, gpu_memory_utilization=args.gpu_mem, max_model_len=4096, trust_remote_code=True)
    res = llm.chat([[{"role": "user", "content": PROMPT[args.tgt].format(text=i["transcript"])}] for i in items],
                   SamplingParams(temperature=0.0, max_tokens=256))
    with open(out, "x", encoding="utf-8") as fh:
        for i, r in zip(items, res):
            t = re.sub(r"<think>.*?</think>", "", r.outputs[0].text, flags=re.S).strip()
            fh.write(json.dumps({**i, "translation": t}, ensure_ascii=False) + "\n")
    print(f"MT_DONE {len(items)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
