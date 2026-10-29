"""English references for real Mandarin meetings (AliMeeting test, build_alimeeting_test.py): each human Chinese
transcript utterance translated by Hy-MT2-30B-A3B (Tencent, FP8), one of the three MT systems behind the AMI Chinese
references, greedy, without reasoning. A model unrelated to Qwen3-Omni, so the references do not share the evaluated
systems' style. Runs in venvs/vllm_omni_030.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.translate_zh2en_refs --items data/alimeeting_si_v1/mt_items.jsonl \
        --out data/alimeeting_si_v1/refs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODEL = "/mnt/speech-h200-fsx/data/data/models/tencent-Hy-MT2-30B-A3B-FP8"
PROMPT = "Translate the following segment into English, without additional explanation.\n\n{text}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096, trust_remote_code=True)
    msgs = [[{"role": "user", "content": PROMPT.format(text=i["zh"])}] for i in items]
    res = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=256))
    with open(out, "x", encoding="utf-8") as fh:
        for i, r in zip(items, res):
            en = re.sub(r"<think>.*?</think>", "", r.outputs[0].text, flags=re.S).strip()
            fh.write(json.dumps({"id": i["id"], "zh": i["zh"], "en": en}, ensure_ascii=False) + "\n")
    print(f"MT_DONE {len(items)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
