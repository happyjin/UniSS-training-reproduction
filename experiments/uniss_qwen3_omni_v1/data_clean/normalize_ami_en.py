"""English references for zh->en (docs/multi_speaker_si_plan_v9_bidir_abc.zh-CN.md §2.4): the AMI transcripts are
upper case, unpunctuated and keep fillers. Each one is cleaned by rule (fillers dropped, immediate word repetitions
collapsed, lower case), then a small LLM restores case and punctuation. The LLM output is kept only if its words, lower
cased and without punctuation, are exactly the rule-cleaned words; otherwise the rule-cleaned text is used with a
capital first letter and a final full stop. No word other than fillers and stutters is ever changed.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.normalize_ami_en --out data/ami_multispk_si_v1/en_norm_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import HESITATIONS

PROMPT = ("Restore capitalization and punctuation in this sentence from a meeting transcript. Do not add, remove, "
          "reorder or change any word. Output only the sentence.\n\nSentence: {}")


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+(?:[-.][a-z0-9']+)*", text.lower())


def rule_clean(raw: str) -> list[str]:
    ws = [w for w in re.findall(r"[A-Za-z0-9'\-.]+", raw or "") if w.upper().strip(".") not in HESITATIONS]
    out = []
    for w in ws:
        w = w.lower()
        if out and out[-1] == w:
            continue
        out.append(w)
    return out


def fallback(ws: list[str]) -> str:
    if not ws:
        return ""
    ws = ["I" if w == "i" else re.sub(r"^i'", "I'", w) for w in ws]
    s = " ".join(ws)
    return s[0].upper() + s[1:] + "."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import pyarrow.dataset as pads
    from vllm import LLM, SamplingParams

    src = pads.dataset(Path(args.data) / "index/source_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "transcript"]).to_pylist()
    if args.limit:
        src = src[: args.limit]
    clean = [rule_clean(r["transcript"]) for r in src]
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1024, seed=0)
    tok = llm.get_tokenizer()
    todo = [i for i, c in enumerate(clean) if len(c) >= 2]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(" ".join(clean[i]))}],
                                       tokenize=False, add_generation_prompt=True) for i in todo]
    gen = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
    llm_out = {i: g.outputs[0].text.strip().splitlines()[0].strip() if g.outputs[0].text.strip() else ""
               for i, g in zip(todo, gen)}
    n_llm = 0
    with open(out, "x", encoding="utf-8") as fh:
        for i, r in enumerate(src):
            y = llm_out.get(i, "")
            if y and words_of(y) == words_of(" ".join(clean[i])):
                text, how = y[0].upper() + y[1:], "llm"
                n_llm += 1
            else:
                text, how = fallback(clean[i]), "rule"
            fh.write(json.dumps({"record_id": r["record_id"], "en_raw": r["transcript"], "en": text, "how": how},
                                ensure_ascii=False) + "\n")
    print(json.dumps({"utterances": len(src), "llm_kept": n_llm, "out": str(out)}))


if __name__ == "__main__":
    main()
