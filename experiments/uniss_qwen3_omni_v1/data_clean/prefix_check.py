"""An independent check of a timing rule: at each segment's write time, does the Thinker,
given only the source heard so far, already translate what has been written?

For each row and each non-final segment, the source prefix heard by the segment's write
time (units ending at or before it) is translated by the Thinker (text, greedy, the
contextual alignment's prompt); the score is the share of the written target -- every
word of this segment and the ones before -- found in that translation (characters for
Chinese, words for English, as multisets). A rule that writes words the heard source
cannot yet support scores low; one that waits needlessly scores high but lags, which
the latency numbers show beside it. Unlike timing_rules' "predictable", nothing here is
defined by any rule's own criterion.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.prefix_check \
        --meta ... --rows 400 --seg omni=seg_omni.parquet --seg hybrid=seg_hybrid.parquet
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
NAMES = {"eng": "English", "cmn": "Chinese"}


def units(text: str, zh: bool) -> collections.Counter:
    text = text.lower()
    return collections.Counter(re.findall(r"[一-鿿]", text) if zh else re.findall(r"[a-z0-9']+", text))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--seg", action="append", required=True, help="name=segments.parquet")
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--chunk-ms", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    segs = {name: {r["key"]: json.loads(r["segments"]) for r in pq.read_table(p, columns=["key", "segments"]).to_pylist()}
            for name, p in (s.split("=", 1) for s in args.seg)}
    common = sorted(set.intersection(*(set(v) for v in segs.values())))
    keys = set(random.Random(args.seed).sample(common, min(args.rows, len(common))))
    meta = {}
    for p in sorted(Path(args.meta).glob("*.parquet")):
        for r in pq.read_table(p, columns=["key", "src_lang", "tgt_lang", "source_words", "target_words"]).to_pylist():
            if r["key"] in keys:
                meta[r["key"]] = r
    tok = AutoTokenizer.from_pretrained(QWEN3)
    template = json.loads((Path(QWEN3) / "chat_template.json").read_text())["chat_template"]
    llm = LLM(model=QWEN3, gpu_memory_utilization=0.85, max_model_len=4096, enable_prefix_caching=True,
              limit_mm_per_prompt={"audio": 0, "image": 0, "video": 0}, seed=0)
    params = SamplingParams(max_tokens=160, temperature=0.0)

    prompts, jobs = {}, []   # one translation per distinct (row, heard prefix)
    for name, by_key in segs.items():
        for key in keys:
            m = meta[key]
            src = json.loads(m["source_words"])
            tgt = json.loads(m["target_words"])
            zh_src, zh_tgt = m["src_lang"] == "cmn", m["tgt_lang"] == "cmn"
            sep = "" if zh_tgt else " "
            for g in by_key[key][:-1]:
                write = g["chunk"] * args.chunk_ms
                heard = sum(1 for w in src if w["end_ms"] <= write)
                prefix = ("" if zh_src else " ").join(w["text"] for w in src[:heard])
                written = sep.join(w["text"] for w in tgt[: g["w1"]])
                pk = (key, heard)
                if pk not in prompts:
                    msg = [{"role": "user", "content": f"Translate the following {NAMES[m['src_lang']]} text into "
                                                       f"{NAMES[m['tgt_lang']]}. Output only the translation.\n\n{prefix}"}]
                    prompts[pk] = tok.apply_chat_template(msg, chat_template=template, add_generation_prompt=True,
                                                          tokenize=False)
                jobs.append((name, m["tgt_lang"], pk, written, zh_tgt, write))
    order = list(prompts)
    outs = llm.generate([TokensPrompt(prompt_token_ids=tok(prompts[k], add_special_tokens=False)["input_ids"])
                         for k in order], params, use_tqdm=False)
    said = {k: o.outputs[0].text for k, o in zip(order, outs)}
    acc = collections.defaultdict(lambda: [0, 0, 0])
    for name, lang, pk, written, zh, _ in jobs:
        w, s = units(written, zh), units(said[pk], zh)
        a = acc[(name, lang)]
        a[0] += sum((w & s).values()); a[1] += sum(w.values()); a[2] += 1
    result = {f"{n}/{l}": {"consistency": round(a[0] / max(1, a[1]), 4), "segments": a[2]} for (n, l), a in sorted(acc.items())}
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
