"""Contextual alignment (Hibiki, 2502.03382) with our own Thinker: for each target
word, the source prefix at which it becomes predictable.

For a row with source words s_1..s_K and the translation, the Thinker (text
only) is asked to translate each prefix s_1..s_k, k = 0..K, and the
translation is teacher-forced after it; vLLM's prompt logprobs give every
translation token's log-probability under each prefix. A target word's
score l_j(k) sums its tokens; it is aligned to the source word whose arrival
raises it most, a_j = argmax_k [l_j(k) - l_j(k-1)] - 1. A lone late spike --
a word aligned more than --spike words after every neighbour within two
places -- is pulled back to the latest of those neighbours, so one outlier
does not delay everything after it once the times are made monotonic.

The prefixes of a row share their beginning, so vLLM's prefix cache makes
each further prefix cost little more than its new word and the translation.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.context_align \
        --meta data/unist_clean_v1/meta --keys keys.parquet --out out.parquet [--shard i --num-shards n]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
NAMES = {"eng": "English", "cmn": "Chinese", "en": "English", "zh": "Chinese"}


def word_spans(text: str, words: list[str]) -> list[tuple[int, int] | None]:
    """Character span of each aligner word in the text, searched in order."""
    spans, pos, low = [], 0, text.lower()
    for w in words:
        w = w.strip().lower()
        k = low.find(w, pos) if w else -1
        if k < 0:
            spans.append(None)
            continue
        spans.append((k, k + len(w)))
        pos = k + len(w)
    return spans


def despike(a: list[int | None], spike: int) -> list[int | None]:
    out = list(a)
    for j, v in enumerate(a):
        if v is None:
            continue
        near = [a[i] for i in range(max(0, j - 2), min(len(a), j + 3)) if i != j and a[i] is not None]
        if near and v > max(near) + spike:
            out[j] = max(near)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--spike", type=int, default=3)
    ap.add_argument("--bounds", default="", help="prefix_bounds.py output; without it, a prefix after every source unit")
    ap.add_argument("--rows-per-call", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {args.out}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    keys = set(pq.read_table(args.keys, columns=["key"]).column("key").to_pylist())
    cols = ["key", "src_lang", "tgt_lang", "translation", "source_words", "target_words", "a68_present"]
    rows = [r for p in sorted(Path(args.meta).glob("*.parquet")) for r in pq.read_table(p, columns=cols).to_pylist()
            if r["key"] in keys and r["a68_present"]]
    rows.sort(key=lambda r: r["key"])
    rows = rows[args.shard :: args.num_shards]
    bounds_of = {}
    if args.bounds:
        mine = {r["key"] for r in rows}
        bounds_of = {r["key"]: json.loads(r["bounds"]) for r in pq.read_table(args.bounds).to_pylist() if r["key"] in mine}
    print(f"shard {args.shard}: {len(rows)} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(QWEN3)
    # The Omni tokenizer carries no template of its own; the processor's is in chat_template.json.
    template = json.loads((Path(QWEN3) / "chat_template.json").read_text())["chat_template"]
    llm = LLM(model=QWEN3, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
              limit_mm_per_prompt={"audio": 0, "image": 0, "video": 0}, enable_prefix_caching=True, seed=0)
    # vLLM skips the prefix cache for any request with prompt logprobs, so every prefix
    # recomputed the whole prompt. Reading it back is safe here: logprobs then cover only
    # the uncached tail, and the translation always follows the growing source prefix,
    # which no earlier request has, so it is never cached (checked below).
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, skip_reading_prefix_cache=False)
    # ... except when a whole prompt was seen before: two rows with the same translation
    # share the empty-prefix prompt word for word. Those few are recomputed without the cache.
    params_uncached = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    out_rows, t0 = [], time.time()
    for s in range(0, len(rows), args.rows_per_call):
        batch = rows[s : s + args.rows_per_call]
        requests, plans = [], []
        for r in batch:
            src = [w["text"] for w in json.loads(r["source_words"])]
            tgt_words = [w["text"] for w in json.loads(r["target_words"])]
            zh_src = r["src_lang"] in ("cmn", "zh")
            enc = tok(r["translation"], add_special_tokens=False, return_offsets_mapping=True)
            spans = word_spans(r["translation"], tgt_words)
            owner = []
            for a, b in enc["offset_mapping"]:
                # by overlap, not by start: an English token carries its leading space
                # (" tell" starts on the space), which lies outside the word's span;
                # and a token may cover several words (Chinese "你知道" is one token),
                # so it counts towards every word it touches
                owner.append([j for j, sp in enumerate(spans) if sp and a < sp[1] and b > sp[0]])
            first = len(requests)
            bounds = bounds_of.get(r["key"], list(range(len(src) + 1)))
            for k in range(len(bounds)):
                prefix = ("" if zh_src else " ").join(src[: bounds[k]])
                msg = [{"role": "user", "content": f"Translate the following {NAMES[r['src_lang']]} text into "
                                                   f"{NAMES[r['tgt_lang']]}. Output only the translation.\n\n{prefix}"}]
                head = tok.apply_chat_template(msg, chat_template=template, add_generation_prompt=True, tokenize=False)
                ids = tok(head, add_special_tokens=False)["input_ids"] + enc["input_ids"]
                requests.append(TokensPrompt(prompt_token_ids=ids))
            plans.append((r, first, len(bounds) - 1, enc["input_ids"], owner, len(tgt_words), bounds))
        outs = llm.generate(requests, params, use_tqdm=False)
        need = {}
        for r, first, K, tids, owner, n_t, bounds in plans:
            for k in range(K + 1):
                need[first + k] = len(tids)
        bad = [i for i, n in need.items()
               if outs[i].prompt_logprobs is None or len(outs[i].prompt_logprobs) < n
               or any(e is None for e in outs[i].prompt_logprobs[-n:])]
        if bad:
            for i, o in zip(bad, llm.generate([requests[i] for i in bad], params_uncached, use_tqdm=False)):
                outs[i] = o
        for r, first, K, tids, owner, n_t, bounds in plans:
            score = [[0.0] * (K + 1) for _ in range(n_t)]
            seen = [False] * n_t
            for k in range(K + 1):
                full = outs[first + k].prompt_logprobs
                if full is None or len(full) < len(tids) or any(e is None for e in full[-len(tids):]):
                    raise RuntimeError(f"{r['key']} prefix {k}: prompt logprobs do not cover the translation")
                lp = full[-len(tids):]
                for t, (tid, entry, words) in enumerate(zip(tids, lp, owner)):
                    if entry is None:
                        continue
                    for j in words:
                        score[j][k] += entry[tid].logprob
                        seen[j] = True
            align = []
            for j in range(n_t):
                if not seen[j] or K == 0:
                    align.append(None)
                    continue
                gains = [score[j][k] - score[j][k - 1] for k in range(1, K + 1)]
                k = max(range(K), key=lambda k: gains[k])       # the (k+1)-th prefix step
                align.append(bounds[k + 1] - 1)                 # its last source unit
            # Chinese sources are aligned per character, so a spike is counted in characters
            spike = args.spike * (2 if r["src_lang"] in ("cmn", "zh") else 1)
            # The whole curve is kept (log-prob of each word under each prefix, 0.01
            # resolution), so other alignment rules can be computed later without the GPU.
            curves = [[round(v, 2) for v in score[j]] if seen[j] else None for j in range(n_t)]
            out_rows.append({"key": r["key"], "context_align": json.dumps(despike(align, spike)),
                             "curves": json.dumps(curves), "bounds": json.dumps(bounds)})
        print(f"shard {args.shard}: {len(out_rows)}/{len(rows)} rows, {len(requests)} prefixes in the last call, "
              f"{len(out_rows) / (time.time() - t0):.1f} rows/s", flush=True)
    pq.write_table(pa.Table.from_pylist(out_rows), args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
