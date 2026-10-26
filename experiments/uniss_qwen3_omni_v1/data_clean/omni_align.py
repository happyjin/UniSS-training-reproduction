"""Word alignment with OmniAlign (2608.18474; zh-en AER 8.5), batched, on our own word units.

OmniAlign's example aligns one pair at a time from its own tokenisation; here the
units are the ones the timing uses -- the forced aligner's source words (English
words; Chinese characters, grouped into jieba words for the alignment and mapped
back to their last character) and the a68 target words -- and many pairs go
through the encoder at once. The method is the example's: hidden states of layer
--layer for each side, subword similarity, softmax both ways, a link where both
exceed --threshold, lifted to words; each link keeps its strength (the larger
min of the two softmaxes over its subword pairs) so a rule can trust only strong
links.

Runs in the align_omni_v1 venv. Shares a GPU with training: the process caps
itself at --memory-fraction of the card.

    python omni_align.py --meta data/unist_clean_v1/meta --keys keys.parquet --out links.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jieba
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers

MODEL = "/opt/dlami/nvme/jasonleeeli/tools/OmniAlign/OmniAlign"


def source_units(source_words: list[dict], zh: bool) -> tuple[list[str], list[int]]:
    """Alignment words, and for each the index of its last forced-aligner unit."""
    texts = [w["text"] for w in source_words]
    if not zh:
        return texts, list(range(len(texts)))
    chars = "".join(texts)
    if len(chars) != len(texts):            # units are not single characters: align them as they are
        return texts, list(range(len(texts)))
    words, last, pos = [], [], 0
    for w in jieba.cut(chars):
        words.append(w)
        pos += len(w)
        last.append(pos - 1)
    return words, last


def encode(tokenizer, words: list[str]) -> tuple[list[int], list[int]]:
    ids, owner = [], []
    for i, w in enumerate(words):
        sub = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) or [tokenizer.unk_token_id]
        ids += sub
        owner += [i] * len(sub)
    return ids, owner


def hidden(model, tokenizer, seqs: list[list[int]], layer: int, device) -> list[torch.Tensor]:
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    limit = tokenizer.model_max_length - 2
    seqs = [s[:limit] for s in seqs]
    width = max(len(s) for s in seqs) + 2
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s) + 2] = torch.tensor([cls] + s + [sep])
        mask[i, : len(s) + 2] = 1
    with torch.no_grad():
        h = model(ids.to(device), attention_mask=mask.to(device), output_hidden_states=True).hidden_states[layer]
    return [h[i, 1 : len(s) + 1].float() for i, s in enumerate(seqs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--memory-fraction", type=float, default=0.05)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {args.out}")
    jieba.setLogLevel(60)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # torch_dtype, not dtype: transformers 4.53 would store "dtype" in the config and fail to print it
    model = transformers.AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

    keys = set(pq.read_table(args.keys, columns=["key"]).column("key").to_pylist())
    cols = ["key", "src_lang", "source_words", "target_words", "a68_present"]
    rows = sorted((r for p in sorted(Path(args.meta).glob("*.parquet")) for r in pq.read_table(p, columns=cols).to_pylist()
                   if r["key"] in keys and r["a68_present"]), key=lambda r: r["key"])[args.shard :: args.num_shards]
    print(f"shard {args.shard}: {len(rows)} rows on {device}", flush=True)

    out = []
    for s in range(0, len(rows), args.batch):
        batch = rows[s : s + args.batch]
        prepared = []
        for r in batch:
            src_words, last = source_units(json.loads(r["source_words"]), r["src_lang"] in ("cmn", "zh"))
            tgt_words = [w["text"] for w in json.loads(r["target_words"])]
            (sid, sown), (tid, town) = encode(tokenizer, src_words), encode(tokenizer, tgt_words)
            prepared.append((r, last, sid, sown, tid, town))
        hs = hidden(model, tokenizer, [p[2] for p in prepared], args.layer, device)
        ht = hidden(model, tokenizer, [p[4] for p in prepared], args.layer, device)
        for (r, last, sid, sown, tid, town), a, b in zip(prepared, hs, ht):
            sim = a @ b.T
            st, ts = torch.softmax(sim, dim=-1), torch.softmax(sim, dim=-2)
            strength = torch.minimum(st, ts)
            best = {}
            for i, j in torch.nonzero((st > args.threshold) & (ts > args.threshold), as_tuple=False).tolist():
                if i >= len(sown) or j >= len(town):
                    continue
                key = (town[j], last[sown[i]])
                best[key] = max(best.get(key, 0.0), float(strength[i, j]))
            links = [{"target_index": t, "source_index": u, "confidence": round(c, 4), "method": "omnialign"}
                     for (t, u), c in sorted(best.items())]
            out.append({"key": r["key"], "omni_links": json.dumps(links)})
        if (s // args.batch) % 50 == 0:
            print(f"shard {args.shard}: {len(out)}/{len(rows)}", flush=True)
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(f"wrote {args.out}: {len(out)} rows", flush=True)


if __name__ == "__main__":
    main()
