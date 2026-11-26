"""sst_hsmm_v1 prereg change 46, stage 0 (D3 pre-training check): how predictable is the next second of SOURCE speech from the
gold source prefix, for the frozen wave17 Thinker in text mode. One teacher-forced forward per source sentence (edev's unique
pairs, open UniST held-out); prompt = user "say a natural spoken <lang> sentence", assistant = the source transcription.
For every non-final read c (s): target = the first canonical token whose first content character lies in a source word that
starts in (c, c + 1 s]; context = all gold text before it (generous upper bound). A read is a "leak" when a context token holds
a character of a word starting after c. Rank ignores pure punctuation / whitespace / special tokens.
Primary metric: top-5 hit on non-leak reads, per direction; GO iff the mean over the two directions >= 0.15.
    CUDA_VISIBLE_DEVICES=0 python -m experiments.sst_hsmm_v1.d3_pre --out eval_outputs/sst_hsmm_v1/d3/pre
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1 import model as MD  # noqa: E402

DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
EDEV = REPO / "eval_outputs/sst_hsmm_v1/edev/data"
ROWS = REPO / "data/unist_simul_v1/rows"
DIRS = ("en2zh", "zh2en")
LANG = {"en2zh": "English", "zh2en": "Chinese"}
NONCONTENT = re.compile(r"^[\s\W_]*$", re.UNICODE)       # punctuation / whitespace only
BS = 16


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def load_rows(ids):
    got = {}
    for f in sorted(ROWS.glob("part-*.parquet")):
        t = pq.read_table(f, columns=["id", "transcription", "source_words", "source_duration_ms"]).to_pylist()
        for r in t:
            if r["id"] in ids and r["id"] not in got:
                got[r["id"]] = r
    return got


def word_spans(text, words):
    """char span of every aligner word inside the transcription (sequential, case-insensitive); None if any is missing"""
    low, p, out = text.lower(), 0, []
    for w in words:
        k = low.find(w["text"].lower(), p)
        if k < 0:
            return None
        out.append((k, k + len(w["text"]), w["start_ms"]))
        p = k + len(w["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: sentences per direction")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "result.json").exists():
        sys.exit(f"exists: {out}/result.json (refuse to overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    ids = {d: list(dict.fromkeys(json.loads(l)["id"].split("__", 1)[1] for l in open(EDEV / f"items_{d}.jsonl"))) for d in DIRS}
    if a.limit:
        ids = {d: v[: a.limit] for d, v in ids.items()}
    rows = load_rows({i for v in ids.values() for i in v})
    log(f"rows {len(rows)} / {sum(map(len, ids.values()))}")
    proc, _ = MD.processor_and_heads()
    tk = proc.tokenizer
    vocab = len(tk)
    ignore = torch.zeros(vocab, dtype=torch.bool)
    for v in range(vocab):
        if v >= 151643 or NONCONTENT.match(tk.decode([v])):
            ignore[v] = True
    # frequency baseline: top-5 content tokens of the training source transcripts, per direction
    freq_top = {}
    for d in DIRS:
        cnt = collections.Counter()
        for l in open(DATA / f"train_{d}.jsonl", encoding="utf-8"):
            for v in tk(json.loads(l)["src"].strip())["input_ids"]:
                if not ignore[v]:
                    cnt[v] += 1
        freq_top[d] = {v for v, _ in cnt.most_common(5)}
        log(f"{d} freq top5 {[tk.decode([v]) for v in freq_top[d]]}")
    dev = "cuda:0"
    thinker, _ = MD.build_thinker(dev, lora=False)
    V = thinker.lm_head.out_features                     # 152064 > len(tokenizer): the padding ids are ignored too
    ign = torch.cat([ignore, torch.ones(V - vocab, dtype=torch.bool)]).to(dev)
    res, per = {}, open(out / "reads.jsonl", "x", encoding="utf-8")
    for d in DIRS:
        conv = [{"role": "user", "content": [{"type": "text", "text": f"Say a natural spoken {LANG[d]} sentence."}]}]
        head = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        items, skip = [], collections.Counter()
        for i in ids[d]:
            r = rows.get(i)
            if r is None:
                skip["no_row"] += 1; continue
            text = r["transcription"].strip()
            ws = word_spans(text, json.loads(r["source_words"]))
            if not ws:
                skip["align"] += 1; continue
            enc = tk(head + text + "<|im_end|>", return_offsets_mapping=True)
            L0 = len(head)
            ch_t = {}                                  # char -> word start_ms
            for k0, k1, st in ws:
                for k in range(k0, k1):
                    ch_t[k] = st
            toks = []                                  # (position, id, first content start_ms, max start_ms over its chars)
            for j, (v, (o0, o1)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
                if o0 < L0 or o0 >= L0 + len(text) or ignore[v]:
                    continue
                ts = [ch_t[k - L0] for k in range(o0, o1) if (k - L0) in ch_t]
                if ts:
                    toks.append((j, v, ts[0], max(ts)))
            items.append((i, enc["input_ids"], toks, int(r["source_duration_ms"])))
        log(f"{d}: {len(items)} sentences, skipped {dict(skip)}")
        agg = collections.Counter()
        for b0 in range(0, len(items), BS):
            batch = items[b0: b0 + BS]
            T = max(len(x[1]) for x in batch)
            inp = torch.full((len(batch), T), tk.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(batch), T), dtype=torch.long)
            for k, x in enumerate(batch):
                inp[k, : len(x[1])] = torch.tensor(x[1]); att[k, : len(x[1])] = 1
            with torch.no_grad():
                lg = thinker(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float()
            for k, (iid, seq, toks, dur) in enumerate(batch):
                for c in range(1, math.ceil(dur / 1000)):
                    lo, hi = c * 1000, c * 1000 + 1000
                    tgt = next((t for t in toks if lo < t[2] <= hi), None)
                    if tgt is None:
                        agg["no_target"] += 1; continue
                    j, v = tgt[0], tgt[1]
                    leak = any(t[3] > lo for t in toks if t[0] < j)
                    z = lg[k, j - 1]
                    lp = float(torch.log_softmax(z, -1)[v])
                    rank = int(((z > z[v]) & ~ign).sum())
                    rec = {"dir": d, "id": iid, "c": c, "leak": leak, "rank": rank, "lp": round(lp, 4),
                           "freq_hit": v in freq_top[d], "tok": tk.decode([v])}
                    per.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for key in (("all",) + (("clean",) if not leak else ("leak",))):
                        agg[f"{key}_n"] += 1
                        agg[f"{key}_hit5"] += rank < 5
                        agg[f"{key}_hit1"] += rank < 1
                        agg[f"{key}_lp"] += lp
                        agg[f"{key}_freq5"] += v in freq_top[d]
            if b0 % (BS * 20) == 0:
                log(f"{d} {b0}/{len(items)}")
        r_ = {"sentences": len(items), "skipped": dict(skip), "no_target_reads": agg["no_target"]}
        for key in ("all", "clean", "leak"):
            n = max(agg[f"{key}_n"], 1)
            r_[key] = {"n": agg[f"{key}_n"], "hit5": agg[f"{key}_hit5"] / n, "hit1": agg[f"{key}_hit1"] / n,
                       "mean_lp": agg[f"{key}_lp"] / n, "freq_hit5": agg[f"{key}_freq5"] / n}
        r_["leak_share"] = agg["leak_n"] / max(agg["all_n"], 1)
        res[d] = r_
        log(f"{d}: {json.dumps(r_)}")
    m = sum(res[d]["clean"]["hit5"] for d in DIRS) / 2
    res["mean_clean_hit5"] = m
    res["GO"] = m >= 0.15
    res["rule"] = "change 46 stage 0: mean over directions of non-leak top-5 hit >= 0.15"
    json.dump(res, open(out / "result.json", "x"), indent=1, ensure_ascii=False)
    log(f"mean clean hit5 {m:.4f} GO={res['GO']}")


if __name__ == "__main__":
    main()
