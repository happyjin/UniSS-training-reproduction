"""change 57 B1 (term-preserving targets, report + training data): rewrite the en2zh training references of
eval_outputs/sst_hsmm_v1/data/train_en2zh.jsonl (the 20k UniST rows E1 trained on) so that RARE English proper names stay in Latin
script, the convention of the open CoVoST dev references (cvdev: rare person / place names kept, well-known ones translated).
Rule (calibrated on the 300 open cvdev sentences only, never on test):
  candidate source token = not sentence-initial, starts with an upper-case letter, not "I"/"I'm"..., and its lower-cased form
  occurs < RARE times in all 557k English UniST transcriptions (eval_outputs/sst_hsmm_v1/b1prep/en_freq.json); consecutive
  candidates form one span.
  The span is replaced only if its OmniAlign links (confidence >= 0.5) point to a non-empty CONTIGUOUS run of target words, every
  link of those target words stays inside the span, the run is CJK-only, and its text occurs exactly once in the reference.
  The run is replaced by the source span text (possessive 's and edge punctuation dropped), with a space on a CJK side.
zh2en rows are not touched. tok is re-tokenised (Qwen3-Omni tokenizer, no special tokens). Writes <out>/train_en2zh.jsonl and
<out>/changes.jsonl.
    python -m experiments.sst_hsmm_v1.b1_rewrite --out eval_outputs/sst_hsmm_v1/b1prep
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
ROWS = REPO / "data/unist_simul_v1/rows"
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
RARE = 15
CJK = re.compile(r"^[一-鿿]+$")
PRON = {"i", "i'm", "i'll", "i've", "i'd"}


def clean_tok(w):
    w = re.sub(r"(['’]s|['’])$", "", w.strip())
    return w.strip(".,;:!?\"“”()[]")


def rewrite(ref, sw, tw, links, freq):
    cand = []
    for i, w in enumerate(sw):
        t = clean_tok(w)
        ok = (i > 0 and t and t[0].isupper() and t.lower() not in PRON and re.match(r"^[A-Za-z][A-Za-z\-]*$", t)
              and freq.get(t.lower(), 0) < RARE)
        cand.append(bool(ok))
    spans, i = [], 0
    while i < len(sw):
        if cand[i]:
            j = i
            while j + 1 < len(sw) and cand[j + 1]:
                j += 1
            spans.append((i, j)); i = j + 1
        else:
            i += 1
    L = [(l["source_index"], l["target_index"]) for l in links if l.get("confidence", 0) >= 0.5]
    out, done = ref, []
    for a, b in spans:
        T = sorted({t for s, t in L if a <= s <= b})
        if not T or T != list(range(T[0], T[-1] + 1)):
            continue
        if any(not (a <= s <= b) for s, t in L if t in T):
            continue
        zh = "".join(tw[t] for t in T)
        if not CJK.match(zh) or out.count(zh) != 1:
            continue
        en = " ".join(clean_tok(sw[k]) for k in range(a, b + 1))
        k = out.index(zh)
        pre = " " if k > 0 and re.match(r"[一-鿿]", out[k - 1]) else ""
        post = " " if k + len(zh) < len(out) and re.match(r"[一-鿿]", out[k + len(zh)]) else ""
        out = out[:k] + pre + en + post + out[k + len(zh):]
        done.append({"en": en, "zh": zh})
    return out, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for x in ("train_en2zh.jsonl", "changes.jsonl"):
        if (out / x).exists():
            sys.exit(f"exists: {out / x} (refuse to overwrite)")
    freq = json.load(open(out / "en_freq.json"))["count"]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(QWEN3))
    rows = [json.loads(l) for l in open(DATA / "train_en2zh.jsonl", encoding="utf-8")]
    want = {r["id"] for r in rows}
    U = {}
    for f in sorted(ROWS.glob("part-*.parquet")):
        for r in pq.read_table(f, columns=["id", "source_words", "target_words", "word_links"]).to_pylist():
            if r["id"] in want and r["id"] not in U:
                U[r["id"]] = r
    n_ch = n_sp = 0
    with open(out / "train_en2zh.jsonl", "x", encoding="utf-8") as fo, open(out / "changes.jsonl", "x", encoding="utf-8") as fc:
        for r in rows:
            u = U.get(r["id"])
            new, done = r["ref"], []
            if u is not None and u["word_links"]:
                sw = [w["text"] for w in json.loads(u["source_words"])]
                tw = [w["text"] for w in json.loads(u["target_words"])]
                new, done = rewrite(r["ref"], sw, tw, json.loads(u["word_links"]), freq)
            if done:
                n_ch += 1; n_sp += len(done)
                fc.write(json.dumps({"id": r["id"], "src": r["src"], "old": r["ref"], "new": new, "spans": done}, ensure_ascii=False) + "\n")
                r = dict(r, ref=new, tok=tk(new, add_special_tokens=False)["input_ids"])
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat = sum(len(re.findall("[A-Za-z]", json.loads(l)["ref"])) for l in open(out / "train_en2zh.jsonl"))
    tot = sum(len(re.sub(r"\s", "", json.loads(l)["ref"])) for l in open(out / "train_en2zh.jsonl"))
    print(json.dumps({"rows": len(rows), "with_alignment": len(U), "rows_changed": n_ch, "spans": n_sp, "latin_per_100ch": round(100 * lat / tot, 2)}))


if __name__ == "__main__":
    main()
