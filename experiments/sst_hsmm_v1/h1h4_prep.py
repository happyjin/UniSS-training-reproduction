"""plan v3 (change 56) data prep for H1 (horizon-limited anticipation) and H4 (natural-monotone tags). CPU only.
For every train row of eval_outputs/sst_hsmm_v1/data/train_<d>.jsonl (the 20k + 20k rows E1 / P2F trained on) reads the UniST rows
(data/unist_simul_v1/rows) and writes one record per row to <out>/prep.jsonl:
  sup  per token of the row's `tok` (the reference tokenisation used in training): the evidence time (ms) of the target word the
       token starts in, under the alignment-only rule support_omni_ms (OmniAlign strong links; NOT hybrid, which mixes in the
       Thinker's own log-probabilities); null for tokens outside any target word or for words without a strong link.
  cm   running maximum of sup (nulls take the previous value, 0 at the start): token k may be written once cm[k] has been heard.
  tau  Kendall tau of the OmniAlign links (confidence >= 0.5), target order vs source order (1 = fully monotone); null if < 3 links.
H1 cuts the target at read c (s) after the last token with cm <= 1000 c + 1000 (horizon 1 s). H4 tags the 30 % most monotone rows
per direction (threshold written to <out>/mono.json).
    python -m experiments.sst_hsmm_v1.h1h4_prep --out eval_outputs/sst_hsmm_v1/h1h4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "eval_outputs/sst_hsmm_v1/data"
ROWS = REPO / "data/unist_simul_v1/rows"
QWEN3 = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DIRS = ("en2zh", "zh2en")
MONO_FRAC = 0.30


def spans(text, words):
    low, p, out = text.lower(), 0, []
    for w in words:
        t = w["text"].lower()
        k = low.find(t, p)
        if k < 0:
            return None
        out.append((k, k + len(t)))
        p = k + len(t)
    return out


def ktau(links):
    L = sorted((l["target_index"], l["source_index"]) for l in links if l.get("confidence", 0) >= 0.5)
    if len(L) < 3:
        return None
    c = d = 0
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            a, b = L[j][0] - L[i][0], L[j][1] - L[i][1]
            if a == 0 or b == 0:
                continue
            if a * b > 0:
                c += 1
            else:
                d += 1
    return None if c + d == 0 else (c - d) / (c + d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "prep.jsonl").exists():
        sys.exit(f"exists: {out}/prep.jsonl (refuse to overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(QWEN3))
    refs = {}
    for d in DIRS:
        for r in map(json.loads, open(DATA / f"train_{d}.jsonl", encoding="utf-8")):
            refs[r["id"]] = r
    got, skip = {}, {"tok": 0, "span": 0, "len": 0}
    cols = ["id", "target_words", "support_omni_ms", "word_links"]
    for f in sorted(ROWS.glob("part-*.parquet")):
        for row in pq.read_table(f, columns=cols).to_pylist():
            r = refs.get(row["id"])
            if r is None or row["id"] in got:
                continue
            ref, tok = r["ref"], (json.loads(r["tok"]) if isinstance(r["tok"], str) else list(r["tok"]))
            enc = tk(ref, return_offsets_mapping=True, add_special_tokens=False)
            if enc["input_ids"] != tok:
                skip["tok"] += 1; continue
            tw = json.loads(row["target_words"]); sup_w = json.loads(row["support_omni_ms"])
            if len(tw) != len(sup_w):
                skip["len"] += 1; continue
            sp = spans(ref, tw)
            if sp is None:
                skip["span"] += 1; continue
            sup = []
            for (o0, o1) in enc["offset_mapping"]:
                while o0 < o1 - 1 and ref[o0].isspace():     # byte-level BPE: " word" starts at the space
                    o0 += 1
                k = next((i for i, (s0, s1) in enumerate(sp) if s0 <= o0 < s1), None)
                sup.append(None if k is None else sup_w[k])
            cm, m = [], 0
            for v in sup:
                if v is not None:
                    m = max(m, int(v))
                cm.append(m)
            got[row["id"]] = {"id": row["id"], "dir": r["dir"], "sup": sup, "cm": cm,
                              "tau": ktau(json.loads(row["word_links"]) if row["word_links"] else [])}
    with open(out / "prep.jsonl", "x", encoding="utf-8") as fh:
        for i in refs:
            if i in got:
                fh.write(json.dumps(got[i]) + "\n")
    mono = {}
    for d in DIRS:
        ts = sorted(x["tau"] for x in got.values() if x["dir"] == d and x["tau"] is not None)
        thr = ts[int(len(ts) * (1 - MONO_FRAC))] if ts else None
        mono[d] = {"threshold": thr, "n_tau": len(ts), "n_rows": sum(1 for x in got.values() if x["dir"] == d),
                   "n_mono": sum(1 for x in got.values() if x["dir"] == d and x["tau"] is not None and x["tau"] >= thr),
                   "tau_mean": sum(ts) / max(1, len(ts)), "tau_q": [ts[int(q * (len(ts) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]}
    json.dump({"skip": skip, "missing": len(refs) - len(got) - sum(skip.values()), "mono": mono}, open(out / "mono.json", "x"), indent=1)
    print(json.dumps({"rows": len(got), "skip": skip, "mono": mono}, indent=1))


if __name__ == "__main__":
    main()
