"""Strict filter for the prefix-constrained start oracle (build_start_oracle.py), from a manual audit of its output.

The first pass accepted a longer start target when the constrained whole translation was within 1 chrF of the free
oracle. Reading samples showed three faults it let through: untranslated English in Chinese drafts cut to a single
letter ("Y", "T"), starts whose meaning the reference does not support ("当时,它相当不错" for "那时，我心中非常清楚",
an added "绝对的"), and words cut in half ("其" of "其他"). A lengthened target is now kept only if
  * the constrained whole is at least as close to the reference as the oracle (chrF, no slack) and 0.85-1.2 its length;
  * the target has no Latin letters in Chinese unless the reference has them, no ellipsis, and (Chinese) ends at a word
    boundary of the constrained whole or at punctuation; (English) ends on a whole word;
  * its content is supported by the reference: >= 70% of its content characters (Chinese, function characters
    excluded) or content words (English, stopwords excluded, matched on their first 4 letters) occur in the reference;
    a target of function words only ("You are in.") is kept only if it is <= 3 units and does not end a sentence.
Otherwise the read keeps its old target (best_L = lcp). Writes <in stem>.strict.jsonl with `best_L` revised and
`reject` giving the reason, and prints the counts. Output: <in stem>.strict2.jsonl (the first strict pass, .strict.jsonl,
lacked the function-word rule).

    python -m experiments.uniss_qwen3_omni_v1.data_clean.filter_start_oracle <start_oracle.jsonl>
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ZH_FUNC = set("的了是在这那个和与而也都就又还被把给对着过吧呢吗啊呀嘛么一些们其之以及或")
EN_STOP = set("a an the of to in on at for with by from and or but that which who whom whose is are was were be been being "
              "am this these those it its as than so if then not no our your their his her my very more most can will would "
              "could should do does did have has had into about just also there here some any each such like what how when "
              "where why i you he she we they me him us them".split())
CJK = re.compile(r"[一-鿿]")


def main() -> None:
    src = Path(sys.argv[1])
    out = src.with_name(src.stem + ".strict2.jsonl")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    import jieba
    why = Counter()
    kept = Counter()
    with open(out, "x", encoding="utf-8") as fh:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            zh = r["dir"] == "en2zh"
            reason = None
            if r["best_L"] > r["lcp"]:
                tgt, full, ref, orc = r["target"], r["full"] or "", r["ref"], r["oracle"]
                n = lambda t: len(t.replace(" ", "")) if zh else len(t.split())
                ratio = n(full) / max(1, n(orc))
                if r["chrf_full"] is None or r["chrf_full"] < r["chrf_oracle"]:
                    reason = "chrF below the oracle"
                elif not 0.85 <= ratio <= 1.2:
                    reason = "length"
                elif "..." in tgt or "…" in tgt or ".." in tgt:
                    reason = "ellipsis"
                elif zh and re.search(r"[A-Za-z]", tgt) and not re.search(r"[A-Za-z]", ref):
                    reason = "latin in Chinese"
                elif zh:
                    pos, ends = 0, set()
                    for w in jieba.lcut(full.replace(" ", "")):
                        pos += len(w)
                        ends.add(pos)
                    t = tgt.replace(" ", "")
                    if not (t and (t[-1] in "，。！？、；：,.!?;:" or len(t) in ends)):
                        reason = "cuts a word"
                    else:
                        content = [c for c in t if CJK.match(c) and c not in ZH_FUNC]
                        if content and sum(c in ref for c in content) / len(content) < 0.7:
                            reason = "not supported by the reference"
                        elif not content and (len(t) > 3 or t[-1] in "。！？.!?"):
                            reason = "function words only"
                else:
                    words = [re.sub(r"[^a-z']", "", w.lower()) for w in tgt.split()]
                    content = [w for w in words if w and w not in EN_STOP]
                    refw = {re.sub(r"[^a-z']", "", w.lower())[:4] for w in ref.split()}
                    if content and sum(w[:4] in refw for w in content) / len(content) < 0.7:
                        reason = "not supported by the reference"
                    elif not content and (len(words) > 3 or tgt.rstrip()[-1:] in ".!?"):
                        reason = "function words only"
                if reason:
                    why[(r["dir"], reason)] += 1
                    r = {**r, "best_L": r["lcp"], "target_rejected": r["target"],
                         "target": ("".join if zh else " ".join)((list(r["draft"].replace(" ", "")) if zh else r["draft"].split())[: r["lcp"]]),
                         "reject": reason}
                else:
                    kept[r["dir"]] += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("kept lengthened:", dict(kept))
    for (d, k), v in sorted(why.items()):
        print(f"  rejected {d} {k}: {v}")


if __name__ == "__main__":
    main()
