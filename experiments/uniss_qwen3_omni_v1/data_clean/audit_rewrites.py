"""Audit of interpreter-style rewrites (rewrite_interpreter_style.py) before they become training data.

Per item: language right, no meta text ("改写如下", "Here is", quotes around the whole), not empty or cut, digits kept,
length ratio to the oracle, chrF of the rewrite and of the oracle to the human reference, chrF of the rewrite to the
oracle (nothing invented). Accepted = all checks pass and chrF(rewrite, ref) >= chrF(oracle, ref) - 1 and the length
ratio is within [0.7, 1.05].
Over the accepted items: corpus BLEU of the rewrites vs the oracles against the references, and, from the oracle dumps,
the share of sentence-start reads (nothing committed) whose draft agrees with the oracle / the rewrite on >= 1 unit.
Writes <in>.audit.jsonl (each item with its checks and `accepted`), <in>.audit.json (summary) and prints random samples.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.audit_rewrites <rewrites.jsonl> [--samples 30]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

REPO = Path(__file__).resolve().parents[3]
DUMPS = "data/si_bayes_lr_v1/dump_*/dumps/*.jsonl,data/si_belief_v1/shard_*/dumps/*.jsonl"
META = re.compile(r"改写|译文如下|以下是|Here is|Here's|rewritten|Rewritten|interpreter|translation:", re.I)
CJK = re.compile(r"[一-鿿]")


def units(t: str, zh: bool) -> list[str]:
    return [c for c in t if not c.isspace()] if zh else t.split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rewrites")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--chrf-slack", type=float, default=1.0)
    args = ap.parse_args()
    src = Path(args.rewrites)
    out_items, out_sum = src.with_suffix(".audit.jsonl"), src.with_suffix(".audit.json")
    for p in (out_items, out_sum):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    chrf = CHRF()
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    audited = []
    for r in rows:
        zh = r["dir"] == "en2zh"
        rw, orc, ref = r["rewrite"].strip(), r["oracle"].strip(), r["ref"].strip()
        lang_ok = (len(CJK.findall(rw)) >= 0.5 * max(1, len(re.sub(r"\W", "", rw)))) if zh else (not CJK.search(rw) and bool(re.search(r"[A-Za-z]", rw)))
        digits_ok = sorted(re.findall(r"\d+", orc)) == sorted(re.findall(r"\d+", rw)) or not re.findall(r"\d+", orc)
        meta_ok = not META.search(rw) and not (rw[:1] in "\"“'「" and rw[-1:] in "\"”'」")
        nonempty = len(units(rw, zh)) > 0
        ratio = len(units(rw, zh)) / max(1, len(units(orc, zh)))
        c_rw, c_or = chrf.sentence_score(rw, [ref]).score, chrf.sentence_score(orc, [ref]).score
        c_fid = chrf.sentence_score(rw, [orc]).score
        acc = lang_ok and digits_ok and meta_ok and nonempty and 0.7 <= ratio <= 1.05 and c_rw >= c_or - args.chrf_slack
        audited.append({**r, "lang_ok": lang_ok, "digits_ok": digits_ok, "meta_ok": meta_ok, "nonempty": nonempty,
                        "len_ratio": round(ratio, 3), "chrf_rewrite_ref": round(c_rw, 2), "chrf_oracle_ref": round(c_or, 2),
                        "chrf_rewrite_oracle": round(c_fid, 2), "accepted": acc})
    with open(out_items, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in audited)

    acc_ids = {a["id"]: a for a in audited if a["accepted"]}
    agree = {d: [0, 0, 0] for d in ("en2zh", "zh2en")}   # start reads, agree with oracle, agree with rewrite
    for f in sorted({f for pat in DUMPS.split(",") for f in glob.glob(str(REPO / pat))}):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("n_committed") != 0 or r["id"] not in acc_ids or "oracle_units" not in r:
                continue
            zh = acc_ids[r["id"]]["dir"] == "en2zh"
            rw = units(acc_ids[r["id"]]["rewrite"], zh)
            dr, ou = r["draft_units"], r["oracle_units"]
            a = agree[acc_ids[r["id"]]["dir"]]
            a[0] += 1
            a[1] += bool(dr) and bool(ou) and dr[0] == ou[0]
            a[2] += bool(dr) and bool(rw) and dr[0] == rw[0]
    summ = {}
    for d in ("en2zh", "zh2en"):
        A = [a for a in audited if a["dir"] == d]
        if not A:
            continue
        acc = [a for a in A if a["accepted"]]
        bleu = BLEU(tokenize="zh" if d == "en2zh" else "13a")
        rate = lambda k: round(sum(a[k] for a in A) / len(A), 3)
        summ[d] = {
            "items": len(A), "accepted": len(acc), "accept_rate": round(len(acc) / len(A), 3),
            "lang_ok": rate("lang_ok"), "digits_ok": rate("digits_ok"), "meta_ok": rate("meta_ok"), "nonempty": rate("nonempty"),
            "len_ratio_p10_p50_p90": [round(float(x), 3) for x in __import__("numpy").percentile([a["len_ratio"] for a in A], [10, 50, 90])],
            "all: bleu_oracle_vs_ref": round(bleu.corpus_score([a["oracle"] for a in A], [[a["ref"] for a in A]]).score, 2),
            "all: bleu_rewrite_vs_ref": round(bleu.corpus_score([a["rewrite"] for a in A], [[a["ref"] for a in A]]).score, 2),
            "accepted: bleu_oracle_vs_ref": round(bleu.corpus_score([a["oracle"] for a in acc], [[a["ref"] for a in acc]]).score, 2) if acc else None,
            "accepted: bleu_rewrite_vs_ref": round(bleu.corpus_score([a["rewrite"] for a in acc], [[a["ref"] for a in acc]]).score, 2) if acc else None,
            "accepted: len_ratio_mean": round(sum(a["len_ratio"] for a in acc) / max(1, len(acc)), 3),
            "accepted: chrf_rewrite_oracle_mean": round(sum(a["chrf_rewrite_oracle"] for a in acc) / max(1, len(acc)), 2),
            "start_reads": agree[d][0],
            "start_first_unit_agree_oracle": round(agree[d][1] / max(1, agree[d][0]), 3),
            "start_first_unit_agree_rewrite": round(agree[d][2] / max(1, agree[d][0]), 3),
        }
    out_sum.write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    rng = random.Random(0)
    for d in ("en2zh", "zh2en"):
        A = [a for a in audited if a["dir"] == d]
        for a in rng.sample(A, min(args.samples, len(A))):
            print(f"\n[{d}] {'OK ' if a['accepted'] else 'REJ'} ratio {a['len_ratio']} chrF ref {a['chrf_oracle_ref']}->{a['chrf_rewrite_ref']}"
                  f"\n  ref : {a['ref']}\n  orc : {a['oracle']}\n  new : {a['rewrite']}")


if __name__ == "__main__":
    main()
