"""Game listening demo (reports/mist_w3po_v1/game_multispk_demo_plan.zh-CN.md §3): speaker-blind BLEU against the SILVER
references. tcpBLEU is not usable here: the silver turns' speakers are clip-local (one id per 30 s pipeline clip), so
the per-speaker matching leaves most references unmatched. Instead, per excerpt, all output segments in time order
(window start_s + segment start_s) form one document and all reference turns in time order the other; BLEU per
excerpt and corpus BLEU over the excerpts (sacrebleu; 13a for English, zh for Chinese). Indicative only.

    python -m experiments.mist_w3po_v1.evaluation.game_blind_bleu --dir zh2en|en2zh --hyp <hyp.jsonl|hyp_asr.jsonl>
        [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sacrebleu.metrics import BLEU

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = a.dir
    W = {w["id"]: w for w in map(json.loads, open(REPO / f"eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1/windows_test_online_{d}.jsonl"))}
    hyp = defaultdict(list)
    for h in map(json.loads, open(a.hyp, encoding="utf-8")):
        w = W.get(h["id"])
        if not w:
            continue
        for s in h.get("segments") or []:
            if (s.get("text") or "").strip():
                hyp[w["meeting"]].append((w["start_s"] + s["start_s"], s["text"].strip()))
    ref = defaultdict(list)
    for r in map(json.loads, open(REPO / f"data/game_esports_si_v1/{d}/refs_windows.jsonl", encoding="utf-8")):
        ref[r["meeting"]].append((r["start_s"], r["utts"][0]["ref"].strip()))
    sep = " " if d == "zh2en" else ""
    bleu = BLEU(tokenize="13a" if d == "zh2en" else "zh")
    ms = sorted(ref)
    H = [sep.join(t for _, t in sorted(hyp[m])) for m in ms]
    R = [sep.join(t for _, t in sorted(ref[m])) for m in ms]
    res = {"dir": d, "hyp": a.hyp, "corpus": round(bleu.corpus_score(H, [R]).score, 2),
           "per_meeting": {m: round(BLEU(tokenize=bleu.tokenizer.signature() if False else ("13a" if d == "zh2en" else "zh"),
                                         effective_order=True).sentence_score(h, [r]).score, 2)
                           for m, h, r in zip(ms, H, R)},
           "len_ratio": round(sum(len(x) for x in H) / max(1, sum(len(x) for x in R)), 3)}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        p = Path(a.out)
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
