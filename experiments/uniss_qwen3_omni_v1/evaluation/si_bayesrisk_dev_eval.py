"""Dev-set (training-domain held-out) metrics for choosing the bayesrisk costs without RealSI.

Each block file is scored on the text the Talker would say ("spoken" when the run cleaned it) against the items'
references, on a modelled speech timeline: a block starts when it is written and the previous one has finished, and
lasts rate s/unit divided by the queue speed-up of si_infer --speed-policy queue (3 steps a second of queue, at most
6, ~1.67% a step). Reported: text BLEU, LAAL on block starts, one-unit blocks, silences >= 1 s inside a sentence
per 100 units, first-speech delay p50.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_bayesrisk_dev_eval --direction en2zh \
        --run name=blocks.jsonl ... --out dev.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.text_metrics import corpus_bleu
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units


def score(path: str, refs: dict, durs: dict, lang: str, rate: float) -> dict:
    hyps, R, lat, one, gaps, first, nunits = [], [], [], 0, 0, [], 0
    nblocks = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] not in refs:
            continue
        end, delays, text = 0.0, [], ""
        for b in r["blocks"]:
            us = units(b["text"], lang)
            text += b["text"]
            if not us:
                continue
            q = max(0.0, end - b["write_ms"]) / 1000
            speed = 1 + min(6.0, 3.0 * q) / 60
            start = max(float(b["write_ms"]), end)
            if end and start - end >= 1000 and not b.get("final"):
                gaps += 1
            dur = len(us) * rate / speed * 1000
            end = start + dur
            delays += [start] * len(us)
            nblocks += 1
            one += len(us) == 1
            if len(delays) == len(us):
                first.append(start / 1000)
        nunits += len(delays)
        hyps.append(text.strip())
        R.append(refs[r["id"]])
        if delays:
            lat.append(laal(delays, durs[r["id"]], len(units(refs[r["id"]], lang))) / 1000)
    return {"bleu": round(corpus_bleu(hyps, R, language=lang)["score"], 2), "laal_s": round(float(np.mean(lat)), 3),
            "one_unit_blocks": round(one / max(nblocks, 1), 3), "gaps_ge1s_per_100u": round(100 * gaps / max(nunits, 1), 2),
            "first_speech_s_p50": round(float(np.median(first)), 2), "sentences": len(hyps)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--items", default="data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl")
    ap.add_argument("--run", action="append", required=True, help="name=blocks.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    lang = "cmn" if args.direction == "en2zh" else "eng"
    rate = 0.191 if lang == "cmn" else 0.339
    refs, durs = {}, {}
    for line in open(args.items.format(d=args.direction), encoding="utf-8"):
        r = json.loads(line)
        refs[r["id"]], durs[r["id"]] = r["translation"], r["dur_ms"]
    res = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        res[name] = score(path, refs, durs, lang, rate)
        print(name, json.dumps(res[name]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
