"""Dev-set (training-domain held-out) metrics for choosing the bayesrisk costs without RealSI.

Each block file is scored on the text the Talker would say ("spoken" when the run cleaned it) against the items'
references, on a modelled speech timeline: a block starts when it is written and the previous one has finished, and
lasts rate s/unit divided by the queue speed-up of si_infer --speed-policy queue (3 steps a second of queue, at most
6, ~1.67% a step). Reported: text BLEU, LAAL on block starts, one-unit blocks, silences >= 1 s inside a sentence
per 100 units, first-speech delay p50.

--timeline full adds the listening measures that decided RealSI (reports/uniss_m2_fluency_results_v2/abrupt/
abrupt.py), on the same modelled timeline: every silence >= 0.5 s between two blocks of a sentence, the seam type
at it (si_fluency_stats.seam_types: S sentence / C clause / P phrase / P-fw after a function word / W inside a word),
whether the source was speaking during it (energy VAD of the item's audio, as abrupt.py's src_vad), and
abrupt = mid-clause seam (P, P-fw, W) while the source speaks, per minute of modelled speech. Pauses inside a block
(the Talker's own) are not modelled.

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


def src_vad(path: str) -> np.ndarray:
    """10 ms frames, True while the source speaks (abrupt.py: 25 dB below the loud 90th percentile, gaps < 0.3 s
    inside speech filled)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    hop = sr // 100
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, bool)
    db = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    loud = db[db > -70]
    on = db > max(np.percentile(loud, 90) - 25, -60) if len(loud) else np.zeros(n, bool)
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if j - i < 30 and i > 0 and j < n:
                on[i:j] = True
            i = j
        else:
            i += 1
    return on


def score(path: str, refs: dict, durs: dict, lang: str, rate: float, audio: dict | None = None) -> dict:
    hyps, R, lat, one, gaps, first, nunits = [], [], [], 0, 0, [], 0
    nblocks = 0
    full = audio is not None
    if full:
        from collections import Counter
        from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import seam_types
        seams_all, seams_pause, n_pause, n_abrupt, n_speaking, speech_s = Counter(), Counter(), 0, 0, 0, 0.0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] not in refs:
            continue
        end, delays, text = 0.0, [], ""
        spans = []
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
            spans.append((start, start + dur, b))
            end = start + dur
            delays += [start] * len(us)
            nblocks += 1
            one += len(us) == 1
            if len(delays) == len(us):
                first.append(start / 1000)
        nunits += len(delays)
        if full and spans:
            speech_s += sum(e - s_ for s_, e, _ in spans) / 1000
            bl = [b for _, _, b in spans]
            types = seam_types(bl, "cmn" if lang == "cmn" else "eng", text) if len(bl) > 1 else []
            seams_all.update(types)
            on = audio[r["id"]]
            for k in range(len(spans) - 1):
                g0, g1 = spans[k][1], spans[k + 1][0]
                if g1 - g0 < 500:
                    continue
                seg = on[int(g0 // 10): int(g1 // 10)]
                speaking = bool(len(seg)) and seg.mean() > 0.5
                n_pause += 1
                n_speaking += speaking
                seams_pause[types[k]] += 1
                n_abrupt += speaking and types[k] in ("P", "P-fw", "W")
        hyps.append(text.strip())
        R.append(refs[r["id"]])
        if delays:
            lat.append(laal(delays, durs[r["id"]], len(units(refs[r["id"]], lang))) / 1000)
    res = {"bleu": round(corpus_bleu(hyps, R, language=lang)["score"], 2), "laal_s": round(float(np.mean(lat)), 3),
           "one_unit_blocks": round(one / max(nblocks, 1), 3), "gaps_ge1s_per_100u": round(100 * gaps / max(nunits, 1), 2),
           "first_speech_s_p50": round(float(np.median(first)), 2), "sentences": len(hyps)}
    if full:
        ns = max(sum(seams_all.values()), 1)
        res.update({"pauses_ge0.5_per_min": round(60 * n_pause / max(speech_s, 1e-9), 2),
                    "abrupt_per_min": round(60 * n_abrupt / max(speech_s, 1e-9), 2),
                    "abrupt_share": round(n_abrupt / max(n_pause, 1), 3),
                    "pause_source_speaking": round(n_speaking / max(n_pause, 1), 3),
                    "seam_mid_phrase_share": round(sum(seams_all[t] for t in ("P", "P-fw", "W")) / ns, 3),
                    "seam_clause_share": round(seams_all["C"] / ns, 3),
                    "seams": {k: round(v / ns, 3) for k, v in seams_all.most_common()}})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--items", default="data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl")
    ap.add_argument("--run", action="append", required=True, help="name=blocks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeline", choices=("gaps", "full"), default="gaps")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    lang = "cmn" if args.direction == "en2zh" else "eng"
    rate = 0.191 if lang == "cmn" else 0.339
    refs, durs, wavs = {}, {}, {}
    for line in open(args.items.format(d=args.direction), encoding="utf-8"):
        r = json.loads(line)
        refs[r["id"]], durs[r["id"]], wavs[r["id"]] = r["translation"], r["dur_ms"], r["audio"]
    audio = {k: src_vad(v) for k, v in wavs.items()} if args.timeline == "full" else None
    res = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        res[name] = score(path, refs, durs, lang, rate, audio)
        print(name, json.dumps(res[name]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
