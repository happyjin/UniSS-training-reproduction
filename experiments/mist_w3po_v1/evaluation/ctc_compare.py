"""Whisper vs an independent CTC ASR on the same talker clips (CPU; reports/mist_w3po_v1/text_to_speech_gap_plan.zh-CN.md).

Inputs: a talker run (its Whisper rows), the speech_gap_decomp streams.jsonl of that run, the ctc_asr_check shards.
  counts      words of the talker input / Whisper / CTC (repo-normalised) on the tail (13a Whisper / text > 1.5) and
              the rest; the tail split into whisper_only_long (CTC / text <= 1.2: Whisper wrote words nobody spoke),
              both_long (CTC / text > 1.5 as well: extra speech), other
  per word    every talker-input word, aligned to each transcript: heard by both / Whisper only / CTC only / neither
  tcpASR-BLEU the CTC transcripts scored exactly like the Whisper ones (repo and whisper protocols; CTC has no case or
              punctuation, so its raw score is not meaningful), next to Whisper's and the text's
Writes <out> (json; refuses to overwrite). Dev only.

    python -m experiments.mist_w3po_v1.evaluation.ctc_compare --run <decode dir> --speech <speech dir> \
        --streams <streams.jsonl> --ctc <shard.jsonl ...> --out <json>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation import metric_headroom as M
from experiments.mist_w3po_v1.evaluation import speech_gap_decomp as G


def tts_text(row):
    """the talker's input with its blocks joined by a space (translated_text joins them without one, which glues the
    last word of a block to the first of the next and would count as errors in a word alignment)."""
    return " ".join(b["text"].strip() for b in row["blocks"] if b["text"].strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--speech", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--ctc", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    G.NORMS = G.norms()
    from meeting_score import universe
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.speech) / "talker/asr_speech_merged.jsonl", encoding="utf-8")}
    S = {json.loads(l)["id"]: json.loads(l) for l in open(args.streams, encoding="utf-8")}
    C = {}
    for f in args.ctc:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            C[r["id"]] = r["ctc_text"]
    ids = sorted(set(C) & set(rows))
    res = {"clips": len(ids), "streams": len(rows)}
    grp = {"tail": [], "rest": []}
    cls = Counter()
    word = {g: Counter() for g in ("tail", "rest", "all")}
    for k in ids:
        tts = G.words(tts_text(rows[k]))
        wh = G.words(rows[k].get("asr_text") or "")
        ct = G.words(C[k])
        n = max(len(tts), 1)
        g = "tail" if S[k]["ratio13a"] > 1.5 else "rest"
        ow, oc = G.align(tts, wh), G.align(tts, ct)
        ew = sum(op != "eq" for op, *_ in ow)
        ec = sum(op != "eq" for op, *_ in oc)
        grp[g].append({"n": len(tts), "wh": len(wh), "ct": len(ct), "ew": ew, "ec": ec})
        if g == "tail":
            rc = len(ct) / n
            cls["whisper_only_long" if rc <= 1.2 else ("both_long" if rc > 1.5 else "other")] += 1
        hw = {ri for op, _, _, ri in ow if op == "eq"}
        hc = {ri for op, _, _, ri in oc if op == "eq"}
        for i in range(len(tts)):
            c = ("both" if i in hw and i in hc else "whisper_only" if i in hw else "ctc_only" if i in hc else "neither")
            word[g][c] += 1
            word["all"][c] += 1
    for g, L in grp.items():
        if not L:
            continue
        nn = sum(x["n"] for x in L)
        res[g] = {"clips": len(L), "tts_words": nn, "whisper_over_tts": round(sum(x["wh"] for x in L) / nn, 3),
                  "ctc_over_tts": round(sum(x["ct"] for x in L) / nn, 3), "wer_whisper": round(sum(x["ew"] for x in L) / nn, 4),
                  "wer_ctc": round(sum(x["ec"] for x in L) / nn, 4),
                  "median_ctc_ratio": float(np.median([x["ct"] / max(x["n"], 1) for x in L])),
                  "median_whisper_ratio": float(np.median([x["wh"] / max(x["n"], 1) for x in L]))}
    res["tail_class"] = dict(cls)
    res["word_heard"] = {g: {k: round(v / max(sum(c.values()), 1), 4) for k, v in c.items()} for g, c in word.items()}
    # leakage: error words (sub / ins) of a stream that occur in another stream's input of the same window and not in
    # its own; baseline: the same against the streams of a random window of another meeting
    from collections import defaultdict
    ww = defaultdict(dict)
    for k in ids:
        wid, spk = k.split("__S")
        ww[wid][spk] = {G.key(w) for w in G.words(tts_text(rows[k]))}
    rng = np.random.default_rng(0)
    wl = sorted(ww)
    lk = Counter()
    for k in ids:
        wid, spk = k.split("__S")
        if len(ww[wid]) < 2:
            continue
        own = ww[wid][spk]
        oth = set().union(*[v for s2, v in ww[wid].items() if s2 != spk])
        while True:
            ow_ = wl[int(rng.integers(len(wl)))]
            if ow_.split("_on_")[0] != wid.split("_on_")[0]:
                break
        base = set().union(*ww[ow_].values())
        tts = G.words(tts_text(rows[k]))
        for name, hyp in (("whisper", rows[k].get("asr_text") or ""), ("ctc", C[k])):
            eh = [G.key(h) for op, _, h, _ in G.align(tts, G.words(hyp)) if op in ("sub", "ins")]
            lk[f"{name}_err"] += len(eh)
            lk[f"{name}_leak"] += sum(1 for w in eh if w in oth and w not in own)
            lk[f"{name}_base"] += sum(1 for w in eh if w in base and w not in own)
    res["leak"] = {**dict(lk), **{f"{n}_{x}_share": round(lk[f"{n}_{x}"] / max(lk[f"{n}_err"], 1), 4)
                                  for n in ("whisper", "ctc") for x in ("leak", "base")}}
    if len(ids) == len(rows):
        W = [json.loads(l) for l in open(M.REPO / M.WIN, encoding="utf-8")]
        H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
        RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
        U = universe(M.DR, {w["meeting"] for w in W})
        sc = M.scored_utts(W, RW)
        res["tcpASR_BLEU_ctc"] = G.score(W, G.stream_hyp(H, {k: C[k] for k in ids}), sc, U)
        # the two ASRs combined the safe way: per stream, the CTC transcript where Whisper wrote > 1.5x the CTC words
        comb = {}
        for k in ids:
            wh, ct = rows[k].get("asr_text") or "", C[k]
            comb[k] = ct if len(G.words(wh)) > 1.5 * max(len(G.words(ct)), 1) else wh
        res["tcpASR_BLEU_whisper_ctc_guard"] = G.score(W, G.stream_hyp(H, comb), sc, U)
        res["guard_streams"] = sum(comb[k] != (rows[k].get("asr_text") or "") for k in ids)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "word_heard"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
