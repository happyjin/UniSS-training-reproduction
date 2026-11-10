"""Best-of-2 talker samples chosen by ASR consistency, from two existing TTS seeds of one decode (CPU; reports/mist_w3po_v1/
text_to_speech_gap_plan.zh-CN.md §B). Per window x speaker stream, one of the two seeds' speech is kept by a selector that
sees only the talker's own input text (never the reference):
  sel_whisper   lower WER of the Whisper transcript against the talker input (the selector IS the evaluation ASR:
                optimistic, an upper bound for best-of-2)
  sel_ctc       lower WER of an independent CTC ASR transcript (wav2vec2-large-960h-lv60-self, ctc_asr_check) against
                the talker input -- the deployable version (select_k with an independent scorer)
  seed1 / seed2 each seed alone (the official two-seed scores)
The kept stream is always scored by its Whisper transcript, exactly as tcpASR-BLEU. Also: how the per-stream errors of
the two seeds agree (systematic = both seeds WER > 0.5 vs one seed only). Writes <out> (json; refuses to overwrite).

    python -m experiments.mist_w3po_v1.evaluation.best_of_seeds --run <decode dir> --speech <seed1 dir> <seed2 dir> \
        --streams <s1.streams.jsonl> <s2.streams.jsonl> --ctc1 <shards> --ctc2 <shards> --out <json>
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--speech", nargs=2, required=True)
    ap.add_argument("--streams", nargs=2, required=True)
    ap.add_argument("--ctc1", nargs="+", required=True)
    ap.add_argument("--ctc2", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    G.NORMS = G.norms()
    from meeting_score import universe
    rows = [{json.loads(l)["id"]: json.loads(l) for l in open(Path(d) / "talker/asr_speech_merged.jsonl", encoding="utf-8")}
            for d in args.speech]
    S = [{json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} for f in args.streams]
    C = [{}, {}]
    for i, fs in enumerate((args.ctc1, args.ctc2)):
        for f in fs:
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                C[i][r["id"]] = r["ctc_text"]
    ids = sorted(set(rows[0]) & set(rows[1]))
    asr = [{k: rows[i][k].get("asr_text") or "" for k in ids} for i in (0, 1)]

    def wer(i, k, hyp):
        t = G.words(tts_text(rows[i][k]))
        return sum(op != "eq" for op, *_ in G.align(t, G.words(hyp))) / max(len(t), 1)
    ww = [{k: S[i][k]["S"] + S[i][k]["D"] + S[i][k]["I"] for k in ids} for i in (0, 1)]
    wc = [{k: wer(i, k, C[i].get(k, "")) for k in ids} for i in (0, 1)]
    sel = {"sel_whisper": {k: 1 if ww[1][k] < ww[0][k] else 0 for k in ids},
           "sel_ctc": {k: 1 if wc[1][k] < wc[0][k] else 0 for k in ids},
           "seed1": {k: 0 for k in ids}, "seed2": {k: 1 for k in ids}}
    W = [json.loads(l) for l in open(M.REPO / M.WIN, encoding="utf-8")]
    H = {h["id"]: h for h in map(json.loads, open(Path(args.run) / "hyp.jsonl", encoding="utf-8"))}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    U = universe(M.DR, {w["meeting"] for w in W})
    sc = M.scored_utts(W, RW)
    res = {"streams": len(ids), "scores": {}, "picked_seed2_share": {}}
    for name, s in sel.items():
        res["scores"][name] = G.score(W, G.stream_hyp(H, {k: asr[s[k]][k] for k in ids}), sc, U)
        res["picked_seed2_share"][name] = round(float(np.mean(list(s.values()))), 4)
    n = {k: max(S[0][k]["n"], 1) for k in ids}
    e0 = np.array([ww[0][k] / n[k] for k in ids])
    e1 = np.array([ww[1][k] / n[k] for k in ids])
    tot = sum(ww[0][k] + ww[1][k] for k in ids)
    both = sum(ww[0][k] + ww[1][k] for i, k in enumerate(ids) if e0[i] > 0.5 and e1[i] > 0.5)
    one = sum(ww[0][k] + ww[1][k] for i, k in enumerate(ids) if (e0[i] > 0.5) != (e1[i] > 0.5))
    res["seed_agreement"] = {"pearson_stream_wer": round(float(np.corrcoef(e0, e1)[0, 1]), 4),
                             "err_share_both_seeds_wer_gt0.5": round(both / max(tot, 1), 4),
                             "err_share_one_seed_wer_gt0.5": round(one / max(tot, 1), 4),
                             "streams_both_gt0.5": int(((e0 > 0.5) & (e1 > 0.5)).sum()),
                             "streams_one_gt0.5": int(((e0 > 0.5) != (e1 > 0.5)).sum()),
                             "mean_wer_seed1": round(float(sum(ww[0].values()) / sum(n.values())), 4),
                             "mean_wer_seed2": round(float(sum(ww[1].values()) / sum(n.values())), 4),
                             "mean_wer_min_of_2": round(float(sum(min(ww[0][k], ww[1][k]) for k in ids) / sum(n.values())), 4)}
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
