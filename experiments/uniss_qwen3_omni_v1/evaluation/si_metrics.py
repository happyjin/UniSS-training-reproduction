"""Scores for si_infer's output: ASR-BLEU on the timeline audio, and when the speech came.

  ASR-BLEU   the paper's ASR (Whisper-large-v3 into English, Paraformer into
             Chinese) on the simultaneous output as heard, waits included;
             SacreBLEU against RealSI's sentence translation, normalised as
             every other score here (evaluation.text_metrics).
  start      first sound: when the first block starts playing plus its leading
             silence codes, from the source's start.
  end        when the last block stops, from the source's end.
  silence    share of the output span, after the first sound, spent waiting.
  LAAL*      length-adaptive average lagging over the output's words (Chinese
             characters, English words), each placed inside its block's play
             time by its position in the block's text -- an estimate until
             the output is force-aligned.
  closing    blocks that closed themselves (their position's closing token), not
             forced by the guard's hard length limit.
  pace       per block of 3+ units, Chinese characters or English syllables a
             second of its speech; stretched: blocks slower than 0.7x their
             sentence's median block pace (a piece drawn out against the rest).

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_metrics --run <si_infer output> [--gpu 4]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

CJK = re.compile(r"[一-鿿]|[A-Za-z0-9']+")
EN = re.compile(r"[A-Za-z0-9']+")


def units(text: str, lang: str) -> list[str]:
    return (CJK if lang == "cmn" else EN).findall(text)


def laal(delays: list[float], src_ms: float, ref_len: int) -> float:
    if not delays:
        return float("nan")
    rate = src_ms / max(len(delays), ref_len, 1)
    total, n = 0.0, 0
    for i, d in enumerate(delays):
        total += d - i * rate
        n += 1
        if d >= src_ms:
            break
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    run = Path(args.run)
    recs = sorted((json.loads(l) for p in glob.glob(str(run / "part_*.jsonl")) for l in open(p, encoding="utf-8")),
                  key=lambda r: r["id"])
    with open(run / "results.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    asr = run / "asr.jsonl"
    if not asr.exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        subprocess.run([sys.executable, "-m", "evaluation.asr_transcribe", "--input", str(run / "results.jsonl"),
                        "--output", str(asr), "--device", "cuda:0", "--batch-size", "8"], check=True, env=env)
    heard = {json.loads(l)["id"]: json.loads(l) for l in open(asr, encoding="utf-8")}

    from evaluation.text_metrics import corpus_bleu
    lang = recs[0]["tgt_lang"]
    hyps = [str(heard.get(r["id"], {}).get("text", heard.get(r["id"], {}).get("asr_text", ""))) for r in recs]
    refs = [r["translation_ref"] for r in recs]
    bleu = corpus_bleu(hyps, refs, language=lang)
    text_bleu = corpus_bleu([r["translated_text"] for r in recs], refs, language=lang)
    lags, starts, ends, sil, closing, blocks = [], [], [], [], [], []
    paces, stretched, judged = [], 0, 0
    for r in recs:
        mine = []
        for b in r["blocks"]:
            n = (len(re.findall(r"[一-鿿]", b["text"])) if lang == "cmn" else
                 sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in EN.findall(b["text"])))
            if n >= 3 and b["dur_ms"] > 0:
                mine.append(n / (b["dur_ms"] / 1000))
        paces += mine
        if len(mine) >= 2:
            med = float(np.median(mine))
            stretched += sum(p < 0.7 * med for p in mine)
            judged += len(mine)
        delays = []
        for b in r["blocks"]:
            u = units(b["text"], lang)
            delays += [b["start_ms"] + (k + 0.5) / max(len(u), 1) * b["dur_ms"] for k in range(len(u))]
            closing.append(bool(b["closed"]))
        lags.append(laal(delays, r["source_duration_ms"], len(units(r["translation_ref"], lang))))
        if r["start_offset_ms"] is not None:
            starts.append(r["start_offset_ms"])
        ends.append(r["end_offset_ms"])
        sil.append(r["silence_ratio"])
        blocks.append(len(r["blocks"]))
    out = {"sentences": len(recs), "asr_bleu": round(float(bleu["score"]), 2), "oracle_text_bleu": round(float(text_bleu["score"]), 2),
           "start_offset_s_p50": round(float(np.median(starts)) / 1000, 2) if starts else None,
           "end_offset_s_p50": round(float(np.median(ends)) / 1000, 2), "laal_est_s_mean": round(float(np.nanmean(lags)) / 1000, 2),
           "silence_ratio_mean": round(float(np.mean(sil)), 3), "blocks_mean": round(float(np.mean(blocks)), 2),
           "closed_unforced": round(float(np.mean(closing)), 3), "no_audio": sum(1 for r in recs if r["error"]),
           "pace_p10_p50": [round(float(x), 2) for x in np.percentile(paces, [10, 50])] if paces else None,
           "stretched_blocks": round(stretched / judged, 3) if judged else None,
           "start_offset_s_p90": round(float(np.percentile(starts, 90)) / 1000, 2) if starts else None,
           "start_under_3s": round(float(np.mean(np.array(starts) < 3000)), 3) if starts else None}
    (run / "metrics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
