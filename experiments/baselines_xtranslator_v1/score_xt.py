"""Score one X-Translator RealSI run with the paper's protocol (R4, paper/baselines_run_plan_xtranslator_v1.md §4).

The numbers come from R1's score_seamless.score() unchanged (same results-row format): ASR-BLEU on asr_speech.jsonl
(speech-only audio, Whisper-large-v3 / Paraformer, evaluation.text_metrics.corpus_bleu, missing ASR = "", coverage
>= 99 %), text BLEU on translated_text (LMT output), D1 = collect_results.realsi_points segment LAAL (block start_ms,
queue included), D0w = laal_defs.sentence_defs D0w (block write_ms). This file only adds X-Translator's own counts.

    PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
        experiments/baselines_xtranslator_v1/score_xt.py <run dir> [--require-asr --out <json>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("score_seamless", REPO / "experiments/s2st_baselines_v1/seamless/score_seamless.py")
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)


def score(run: Path, require_asr: bool) -> dict:
    out = SS.score(run, require_asr)
    recs = SS.merged(run)
    point = recs[0]["config"]["point"]
    direction = recs[0]["config"]["direction"]
    segs = [s for r in recs for s in r["segments"]]
    res = {"system": "X-Translator (Qwen3-ASR-1.7B + LMT-60-8B + X-Voice 0.4B Stage2, its own default stack)",
           "point": point, "direction": direction, "n": len(recs), **out}
    res["n_no_output"] = sum(1 for r in recs if not r["blocks"])
    res["n_error_rows"] = sum(1 for r in recs if r.get("error"))
    res["n_segments"] = len(segs)
    res["n_tts_fail_segments"] = sum(1 for s in segs if s.get("tts_error"))
    res["n_mt_fail_segments"] = sum(1 for s in segs if s.get("mt_error"))
    res["n_unspoken_sentences"] = sum(1 for r in recs if r.get("unspoken_text"))
    res["blocks_per_sentence"] = round(st.mean(len(r["blocks"]) for r in recs), 2)
    X = {r["id"]: r["source_duration_ms"] for r in recs}
    res["n_last_commit_before_end"] = sum(1 for r in recs if r["blocks"] and r["blocks"][-1]["write_ms"] < r["source_duration_ms"])
    res["share_units_written_at_end"] = None
    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units
    tot = end = 0
    for r in recs:
        for b in r["blocks"]:
            n = len(units(b["text"], r["tgt_lang"]))
            tot += n
            if b["write_ms"] >= X[r["id"]]:
                end += n
    res["share_units_written_at_end"] = round(end / tot, 4) if tot else None
    c = Counter()
    for r in recs:
        for k, v in (r.get("counters") or {}).items():
            if isinstance(v, (int, float)):
                c[k] += v
    res["speaker_boundary_flushes"] = c.get("boundary_flushes", 0)
    res["embedding_backends"] = dict(Counter((r.get("counters") or {}).get("embedding_backend", "?") for r in recs))
    res["registration"] = "paper/baselines_run_plan_xtranslator_v1.md (+ amendment 1)"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--require-asr", action="store_true")
    ap.add_argument("--out", default="", help="refuses an existing file")
    a = ap.parse_args()
    r = score(Path(a.run), a.require_asr)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("config", "asr_bleu_detail")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
