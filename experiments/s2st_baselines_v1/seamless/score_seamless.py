"""Score SeamlessStreaming runs with the single-speaker paper's RealSI protocol (R1, paper/baselines_run_plan_seamless_v1.md).

Per run folder (realsi_<dir>, or a dev calibration config folder) holding part_*.jsonl from run_seamless.py:
  results.jsonl   the parts merged, sorted by id (written once, never overwritten)
  ASR-BLEU        asr_speech.jsonl (evaluation.asr_transcribe on results_speechonly.jsonl, written by
                  reports/uniss_si_v4/speech_only.py) scored as paper/bayes_simul_s2st_v11/scripts/collect_results.py
                  realsi_points: evaluation.text_metrics.corpus_bleu, missing ASR = "", coverage >= 99 %
  text BLEU       the agent's own text output (translated_text), same corpus_bleu
  D1              segment LAAL: realsi_points verbatim (each unit at its block's start_ms; si_metrics.laal; mean over
                  sentences with an output unit)
  D0w             laal_defs.sentence_defs()["D0w"] (each unit at its block's emission time, SimulEval arithmetic)
  also D0 (SimulEval default, one delay per speech segment), D2, StartOffset / EndOffset medians.

    PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
        experiments/s2st_baselines_v1/seamless/score_seamless.py <run folder> [--out <json>] [--require-asr]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112"))

from evaluation.text_metrics import corpus_bleu  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units  # noqa: E402
import laal_defs  # noqa: E402


def merged(run: Path) -> list[dict]:
    res = run / "results.jsonl"
    if not res.exists():
        recs = sorted((json.loads(l) for p in sorted(glob.glob(str(run / "part_*.jsonl")))
                       for l in open(p, encoding="utf-8")), key=lambda r: r["id"])
        ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids)), "duplicate ids across parts"
        with open(res, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    return [json.loads(l) for l in open(res, encoding="utf-8")]


def score(run: Path, require_asr: bool = False) -> dict:
    recs = merged(run)
    lang = recs[0]["tgt_lang"]
    refs = [r["translation_ref"] for r in recs]
    out = {"run": str(run), "n_sentences": len(recs), "tgt_lang": lang, "config": recs[0].get("config"),
           "errors": sum(1 for r in recs if r.get("error")), "no_speech": sum(1 for r in recs if not r["blocks"]),
           "restarts": sum(r.get("restarts", 0) for r in recs)}
    out["text_bleu"] = round(corpus_bleu([r["translated_text"] for r in recs], refs, language=lang)["score"], 2)
    asr = run / "asr_speech.jsonl"
    if asr.exists():
        heard = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(asr, encoding="utf-8"))}
        out["asr_coverage"] = round(sum(1 for r in recs if r["id"] in heard) / len(recs), 4)
        b = corpus_bleu([heard.get(r["id"], "") for r in recs], refs, language=lang)
        out["asr_bleu"] = round(b["score"], 2)
        out["asr_bleu_detail"] = {k: b[k] for k in ("precisions", "bp", "sys_len", "ref_len", "tokenizer")}
        if out["asr_coverage"] < 0.99:
            out["asr_bleu_invalid"] = "coverage < 99 %"
    elif require_asr:
        sys.exit(f"no asr_speech.jsonl in {run}")
    # D1: collect_results.realsi_points verbatim
    seg = []
    for r in recs:
        ds = [bl["start_ms"] for bl in r["blocks"] for _ in units(bl["text"], lang)]
        if ds:
            seg.append(laal(ds, r["source_duration_ms"], len(units(r["translation_ref"], lang))))
    out["laal_d1"] = round(st.mean(seg) / 1000, 3) if seg else None
    out["n_laal"] = len(seg)
    per = [p for p in (laal_defs.sentence_defs(r) for r in recs) if p is not None]
    for k in ("D0w", "D0", "D2", "D1"):
        out[f"laal_{k.lower()}_defs"] = round(st.mean(p[k] for p in per) / 1000, 3) if per else None
    out["laal_d0w"] = out.pop("laal_d0w_defs")
    assert out["laal_d1_defs"] is None or abs(out["laal_d1_defs"] - out["laal_d1"]) < 1e-3, (out["laal_d1_defs"], out["laal_d1"])
    so = [r["start_offset_ms"] for r in recs if r["blocks"]]
    eo = [r["end_offset_ms"] for r in recs if r["blocks"]]
    out["start_offset_s_p50"] = round(st.median(so) / 1000, 2) if so else None
    out["end_offset_s_p50"] = round(st.median(eo) / 1000, 2) if eo else None
    out["segments_per_sentence"] = round(st.mean(len(r["blocks"]) for r in recs), 2)
    out["source_s_mean"] = round(st.mean(r["source_duration_ms"] for r in recs) / 1000, 2)
    out["compute_s_per_sentence"] = round(st.mean(r.get("elapsed_s", 0) for r in recs), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="", help="write the (single) run's score json here (refuses an existing file)")
    ap.add_argument("--require-asr", action="store_true")
    a = ap.parse_args()
    res = [score(Path(r), a.require_asr) for r in a.runs]
    if a.out:
        assert len(res) == 1
        with open(a.out, "x", encoding="utf-8") as fh:
            json.dump(res[0], fh, ensure_ascii=False, indent=1)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k not in ("config", "asr_bleu_detail")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
