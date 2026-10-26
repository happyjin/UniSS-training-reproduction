"""The rest of the plan's metrics for an si_infer run, beside si_metrics.py.

  A.PCP     AutoPCP (multilingual v2 on wav2vec2-XLSR layer 9, symmetrised: the
            same settings as every A.PCP here) between the source and the output
            speech alone (no waits), as SimulS2ST-Omni scores prosody.
  SIM-src   WavLM speaker-verification cosine between the output speech and the
            source speaker -- RealSI has no reference speech in the target
            language, so the voice is checked against the person who spoke.
  LAAL      on aligned words: the ASR transcript of the output is force-aligned
            (Qwen3-ForcedAligner) to the speech alone, each word's end mapped to
            the timeline through its block's play time; length-adaptive average
            lagging against the source duration (ms, computation not counted).
            Also StartOffset (first word's start) from the same alignment.

The alignment runs in the aligner's venv (--align-python); the rest here.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_extra_metrics --run <si_infer output dir> --gpu 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

MODELS = "/opt/dlami/nvme/jasonleeeli/evaluation_models"
ALIGN_PY = "/opt/dlami/nvme/jasonleeeli/venvs/uniss-formal-align-v1/bin/python"


def align_main(run: Path) -> None:
    """In the aligner's venv: word times of the ASR transcript in each output's speech alone."""
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from experiments.uniss_qwen3_omni_v1.evaluation.realsi_align import ALIGNER

    model = Qwen3ForcedAligner.from_pretrained(ALIGNER, dtype=torch.bfloat16, device_map="cuda:0")
    recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    heard = {json.loads(l)["id"]: json.loads(l).get("asr_text", "") for l in open(run / "asr.jsonl", encoding="utf-8")}
    out = []
    for s in range(0, len(recs), 16):
        batch = [r for r in recs[s: s + 16] if heard.get(r["id"], "").strip()]
        if not batch:
            continue
        audio = [(sf.read(r["speech_path"], dtype="float32")[0], 16000) for r in batch]
        lang = ["Chinese" if r["tgt_lang"] == "cmn" else "English" for r in batch]
        for r, words in zip(batch, model.align(audio=audio, text=[heard[r["id"]] for r in batch], language=lang)):
            out.append({"id": r["id"], "words": [[float(w.start_time) * 1000, float(w.end_time) * 1000] for w in words]})
    (run / "aligned_words.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out), encoding="utf-8")


def to_timeline(ms: float, blocks: list[dict]) -> float:
    for b in blocks:
        if ms < b["speech_at_ms"] + b["dur_ms"] or b is blocks[-1]:
            return b["start_ms"] + max(0.0, ms - b["speech_at_ms"])
    return ms


def laal(delays: list[float], src_ms: float, ref_len: int) -> float:
    rate = src_ms / max(len(delays), ref_len, 1)
    total, n = 0.0, 0
    for i, d in enumerate(delays):
        total += d - i * rate
        n += 1
        if d >= src_ms:
            break
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--align-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    run = Path(args.run)
    if args.align_only:
        align_main(run)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=".")
    recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    speech = run / "results_speech.jsonl"     # audio_path -> the speech alone; reference -> the source speaker
    speech.write_text("".join(json.dumps({**r, "mode": "si", "audio_path": r["speech_path"],
                                          "reference_audio_path": r["source_audio_path"]}, ensure_ascii=False) + "\n"
                              for r in recs if not r.get("error")), encoding="utf-8")
    if not (run / "apcp").exists():
        subprocess.run([sys.executable, "-m", "evaluation.autopcp_metrics", "--input", str(speech), "--output-dir",
                        str(run / "apcp"), "--comparator-path", f"{MODELS}/AutoPCP-multilingual-v2", "--encoder-model",
                        f"{MODELS}/wav2vec2-large-xlsr-53", "--device", "cuda:0", "--pick-layer", "9", "--symmetrize",
                        "--batch-size", "8"], check=True, env=env)
    if not (run / "sim_src.json").exists():
        subprocess.run([sys.executable, "-m", "experiments.uniss_omni_s2st_v1.evaluation.sim_o", "--input", str(speech),
                        "--output", str(run / "sim_src.json"), "--model", f"{MODELS}/wavlm-base-plus-sv",
                        "--device", "cuda:0"], check=True, env=env)
    if not (run / "aligned_words.jsonl").exists():
        subprocess.run([ALIGN_PY, "-m", "experiments.uniss_qwen3_omni_v1.evaluation.si_extra_metrics", "--run", str(run),
                        "--align-only"], check=True, env=env)
    apcp = [json.loads(l)["autopcp_score"] for p in glob.glob(str(run / "apcp" / "*.jsonl")) for l in open(p)
            if "autopcp_score" in json.loads(l)]
    sim = json.load(open(run / "sim_src.json"))
    sim_mean = float(next(iter(sim["groups"].values()))["mean"])
    words = {json.loads(l)["id"]: json.loads(l)["words"] for l in open(run / "aligned_words.jsonl")}
    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units
    lags, starts = [], []
    for r in recs:
        w = words.get(r["id"])
        if not w:
            continue
        ends = [to_timeline(e, r["blocks"]) for _, e in w]
        lags.append(laal(ends, r["source_duration_ms"], len(units(r["translation_ref"], r["tgt_lang"]))))
        starts.append(to_timeline(w[0][0], r["blocks"]))
    out = {"apcp": round(float(np.mean(apcp)), 3) if apcp else None, "sim_src": round(sim_mean, 3),
           "laal_s": round(float(np.mean(lags)) / 1000, 2) if lags else None,
           "start_offset_aligned_s_p50": round(float(np.median(starts)) / 1000, 2) if starts else None,
           "aligned_sentences": len(lags)}
    (run / "metrics_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
