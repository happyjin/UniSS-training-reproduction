"""v11 P1-4 (d): a second ASR on the EXISTING speech-only audio of the streaming test runs, to check that the ASR-BLEU
gain of the playback layer is not specific to Whisper / Paraformer (the PA Talker picks candidates with a wav2vec2 CTC
recogniser). English output: NVIDIA parakeet-tdt-0.6b-v2 (NeMo, run with venvs/nemo_diar); Chinese output: Whisper
large-v3 with language=zh and OpenCC t2s (run with conda_envs/uniss-eval). Reads <speech run>/talker/asr_speech_merged.jsonl
(ids, audio_path = speech-only wav), writes eval_outputs/mist_w3po_v1/v11_metrics/asr2/<run>/asr2_speech.jsonl
({id, asr_text, asr_model}); refuses to overwrite. Hypothesis lines and scores: asr_hyp + v11_metrics (std only).

    <python> experiments/mist_w3po_v1/evaluation/v11_asr2.py --backend parakeet|whisper_zh --runs <run> [<run> ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
OUT = REPO / "eval_outputs/mist_w3po_v1/v11_metrics/asr2"


def rows_of(run: str) -> list[dict]:
    rows = [json.loads(l) for l in open(SPD / run / "talker/asr_speech_merged.jsonl", encoding="utf-8")]
    return [r for r in rows if r.get("audio_path") and Path(r["audio_path"]).exists()]


def parakeet(paths: list[str]) -> list[str]:
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
    out = m.transcribe(paths, batch_size=32, verbose=False)
    return [(o.text if hasattr(o, "text") else str(o)).strip() for o in out]


def whisper_zh(paths: list[str], durs: list[float]) -> list[str]:
    """Whisper loops on some short clips ('U.M.U.M...'): every transcript is cut to ceil(8 chars/s * duration) + 4
    characters, one fixed rule for every system (Chinese speech runs at about 4-5 characters per second)."""
    import math
    import torch
    from transformers import pipeline
    import opencc
    cc = opencc.OpenCC("t2s")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                   device="cuda:0", chunk_length_s=30)
    import soundfile as sf
    import librosa

    def load(p):
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        return {"raw": x, "sampling_rate": 16000}
    res = []
    B = 16
    for i in range(0, len(paths), B):
        o = asr([load(p) for p in paths[i:i + B]], batch_size=B,
                generate_kwargs={"language": "chinese", "task": "transcribe"})
        res += [cc.convert(x["text"]).strip()[:math.ceil(8 * d) + 4] for x, d in zip(o, durs[i:i + B])]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["parakeet", "whisper_zh"])
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    for run in a.runs:
        o = OUT / run / "asr2_speech.jsonl"
        if o.exists():
            print(f"{o} exists; kept", flush=True)
            continue
        rows = rows_of(run)
        rows = sorted(rows, key=lambda r: float(r.get("audio_duration_seconds") or 0))
        P = [r["audio_path"] for r in rows]
        texts = parakeet(P) if a.backend == "parakeet" else whisper_zh(P, [float(r.get("audio_duration_seconds") or 0) for r in rows])
        o.parent.mkdir(parents=True, exist_ok=True)
        tmp = o.with_suffix(".jsonl.part")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r, t in zip(rows, texts):
                fh.write(json.dumps({"id": r["id"], "asr_text": t,
                                     "asr_model": "nvidia/parakeet-tdt-0.6b-v2" if a.backend == "parakeet"
                                     else "openai/whisper-large-v3+zh+t2s"}, ensure_ascii=False) + "\n")
        os.rename(tmp, o)
        print(f"{run}: {len(rows)} items -> {o}", flush=True)


if __name__ == "__main__":
    main()
