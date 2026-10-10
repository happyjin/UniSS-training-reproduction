"""SIM-O: speaker similarity against the paired real reference target speech.

Table 1 of arXiv 2607.19810 defines it as WavLM speaker embeddings with
cosine similarity "against the paired real reference target speech" -- one
reference per utterance, not a single fixed voice. The repository's
existing speaker_similarity.py scores against one shared reference, which
is the right thing for the experiment it belongs to and the wrong thing
here, so this is separate rather than a change to it.

Deviation to state plainly: the paper uses WavLM-**Large**. This box has
``wavlm-base-plus-sv`` and no network, so the numbers here are on a
smaller encoder and are not directly comparable to the paper's column.
The ranking between our own checkpoints is still meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="results.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--model",
        default="/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv",
    )
    ap.add_argument("--device", default="cuda:0")
    # WavLM's x-vector head is a TDNN; a waveform shorter than its receptive
    # field raises inside conv1d rather than returning anything. A tenth of
    # a second is far below any real utterance and well above the kernel.
    ap.add_argument("--min-seconds", type=float, default=0.2)
    args = ap.parse_args()

    from transformers import AutoFeatureExtractor, WavLMForXVector

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model = (
        WavLMForXVector.from_pretrained(args.model, local_files_only=True)
        .to(args.device)
        .eval()
    )

    def embed(path: str) -> torch.Tensor | None:
        try:
            waveform, rate = sf.read(path, dtype="float32")
        except Exception:
            return None
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform.shape[0] < args.min_seconds * rate:
            return None
        inputs = extractor(waveform, sampling_rate=rate, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            return model(**inputs).embeddings[0].float()

    scores: dict[str, list[float]] = {}
    per_sample = []
    skipped = 0
    for row in rows:
        generated, reference = row.get("audio_path"), row.get("reference_audio_path")
        if not generated or not reference:
            skipped += 1
            continue
        try:
            a, b = embed(generated), embed(reference)
        except Exception:
            a = b = None
        if a is None or b is None:
            skipped += 1
            continue
        value = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        key = f"{row['src_lang']}->{row['tgt_lang']}"
        scores.setdefault(key, []).append(value)
        per_sample.append({"id": row["id"], "direction": key, "sim_o": value})

    report = {
        "definition": "SIM-O: cosine similarity of WavLM x-vectors between the"
        " generated translation and its own paired real reference target speech.",
        "encoder": args.model,
        "encoder_note": "the paper uses WavLM-Large; this is a smaller encoder,"
        " so absolute values are not comparable to its column",
        "skipped": skipped,
        "groups": {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "sample_count": len(values),
            }
            for key, values in sorted(scores.items())
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out.parent / "per_sample_sim_o.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_sample) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["groups"], ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
