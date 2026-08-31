#!/usr/bin/env python3
"""Experiment 0-A: does the streaming inference path see the codes it trained on?

The content-first SFT consumed precomputed offline WhisperVQ codes
(``source_glm`` in the dense pack, produced by the non-causal GLM4 tokenizer).
The rollout cascade instead derives codes online from a block-causal frontend
and re-quantizes the pre-VQ hidden with an L2 argmin
(``event_policy_cascade.py`` lines 116-127 and ``model_loader.py`` lines
155-190).  If those two code streams disagree, the trained
``frontend_adapter``/``frontend_projection`` residual and the GLM semantic
embeddings both receive out-of-distribution input at inference, and the
observed ASR teacher similarity of 0.048 is an inference-path artefact rather
than a capability gap.

This module measures the disagreement.  It writes only into its own report
path and imports established modules without mutating them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage00_baseline.shared_causal_frontend import (
    BLOCK_SAMPLES,
    SAMPLE_RATE,
    SharedCausalWhisperVQFrontend,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from uniss.speech_tokenizer.glm4.utils import extract_speech_token


SCHEMA = "uniss_content_first_bridge_parity_v1"


def unique_components(
    episodes_path: Path, limit: int
) -> list[dict[str, object]]:
    """Return the first ``limit`` distinct components in deterministic order."""

    seen: set[str] = set()
    selected: list[dict[str, object]] = []
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            for component in episode.get("components", []):
                sample_id = str(component["sample_id"])
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                selected.append(
                    {
                        "episode_id": str(episode["episode_id"]),
                        "src_lang": str(episode["src_lang"]),
                        "sample_id": sample_id,
                        "source_audio": str(component["source_audio"]),
                        "duration_ms": int(component["duration_ms"]),
                        "source_glm_length": int(component["source_glm_length"]),
                    }
                )
                if len(selected) >= limit:
                    return selected
    return selected


def read_waveform(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz, expected {SAMPLE_RATE}")
    if waveform.ndim != 1:
        waveform = waveform[:, 0]
    return np.ascontiguousarray(waveform, dtype=np.float32)


def nearest_codes(codebook: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Reproduce ``model_loader._nearest_codes`` exactly (float32 L2 argmin)."""

    book = codebook.to(device=hidden.device, dtype=torch.float32)
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    code_norm = book.square().sum(dim=1)
    pieces: list[torch.Tensor] = []
    for start in range(0, len(flat), 4096):
        value = flat[start : start + 4096]
        distance = (
            value.square().sum(dim=1, keepdim=True)
            + code_norm.unsqueeze(0)
            - 2.0 * value @ book.t()
        )
        pieces.append(distance.argmin(dim=1))
    return torch.cat(pieces).reshape(hidden.shape[:-1])


@torch.inference_mode()
def stream_codes(
    frontend: SharedCausalWhisperVQFrontend,
    codebook: torch.Tensor,
    waveform: np.ndarray,
) -> tuple[list[int], list[int], int]:
    """Push PCM exactly as the rollout cascade does and collect both codes.

    Returns ``(frontend_token_ids, bridge_argmin_codes, encoder_resets)``.  The
    first is what the frontend itself quantized; the second is what the
    content-first bridge recomputes from ``pre_vq_hidden``.  The cascade uses
    the second one.
    """

    state = None
    own: list[int] = []
    bridge: list[int] = []
    for start in range(0, len(waveform), BLOCK_SAMPLES):
        stop = min(len(waveform), start + BLOCK_SAMPLES)
        output = frontend.push(
            waveform[start:stop], state, is_final=stop == len(waveform)
        )
        state = output.state
        own.extend(int(value) for value in output.token_ids[0].tolist())
        hidden = output.pre_vq_hidden[0]
        bridge.extend(
            int(value) for value in nearest_codes(codebook, hidden).tolist()
        )
    if state is None or not state.finalized:
        raise RuntimeError("streaming frontend did not finalize")
    return own, bridge, int(state.encoder_resets)


def agreement(left: Sequence[int], right: Sequence[int]) -> dict[str, object]:
    """Prefix-aligned exact agreement over the shared length."""

    shared = min(len(left), len(right))
    matches = sum(1 for index in range(shared) if left[index] == right[index])
    return {
        "left_length": len(left),
        "right_length": len(right),
        "compared": shared,
        "matches": matches,
        "agreement": (matches / shared) if shared else 0.0,
        "length_equal": len(left) == len(right),
    }


def embedding_similarity(
    codebook: torch.Tensor, left: Sequence[int], right: Sequence[int]
) -> dict[str, float]:
    """Cosine similarity between the codebook vectors the two streams select."""

    shared = min(len(left), len(right))
    if not shared:
        return {"mean_cosine": 0.0, "p05_cosine": 0.0}
    book = codebook.float().cpu()
    a = book[torch.tensor(list(left[:shared]), dtype=torch.long)]
    b = book[torch.tensor(list(right[:shared]), dtype=torch.long)]
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
    }


def offline_codes(
    encoder, extractor: WhisperFeatureExtractor, waveform: np.ndarray
) -> list[int]:
    """Non-causal GLM4 tokenizer codes: the exact training-time source."""

    tokens = extract_speech_token(
        encoder, extractor, [(torch.from_numpy(waveform).unsqueeze(0), SAMPLE_RATE)]
    )
    return [int(value) for value in tokens[0]]


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("bridge parity requires CUDA (offline tokenizer is CUDA-only)")
    torch.cuda.set_device(device)

    trainable = (
        TrainableSharedCausalWhisperVQ(args.whispervq_model, gradient_checkpointing=False)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    frontend = SharedCausalWhisperVQFrontend(
        trainable.encoder, trainable.mel_filters, device=device
    )
    frontend.requires_grad_(False).eval()
    codebook = trainable.codebook.detach()
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(args.whispervq_model), local_files_only=True
    )

    components = unique_components(args.episodes, args.components)
    if not components:
        raise RuntimeError(f"no components found in {args.episodes}")

    rows: list[dict[str, object]] = []
    for component in components:
        waveform = read_waveform(Path(str(component["source_audio"])))
        own, bridge, resets = stream_codes(frontend, codebook, waveform)
        offline = offline_codes(trainable.encoder, extractor, waveform)
        row = {
            **component,
            "waveform_samples": int(len(waveform)),
            "encoder_resets": resets,
            "offline_length": len(offline),
            "stream_length": len(bridge),
            "declared_source_glm_length": int(component["source_glm_length"]),
            "offline_matches_declared_length": len(offline)
            == int(component["source_glm_length"]),
            "frontend_vs_bridge": agreement(own, bridge),
            "offline_vs_bridge": agreement(offline, bridge),
            "offline_vs_frontend": agreement(offline, own),
            "offline_vs_bridge_embedding": embedding_similarity(
                codebook, offline, bridge
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "offline_length": row["offline_length"],
                    "stream_length": row["stream_length"],
                    "offline_vs_bridge_agreement": row["offline_vs_bridge"]["agreement"],
                    "frontend_vs_bridge_agreement": row["frontend_vs_bridge"][
                        "agreement"
                    ],
                    "mean_cosine": row["offline_vs_bridge_embedding"]["mean_cosine"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def mean(selector) -> float:
        values = [float(selector(row)) for row in rows]
        return sum(values) / len(values) if values else 0.0

    summary = {
        "components": len(rows),
        "offline_vs_bridge_agreement_mean": mean(
            lambda row: row["offline_vs_bridge"]["agreement"]
        ),
        "offline_vs_bridge_agreement_min": min(
            float(row["offline_vs_bridge"]["agreement"]) for row in rows
        ),
        "offline_vs_frontend_agreement_mean": mean(
            lambda row: row["offline_vs_frontend"]["agreement"]
        ),
        "frontend_vs_bridge_agreement_mean": mean(
            lambda row: row["frontend_vs_bridge"]["agreement"]
        ),
        "offline_vs_bridge_mean_cosine": mean(
            lambda row: row["offline_vs_bridge_embedding"]["mean_cosine"]
        ),
        "length_equal_fraction": mean(
            lambda row: 1.0 if row["offline_vs_bridge"]["length_equal"] else 0.0
        ),
        "declared_length_match_fraction": mean(
            lambda row: 1.0 if row["offline_matches_declared_length"] else 0.0
        ),
    }
    agree = float(summary["offline_vs_bridge_agreement_mean"])
    summary["verdict"] = (
        "inference_path_consistent"
        if agree >= 0.99
        else "inference_path_mismatch_is_primary_suspect"
        if agree < 0.95
        else "inconclusive_between_0.95_and_0.99"
    )
    return {
        "schema_version": SCHEMA,
        "experiment": "0-A_bridge_parity",
        "question": (
            "do the block-causal codes consumed at inference match the offline "
            "codes the content-first SFT was trained on"
        ),
        "whispervq_model": str(args.whispervq_model),
        "episodes": str(args.episodes),
        "decision_thresholds": {
            "consistent": ">=0.99 offline_vs_bridge agreement",
            "mismatch_primary_suspect": "<0.95 offline_vs_bridge agreement",
        },
        "summary": summary,
        "components_detail": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.components <= 0:
        raise ValueError("--components must be positive")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
