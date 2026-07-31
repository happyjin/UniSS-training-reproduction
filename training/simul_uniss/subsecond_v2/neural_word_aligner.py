"""Batched multilingual contextual word alignment for formal Stage-A A6."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


def links_from_similarity(
    similarity: torch.Tensor,
    source_words: Sequence[str],
    target_words: Sequence[str],
    *,
    mutual_threshold: float = 0.35,
    union_threshold: float = 0.55,
) -> list[dict[str, object]]:
    """Extract auditable mutual-nearest and high-confidence union links."""

    if similarity.shape != (len(source_words), len(target_words)):
        raise ValueError("similarity shape does not match word counts")
    if not source_words or not target_words:
        return []
    source_best = similarity.argmax(dim=1)
    target_best = similarity.argmax(dim=0)
    links: dict[tuple[int, int], dict[str, object]] = {}
    for source_index, target_tensor in enumerate(source_best):
        target_index = int(target_tensor)
        confidence = float(similarity[source_index, target_index])
        mutual = int(target_best[target_index]) == source_index
        if (mutual and confidence >= mutual_threshold) or confidence >= union_threshold:
            links[(source_index, target_index)] = {
                "source_index": source_index,
                "target_index": target_index,
                "confidence": confidence,
                "method": "neural_mutual_nearest" if mutual else "neural_source_union",
            }
    for target_index, source_tensor in enumerate(target_best):
        source_index = int(source_tensor)
        confidence = float(similarity[source_index, target_index])
        mutual = int(source_best[source_index]) == target_index
        if (mutual and confidence >= mutual_threshold) or confidence >= union_threshold:
            links[(source_index, target_index)] = {
                "source_index": source_index,
                "target_index": target_index,
                "confidence": confidence,
                "method": "neural_mutual_nearest" if mutual else "neural_target_union",
            }
    # Exact strings, numbers, and punctuation are reliable anchors even when
    # contextual embeddings are diffuse.
    for source_index, source in enumerate(source_words):
        source_key = source.strip().casefold()
        if not source_key:
            continue
        for target_index, target in enumerate(target_words):
            if source_key == target.strip().casefold():
                links[(source_index, target_index)] = {
                    "source_index": source_index,
                    "target_index": target_index,
                    "confidence": 1.0,
                    "method": "exact_lexical_anchor",
                }
    return sorted(links.values(), key=lambda value: (int(value["target_index"]), int(value["source_index"])))


class BatchedNeuralWordAligner:
    """mBERT/XLM-R contextual aligner with GPU-batched encoder inference."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda:0",
        batch_size: int = 128,
        maximum_tokens: int = 256,
        mutual_threshold: float = 0.35,
        union_threshold: float = 0.55,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.maximum_tokens = maximum_tokens
        self.mutual_threshold = mutual_threshold
        self.union_threshold = union_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ).eval().to(self.device)

    @staticmethod
    def _pool_words(
        hidden: torch.Tensor, word_ids: Sequence[int | None], word_count: int
    ) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        for word_index in range(word_count):
            positions = [index for index, value in enumerate(word_ids) if value == word_index]
            if not positions:
                vectors.append(hidden.new_zeros(hidden.shape[-1]))
            else:
                vectors.append(hidden[positions].mean(dim=0))
        return F.normalize(torch.stack(vectors).float(), dim=-1)

    @torch.inference_mode()
    def _encode(self, batches: Sequence[Sequence[str]]) -> list[torch.Tensor]:
        encoded = self.tokenizer(
            list(batches),
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.maximum_tokens,
            return_tensors="pt",
        )
        word_ids = [encoded.word_ids(index) for index in range(len(batches))]
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(**model_inputs, output_hidden_states=True)
            hidden = torch.stack(output.hidden_states[-4:]).mean(dim=0)
        return [
            self._pool_words(hidden[index], word_ids[index], len(words)).cpu()
            for index, words in enumerate(batches)
        ]

    def align_batch(
        self,
        source_batches: Sequence[Sequence[str]],
        target_batches: Sequence[Sequence[str]],
    ) -> list[list[dict[str, object]]]:
        if len(source_batches) != len(target_batches):
            raise ValueError("source and target alignment batch sizes differ")
        result: list[list[dict[str, object]]] = []
        for start in range(0, len(source_batches), self.batch_size):
            sources = source_batches[start : start + self.batch_size]
            targets = target_batches[start : start + self.batch_size]
            source_vectors = self._encode(sources)
            target_vectors = self._encode(targets)
            for source_words, target_words, source, target in zip(
                sources, targets, source_vectors, target_vectors
            ):
                result.append(
                    links_from_similarity(
                        source @ target.T,
                        source_words,
                        target_words,
                        mutual_threshold=self.mutual_threshold,
                        union_threshold=self.union_threshold,
                    )
                )
        return result
