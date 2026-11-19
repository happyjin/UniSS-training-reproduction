"""Dataset over the Stage 2 manifests for the ASR, S2TT and MT task families.

One class serves all three because they differ only in which fields the prompt
builder consumes; the three families share one manifest schema. Samples come out in the shape
`collators/multitask.py` expects: `system_prompt` / `user_payload` /
`assistant_payload`, plus `audio` for audio-input tasks.

The audio-input tasks leave `<|AUDIO_SPAN|>` in the user payload as a
placeholder. Only the collator knows the encoder's frame count for a given clip,
so it performs the substitution; a dataset that guessed would desynchronise the
`<|AUDIO|>` placeholders from the scattered encoder outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from torch.utils.data import Dataset

from simuls2st_omni.train.datasets.audio_refs import AudioRefError, AudioRefResolver
from simuls2st_omni.train.datasets.subsample import subsample_uniform
from simuls2st_omni.train.prompt_formats import (
    TASK_S2TT_ASR,
    TASK_S2TT_TRANSLATE,
    TASK_T2T_TRANSLATE,
    build_audio_span,
    build_prompt_bundle,
)

logger = logging.getLogger(__name__)

# Literal the collator looks for when expanding the audio span to the clip's
# real frame count (`collators/multitask.py`).
AUDIO_SPAN_MARKER = "<|AUDIO_SPAN|>"

# Thinker-side audio cost: the collator's `audio_token_count` yields 25 tokens
# per second (40ms per audio token). Used so audio and text batches are packed
# in the same currency.
AUDIO_TOKENS_PER_SEC = 25.0
# Crude characters-per-token ratio for length estimation. It only feeds
# bucketing, never correctness.
CHARS_PER_TOKEN = 3.0
# Assumed clip length when a corpus records no duration (LibriSpeech). Reading
# the audio instead would make sampler setup O(corpus) in IO.
ASSUMED_DURATION_SEC = 12.0


def _manifest_duration_sec(record: dict) -> float:
    duration = record.get("duration")
    if duration is not None:
        return float(duration)
    # LibriSpeech rows omit duration; same stand-in the sampler uses.
    return ASSUMED_DURATION_SEC if record.get("audio") else 0.0


class ManifestDataset(Dataset):
    """ASR / S2TT / MT samples backed by JSONL manifests."""

    def __init__(
        self,
        manifest_dirs: Iterable[str | Path],
        corpus_roots: dict[str, str | Path],
        task_type: Optional[str] = None,
        max_samples: int = 0,
        max_hours: float = 0.0,
        sample_seed: int = 0,
        max_duration: float = 30.0,
        min_duration: float = 0.2,
        max_chars: int = 2000,
        asr_prob: float = 0.0,
        require_source_text: bool = False,
        seed: int = 0,
    ):
        """
        manifest_dirs: directories of `*.jsonl` shards to read.
        corpus_roots: corpus key -> directory its `audio` relpaths resolve
            against. The key is the manifest's `corpus` field, defaulting to the
            manifest directory name.
        task_type: override each record's own `task_type`. Forcing S2TT records
            to ASR is how the paper's ASR substitution is realised; the split is
            done by the caller as a separate pool so a batch stays single-task.
        require_source_text: drop records without a source transcript, which an
            ASR-substituted view cannot use as a target.
        """
        self.records: list[dict] = []
        self.task_override = task_type
        self.asr_prob = float(asr_prob)
        self.seed = int(seed)
        self._resolver: Optional[AudioRefResolver] = None
        self.corpus_roots = {k: Path(v) for k, v in corpus_roots.items()}

        dropped = {"duration": 0, "text": 0, "unresolved": 0, "no_source": 0}
        for manifest_dir in manifest_dirs:
            manifest_dir = Path(manifest_dir)
            assert manifest_dir.is_dir(), f"missing manifest dir: {manifest_dir}"
            default_corpus = manifest_dir.name
            for shard in sorted(manifest_dir.glob("*.jsonl")):
                with shard.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        target = (record.get("target_text") or "").strip()
                        source = (record.get("source_text") or "").strip()
                        if not target or len(target) > max_chars:
                            dropped["text"] += 1
                            continue
                        if len(source) > max_chars:
                            dropped["text"] += 1
                            continue
                        if require_source_text and not source:
                            dropped["no_source"] += 1
                            continue
                        ref = record.get("audio")
                        if ref and str(ref).startswith("gigaspeech:"):
                            # Manifest built without --gigaspeech-index, so the
                            # audio cannot be fetched.
                            dropped["unresolved"] += 1
                            continue
                        duration = record.get("duration")
                        if duration is not None:
                            duration = float(duration)
                            if duration < min_duration or duration > max_duration:
                                dropped["duration"] += 1
                                continue
                        self.records.append(
                            {
                                "id": record["id"],
                                "task_type": record["task_type"],
                                "audio": ref,
                                "corpus": record.get("corpus", default_corpus),
                                "src_lang": record["src_lang"],
                                "tgt_lang": record["tgt_lang"],
                                "source_text": source,
                                "target_text": target,
                                "duration": duration,
                                "offset": record.get("offset", 0.0),
                            }
                        )

        n_pool = len(self.records)
        self.records = subsample_uniform(
            self.records,
            max_samples=max_samples,
            max_hours=max_hours,
            seed=sample_seed,
            duration_sec_fn=_manifest_duration_sec,
        )
        assert self.records, f"no usable records in {list(manifest_dirs)}"
        logger.info(
            "ManifestDataset: %d samples from pool %d (dropped %s, max_hours=%s)",
            len(self.records),
            n_pool,
            dropped,
            max_hours or "off",
        )

    def __len__(self) -> int:
        return len(self.records)

    def token_length(self, index: int) -> int:
        """Thinker-side cost estimate for the batch sampler."""
        record = self.records[index]
        if record["duration"]:
            audio_tokens = record["duration"] * AUDIO_TOKENS_PER_SEC
        elif record["audio"]:
            audio_tokens = ASSUMED_DURATION_SEC * AUDIO_TOKENS_PER_SEC
        else:
            audio_tokens = 0.0
        text_tokens = (
            len(record["source_text"]) + len(record["target_text"])
        ) / CHARS_PER_TOKEN
        return max(1, int(audio_tokens + text_tokens))

    # `build_sampler` calls `code_length` for Stage 1; alias it so the same
    # helper works for text/audio tasks without special-casing.
    code_length = token_length

    def _resolve_audio(self, record: dict):
        if self._resolver is None:
            # Built lazily so each DataLoader worker gets its own handle cache
            # after fork rather than sharing one across processes.
            self._resolver = AudioRefResolver(self.corpus_roots)
        return self._resolver.resolve(
            record["audio"],
            corpus=record["corpus"],
            offset=float(record["offset"] or 0.0),
            duration=record["duration"],
        )

    def __getitem__(self, index: int) -> dict:
        last_err: Optional[AudioRefError] = None
        n = len(self.records)
        for step in range(min(8, n)):
            try:
                return self._sample_at((index + step) % n)
            except AudioRefError as exc:
                last_err = exc
                logger.warning("skip unreadable audio %s", exc)
        assert last_err is not None
        raise last_err

    def _sample_at(self, index: int) -> dict:
        record = self.records[index]
        task = self.task_override or record["task_type"]
        target_text = record["target_text"]
        tgt_lang = record["tgt_lang"]

        # ASR substitution: an S2TT sample whose target is rewritten as its
        # source transcript, which the paper uses to compensate for the small
        # ASR pool. `task_override` selects it, so the decision is made by
        # whichever task pool the sampler drew from -- never per sample. Drawing
        # here would put two task types in one batch, which the collator
        # rejects because `compute_loss` reads a single scalar `task_type`.
        if task == TASK_S2TT_ASR and record["task_type"] == TASK_S2TT_TRANSLATE:
            assert record["source_text"], (
                f"{record['id']}: ASR substitution needs a source transcript"
            )
            target_text = record["source_text"]
            tgt_lang = record["src_lang"]

        bundle = build_prompt_bundle(
            task_type=task,
            text=record["source_text"] if task == TASK_T2T_TRANSLATE else "",
            src_text=record["source_text"],
            target_text=target_text,
            src_lang=record["src_lang"],
            tgt_lang=tgt_lang,
            # A count of 1 makes `build_user_payload` emit the full audio
            # structure (language prefix + speech start/patch/end); the single
            # patch token is then swapped for the `<|AUDIO_SPAN|>` marker that
            # the collator expands once it knows the clip's real frame count.
            # Passing 0 here would drop the audio payload entirely.
            n_audio_tokens=1 if record["audio"] else 0,
        )
        user_payload = bundle.user_payload
        if record["audio"]:
            one_token_span = build_audio_span(1)
            assert one_token_span in user_payload, (
                f"{record['id']}: task {task} produced no audio span to replace"
            )
            user_payload = user_payload.replace(one_token_span, AUDIO_SPAN_MARKER)

        sample = {
            "id": record["id"],
            "task_type": task,
            "system_prompt": bundle.system_prompt,
            "user_payload": user_payload,
            "assistant_payload": bundle.assistant_payload,
        }
        if record["audio"]:
            sample["audio"] = self._resolve_audio(record)
        return sample
