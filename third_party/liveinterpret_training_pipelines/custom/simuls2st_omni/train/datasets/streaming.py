"""Streaming-trajectory dataset for Stage 3.

Reads S2ST manifests whose trajectories live on config D's 1000ms
WAIT/WRITE grid (0-based `step_id`),
and emits the per-chunk structure `collators/streaming.py` consumes.

Latency multiplier
------------------
`m` merges m adjacent base windows into one chunk: windows sharing
`window_idx // m` are grouped, their texts concatenated and their code spans
unioned. Because that is integer arithmetic on the base grid, all twelve
latency settings come from one copy of the data instead of twelve. Inference
reads `m * 1000ms` of source audio per chunk.

`m` is drawn uniformly from {1..12} per `__getitem__`, keyed on (seed, index)
so a sample's setting is stable within a run and the batch sampler's length
estimate stays meaningful.

Thinker format
--------------
Each chunk is tokenized as:

    chunk_ids = user_ids + asst_prefix_ids + asst_text_ids + asst_suffix_ids
    reply_span = [len(user_ids) + len(asst_prefix_ids), len(chunk_ids))

The system prompt is prepended to chunk 0 only, and a wait chunk carries
`<|idle|>` as its assistant text. Spans are then offset into the concatenated
sequence, since the collator indexes `thinker_cond` globally.

Samples are single-utterance: one utterance is one trajectory, matching how the
teacher trajectories were built.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from torch.utils.data import Dataset

from simuls2st_omni.train.collators.streaming import StreamingChunk
from simuls2st_omni.train.modeling_dual_head import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IDLE_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
)
from simuls2st_omni.train.prompt_formats import (
    TASK_STREAMING_S2S_TRANSLATE,
    build_audio_span,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE_16K = 16_000
# Thinker audio tokens: 40ms each. Config D at 1s is 25 audio tokens per
# base window. `m` multiplies that, matching the agent's cap.
AUDIO_FRAME_MS = 40
DEFAULT_BASE_WINDOW_MS = 1000
MAX_LATENCY_MULTIPLIER = 12


class StreamingS2STDataset(Dataset):
    """Streaming S2ST trajectories over DualCodec targets."""

    def __init__(
        self,
        root: str | Path,
        tokenizer,
        si_root: str | Path,
        directions: Optional[Iterable[str]] = None,
        latency_multipliers: Optional[Iterable[int]] = None,
        max_samples: int = 0,
        max_codes: int = 1500,
        min_write_chunks: int = 1,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.si_root = Path(si_root)
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.latency_multipliers = sorted(
            latency_multipliers
            if latency_multipliers
            else range(1, MAX_LATENCY_MULTIPLIER + 1)
        )
        assert self.latency_multipliers, "no latency multiplier to sample"
        assert all(m >= 1 for m in self.latency_multipliers)

        self.records: list[dict] = []
        self._code_arrays: dict[str, np.ndarray] = {}
        # Prompt text is identical for every sample of a direction/latency, so
        # tokenizing it once per combination avoids millions of repeat calls.
        self._system_ids: dict[tuple[str, str, int], list[int]] = {}
        self._user_ids: dict[tuple[str, int], list[int]] = {}
        self._prefix_ids: Optional[list[int]] = None

        manifest_root = self.root / "manifest"
        assert manifest_root.is_dir(), f"missing manifest dir: {manifest_root}"
        dir_names = (
            list(directions)
            if directions
            else sorted(p.name for p in manifest_root.iterdir() if p.is_dir())
        )

        dropped = {"codes": 0, "chunks": 0}
        for direction in dir_names:
            dir_path = manifest_root / direction
            if not dir_path.is_dir():
                logger.warning("skip missing manifest dir %s", dir_path)
                continue
            for shard in sorted(dir_path.glob("*.jsonl")):
                with shard.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        trajectory = record["trajectory"]
                        if len(trajectory["chunks"]) < min_write_chunks:
                            dropped["chunks"] += 1
                            continue
                        n_codes = self._code_len(record["codes"])
                        if n_codes <= 0 or n_codes > max_codes:
                            dropped["codes"] += 1
                            continue
                        self.records.append(record)
                        if max_samples and len(self.records) >= max_samples:
                            break
                if max_samples and len(self.records) >= max_samples:
                    break
            if max_samples and len(self.records) >= max_samples:
                break

        assert self.records, f"no usable records under {manifest_root}"
        logger.info(
            "StreamingS2STDataset: %d samples (dropped %s)", len(self.records), dropped
        )

    @staticmethod
    def _code_len(ref: str) -> int:
        _, start, end = ref.rsplit(":", 2)
        return int(end) - int(start)

    def _codes(self, ref: str) -> np.ndarray:
        rel, start, end = ref.rsplit(":", 2)
        array = self._code_arrays.get(rel)
        if array is None:
            path = self.root / "codes" / rel
            assert path.exists(), f"missing code file: {path}"
            # mmap so worker processes share page cache instead of copying.
            array = np.load(path, mmap_mode="r")
            self._code_arrays[rel] = array
        return np.asarray(array[int(start) : int(end)], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.records)

    def _pick_latency(self, index: int) -> int:
        """Uniform draw over the configured multipliers, stable per (seed, index)."""
        draw = (hash((self.seed, index, "m")) & 0xFFFFFFFF) / 0x100000000
        return self.latency_multipliers[
            min(
                int(draw * len(self.latency_multipliers)),
                len(self.latency_multipliers) - 1,
            )
        ]

    def _lang_names(self, direction: str) -> tuple[str, str]:
        src, _, tgt = direction.partition("2")
        names = {"en": "English", "zh": "Chinese"}
        return names.get(src, "English"), names.get(tgt, "Chinese")

    def _system_token_ids(self, direction: str, latency: int) -> list[int]:
        key = (direction, "sys", latency)
        ids = self._system_ids.get(key)
        if ids is None:
            src_lang, tgt_lang = self._lang_names(direction)
            prompt = build_system_prompt(
                task_type=TASK_STREAMING_S2S_TRANSLATE,
                base_system_prompt=DEFAULT_SYSTEM_PROMPT,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                latency=latency,
            )
            text = f"{DEFAULT_BOS_TOKEN}system\n{prompt}{DEFAULT_EOS_TOKEN}\n"
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            self._system_ids[key] = ids
        return ids

    def _user_token_ids(self, direction: str, n_audio_tokens: int) -> list[int]:
        key = (direction, n_audio_tokens)
        ids = self._user_ids.get(key)
        if ids is None:
            src_lang, _ = self._lang_names(direction)
            payload = src_lang + build_audio_span(n_audio_tokens)
            text = f"{DEFAULT_BOS_TOKEN}user\n{payload}{DEFAULT_EOS_TOKEN}\n"
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            self._user_ids[key] = ids
        return ids

    def _asst_prefix_token_ids(self) -> list[int]:
        if self._prefix_ids is None:
            self._prefix_ids = self.tokenizer(
                f"{DEFAULT_BOS_TOKEN}assistant\n", add_special_tokens=False
            ).input_ids
        return self._prefix_ids

    def _regrid(self, trajectory: dict, latency: int) -> list[dict]:
        """Merge base windows into `latency`-wide chunks.

        Returns one entry per emitted chunk, in window order, each with the
        concatenated text and the union of code spans. Windows with no write are
        wait steps and carry empty text.
        """
        by_group: dict[int, dict] = {}
        for chunk in trajectory["chunks"]:
            group = int(chunk["window_idx"]) // latency
            entry = by_group.get(group)
            if entry is None:
                by_group[group] = {
                    "group": group,
                    "texts": [chunk["text"]],
                    "code_start": int(chunk["code_start"]),
                    "code_end": int(chunk["code_end"]),
                }
            else:
                entry["texts"].append(chunk["text"])
                # Spans are monotonic and disjoint by construction, so the union
                # is just the outer bounds.
                entry["code_start"] = min(entry["code_start"], int(chunk["code_start"]))
                entry["code_end"] = max(entry["code_end"], int(chunk["code_end"]))

        n_groups = -(-int(trajectory["num_windows"]) // latency)  # ceil
        out: list[dict] = []
        for group in range(n_groups):
            entry = by_group.get(group)
            if entry is None:
                out.append({"text": "", "code_start": 0, "code_end": 0})
            else:
                out.append(
                    {
                        "text": " ".join(t for t in entry["texts"] if t).strip(),
                        "code_start": entry["code_start"],
                        "code_end": entry["code_end"],
                    }
                )
        return out

    def code_length(self, index: int) -> int:
        """Talker-side cost estimate for the batch sampler.

        Each chunk adds its codes plus mask/pad/bos/eos slots. The mask region is
        one slot per thinker token, so it dominates; estimated at the mean
        latency rather than the drawn one, which is enough for bucketing.
        """
        record = self.records[index]
        n_codes = self._code_len(record["codes"])
        n_windows = int(record["trajectory"]["num_windows"])
        base_ms = int(record["trajectory"].get("base_window_ms") or DEFAULT_BASE_WINDOW_MS)
        tokens_per_window = max(1, base_ms // AUDIO_FRAME_MS)
        mean_m = sum(self.latency_multipliers) / len(self.latency_multipliers)
        n_chunks = max(1, int(n_windows / mean_m))
        per_chunk = tokens_per_window * mean_m + 12
        return max(1, int(n_codes + n_chunks * per_chunk))

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        direction = record["direction"]
        latency = self._pick_latency(index)
        codes = self._codes(record["codes"])
        groups = self._regrid(record["trajectory"], latency)

        # One chunk consumes `latency` base windows of source audio. The agent
        # caps placeholders at `(base_window_ms/40) * m`, and a full chunk
        # always hits that cap, so train and inference agree on counts.
        base_ms = int(record["trajectory"].get("base_window_ms") or DEFAULT_BASE_WINDOW_MS)
        n_audio_tokens = max(1, (base_ms // AUDIO_FRAME_MS) * latency)
        user_ids = self._user_token_ids(direction, n_audio_tokens)
        prefix_ids = self._asst_prefix_token_ids()
        system_ids = self._system_token_ids(direction, latency)

        thinker_ids: list[int] = []
        chunks: list[StreamingChunk] = []
        for position, group in enumerate(groups):
            text = group["text"] or DEFAULT_IDLE_TOKEN
            asst_text = f"{text}{DEFAULT_EOS_TOKEN}\n"
            asst_ids = self.tokenizer(asst_text, add_special_tokens=False).input_ids

            # System prompt is prepended on chunk 0 only.
            lead = list(system_ids) if position == 0 else []
            chunk_ids = lead + list(user_ids) + list(prefix_ids) + list(asst_ids)
            base = len(thinker_ids)
            reply_start = base + len(lead) + len(user_ids) + len(prefix_ids)
            reply_end = base + len(chunk_ids)
            thinker_ids.extend(chunk_ids)

            chunks.append(
                StreamingChunk(
                    thinker_span=(base, base + len(chunk_ids)),
                    reply_span=(reply_start, reply_end),
                    codes=codes[group["code_start"] : group["code_end"]].tolist(),
                )
            )

        return {
            "id": record["id"],
            "task_type": TASK_STREAMING_S2S_TRANSLATE,
            "thinker_token_ids": thinker_ids,
            "chunks": chunks,
            "latency_multiplier": latency,
        }
