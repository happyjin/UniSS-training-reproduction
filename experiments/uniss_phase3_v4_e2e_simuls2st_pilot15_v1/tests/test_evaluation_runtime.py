from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.runtime import (
    PersistentInterleavedSession,
    append_only_commit,
    generate_mt_prefix,
    mt_prompt_ids,
)
from training import constants_uniss as c


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10 + index for index, _ in enumerate(text.split())]

    def decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens
        return " ".join(f"t{int(value)}" for value in values)

    def __len__(self):
        return c.VOCAB_SIZE


class _ScriptedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(c.VOCAB_SIZE, 4)
        self.calls: list[tuple[int, ...]] = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del use_cache, return_dict
        length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if input_ids is not None:
            values = tuple(int(value) for value in input_ids[0].tolist())
        else:
            values = (-1,) * int(length)
        self.calls.append(values)
        logits = torch.full((1, int(length), c.VOCAB_SIZE), -1000.0)
        # Stateless MT: one base-vocabulary token, then END_CONTENT.
        if values and values[-1] == c.TOKEN_START_CONTENT:
            logits[:, -1, 42] = 10.0
        elif values and values[-1] == 42:
            logits[:, -1, c.TOKEN_END_CONTENT] = 10.0
        # Interleaved source blocks arrive as embeddings.  Prefer illegal EOS
        # over WAIT so the runtime test proves non-final EOS is masked.
        elif values and values[-1] == -1:
            logits[:, -1, c.TOKEN_EOS] = 20.0
            logits[:, -1, c.TOKEN_WAIT_READ] = 10.0
        else:
            logits[:, -1, c.TOKEN_START_GLM] = 10.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=int(past_key_values or 0) + int(length),
        )


class _RepeatingFamilyModel(nn.Module):
    """Prefer ASR forever unless the runtime applies the task grammar."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(c.VOCAB_SIZE, 4)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:, 0] = torch.arange(
                c.VOCAB_SIZE, dtype=self.embedding.weight.dtype
            )

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del use_cache, return_dict
        length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if input_ids is not None:
            last = int(input_ids[0, -1])
        else:
            last = int(round(float(inputs_embeds[0, -1, 0])))
        logits = torch.full((1, int(length), c.VOCAB_SIZE), -1000.0)
        if last == c.TOKEN_END_GLM:
            logits[:, -1, c.TOKEN_WRITE_GENERATE] = 30.0
        elif last == c.TOKEN_WRITE_GENERATE:
            logits[:, -1, c.TOKEN_TASK_ASR] = 30.0
            logits[:, -1, c.TOKEN_TASK_S2T_TRANSLATION] = 20.0
            logits[:, -1, c.TOKEN_TASK_TTS] = 10.0
        elif last == c.TOKEN_START_CONTENT:
            logits[:, -1, 42] = 30.0
        elif last == 42:
            logits[:, -1, c.TOKEN_END_CONTENT] = 30.0
        elif last == c.TOKEN_END_CONTENT:
            logits[:, -1, c.TOKEN_WRITE_GENERATE] = 30.0
        elif last == c.TOKEN_START_SEMANTIC:
            logits[:, -1, c.BICODEC_SEMANTIC_OFFSET + 3] = 30.0
        elif last == c.BICODEC_SEMANTIC_OFFSET + 3:
            logits[:, -1, c.TOKEN_END_SEMANTIC] = 30.0
        elif last == c.TOKEN_END_SEMANTIC:
            logits[:, -1, c.TOKEN_START_GLM] = 30.0
        else:
            logits[:, -1, c.TOKEN_START_GLM] = 30.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=int(past_key_values or 0) + int(length),
        )


def test_mt_prompt_and_append_only_commit() -> None:
    tokenizer = _Tokenizer()
    prompt = mt_prompt_ids(tokenizer, "one two", "cmn")
    assert prompt[:3] == (
        c.TOKEN_TASK_T2T_TRANSLATION,
        c.TOKEN_CMN,
        c.TOKEN_START_CONTENT,
    )
    assert prompt[-3:] == (
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_CMN,
        c.TOKEN_START_CONTENT,
    )
    assert append_only_commit("one", "one two", "eng") == ("one two", False)
    assert append_only_commit("one", "other", "eng") == ("one", True)


def test_mt_generation_uses_restricted_text_grammar() -> None:
    model = _ScriptedModel()
    text, ids, reached = generate_mt_prefix(
        model, _Tokenizer(), "one two", "cmn", max_tokens=4
    )
    assert text == "t42"
    assert ids == (42,)
    assert reached is True


def test_interleaved_session_commits_wait_without_forced_write() -> None:
    model = _ScriptedModel()
    trajectory = SimpleNamespace(
        tgt_lang="cmn",
        src_lang="eng",
        speaker_global=tuple(range(32)),
    )
    session = PersistentInterleavedSession(
        model,
        _Tokenizer(),
        torch.randn(2, 4),
        trajectory,
    )
    event = SimpleNamespace(
        event_index=0,
        source_end_ms=160,
        source_final=False,
        source_glm_start=0,
        source_glm_end=2,
    )
    row = session.run_event(
        event,
        max_fragments=2,
        max_text_tokens=4,
        max_semantic_tokens=4,
    )
    assert row.chosen_continuations == ("WAIT",)
    assert not row.asr_deltas and not row.mt_deltas and not row.semantic_tokens
    assert row.malformed_segments == 0


def test_interleaved_session_allows_eos_only_after_final_source_event() -> None:
    model = _ScriptedModel()
    trajectory = SimpleNamespace(
        tgt_lang="cmn",
        src_lang="eng",
        speaker_global=tuple(range(32)),
    )
    session = PersistentInterleavedSession(
        model,
        _Tokenizer(),
        torch.randn(2, 4),
        trajectory,
    )
    event = SimpleNamespace(
        event_index=0,
        source_end_ms=160,
        source_final=True,
        source_glm_start=0,
        source_glm_end=2,
    )
    row = session.run_event(
        event,
        max_fragments=2,
        max_text_tokens=4,
        max_semantic_tokens=4,
    )
    assert row.chosen_continuations == ("EOS",)
    assert row.malformed_segments == 0


def test_interleaved_session_masks_repeated_families_and_reaches_semantic() -> None:
    model = _RepeatingFamilyModel()
    trajectory = SimpleNamespace(
        tgt_lang="cmn",
        src_lang="eng",
        speaker_global=tuple(range(32)),
    )
    session = PersistentInterleavedSession(
        model,
        _Tokenizer(),
        torch.randn(2, 4),
        trajectory,
    )
    event = SimpleNamespace(
        event_index=0,
        source_end_ms=160,
        source_final=False,
        source_glm_start=0,
        source_glm_end=2,
    )
    row = session.run_event(
        event,
        max_fragments=4,
        max_text_tokens=4,
        max_semantic_tokens=4,
    )
    assert row.chosen_continuations == (
        "WRITE_ASR",
        "WRITE_MT",
        "WRITE_SEMANTIC",
        "READ_NEXT",
    )
    assert row.asr_deltas == ("t42",)
    assert row.mt_deltas == ("t42",)
    assert row.semantic_tokens == (3,)
    assert row.malformed_segments == 0
