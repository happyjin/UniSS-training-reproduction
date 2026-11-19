"""Behavior tests for token-level Thinker→Talker inference scheduling."""

from types import SimpleNamespace

import pytest
import torch

from custom.qwen_omnispeak.configuration_qwen_omnispeak import Qwen3TtsTalkerConfig
from custom.qwen_omnispeak.modeling_qwen_omnispeak import (
    Qwen3TtsTalkerForConditionalGeneration,
)
from inference.code_predictor_runtime import CodePredictorRunner
from inference.online_pipeline import (
    IncrementalThinkerSession,
    OnlineTalkerSession,
    ResidualFrameBatcher,
    ThinkerTalkerScheduler,
    _TalkerForwardModule,
)
from inference.online_s2s import build_online_control_prefix


class _ScriptedThinker(torch.nn.Module):
    """Tiny causal model whose selected hidden is the input token id."""

    def __init__(self, sampled_tokens):
        super().__init__()
        self.sampled_tokens = list(sampled_tokens)
        self.embedding = torch.nn.Embedding(64, 2)
        with torch.no_grad():
            self.embedding.weight[:, 0] = torch.arange(64)
            self.embedding.weight[:, 1] = -torch.arange(64)
        self.forward_inputs = []
        self.cache_positions = []

    def get_input_embeddings(self):
        return self.embedding

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        del kwargs
        if cache_position is None:
            raise TypeError("cache_position is required")
        self.cache_positions.append(cache_position.detach().clone())
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    ):
        del attention_mask, use_cache, output_hidden_states, return_dict
        self.forward_inputs.append(input_ids.detach().clone())
        hidden = self.embedding(input_ids)
        logits = torch.full((1, input_ids.size(1), 64), -1000.0)
        token = self.sampled_tokens[len(self.forward_inputs) - 1]
        logits[:, -1, token] = 1000.0
        next_cache = 1 if past_key_values is None else past_key_values + 1
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden * 0.0, hidden),
            past_key_values=next_cache,
        )


def _greedy(logits):
    return torch.argmax(logits, dim=-1)


def _session(sampled_tokens, *, max_new_tokens=8):
    thinker = _ScriptedThinker(sampled_tokens)
    session = IncrementalThinkerSession(
        thinker=thinker,
        input_ids=torch.tensor([[10, 11]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        thinker_kwargs={},
        stop_ids={2},
        max_new_tokens=max_new_tokens,
        hidden_selection=-1,
        sample_fn=_greedy,
    )
    return thinker, session


def test_incremental_thinker_commits_the_sampled_token_one_decode_later():
    """The Talker condition must use token 21's own hidden, not its predictor."""
    thinker, session = _session([21, 22, 2])

    assert session.prefill() == 21
    committed = session.step()

    assert committed.token_id == 21
    torch.testing.assert_close(
        committed.hidden, torch.tensor([[[21.0, -21.0]]])
    )
    torch.testing.assert_close(committed.embedding, committed.hidden)
    assert session.pending_token_id == 22
    assert thinker.forward_inputs[0].tolist() == [[10, 11]]
    assert thinker.forward_inputs[1].tolist() == [[21]]
    assert [value.tolist() for value in thinker.cache_positions] == [[0, 1], [2]]


def test_incremental_thinker_excludes_eos_from_generated_and_committed_tokens():
    """A sampled stop token must close the text watermark without conditioning Talker."""
    _, session = _session([21, 2])

    session.prefill()
    first = session.step()

    assert first.token_id == 21
    assert session.done is True
    assert session.pending_token_id is None
    assert session.generated_ids == [21]
    assert session.step() is None


def test_incremental_thinker_reports_each_non_eos_text_snapshot():
    """Removing the callback would make the web demo lose live translation text."""
    thinker = _ScriptedThinker([21, 22, 2])
    snapshots = []
    session = IncrementalThinkerSession(
        thinker=thinker,
        input_ids=torch.tensor([[10, 11]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        thinker_kwargs={},
        stop_ids={2},
        max_new_tokens=8,
        hidden_selection=-1,
        sample_fn=_greedy,
        token_callback=lambda token_ids: snapshots.append(list(token_ids)),
    )

    session.prefill()
    session.step()

    assert snapshots == [[21], [21, 22]]


def test_incremental_thinker_commits_last_token_at_generation_limit():
    """max_new_tokens must not drop the final token's Talker condition."""
    thinker, session = _session([21, 22], max_new_tokens=1)

    session.prefill()
    committed = session.step()

    assert committed.token_id == 21
    assert session.done is True
    assert session.generated_ids == [21]
    assert len(thinker.forward_inputs) == 2


def test_incremental_thinker_handles_immediate_eos_without_a_pending_token():
    """An empty translation must close cleanly and never fabricate a condition."""
    _, session = _session([2])

    assert session.prefill() is None
    assert session.done is True
    assert session.generated_ids == []
    assert session.step() is None


class _RecordingTalkerBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.codec_embedding = torch.nn.Embedding(32, 2)
        self.text_embedding = torch.nn.Embedding(32, 2)
        with torch.no_grad():
            for token_id in range(32):
                self.codec_embedding.weight[token_id] = torch.tensor(
                    [float(token_id), float(token_id) + 0.25]
                )
                self.text_embedding.weight[token_id] = torch.tensor(
                    [float(token_id), -float(token_id)]
                )
        self.calls = []

    def forward_cached(self, inputs_embeds, past_key_values=None):
        self.calls.append(inputs_embeds.detach().clone())
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class _FakeOnlineModel:
    def __init__(self, *, full_add=True):
        backbone = _RecordingTalkerBackbone()
        self.talker = SimpleNamespace(
            model=backbone,
            text_projection=torch.nn.Identity(),
            codec_head=torch.nn.Identity(),
        )
        self.thinker_adapter = torch.nn.Identity()
        self.config = SimpleNamespace(
            full_add_layout=full_add,
            thinker_signal_type="hidden_plus_embed",
            talker_config=SimpleNamespace(
                tts_pad_token_id=3,
                tts_bos_token_id=4,
            ),
        )


def _online_talker(*, full_add=True, compile_model=False, compile_fn=torch.compile):
    model = _FakeOnlineModel(full_add=full_add)
    session = OnlineTalkerSession(
        model=model,
        control_prefix_ids=torch.tensor([[5, 6, 7, 8]]),
        prompt_hidden=torch.tensor([[[1.0, 2.0]]]),
        prompt_embeddings=torch.tensor([[[0.5, 1.5]]]),
        full_add_prompt_len=1 if full_add else None,
        compile_model=compile_model,
        compile_fn=compile_fn,
    )
    return model, session


def test_online_talker_splits_metadata_prefill_from_first_reply_condition():
    """codec_bos must wait for the first committed reply condition."""
    model, session = _online_talker(full_add=True)

    session.prefill_metadata()
    condition = SimpleNamespace(
        hidden=torch.tensor([[[2.0, 3.0]]]),
        embedding=torch.tensor([[[4.0, 5.0]]]),
    )
    logits, hidden = session.start(condition, thinker_done=False)

    metadata = model.talker.model.calls[0]
    assert metadata.shape == (1, 3, 2)
    torch.testing.assert_close(
        metadata[:, 0],
        model.talker.model.codec_embedding(torch.tensor([[5]]))[:, 0]
        + torch.tensor([[1.5, 3.5]]),
    )
    torch.testing.assert_close(
        metadata[:, 2],
        model.talker.model.codec_embedding(torch.tensor([[7]]))[:, 0]
        + model.talker.model.text_embedding(torch.tensor([[4]]))[:, 0],
    )
    expected_first = (
        model.talker.model.codec_embedding(torch.tensor([[8]]))
        + torch.tensor([[[6.0, 8.0]]])
    )
    torch.testing.assert_close(model.talker.model.calls[1], expected_first)
    torch.testing.assert_close(logits, expected_first)
    torch.testing.assert_close(hidden, expected_first)


def test_online_talker_rejects_pad_before_thinker_eos_and_allows_it_after():
    """The Talker must never advance beyond the committed text watermark."""
    model, session = _online_talker(full_add=False)
    session.prefill_metadata()

    with pytest.raises(RuntimeError, match="watermark"):
        session.start(None, thinker_done=False)

    session.start(None, thinker_done=True)
    session.step(9, None, thinker_done=True)
    expected_pad = model.talker.model.text_embedding(torch.tensor([[3]]))
    expected_step = model.talker.model.codec_embedding(torch.tensor([[9]])) + expected_pad
    torch.testing.assert_close(model.talker.model.calls[-1], expected_step)


def test_online_talker_compile_wraps_the_registered_backbone_module():
    """Dynamo must receive a module so parameters and RoPE buffers stay state."""
    targets = []

    def fake_compile(target, **kwargs):
        targets.append((target, kwargs))
        return target

    model, session = _online_talker(
        full_add=False, compile_model=True, compile_fn=fake_compile
    )
    session.prefill_metadata()

    assert len(targets) == 1
    target, kwargs = targets[0]
    assert isinstance(target, torch.nn.Module)
    assert target.model is model.talker.model
    assert kwargs == {"mode": "default", "fullgraph": False}


def test_talker_compile_entry_avoids_nested_inference_mode_during_aot_capture():
    """AOT capture must not create weak-referenced inference RoPE buffers."""
    from transformers.cache_utils import DynamicCache

    talker = _tiny_real_talker()
    target = _TalkerForwardModule(talker.model)
    compiled = torch.compile(target, backend="aot_eager", fullgraph=False)
    embedding = torch.randn(1, 1, talker.config.hidden_size)

    with torch.inference_mode():
        output = compiled(embedding, DynamicCache())

    assert output.last_hidden_state.shape == embedding.shape
    assert output.past_key_values.get_seq_length() == 1


def _tiny_real_talker():
    config = Qwen3TtsTalkerConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        codec_vocab_size=32,
        num_code_groups=3,
        code_predictor_hidden_size=16,
        code_predictor_intermediate_size=32,
        code_predictor_num_layers=1,
        code_predictor_num_attention_heads=4,
        code_predictor_num_key_value_heads=2,
        code_predictor_head_dim=4,
        code_predictor_vocab_size=16,
        max_position_embeddings=32,
    )
    return Qwen3TtsTalkerForConditionalGeneration(config).float().eval()


def test_residual_frame_batcher_matches_individual_real_predictor_outputs():
    """Batching frames must preserve greedy residual codes and frame order."""
    torch.manual_seed(17)
    talker = _tiny_real_talker()
    runner = CodePredictorRunner(talker, use_kv_cache=True)
    h0 = torch.randn(1, 1, talker.config.hidden_size)
    h1 = torch.randn(1, 1, talker.config.hidden_size)
    m0 = torch.tensor([3])
    m1 = torch.tensor([5])
    expected = [runner.predict(h0, m0)[0], runner.predict(h1, m1)[0]]
    batcher = ResidualFrameBatcher(runner, batch_size=2)

    assert batcher.add(m0, h0) == []
    ready = batcher.add(m1, h1)

    assert len(ready) == 2
    assert int(ready[0].main_code.item()) == 3
    assert int(ready[1].main_code.item()) == 5
    assert torch.equal(ready[0].residual_codes, expected[0])
    assert torch.equal(ready[1].residual_codes, expected[1])


def test_residual_frame_batcher_flushes_a_partial_tail():
    """End-of-stream must not drop frames below the configured batch size."""
    torch.manual_seed(19)
    talker = _tiny_real_talker()
    runner = CodePredictorRunner(talker)
    hidden = torch.randn(1, 1, talker.config.hidden_size)
    batcher = ResidualFrameBatcher(runner, batch_size=4)

    batcher.add(torch.tensor([7]), hidden)
    ready = batcher.flush()

    assert len(ready) == 1
    assert int(ready[0].main_code.item()) == 7
    assert batcher.pending == 0


def test_residual_frame_batcher_pads_partial_tail_to_stable_compile_shape():
    """A compiled predictor must not see a new batch shape at end-of-stream."""

    class RecordingRunner:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, codec_hiddens, main_codes):
            self.batch_sizes.append(int(main_codes.shape[0]))
            return main_codes[:, None].repeat(1, 3)

    runner = RecordingRunner()
    batcher = ResidualFrameBatcher(runner, batch_size=4)
    hidden = torch.zeros(1, 1, 8)
    batcher.add(torch.tensor([7]), hidden)
    batcher.add(torch.tensor([9]), hidden)

    ready = batcher.flush()

    assert runner.batch_sizes == [4]
    assert [int(frame.main_code.item()) for frame in ready] == [7, 9]


class _FakeThinkerSession:
    def __init__(self, conditions):
        self.conditions = list(conditions)
        self.done = False
        self.events = []

    def prefill(self):
        self.events.append("thinker_prefill")
        if not self.conditions:
            self.done = True

    def step(self):
        self.events.append("thinker_step")
        if not self.conditions:
            self.done = True
            return None
        condition = self.conditions.pop(0)
        if not self.conditions:
            self.done = True
        return condition


class _FakeTalkerSession:
    def __init__(self, shared_events):
        self.shared_events = shared_events

    def prefill_metadata(self):
        self.shared_events.append("talker_metadata")

    def start(self, condition, *, thinker_done):
        token = None if condition is None else condition.token_id
        self.shared_events.append(("talker_start", token, thinker_done))
        return torch.tensor([[[0.0]]]), torch.tensor([[[100.0]]])

    def step(self, main_id, condition, *, thinker_done):
        token = None if condition is None else condition.token_id
        self.shared_events.append(
            ("talker_step", int(main_id.item()), token, thinker_done)
        )
        return torch.tensor([[[0.0]]]), torch.tensor([[[101.0]]])


def test_scheduler_never_advances_talker_past_text_watermark():
    """Pre-EOS frames require committed conditions; only post-EOS may use pad."""
    shared_events = []
    conditions = [
        SimpleNamespace(token_id=21),
        SimpleNamespace(token_id=22),
    ]
    thinker = _FakeThinkerSession(conditions)
    talker = _FakeTalkerSession(shared_events)
    mains = iter([10, 11, 12, 99])
    scheduler = ThinkerTalkerScheduler(
        thinker_session=thinker,
        talker_session=talker,
        start_tokens=1,
        max_codec_frames=8,
        codec_eos_id=99,
        sample_main_fn=lambda logits, previous: torch.tensor(
            [next(mains)], dtype=torch.long
        ),
    )

    frames = list(scheduler.run())

    assert [int(frame.main_code.item()) for frame in frames] == [10, 11, 12]
    assert shared_events == [
        "talker_metadata",
        ("talker_start", 21, False),
        ("talker_step", 10, 22, True),
        ("talker_step", 11, None, True),
        ("talker_step", 12, None, True),
    ]


def test_scheduler_waits_for_configured_start_watermark():
    """A two-token startup buffer must commit twice before Talker starts."""
    shared_events = []
    thinker = _FakeThinkerSession(
        [SimpleNamespace(token_id=31), SimpleNamespace(token_id=32)]
    )
    scheduler = ThinkerTalkerScheduler(
        thinker_session=thinker,
        talker_session=_FakeTalkerSession(shared_events),
        start_tokens=2,
        max_codec_frames=1,
        codec_eos_id=99,
        sample_main_fn=lambda logits, previous: torch.tensor([7]),
    )

    list(scheduler.run())

    assert thinker.events == ["thinker_prefill", "thinker_step", "thinker_step"]
    assert shared_events[1] == ("talker_start", 31, True)


def test_online_control_prefix_preserves_full_add_and_speaker_layout():
    """Online prefix positions must match the existing full-add inference template."""
    model_config = SimpleNamespace(
        codec_mask_id=9,
        codec_bos_id=8,
        codec_eos_id=10,
        codec_pad_id=7,
        codec_think_id=5,
        codec_nothink_id=4,
        codec_think_bos_id=6,
        codec_think_eos_id=11,
        codec_language_id={"chinese": 12},
        talker_config=SimpleNamespace(
            num_code_groups=3, code_predictor_vocab_size=16
        ),
    )

    prefix, speaker_slot = build_online_control_prefix(
        model_config,
        target_lang="zh",
        prompt_len=2,
        full_add=True,
        use_speaker=True,
    )

    assert prefix == [9, 9, 5, 6, 12, 11, 7, 7, 8]
    assert speaker_slot == 6
