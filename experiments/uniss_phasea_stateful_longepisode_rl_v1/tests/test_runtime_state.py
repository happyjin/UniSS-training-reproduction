from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.commit import (
    AppendOnlyDeltaCommitter,
    StablePrefixCommitter,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.state import (
    StreamingSessionState,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.tts_queue import (
    QueueStatus,
    TTSQueue,
)


def test_stable_prefix_never_rolls_back_committed_tokens():
    value = StablePrefixCommitter(holdback=1)
    assert value.update([1, 2, 3]) == []
    assert value.update([1, 2, 3, 4]) == [1, 2]
    assert value.update([9, 9, 9]) == []
    assert value.committed == [1, 2]
    assert value.revision_conflicts == 1


def test_delta_committer_uses_immutable_prefix_and_flushes_only_at_true_final():
    value = AppendOnlyDeltaCommitter(holdback=1)
    assert value.update([10, 11, 12]) == []
    assert value.update([10, 11, 13]) == [10]
    assert value.committed == [10]
    assert value.update([11, 13, 14]) == [11]
    assert value.force_pending() == [13, 14]
    assert value.committed == [10, 11, 13, 14]


def test_tts_text_survives_empty_or_unhealthy_generation_until_ack():
    queue = TTSQueue()
    item = queue.append([1, 2], "hello world", 640)
    queue.begin(item.item_id)
    queue.acknowledge(
        item.item_id,
        semantic_tokens=0,
        continuation_count=0,
        audio_samples=0,
        healthy=False,
    )
    assert queue.pending[0].text == "hello world"
    assert queue.pending[0].status == QueueStatus.PENDING
    queue.begin(item.item_id)
    queue.acknowledge(
        item.item_id,
        semantic_tokens=321,
        continuation_count=1,
        audio_samples=64000,
        healthy=True,
    )
    assert queue.pending == []
    assert queue.acknowledged[0].continuation_count == 1


def test_memory_rollover_keeps_frontend_and_global_text_state():
    marker = object()
    state = StreamingSessionState(frontend_state=marker)
    state.asr_segment_committer.previous = [4, 5, 6]
    state.speech_embedding_ring.extend(["a", "b"])
    assert state.finalize_asr_segment() == [4, 5, 6]
    assert state.frontend_state is marker
    assert state.asr_committed_ids == [4, 5, 6]
    assert state.speech_embedding_ring == []
    assert not state.source_finished
    assert state.memory_rollovers == 1


def test_artificial_boundary_does_not_finalize_source():
    state = StreamingSessionState()
    state.mark_artificial_boundary()
    assert not state.source_finished
    assert state.artificial_boundary_finalizations == 1
    state.mark_true_final()
    assert state.source_finished
