from __future__ import annotations

from training import constants_uniss as c
from web_demo.runtime_parity_streaming_v12.inference import MicroblockPromptSession
from web_demo.runtime_parity_streaming_v2.tests.test_session import RecordingBackend


def _prepared_session():
    backend = RecordingBackend()
    session = MicroblockPromptSession(
        backend, target_lang="cmn", speaker_global=tuple(range(32))
    )
    session.begin_tick([3, 5])
    session.begin_write()
    session.append_text_ids([101])
    session.end_text_with_hidden()
    return backend, session


def test_intermediate_microblock_enters_main_cache_and_returns_hidden() -> None:
    backend, session = _prepared_session()
    result = session.append_semantic_microblock_with_hidden([7, 8, 9, 10])
    assert result.last_hidden == ("hidden_at", len(session.transcript) - 1)
    assert backend.calls[-1].capture_last_hidden
    assert backend.calls[-1].values == tuple(c.encode_bicodec_semantic([7, 8, 9, 10]))
    tick = session.commit_final_semantic_microblock([11, 12])
    assert tick.semantic_codes == (7, 8, 9, 10, 11, 12)
    assert session.transcript[-1] == c.TOKEN_END_SEMANTIC


def test_final_microblock_is_one_dispatch_and_naturally_closes_write() -> None:
    backend, session = _prepared_session()
    calls_before = len(backend.calls)
    tick = session.commit_final_semantic_microblock([20, 21, 22])
    assert len(backend.calls) == calls_before + 1
    assert backend.calls[-1].values == (
        *c.encode_bicodec_semantic([20, 21, 22]),
        c.TOKEN_END_SEMANTIC,
    )
    assert tick.semantic_codes == (20, 21, 22)
