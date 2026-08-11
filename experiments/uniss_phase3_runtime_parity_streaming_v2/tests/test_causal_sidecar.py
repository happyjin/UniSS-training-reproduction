from experiments.uniss_phase3_runtime_parity_streaming_v2.data.causal_sidecar import (
    runtime_commit_end_times,
)


def test_runtime_commit_times_include_right_context() -> None:
    assert runtime_commit_end_times(1000, 8) == [
        320,
        320,
        480,
        480,
        640,
        640,
        800,
        800,
    ]


def test_runtime_commit_times_flush_at_eos() -> None:
    assert runtime_commit_end_times(220, 3) == [220, 220, 220]
