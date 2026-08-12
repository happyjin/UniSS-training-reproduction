import inspect

from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize14_dagger_prefix.pretrain_generalize14 import (
    RuntimeParityGeneralize14Objective,
    V14_WEIGHTS,
)


def test_latency_and_model_prefix_recovery_have_optimization_mass() -> None:
    assert V14_WEIGHTS["deadline_survival"] > 0
    assert V14_WEIGHTS["runtime_prefix_recovery"] > 0
    assert V14_WEIGHTS["phase3_replay"] > 0


def test_trajectory_keeps_one_full_vocabulary_ce_graph() -> None:
    source = inspect.getsource(RuntimeParityGeneralize14Objective.trajectory)
    assert source.count("token_cross_entropy_values(") == 1
    assert "super().trajectory(" not in source


def test_runtime_probe_does_not_build_full_vocabulary_output_layer() -> None:
    from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize14_dagger_prefix import (
        pretrain_generalize14 as module,
    )

    source = inspect.getsource(module._probe_output_processor)
    assert 'kwargs["output_layer"]' not in source
