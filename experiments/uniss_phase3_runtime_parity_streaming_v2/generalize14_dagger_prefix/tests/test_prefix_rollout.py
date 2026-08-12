import torch

from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize14_dagger_prefix.prefix_rollout import (
    apply_prefix_predictions,
    expand_recovery_mask,
    prefix_schedule,
)


def test_schedule_warms_up_then_reaches_two_round_half_rollin() -> None:
    assert prefix_schedule(0.05).rounds == 0
    middle = prefix_schedule(0.45)
    assert middle.rounds == 2
    assert 0.25 < middle.probability < 0.45
    final = prefix_schedule(1.0)
    assert final.rounds == 2
    assert final.probability == 0.5


def test_predictions_shift_into_matching_input_positions() -> None:
    tokens = torch.tensor([[10, 11, 12, 13, 14]])
    predicted = torch.tensor([[21, 22, 23, 24, 25]])
    eligible = torch.tensor([[False, True, True, False, False]])
    updated, corrupted = apply_prefix_predictions(
        tokens, predicted, eligible, probability=1.0
    )
    assert updated.tolist() == [[10, 11, 22, 23, 14]]
    assert corrupted.tolist() == [[False, False, True, True, False]]


def test_recovery_mask_never_crosses_batch_rows() -> None:
    corrupted = torch.tensor(
        [[False, False, True, False, False], [True, False, False, False, False]]
    )
    recovery = expand_recovery_mask(corrupted, horizon=3)
    assert recovery.tolist() == [
        [False, False, True, True, True],
        [True, True, True, False, False],
    ]
