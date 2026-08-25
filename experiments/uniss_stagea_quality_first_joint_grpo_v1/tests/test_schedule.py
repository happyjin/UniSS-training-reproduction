from torch.utils.data import Dataset

from experiments.uniss_stagea_quality_first_joint_grpo_v1.training.schedule import (
    OneFamilyCoverageSchedule,
)


class Values(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        return index


def test_one_coverage_is_deterministic_and_complete():
    left = OneFamilyCoverageSchedule(
        Values(), total_samples=12, global_batch_size=4, data_parallel_group_size=2, shuffle_seed=7, split="train"
    )
    right = OneFamilyCoverageSchedule(
        Values(), total_samples=12, global_batch_size=4, data_parallel_group_size=2, shuffle_seed=7, split="train"
    )
    assert [left[index] for index in range(12)] == [right[index] for index in range(12)]
    assert sorted(left[index] for index in range(11)) == list(range(11))
    assert len(left) == 12
