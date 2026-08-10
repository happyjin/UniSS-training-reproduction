from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.schema import (
    Action,
    TrajectoryRecord,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    ROLE_ACTION,
    ROLE_KD,
    ROLE_SEMANTIC,
    build_trajectory_token_sample,
    pack_trajectory_samples,
    shift_trajectory_sample,
)
from training import constants_uniss as c


def _record(*, supported: int, forced: bool = False) -> TrajectoryRecord:
    stable = supported
    natural = Action.WRITE if supported else Action.READ
    deadline = Action.WRITE if forced or supported else Action.READ
    return TrajectoryRecord(
        sample_id="sample",
        shard=0,
        row_index=1,
        src_lang="eng",
        tgt_lang="cmn",
        source_duration_ms=1600,
        chunk_end_ms=800 if forced else 480,
        future_1_end_ms=960,
        future_2_end_ms=1120,
        causal_source_glm=(1, 2, 3),
        future_1_source_glm=(1, 2, 3, 4),
        future_2_source_glm=(1, 2, 3, 4, 5),
        frontend_token_cache="bundle.npz::causal:1",
        translation_ids=(10, 11, 12),
        teacher_prefix_topk_path="bundle.npz::teacher:8",
        teacher_future_1_topk_path="bundle.npz::teacher:9",
        teacher_future_2_topk_path="bundle.npz::teacher:10",
        teacher_full_topk_path="bundle.npz::teacher:11",
        previous_committed_length=0,
        stable_target_length=stable,
        new_supported_count=supported,
        support_bucket=min(supported, 4),
        safe_commit_mask=tuple(index < stable for index in range(3)),
        natural_action_target=natural,
        deadline_action_target=deadline,
        deadline_forced_target=forced,
        target_text_delta_ids=tuple((10, 11, 12)[:stable]),
        semantic_history_start=0,
        semantic_history_end=0,
        semantic_target_start=0,
        semantic_target_end=8,
        speaker_global=tuple(range(32)),
    ).with_checksum()


class TrajectoryPackingTest(unittest.TestCase):
    def test_action_role_is_aligned_to_next_token_label(self) -> None:
        sample = build_trajectory_token_sample(_record(supported=1), list(range(16)))
        action_index = sample.input_ids.index(c.TOKEN_WRITE_GENERATE)
        shifted = shift_trajectory_sample(sample)
        self.assertEqual(shifted.labels[action_index - 1], c.TOKEN_WRITE_GENERATE)
        self.assertEqual(shifted.token_roles[action_index - 1], ROLE_ACTION)
        self.assertIn(ROLE_SEMANTIC, shifted.token_roles)

    def test_deadline_forced_write_does_not_use_hard_future_content(self) -> None:
        sample = build_trajectory_token_sample(
            _record(supported=0, forced=True),
            list(range(16)),
            anticipation_ids=(101, 102),
        )
        action_index = sample.input_ids.index(c.TOKEN_WRITE_GENERATE)
        self.assertEqual(sample.input_ids[action_index + 4 : action_index + 6], (101, 102))
        self.assertEqual(sample.token_roles[action_index + 4 : action_index + 6], (ROLE_KD, ROLE_KD))
        self.assertNotIn(ROLE_SEMANTIC, sample.token_roles)
        shifted = shift_trajectory_sample(sample)
        self.assertTrue(
            all(mask == 0.0 for mask, role in zip(shifted.loss_mask, shifted.token_roles) if role == ROLE_KD)
        )

    def test_packed_sidecars_align_with_boundaries(self) -> None:
        write = shift_trajectory_sample(
            build_trajectory_token_sample(_record(supported=1), list(range(16)))
        )
        read = shift_trajectory_sample(
            build_trajectory_token_sample(_record(supported=0), list(range(16)))
        )
        packed = list(pack_trajectory_samples((write, read), seq_length=160))
        self.assertEqual(len(packed), 1)
        item = packed[0]
        self.assertEqual(len(item["trajectory_sidecars"]), 2)
        self.assertEqual(len(item["sample_boundaries"]), 2)
        self.assertEqual(len(item["tokens"]), 160)
        self.assertEqual(len(item["token_roles"]), 160)
        self.assertTrue(any(item["loss_mask"]))


if __name__ == "__main__":
    unittest.main()
