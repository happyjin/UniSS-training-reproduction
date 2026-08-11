from __future__ import annotations

import unittest

import torch

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.trajectory_packing import (
    ROLE_ACTION,
    ROLE_KD,
    ROLE_OBSERVED,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training.losses import (
    grouped_deadline_survival_term,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.packing import (
    build_token_sample,
    shift_sample,
)
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schedule import tick_times
from experiments.uniss_true_subsecond_pilot15_epoch1_v2.data.schema import (
    Action,
    TrajectoryRecord,
)


def record(**overrides) -> TrajectoryRecord:
    values = dict(
        sample_id="sample-1",
        shard=0,
        row_index=3,
        src_lang="eng",
        tgt_lang="cmn",
        source_duration_ms=4000,
        chunk_end_ms=480,
        future_1_end_ms=640,
        future_2_end_ms=800,
        trajectory_kind="fixed_480",
        causal_source_glm=(1, 2, 3),
        future_1_source_glm=(1, 2, 3, 4),
        future_2_source_glm=(1, 2, 3, 4, 5),
        frontend_token_cache="/tmp/cache.npz::causal:0",
        translation_ids=(101, 102, 103, 104),
        teacher_prefix_topk_path="/tmp/cache.npz::teacher:0",
        teacher_future_1_topk_path="/tmp/cache.npz::teacher:1",
        teacher_future_2_topk_path="/tmp/cache.npz::teacher:2",
        teacher_full_topk_path="/tmp/cache.npz::teacher:3",
        previous_committed_length=2,
        stable_target_length=3,
        new_supported_count=1,
        support_bucket=1,
        safe_commit_mask=(True, True, True, False),
        natural_action_target=Action.WRITE,
        deadline_action_target=Action.WRITE,
        deadline_forced_target=False,
        deadline_loss_enabled=True,
        target_text_delta_ids=(103,),
        semantic_history_start=0,
        semantic_history_end=8,
        semantic_target_start=8,
        semantic_target_end=16,
        speaker_global=tuple(range(32)),
    )
    values.update(overrides)
    return TrajectoryRecord(**values).with_checksum()


class RepairedDataTest(unittest.TestCase):
    def test_schedule_contains_exact_deadline_and_is_deterministic(self):
        first = tick_times("abc", 8000)
        self.assertEqual(first, tick_times("abc", 8000))
        self.assertEqual(first[:4], (320, 480, 640, 800))
        self.assertEqual(len(first), 5)
        self.assertNotIn(800, tick_times("short", 740))

    def test_semantic_cursor_and_short_final_block(self):
        value = record(semantic_target_end=13)
        self.assertEqual(value.semantic_target_start, value.semantic_history_end)
        self.assertEqual(value.semantic_target_end - value.semantic_target_start, 5)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            record(semantic_target_start=9, semantic_target_end=14)

    def test_history_is_observed_without_repeated_ce(self):
        value = record()
        sample = build_token_sample(value, list(range(100)))
        shifted = shift_sample(sample)
        previous = list(value.translation_ids[: value.previous_committed_length])
        starts = [
            index
            for index in range(len(sample.input_ids) - len(previous) + 1)
            if list(sample.input_ids[index : index + len(previous)]) == previous
        ]
        self.assertTrue(starts)
        self.assertTrue(
            all(
                role == ROLE_OBSERVED
                for role in sample.token_roles[starts[0] : starts[0] + len(previous)]
            )
        )
        observed_labels = [
            label
            for label, role, mask in zip(
                shifted.labels, shifted.token_roles, shifted.loss_mask
            )
            if role == ROLE_OBSERVED and mask == 0.0
        ]
        self.assertTrue(set(previous).issubset(observed_labels))

    def test_forced_write_has_no_hard_action_or_content_ce(self):
        forced = record(
            chunk_end_ms=800,
            future_1_end_ms=960,
            future_2_end_ms=1120,
            trajectory_kind="fixed_800",
            previous_committed_length=0,
            stable_target_length=0,
            new_supported_count=0,
            support_bucket=0,
            safe_commit_mask=(False, False, False, False),
            natural_action_target=Action.READ,
            deadline_action_target=Action.WRITE,
            deadline_forced_target=True,
            target_text_delta_ids=(),
            semantic_history_end=0,
            semantic_target_start=0,
            semantic_target_end=0,
        )
        shifted = shift_sample(
            build_token_sample(forced, list(range(100)), anticipation_ids=(120, 121))
        )
        action_masks = [
            mask
            for role, mask in zip(shifted.token_roles, shifted.loss_mask)
            if role == ROLE_ACTION
        ]
        kd_masks = [
            mask
            for role, mask in zip(shifted.token_roles, shifted.loss_mask)
            if role == ROLE_KD
        ]
        self.assertEqual(action_masks, [0.0])
        self.assertTrue(kd_masks)
        self.assertEqual(set(kd_masks), {0.0})

    def test_deadline_mask_preserves_v1_default(self):
        logits = torch.tensor(
            [[3.0, -3.0], [3.0, -3.0], [3.0, -3.0], [3.0, -3.0]]
        )
        groups = torch.tensor([1, 1, 2, 2])
        ticks = torch.tensor([640, 800, 640, 800])
        soft = torch.full((4,), 640)
        hard = torch.full((4,), 800)
        legacy = grouped_deadline_survival_term(logits, groups, ticks, soft, hard)
        all_enabled = grouped_deadline_survival_term(
            logits, groups, ticks, soft, hard, torch.ones(4, dtype=torch.bool)
        )
        one_session = grouped_deadline_survival_term(
            logits,
            groups,
            ticks,
            soft,
            hard,
            torch.tensor([True, True, False, False]),
        )
        torch.testing.assert_close(legacy.mean, all_enabled.mean)
        self.assertLess(one_session.denominator, legacy.denominator)

    def test_forced_write_only_at_exact_800ms(self):
        with self.assertRaisesRegex(ValueError, "exact hard deadline"):
            record(
                chunk_end_ms=960,
                future_1_end_ms=1120,
                future_2_end_ms=1280,
                previous_committed_length=0,
                stable_target_length=0,
                new_supported_count=0,
                support_bucket=0,
                safe_commit_mask=(False, False, False, False),
                natural_action_target=Action.READ,
                deadline_action_target=Action.WRITE,
                deadline_forced_target=True,
                target_text_delta_ids=(),
                semantic_history_end=0,
                semantic_target_start=0,
                semantic_target_end=0,
            )


if __name__ == "__main__":
    unittest.main()
