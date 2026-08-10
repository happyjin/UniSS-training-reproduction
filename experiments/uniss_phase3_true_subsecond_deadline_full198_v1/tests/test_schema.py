from __future__ import annotations

import unittest

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.schema import (
    Action,
    TrajectoryPlan,
    TrajectoryRecord,
)


class SchemaTest(unittest.TestCase):
    def test_plan_rejects_future_past_duration(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryPlan(
                sample_id="x",
                shard=0,
                row_index=1,
                src_lang="eng",
                tgt_lang="cmn",
                source_duration_ms=640,
                chunk_end_ms=480,
                future_1_end_ms=640,
                future_2_end_ms=800,
                trajectory_kind="early",
                source_glm_length=8,
                source_bicodec_length=32,
                target_bicodec_length=32,
            )

    def test_record_checksum_round_trip(self) -> None:
        record = TrajectoryRecord(
            sample_id="x",
            shard=0,
            row_index=1,
            src_lang="eng",
            tgt_lang="cmn",
            source_duration_ms=960,
            chunk_end_ms=640,
            future_1_end_ms=800,
            future_2_end_ms=960,
            causal_source_glm=(1, 2),
            future_1_source_glm=(1, 2, 3),
            future_2_source_glm=(1, 2, 3, 4),
            frontend_token_cache="cache.npz::causal:0",
            translation_ids=(10, 11, 12),
            teacher_prefix_topk_path="p.npz",
            teacher_future_1_topk_path="f1.npz",
            teacher_future_2_topk_path="f2.npz",
            teacher_full_topk_path="full.npz",
            previous_committed_length=0,
            stable_target_length=1,
            new_supported_count=1,
            support_bucket=1,
            safe_commit_mask=(True, False, False),
            natural_action_target=Action.WRITE,
            deadline_action_target=Action.WRITE,
            deadline_forced_target=False,
            target_text_delta_ids=(10,),
            semantic_history_start=0,
            semantic_history_end=0,
            semantic_target_start=0,
            semantic_target_end=8,
            speaker_global=tuple(range(32)),
        ).with_checksum()
        restored = TrajectoryRecord.from_dict(record.to_dict())
        self.assertEqual(restored, record)

    def test_natural_write_requires_supported_content(self) -> None:
        values = dict(
            sample_id="x",
            shard=0,
            row_index=1,
            src_lang="eng",
            tgt_lang="cmn",
            source_duration_ms=960,
            chunk_end_ms=640,
            future_1_end_ms=800,
            future_2_end_ms=960,
            causal_source_glm=(1,),
            future_1_source_glm=(1,),
            future_2_source_glm=(1,),
            frontend_token_cache="cache.npz::causal:0",
            translation_ids=(10,),
            teacher_prefix_topk_path="p",
            teacher_future_1_topk_path="f1",
            teacher_future_2_topk_path="f2",
            teacher_full_topk_path="full",
            previous_committed_length=0,
            stable_target_length=0,
            new_supported_count=0,
            support_bucket=0,
            safe_commit_mask=(False,),
            natural_action_target=Action.WRITE,
            deadline_action_target=Action.WRITE,
            deadline_forced_target=True,
            target_text_delta_ids=(),
            semantic_history_start=0,
            semantic_history_end=0,
            semantic_target_start=0,
            semantic_target_end=8,
            speaker_global=tuple(range(32)),
        )
        with self.assertRaises(ValueError):
            TrajectoryRecord(**values)


if __name__ == "__main__":
    unittest.main()
