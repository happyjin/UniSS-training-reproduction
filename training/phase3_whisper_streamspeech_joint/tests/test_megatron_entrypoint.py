from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.phase3_whisper_streamspeech_joint.pretrain_joint_megatron import (
    lr_group_values,
    parse_chunks,
    validate_joint_args,
)


class MegatronEntrypointTest(unittest.TestCase):
    def test_chunks_and_lr_groups_match_the_plan(self) -> None:
        self.assertEqual(parse_chunks("320,640,offline"), (320, 640, None))
        values = lr_group_values(1e-4, 1e-5)
        self.assertAlmostEqual(values["uniss_lr_qwen"]["max_lr"], 2e-6)
        self.assertAlmostEqual(values["uniss_lr_qwen_io"]["max_lr"], 1e-6)
        self.assertAlmostEqual(values["uniss_lr_whisper_bottom"]["max_lr"], 5e-6)

    def test_validation_requires_megatron_phase3_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in (
                "joint_train_manifest",
                "joint_valid_manifest",
                "joint_tokenizer_map_dir",
                "joint_phase3_replay_packed",
                "joint_phase3_replay_offsets",
                "joint_whisper_model",
                "joint_phase3_model",
                "joint_direction_index_dir",
            ):
                (path / name).touch()
            args = SimpleNamespace(
                micro_batch_size=1,
                global_batch_size=128,
                seq_length=18_000,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                joint_replay_probability=0.2,
                joint_chunks="320,640,960,1280,offline",
                joint_right_context_ms=80,
                **{name: str(path / name) for name in (
                    "joint_train_manifest",
                    "joint_valid_manifest",
                    "joint_tokenizer_map_dir",
                    "joint_phase3_replay_packed",
                    "joint_phase3_replay_offsets",
                    "joint_whisper_model",
                    "joint_phase3_model",
                    "joint_direction_index_dir",
                )},
            )
            validate_joint_args(args)
            args.micro_batch_size = 2
            validate_joint_args(args)
            args.global_batch_size = 64
            with self.assertRaises(ValueError):
                validate_joint_args(args)


if __name__ == "__main__":
    unittest.main()
