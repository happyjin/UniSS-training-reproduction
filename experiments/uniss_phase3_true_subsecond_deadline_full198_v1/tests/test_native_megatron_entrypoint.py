from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.training import (
    pretrain_true_subsecond_megatron as entrypoint,
)


class NativeMegatronEntrypointTest(unittest.TestCase):
    def test_steady_state_rerun_checkpoint_is_a_strict_load_template(self) -> None:
        value = {
            "mode": "validate_results",
            "state": "not_running_yet",
            "current_iteration": 5,
            "sharded": object(),
        }
        self.assertTrue(entrypoint._is_complete_rerun_checkpoint_state(value))
        value.pop("sharded")
        self.assertTrue(entrypoint._is_complete_rerun_checkpoint_state(value))
        self.assertFalse(entrypoint._is_complete_rerun_checkpoint_state({"state": "x"}))

    def test_lr_groups_match_frozen_plan(self) -> None:
        args = SimpleNamespace(
            lr=5e-5,
            true_lr_qwen_lora=1e-5,
            true_lr_frontend=5e-6,
            true_lr_new_heads=5e-5,
            true_min_lr=1e-6,
        )
        values = entrypoint.lr_group_values(args)
        self.assertEqual(values["uniss_lr_qwen_lora"]["max_lr"], 1e-5)
        self.assertEqual(values["uniss_lr_frontend"]["max_lr"], 5e-6)
        self.assertTrue(values["uniss_lr_frontend"]["uniss_dynamic_frontend_lr"])
        self.assertEqual(values["uniss_lr_new_heads"]["max_lr"], 5e-5)

    def test_phase3_fingerprint_detects_loaded_embedding(self) -> None:
        model = nn.Module()
        model.embedding = nn.Module()
        model.embedding.word_embeddings = nn.Embedding(8, 6)
        with torch.no_grad():
            model.embedding.word_embeddings.weight.copy_(
                torch.arange(48, dtype=torch.float32).reshape(8, 6)
            )
        rows, columns = [1, 7], [0, 3, 5]
        expected = (
            model.embedding.word_embeddings.weight[rows][:, columns].float().tolist()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fingerprint.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "uniss_phase3_embedding_fingerprint_v1",
                        "rows": rows,
                        "columns": columns,
                        "values": expected,
                    }
                )
            )
            entrypoint.verify_phase3_fingerprint(model, path)
            self.assertTrue(model._true_phase3_fingerprint_verified)

    def test_output_processor_uses_fixed_metric_schema(self) -> None:
        class FakeLayer:
            def __call__(self, hidden, **_kwargs):
                logits = torch.cat((hidden, hidden[..., :1]), dim=-1)
                return logits, None

        class FakeObjective:
            def replay(self, logits, labels, loss_mask):
                self.shapes = (logits.shape, labels.shape, loss_mask.shape)
                return object()

        objective = FakeObjective()
        metrics = OrderedDict(
            (name, torch.tensor(float(index + 1)))
            for index, name in enumerate(entrypoint.METRIC_NAMES)
        )
        kwargs = {
            "context": {
                "objective": objective,
                "sample_kind": "replay",
                "batch": {},
                "progress": 0.5,
                "frontend_residual_rms": torch.tensor(0.0),
                "word_embedding_weight": torch.empty(0),
            },
            "hidden_states": torch.randn(4, 1, 3),
            "output_layer": FakeLayer(),
            "output_weight": None,
            "runtime_gather_output": None,
            "scale_logits": lambda value: value,
            "labels": torch.zeros(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4),
        }
        with patch.object(
            entrypoint,
            "distributed_weighted_objective",
            return_value=(torch.tensor(2.0), metrics),
        ):
            output = entrypoint._joint_output_processor(**kwargs)
        self.assertEqual(tuple(output.shape), (1 + len(entrypoint.METRIC_NAMES),))
        self.assertEqual(objective.shapes, (torch.Size([4, 4]), torch.Size([4]), torch.Size([4])))
        self.assertEqual(float(output[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
