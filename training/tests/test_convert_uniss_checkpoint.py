import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from training import convert_uniss_checkpoint as convert


class FakeProvider:
    instances = []

    def __init__(self):
        self.gradient_accumulation_fusion = True
        self.finalized = False
        self.provide_calls = []
        FakeProvider.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []

    def finalize(self):
        self.finalized = True

    def provide_distributed_model(self, **kwargs):
        self.provide_calls.append(kwargs)
        return ["fake_megatron_model"]


class FakeBridge:
    import_calls = []
    export_calls = []
    from_hf_calls = []
    save_calls = []

    @classmethod
    def reset(cls):
        cls.import_calls = []
        cls.export_calls = []
        cls.from_hf_calls = []
        cls.save_calls = []
        FakeProvider.reset()

    @classmethod
    def import_ckpt(cls, **kwargs):
        cls.import_calls.append(kwargs)

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        cls.from_hf_calls.append((args, kwargs))
        return cls()

    def to_megatron_provider(self, **kwargs):
        self.to_megatron_provider_kwargs = kwargs
        return FakeProvider()

    def save_megatron_model(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))

    def export_ckpt(self, **kwargs):
        self.export_calls.append(kwargs)


class ConvertUniSSCheckpointTest(unittest.TestCase):
    def setUp(self):
        FakeBridge.reset()

    def test_torch_dtype_from_name(self):
        self.assertIs(convert.torch_dtype_from_name(None), None)
        self.assertIs(convert.torch_dtype_from_name("bfloat16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            convert.torch_dtype_from_name("bad")

    def test_import_dry_run_checks_hf_path_and_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_model = Path(tmpdir) / "hf"
            megatron_path = Path(tmpdir) / "megatron"
            hf_model.mkdir()
            args = convert.parse_args(
                [
                    "import",
                    "--hf-model",
                    str(hf_model),
                    "--megatron-path",
                    str(megatron_path),
                    "--dry-run",
                ]
            )
            summary = convert.import_hf_to_megatron(args, autobridge=FakeBridge)
            self.assertEqual(summary.direction, "import")
            self.assertTrue(megatron_path.exists())
            self.assertEqual(FakeBridge.import_calls, [])

    def test_import_calls_autobridge_with_local_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_model = Path(tmpdir) / "hf"
            megatron_path = Path(tmpdir) / "megatron"
            hf_model.mkdir()
            args = convert.parse_args(
                [
                    "import",
                    "--hf-model",
                    str(hf_model),
                    "--megatron-path",
                    str(megatron_path),
                    "--torch-dtype",
                    "bfloat16",
                ]
            )
            convert.import_hf_to_megatron(args, autobridge=FakeBridge)
            self.assertEqual(FakeBridge.from_hf_calls[0][0], (str(hf_model),))
            self.assertTrue(FakeBridge.from_hf_calls[0][1]["local_files_only"])
            self.assertIs(FakeBridge.from_hf_calls[0][1]["torch_dtype"], torch.bfloat16)
            provider = FakeProvider.instances[0]
            self.assertFalse(provider.gradient_accumulation_fusion)
            self.assertTrue(provider.finalized)
            self.assertEqual(
                provider.provide_calls[0],
                {"wrap_with_ddp": False, "use_cpu_initialization": True},
            )
            save_args, save_kwargs = FakeBridge.save_calls[0]
            self.assertEqual(save_args, (["fake_megatron_model"], str(megatron_path)))
            self.assertEqual(save_kwargs["hf_tokenizer_path"], str(hf_model))
            self.assertTrue(save_kwargs["low_memory_save"])

    def test_export_calls_autobridge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_model = Path(tmpdir) / "hf"
            megatron_path = Path(tmpdir) / "megatron"
            hf_output = Path(tmpdir) / "hf-out"
            hf_model.mkdir()
            megatron_path.mkdir()
            args = convert.parse_args(
                [
                    "export",
                    "--hf-model",
                    str(hf_model),
                    "--megatron-path",
                    str(megatron_path),
                    "--hf-output",
                    str(hf_output),
                    "--strict",
                    "--no-progress",
                ]
            )
            convert.export_megatron_to_hf(args, autobridge=FakeBridge)
            self.assertEqual(FakeBridge.from_hf_calls[0][0], (str(hf_model),))
            self.assertTrue(FakeBridge.from_hf_calls[0][1]["local_files_only"])
            self.assertEqual(FakeBridge.export_calls[0]["megatron_path"], str(megatron_path))
            self.assertEqual(FakeBridge.export_calls[0]["hf_path"], str(hf_output))
            self.assertFalse(FakeBridge.export_calls[0]["show_progress"])
            self.assertTrue(FakeBridge.export_calls[0]["strict"])

    def test_export_requires_hf_output(self):
        args = argparse.Namespace(
            hf_model=Path("."),
            megatron_path=Path("."),
            hf_output=None,
            direction="export",
            torch_dtype=None,
            trust_remote_code=False,
            strict=False,
            dry_run=False,
            no_progress=True,
        )
        with self.assertRaises(ValueError):
            convert.export_megatron_to_hf(args, autobridge=FakeBridge)


if __name__ == "__main__":
    unittest.main()
