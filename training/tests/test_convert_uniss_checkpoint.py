import argparse
import json
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

    def test_torch_checkpoint_no_dist_compat_drops_legacy_kwarg(self):
        import torch.distributed.checkpoint as checkpoint

        original_load = checkpoint.load
        calls = []

        def fake_load(state_dict, *, storage_reader=None):
            calls.append((state_dict, storage_reader))
            return "loaded"

        try:
            checkpoint.load = fake_load
            convert.patch_torch_distributed_checkpoint_no_dist()
            result = checkpoint.load({"common": object()}, storage_reader="reader", no_dist=True)
        finally:
            checkpoint.load = original_load

        self.assertEqual(result, "loaded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "reader")

    def test_resolve_latest_iter_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "iter_0000002").mkdir()
            (root / "iter_0000010").mkdir()
            self.assertEqual(convert.resolve_latest_iter_dir(root), root / "iter_0000010")
            self.assertEqual(convert.resolve_latest_iter_dir(root / "iter_0000002"), root / "iter_0000002")

    def test_sync_hf_config_vocab_size_reads_safetensors_embedding(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            hf_dir = Path(tmpdir)
            (hf_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen2", "vocab_size": 3}) + "\n",
                encoding="utf-8",
            )
            save_file({"model.embed_tokens.weight": torch.zeros(5, 2)}, hf_dir / "model.safetensors")

            vocab_size = convert.sync_hf_config_vocab_size(hf_dir)
            config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(vocab_size, 5)
        self.assertEqual(config["vocab_size"], 5)

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

    def test_export_model_type_uses_megatron_lm_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_model = Path(tmpdir) / "hf"
            megatron_path = Path(tmpdir) / "megatron"
            iter_dir = megatron_path / "iter_0000007"
            hf_output = Path(tmpdir) / "hf-out"
            hf_model.mkdir()
            iter_dir.mkdir(parents=True)
            args = convert.parse_args(
                [
                    "export",
                    "--hf-model",
                    str(hf_model),
                    "--megatron-path",
                    str(megatron_path),
                    "--hf-output",
                    str(hf_output),
                    "--model-type",
                    "gpt",
                    "--no-progress",
                ]
            )
            calls = []

            class FakeContext:
                def __enter__(self):
                    calls.append(("enter",))

                def __exit__(self, exc_type, exc, tb):
                    calls.append(("exit", exc_type))

            def fake_load_megatron_model(path, **kwargs):
                calls.append(("load", path, kwargs))
                return "megatron-model"

            class FakeBridgeWithSave(FakeBridge):
                def save_hf_pretrained(self, model, path, **kwargs):
                    calls.append(("save", model, path, kwargs))

            original_load = convert.load_autobridge
            original_prepare = convert.ensure_default_bridge_runtime
            real_import = __import__

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "megatron.bridge.training.model_load_save":
                    return argparse.Namespace(
                        load_megatron_model=fake_load_megatron_model,
                        temporary_distributed_context=lambda backend: FakeContext(),
                    )
                return real_import(name, globals, locals, fromlist, level)

            try:
                convert.load_autobridge = lambda: FakeBridgeWithSave
                convert.ensure_default_bridge_runtime = lambda: calls.append(("prepare",))
                import builtins

                original_import = builtins.__import__
                builtins.__import__ = fake_import
                convert.export_megatron_to_hf(args)
            finally:
                convert.load_autobridge = original_load
                convert.ensure_default_bridge_runtime = original_prepare
                builtins.__import__ = original_import

            self.assertIn(("prepare",), calls)
            self.assertIn(
                (
                    "load",
                    str(iter_dir),
                    {"model_type": "gpt", "use_cpu_init": True, "skip_temp_dist_context": True},
                ),
                calls,
            )
            self.assertTrue(any(call[0] == "save" and call[2] == str(hf_output) for call in calls))

    def test_export_default_bridge_prepares_runtime(self):
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
                    "--no-progress",
                ]
            )
            prepared = []
            original_load = convert.load_autobridge
            original_prepare = convert.ensure_default_bridge_runtime
            try:
                convert.load_autobridge = lambda: FakeBridge
                convert.ensure_default_bridge_runtime = lambda: prepared.append(True)
                convert.export_megatron_to_hf(args)
            finally:
                convert.load_autobridge = original_load
                convert.ensure_default_bridge_runtime = original_prepare
            self.assertEqual(prepared, [True])
            self.assertEqual(FakeBridge.export_calls[0]["megatron_path"], str(megatron_path))

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
