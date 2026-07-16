"""Convert UniSS checkpoints between Hugging Face and Megatron formats.

The training entrypoint uses Megatron-LM native checkpoints, while inference
and audio evaluation use Hugging Face checkpoints. This wrapper keeps the
conversion commands reproducible and local-path only for the UniSS runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_MODEL = REPO_ROOT / "checkpoints" / "qwen2_1p5b_uniss_vocab_hf"
DEFAULT_MEGATRON_PATH = REPO_ROOT / "checkpoints" / "qwen2_1p5b_uniss_vocab"
DEFAULT_HF_EXPORT = REPO_ROOT / "checkpoints" / "exported_hf" / "uniss_export"
DEFAULT_MEGATRON_BRIDGE_ROOT = REPO_ROOT / "third_party" / "Megatron-Bridge" / "src"
DEFAULT_MEGATRON_LM_ROOT = REPO_ROOT / "third_party" / "Megatron-LM"


@dataclass(frozen=True)
class ConversionSummary:
    direction: str
    hf_model: str
    megatron_path: str
    hf_output: str | None
    torch_dtype: str | None
    trust_remote_code: bool
    gradient_accumulation_fusion: bool | None
    strict: bool | None
    dry_run: bool


def ensure_bridge_import_path(
    *,
    bridge_root: Path = DEFAULT_MEGATRON_BRIDGE_ROOT,
    megatron_lm_root: Path = DEFAULT_MEGATRON_LM_ROOT,
) -> None:
    for path in (bridge_root, megatron_lm_root, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def torch_dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_autobridge():
    ensure_bridge_import_path()
    from megatron.bridge import AutoBridge

    return AutoBridge


def ensure_modelopt_checkpoint_plugin_exports() -> None:
    """Expose ModelOpt mcore checkpoint helpers where Megatron Bridge imports them.

    Megatron Bridge imports these helpers from ``modelopt.torch.opt.plugins``.
    Some ModelOpt releases keep the functions in the
    ``plugins.mcore_dist_checkpointing`` submodule without re-exporting them.
    Patching the module object keeps the compatibility fix local to this process
    and avoids editing site-packages or vendored Bridge code.
    """

    ensure_bridge_import_path()
    try:
        import modelopt.torch.opt.plugins as plugins
    except ImportError as exc:
        raise ImportError(
            "Megatron Bridge checkpoint saving requires nvidia-modelopt. "
            "Install it without changing the training torch stack, for example: "
            "pip install nvidia-modelopt==0.44.0 --no-deps"
        ) from exc

    required = ("restore_modelopt_state", "save_modelopt_state", "save_sharded_modelopt_state")
    if all(hasattr(plugins, name) for name in required):
        return

    from modelopt.torch.opt.plugins import mcore_dist_checkpointing

    for name in required:
        setattr(plugins, name, getattr(mcore_dist_checkpointing, name))


def ensure_transformers_bridge_symbols() -> None:
    """Provide optional Transformer symbols imported by broad Bridge modules."""

    import transformers

    if not hasattr(transformers, "Qwen3VLProcessor"):
        transformers.Qwen3VLProcessor = transformers.AutoProcessor
    if not hasattr(transformers, "PreTrainedConfig") and hasattr(transformers, "PretrainedConfig"):
        transformers.PreTrainedConfig = transformers.PretrainedConfig


def ensure_bridge_training_data_import_stubs() -> None:
    """Avoid importing optional Bridge data backends during checkpoint save.

    Bridge checkpoint saving imports ``megatron.bridge.training.config``, whose
    module-level imports pull in every data builder, including optional Energon
    and VLM processors. Conversion does not build datasets, so lightweight class
    stubs are sufficient for those config type references.
    """

    ensure_bridge_import_path()

    data_root = DEFAULT_MEGATRON_BRIDGE_ROOT / "megatron" / "bridge" / "data"
    builders_root = data_root / "builders"
    sources_root = data_root / "sources"

    def package(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module

    class DatasetConfigStub:
        def validate(self) -> None:
            return None

        def finalize(self) -> None:
            return None

    package("megatron.bridge.data", data_root)
    package("megatron.bridge.data.builders", builders_root)
    package("megatron.bridge.data.sources", sources_root)

    stubs = {
        "megatron.bridge.data.builders.direct_hf_sft": {
            "DirectHFSFTDatasetConfig": DatasetConfigStub,
        },
        "megatron.bridge.data.builders.energon": {
            "EnergonDatasetConfig": DatasetConfigStub,
        },
        "megatron.bridge.data.builders.gpt_sft": {
            "FinetuningDatasetConfig": DatasetConfigStub,
            "GPTSFTDatasetConfig": DatasetConfigStub,
        },
        "megatron.bridge.data.builders.mock_vlm_sft": {
            "MockVLMSFTDatasetConfig": DatasetConfigStub,
        },
        "megatron.bridge.data.sources.hf": {
            "HFDatasetSourceConfig": DatasetConfigStub,
        },
    }

    for module_name, attrs in stubs.items():
        module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module


def ensure_bridge_model_symbols() -> None:
    """Provide model symbols expected by broad Bridge training imports."""

    ensure_bridge_import_path()
    import megatron.bridge.models as bridge_models

    if not hasattr(bridge_models, "T5ModelProvider"):
        bridge_models.T5ModelProvider = bridge_models.GPTModelProvider


def build_summary(args: argparse.Namespace) -> ConversionSummary:
    hf_output = getattr(args, "hf_output", None)
    return ConversionSummary(
        direction=args.direction,
        hf_model=str(args.hf_model),
        megatron_path=str(args.megatron_path),
        hf_output=str(hf_output) if hf_output is not None else None,
        torch_dtype=args.torch_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        gradient_accumulation_fusion=(
            bool(args.gradient_accumulation_fusion) if args.direction == "import" else None
        ),
        strict=bool(args.strict) if args.direction == "export" else None,
        dry_run=bool(args.dry_run),
    )


def import_hf_to_megatron(args: argparse.Namespace, *, autobridge: Any | None = None) -> ConversionSummary:
    require_path(args.hf_model, "HF model")
    args.megatron_path.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args)
    if args.dry_run:
        return summary

    uses_default_bridge = autobridge is None
    bridge_cls = load_autobridge() if uses_default_bridge else autobridge
    kwargs: dict[str, object] = {
        "local_files_only": True,
    }
    dtype = torch_dtype_from_name(args.torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    bridge = bridge_cls.from_hf_pretrained(str(args.hf_model), **kwargs)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.gradient_accumulation_fusion = bool(args.gradient_accumulation_fusion)
    if hasattr(provider, "finalize"):
        provider.finalize()
    megatron_model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
    )

    hf_tokenizer_kwargs = {}
    model_bridge = getattr(bridge, "_model_bridge", None)
    if model_bridge is not None and hasattr(model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = model_bridge.get_hf_tokenizer_kwargs()
    if args.trust_remote_code:
        hf_tokenizer_kwargs.setdefault("trust_remote_code", True)

    if uses_default_bridge:
        ensure_modelopt_checkpoint_plugin_exports()
        ensure_transformers_bridge_symbols()
        ensure_bridge_training_data_import_stubs()
        ensure_bridge_model_symbols()
    bridge.save_megatron_model(
        megatron_model,
        str(args.megatron_path),
        hf_tokenizer_path=str(args.hf_model),
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        low_memory_save=True,
    )
    return summary


def export_megatron_to_hf(args: argparse.Namespace, *, autobridge: Any | None = None) -> ConversionSummary:
    require_path(args.hf_model, "HF reference model")
    require_path(args.megatron_path, "Megatron checkpoint")
    if args.hf_output is None:
        raise ValueError("--hf-output is required for export")
    args.hf_output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args)
    if args.dry_run:
        return summary

    bridge_cls = autobridge or load_autobridge()
    bridge = bridge_cls.from_hf_pretrained(
        str(args.hf_model),
        local_files_only=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    bridge.export_ckpt(
        megatron_path=str(args.megatron_path),
        hf_path=str(args.hf_output),
        show_progress=not args.no_progress,
        strict=bool(args.strict),
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="direction", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hf-model", type=Path, default=DEFAULT_HF_MODEL)
    common.add_argument("--megatron-path", type=Path, default=DEFAULT_MEGATRON_PATH)
    common.add_argument("--trust-remote-code", action="store_true")
    common.add_argument("--dry-run", action="store_true")

    import_parser = subparsers.add_parser("import", parents=[common], help="HF checkpoint -> Megatron checkpoint")
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    import_parser.add_argument(
        "--gradient-accumulation-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Megatron gradient accumulation fusion during conversion. "
            "Defaults to disabled because the non-TE path requires the Apex "
            "fused_weight_gradient_mlp_cuda extension."
        ),
    )

    export_parser = subparsers.add_parser("export", parents=[common], help="Megatron checkpoint -> HF checkpoint")
    export_parser.add_argument("--hf-output", type=Path, default=DEFAULT_HF_EXPORT)
    export_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default=None)
    export_parser.add_argument("--strict", action="store_true")
    export_parser.add_argument("--no-progress", action="store_true")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.direction == "import":
        summary = import_hf_to_megatron(args)
    elif args.direction == "export":
        summary = export_megatron_to_hf(args)
    else:
        raise RuntimeError(f"Unsupported direction: {args.direction}")

    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
