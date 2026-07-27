"""Export a Stage7A action head as an untied Hugging Face/vLLM model.

The Stage6 Qwen checkpoint ties its input embedding and output LM head.  A
Stage7A checkpoint contains an independently trained two-row WAIT/WRITE output
projection, so mutating the tied weight would also mutate how previous action
tokens are embedded.  This exporter first creates a fully independent output
head, copies all Stage6 vocabulary rows, and then replaces only WAIT/WRITE.
The resulting directory can be passed to the unchanged Stage4/6 free-running
streaming generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.simul_uniss.stage7a.policy import ACTION_IDS
from training.simul_uniss.stage7a.train import CHECKPOINT_SCHEMA


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def install_action_head(
    model: nn.Module, action_state: dict[str, torch.Tensor]
) -> dict[str, object]:
    old_head = model.get_output_embeddings()
    input_embedding = model.get_input_embeddings()
    if not isinstance(old_head, nn.Linear) or not isinstance(input_embedding, nn.Embedding):
        raise TypeError("expected linear LM head and embedding input layer")
    action_weight = action_state.get("projection.weight")
    expected = (len(ACTION_IDS), old_head.in_features)
    if action_weight is None or tuple(action_weight.shape) != expected:
        raise ValueError(
            f"action projection has shape {getattr(action_weight, 'shape', None)}, "
            f"expected {expected}"
        )

    input_rows_before = input_embedding.weight[list(ACTION_IDS)].detach().cpu().clone()
    output_rows_before = old_head.weight[list(ACTION_IDS)].detach().cpu().clone()
    new_head = nn.Linear(
        old_head.in_features,
        old_head.out_features,
        bias=False,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    with torch.no_grad():
        new_head.weight.copy_(old_head.weight)
        new_head.weight.index_copy_(
            0,
            torch.tensor(ACTION_IDS, device=new_head.weight.device),
            action_weight.to(device=new_head.weight.device, dtype=new_head.weight.dtype),
        )
    model.config.tie_word_embeddings = False
    model.set_output_embeddings(new_head)

    if model.get_output_embeddings().weight.data_ptr() == input_embedding.weight.data_ptr():
        raise RuntimeError("exported output head is still tied to input embeddings")
    if not torch.equal(
        input_embedding.weight[list(ACTION_IDS)].detach().cpu(), input_rows_before
    ):
        raise RuntimeError("installing action head changed input embedding rows")
    return {
        "action_ids": list(ACTION_IDS),
        "input_rows_unchanged": True,
        "old_action_rows_sha256": hashlib.sha256(
            output_rows_before.contiguous().float().numpy().tobytes()
        ).hexdigest(),
        "new_action_rows_sha256": hashlib.sha256(
            new_head.weight[list(ACTION_IDS)]
            .detach()
            .cpu()
            .contiguous()
            .float()
            .numpy()
            .tobytes()
        ).hexdigest(),
    }


def export_policy_model(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected checkpoint schema: {checkpoint.get('schema_version')}")
    base_model = Path(args.base_model or checkpoint["base_model"]).resolve()
    output_dir = Path(args.output_dir).resolve()
    temporary = output_dir.with_name(f".{output_dir.name}.partial.{os.getpid()}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary directory {temporary}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    installation = install_action_head(model, checkpoint["action_head"])
    model.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.save_pretrained(temporary)

    model_files = sorted(temporary.glob("model*.safetensors"))
    if not model_files:
        raise RuntimeError("export did not produce safetensors weights")
    manifest = {
        "schema_version": "simul_uniss_stage7a_policy_export_v1",
        "base_model": str(base_model),
        "base_model_sha256": sha256_file(base_model / "model.safetensors"),
        "action_checkpoint": str(checkpoint_path),
        "action_checkpoint_sha256": sha256_file(checkpoint_path),
        "action_checkpoint_step": checkpoint.get("step"),
        "action_checkpoint_mode": checkpoint.get("mode"),
        "tie_word_embeddings": False,
        "installation": installation,
        "weight_files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in model_files
        },
    }
    (temporary / "stage7a_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (temporary / "EXPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
    os.replace(temporary, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **manifest}, sort_keys=True))
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    export_policy_model(parse_args(argv))


if __name__ == "__main__":
    main()
