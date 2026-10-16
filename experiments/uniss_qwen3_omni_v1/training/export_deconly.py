"""Pull the decoder-only route's trained tensors out of a Megatron save.

What trains here is not what trains in the two-stream route: there is no
talker and no markers, and the code path is a pair of new modules on the
thinker. So this cannot reuse export_phase_d, whose prefixes would match
nothing and leave an empty payload.

The adapter is the part worth guarding. Its B matrices start at zero, so
an all-zero B means the checkpoint never trained -- a believable file
that scores a frozen model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

PREFIXES = ("audio_embed.", "global_embed.", "audio_head.")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _is_adapter(key: str) -> bool:
    parts = key.split(".")
    return (parts[0] == "thinker" and len(parts) >= 3
            and parts[-1] in ("a", "b") and parts[-2] in LORA_TARGETS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output", required=True)
    # The all-zero-B guard below is the only thing standing between a
    # believable file and a frozen model scored as if it trained. With no
    # adapter at all that guard has nothing to check, so the absence has
    # to be declared rather than inferred.
    ap.add_argument("--no-lora", action="store_true",
                    help="the run was trained with UNISS_DECONLY_LORA_FROM=none")
    args = ap.parse_args()

    source = Path(args.checkpoint_dir)
    if not (source / "metadata.json").is_file() and not any(source.glob("*.distcp")):
        raise SystemExit(f"{source} does not look like a torch_dist checkpoint")

    reader = dcp.FileSystemReader(str(source))
    metadata = reader.read_metadata()
    wanted = {
        key: torch.empty(
            value.size,
            dtype=value.properties.dtype if hasattr(value, "properties") else torch.bfloat16,
        )
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith(PREFIXES) or _is_adapter(key)
    }
    if not wanted:
        sample = list(metadata.state_dict_metadata)[:5]
        raise SystemExit(f"no decoder-only tensors in {source}; keys look like {sample}")
    dcp.load(state_dict=wanted, storage_reader=reader)

    modules: dict[str, dict[str, torch.Tensor]] = {
        "audio_embed": {}, "global_embed": {}, "audio_head": {}}
    adapter: dict[str, torch.Tensor] = {}
    for key, tensor in wanted.items():
        if _is_adapter(key):
            # lora.load_state_dict matches on lora_name, relative to the
            # thinker.
            adapter[key[len("thinker."):]] = tensor
            continue
        head, _, rest = key.partition(".")
        modules[head][rest] = tensor

    for name, state in modules.items():
        if not state:
            raise SystemExit(f"{name} is missing from {source}")
    if not adapter and not args.no_lora:
        raise SystemExit(
            "no LoRA tensors found; the thinker would load unadapted."
            " Pass --no-lora if the run was trained with"
            " UNISS_DECONLY_LORA_FROM=none.")
    if adapter and args.no_lora:
        raise SystemExit(
            f"--no-lora was passed but {len(adapter)} adapter tensors are"
            " present; this checkpoint did train the thinker")
    if len(adapter) % 2:
        raise SystemExit(f"{len(adapter)} adapter tensors is odd; each layer needs a and b")
    if adapter and not any(k.endswith(".b") and float(t.float().abs().max()) > 0
                           for k, t in adapter.items()):
        raise SystemExit("every LoRA B is still zero; this checkpoint did not train")
    # With the thinker frozen the embeddings are the only evidence the run
    # happened, so they carry the check instead.
    if not adapter and float(modules["audio_head"]["weight"].float().abs().max()) == 0:
        raise SystemExit("audio_head is all zero; this checkpoint did not train")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_embed": modules["audio_embed"],
        "global_embed": modules["global_embed"],
        "audio_head": modules["audio_head"],
        "thinker_lora": adapter,
        "thinker_frozen": not adapter,
        "step": int(source.name.split("_")[-1]),
        "source": str(source),
    }
    torch.save(payload, out)
    print(json.dumps({
        "output": str(out),
        "step": payload["step"],
        "lora_tensors": len(adapter),
        "lora_layers": len(adapter) // 2,
        "thinker_frozen": not adapter,
        "audio_embed": list(modules["audio_embed"]["weight"].shape),
        "audio_head": list(modules["audio_head"]["weight"].shape),
    }, indent=2))


if __name__ == "__main__":
    main()
