"""Pull E2's trained tensors out of a Megatron save, in a form inference loads.

Writes two things into --output:
  uniss/     UniSS in Hugging Face form -- config and tokenizer from the
             model it started from, weights from the save -- so vLLM serves
             it as it served the untrained one;
  added.pt   the three added inputs: hid_proj, hid_gate, pros_proj, dur_proj.

The gate starts at zero, so an all-zero gate means the hidden-state input
never trained; the export reports it rather than refusing, since a run can
legitimately learn to leave it shut.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

ADDED = ("hid_proj.", "hid_gate", "pros_proj.", "dur_proj.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True, help="an iter_NNNNNNN directory")
    ap.add_argument("--base", default="pretrained_models/UniSS_full", help="the UniSS E2 started from")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source, out = Path(args.checkpoint_dir), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    reader = dcp.FileSystemReader(str(source))
    meta = reader.read_metadata()
    wanted = {k: torch.empty(v.size, dtype=v.properties.dtype if hasattr(v, "properties") else torch.bfloat16)
              for k, v in meta.state_dict_metadata.items()
              if k.startswith("talker.") or k.startswith(ADDED)}
    if not any(k.startswith("talker.") for k in wanted):
        raise SystemExit(f"no talker tensors in {source}; keys look like {list(meta.state_dict_metadata)[:5]}")
    dcp.load(state_dict=wanted, storage_reader=reader)

    from safetensors.torch import load_file, save_file
    base = Path(args.base)
    state = load_file(str(base / "model.safetensors"))
    replaced = 0
    for key, tensor in wanted.items():
        if key.startswith("talker."):
            name = key[len("talker."):]
            if name in state:
                if tuple(state[name].shape) != tuple(tensor.shape):
                    raise SystemExit(f"{name}: shape {tuple(tensor.shape)} != base {tuple(state[name].shape)}")
                state[name] = tensor.to(state[name].dtype).contiguous()
                replaced += 1
    if replaced < len(state) * 0.9:
        raise SystemExit(f"only {replaced} of {len(state)} UniSS tensors found in the save")

    (out / "uniss").mkdir(parents=True)
    for f in base.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / "uniss" / f.name)
    save_file(state, str(out / "uniss" / "model.safetensors"), metadata={"format": "pt"})
    added = {k: v for k, v in wanted.items() if k.startswith(ADDED)}
    torch.save({"added": added, "step": int(source.name.split("_")[-1]), "source": str(source)}, out / "added.pt")
    gate = added.get("hid_gate")
    print(json.dumps({"output": str(out), "uniss_tensors_replaced": replaced, "uniss_tensors": len(state),
                      "added": {k: list(v.shape) for k, v in added.items()},
                      "gate_abs_mean": None if gate is None else float(gate.float().abs().mean()),
                      # a run with --e2-no-added has none of them: added.pt is then empty
                      "pros_abs_mean": float(added["pros_proj.weight"].float().abs().mean()) if added else None,
                      "dur_abs_mean": float(added["dur_proj.weight"].float().abs().mean()) if added else None},
                     indent=2))


if __name__ == "__main__":
    main()
