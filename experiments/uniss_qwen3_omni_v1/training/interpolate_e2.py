"""An E2 export part-way between untrained UniSS and a trained step (WiSE-FT).

UniSS's weights become base + alpha * (trained - base). The added inputs all
start at zero bar hid_proj, whose effect is gated by hid_gate, so they are
scaled by alpha (hid_proj is kept as trained). The result has export_e2's
layout -- <out>/uniss/ and <out>/added.pt -- so e2_infer reads it as is
(run_e2_probe.sh: EXPORT_DIR=<out>).

    python experiments/uniss_qwen3_omni_v1/training/interpolate_e2.py \
        --tuned <run>/full_iter4000_r2/export --alpha 0.5 --output <run>/wise_iter4000_a050/export
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(REPO / "pretrained_models/UniSS_full"))
    ap.add_argument("--tuned", required=True, help="an export_e2 output directory")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    tuned = Path(args.tuned)
    base = load_file(str(Path(args.base) / "model.safetensors"))
    trained = load_file(str(tuned / "uniss" / "model.safetensors"))
    if set(base) != set(trained):
        raise SystemExit(f"keys differ: {sorted(set(base) ^ set(trained))[:5]}")
    mixed = {k: (base[k].float() + args.alpha * (trained[k].float() - base[k].float())).to(trained[k].dtype)
             for k in trained}

    (out / "uniss").mkdir(parents=True)
    save_file(mixed, str(out / "uniss" / "model.safetensors"), metadata={"format": "pt"})
    for f in (tuned / "uniss").iterdir():
        if f.name != "model.safetensors":
            shutil.copy2(f, out / "uniss" / f.name)
    added = torch.load(tuned / "added.pt", map_location="cpu")
    added["added"] = {k: v if k == "hid_proj.weight" else (args.alpha * v.float()).to(v.dtype)
                      for k, v in added["added"].items()}
    added["source"] = f"interpolate_e2: base {args.base} + {args.alpha} x ({tuned} - base)"
    torch.save(added, out / "added.pt")
    print(f"wrote {out}  alpha {args.alpha}  {len(mixed)} tensors", flush=True)


if __name__ == "__main__":
    main()
