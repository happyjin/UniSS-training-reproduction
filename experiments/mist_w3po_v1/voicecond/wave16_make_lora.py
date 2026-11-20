"""Robust wave 16 NCWI (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md, section 3): one trained LoRA
(wave5_m_train.py / wave14_train.py adapters_final.pt: keys layers.<i>.self_attn.<q|k|v|o>_proj.{a,b}, a = [r, in],
b = [out, r], fp32) -> PEFT adapters vLLM serves, one directory per WiSE coefficient alpha > 0:
<out>/a<alpha:.4f>/{adapter_config.json, adapter_model.safetensors}, keys base_model.model.thinker.model.<stem>.lora_A/B.weight
(A = a, B = b unchanged), r = rank, lora_alpha = alpha x the trained lora_alpha -- vLLM's scale lora_alpha / r is then
alpha x the training scale (alpha 0.5 = WiSE 0.5 = the merged *_wise05 copy). alpha = 0 needs no adapter (the base).
--check-merged DIR --check-alpha A: on 8 projections (layers 0 / 15 / 31 / 47, q and o), bf16(base + (lora_alpha / r) B A)
-- wave5_m_merge.py's arithmetic -- must reproduce DIR's merged weight in >= 99.9 % of the elements; exits otherwise. Also
printed: |merged - base - delta| / |delta|, the bf16 rounding the merged copy carries (the unmerged LoRA does not).
Refuses to overwrite.
    <qwen3omni python> experiments/mist_w3po_v1/voicecond/wave16_make_lora.py --adapters <run>/adapters_final.pt \
        --alphas 0.4,0.5,1 --out <dir> [--check-merged <merged dir> --check-alpha 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"


def tensor(model_dir: Path, key: str) -> torch.Tensor:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(model_dir / idx[key]), framework="pt") as f:
        return f.get_tensor(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--alphas", required=True, help="comma-separated WiSE coefficients; 0 is skipped (no adapter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-merged", default="")
    ap.add_argument("--check-alpha", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    meta = json.load(open(Path(a.adapters).with_suffix(".json")))
    r, la = int(meta["rank"]), float(meta["alpha"])
    st = torch.load(a.adapters, map_location="cpu")
    stems = sorted(k[:-2] for k in st if k.endswith(".a"))
    assert stems and all(s + ".b" in st for s in stems) and len(st) == 2 * len(stems), "adapter keys"
    mods = sorted({s.rsplit(".", 1)[1] for s in stems})
    assert set(mods) <= {"q_proj", "k_proj", "v_proj", "o_proj"}, mods
    tensors = {}
    for s in stems:
        A, B = st[s + ".a"].float().contiguous(), st[s + ".b"].float().contiguous()
        assert A.shape[0] == r and B.shape[1] == r, (s, A.shape, B.shape)
        tensors[f"base_model.model.thinker.model.{s}.lora_A.weight"] = A
        tensors[f"base_model.model.thinker.model.{s}.lora_B.weight"] = B
    alphas = sorted({round(float(x), 4) for x in a.alphas.split(",") if x.strip()})
    assert all(0.0 <= x <= 1.0 for x in alphas), alphas
    if a.check_merged:
        md = Path(a.check_merged)
        worst = 1.0
        for s in [f"layers.{i}.self_attn.{p}" for i in (0, 15, 31, 47) for p in ("q_proj", "o_proj")]:
            key = f"thinker.model.{s}.weight"
            m_, w0 = tensor(md, key), tensor(BASE, key)
            got = (a.check_alpha * la / r) * (st[s + ".b"].float() @ st[s + ".a"].float())
            # wave5_m_merge.py's own arithmetic: bf16(W0 + delta); the adapter must reproduce the merged copy
            same = float(((w0.float() + got).to(m_.dtype) == m_).float().mean())
            worst = min(worst, same)
            want = m_.float() - w0.float()
            rel = float((got - want).norm() / want.norm())   # information: bf16 rounding of the merged copy vs exact delta
            print(f"check {s}: re-merge identical {same:.6f}; |merged - base - delta| / |delta| {rel:.4f}", flush=True)
        if not worst >= 0.999:
            raise SystemExit(f"adapter != merged copy {md}: only {worst:.6f} of the re-merged elements identical")
        print(f"adapter check OK against {md} at alpha {a.check_alpha}: re-merge identical >= {worst:.6f}", flush=True)
    out.mkdir(parents=True)
    made = []
    for x in alphas:
        if x == 0.0:
            continue
        dd = out / f"a{x:.4f}"
        dd.mkdir()
        save_file(tensors, str(dd / "adapter_model.safetensors"), metadata={"format": "pt"})
        cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": x * la, "lora_dropout": 0.0,
               "bias": "none", "target_modules": mods, "use_rslora": False, "use_dora": False, "modules_to_save": None,
               "base_model_name_or_path": str(BASE),
               "wave16": {"wise_alpha": x, "trained_lora_alpha": la, "scale": x * la / r, "adapters": str(a.adapters)}}
        (dd / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
        made.append(dd.name)
    print(json.dumps({"out": str(out), "adapters": made, "n_modules": len(stems), "rank": r, "trained_alpha": la}), flush=True)


if __name__ == "__main__":
    main()
