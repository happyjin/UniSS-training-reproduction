"""Robust K1 (reports/mist_w3po_v1/robust_k1_unmerged_prereg.zh-CN.md): the task-vector difference tau_noisy - tau_clean
as ONE exact LoRA (no weight merge), for vLLM to serve on the base Qwen3-Omni Thinker.

Inputs: two wave14_train.py / r1_train.py adapters (adapters_final.pt: keys layers.<i>.self_attn.<q|k|v|o>_proj.{a,b},
a = [r, in], b = [out, r], fp32; adapters_final.json: rank, alpha). With s_x = alpha_x / rank_x:

    A = [a_n ; a_c]             (r_n + r_c rows)
    B = [s_n b_n , -s_c b_c]    (r_n + r_c columns)      ->  B A = s_n b_n a_n - s_c b_c a_c = dW_n - dW_c  exactly

Written as a PEFT adapter (wave16_make_lora.py layout: keys base_model.model.thinker.model.<stem>.lora_A/B.weight, fp32)
with r = lora_alpha = r_n + r_c, so vLLM's scale lora_alpha / r = 1 (both scalings are already folded into B).
Checks (fp32, every adapted module, from the saved file and its config): max |scale * B A - (dW_n - dW_c)| and the
relative Frobenius error; also, for information, the relative error when vLLM casts the factors to bf16
(||bf16(B) bf16(A) - D|| / ||D||) next to R1's merged-copy bf16 error (0.17 mean). Exits non-zero if the fp32 relative
error exceeds 1e-5 in any module. Refuses to overwrite.

    <qwen3omni python> experiments/mist_w3po_v1/voicecond/k1_make_lora.py --noisy <run>/adapters_final.pt \
        --clean <run>/adapters_final.pt --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"


def load(path: str):
    meta = json.load(open(Path(path).with_suffix(".json")))
    r, la = int(meta["rank"]), float(meta["alpha"])
    st = torch.load(path, map_location="cpu")
    stems = sorted(k[:-2] for k in st if k.endswith(".a"))
    assert stems and all(s + ".b" in st for s in stems) and len(st) == 2 * len(stems), path
    for s in stems:
        assert st[s + ".a"].shape[0] == r and st[s + ".b"].shape[1] == r, (path, s)
    return st, stems, r, la / r, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    sn, stems, rn, scn, mn = load(a.noisy)
    sc, stems_c, rc, scc, mc = load(a.clean)
    assert stems == stems_c, "the two adapters cover different modules"
    mods = sorted({s.rsplit(".", 1)[1] for s in stems})
    assert set(mods) <= {"q_proj", "k_proj", "v_proj", "o_proj"}, mods
    R = rn + rc
    tensors = {}
    for s in stems:
        A = torch.cat([sn[s + ".a"].float(), sc[s + ".a"].float()], 0).contiguous()
        B = torch.cat([scn * sn[s + ".b"].float(), -scc * sc[s + ".b"].float()], 1).contiguous()
        tensors[f"base_model.model.thinker.model.{s}.lora_A.weight"] = A
        tensors[f"base_model.model.thinker.model.{s}.lora_B.weight"] = B
    out.mkdir(parents=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": R, "lora_alpha": float(R), "lora_dropout": 0.0,
           "bias": "none", "target_modules": mods, "use_rslora": False, "use_dora": False, "modules_to_save": None,
           "base_model_name_or_path": str(BASE),
           "k1": {"noisy": a.noisy, "clean": a.clean, "rank_noisy": rn, "rank_clean": rc, "scale_noisy": scn,
                  "scale_clean": scc, "meta_noisy": mn, "meta_clean": mc}}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))

    # ---- verification from the saved files
    cfg2 = json.load(open(out / "adapter_config.json"))
    scale = cfg2["lora_alpha"] / cfg2["r"]
    sv = load_file(str(out / "adapter_model.safetensors"))
    worst_abs, worst_rel, rel_bf16, rel_norm = 0.0, 0.0, [], []
    for s in stems:
        A = sv[f"base_model.model.thinker.model.{s}.lora_A.weight"].double()
        B = sv[f"base_model.model.thinker.model.{s}.lora_B.weight"].double()
        want = scn * (sn[s + ".b"].double() @ sn[s + ".a"].double()) - scc * (sc[s + ".b"].double() @ sc[s + ".a"].double())
        got32 = scale * (B.float() @ A.float()).double()   # fp32 product, as the check asks
        err = (got32 - want).abs().max().item()
        rel = ((got32 - want).norm() / want.norm()).item()
        worst_abs, worst_rel = max(worst_abs, err), max(worst_rel, rel)
        gb = scale * (B.to(torch.bfloat16).float() @ A.to(torch.bfloat16).float()).double()
        rel_bf16.append(((gb - want).norm() / want.norm()).item())
        rel_norm.append((want.norm() / (scn * (sn[s + ".b"].double() @ sn[s + ".a"].double())).norm()).item())
    res = {"out": str(out), "n_modules": len(stems), "rank": R, "lora_alpha": float(R), "scale_noisy": scn,
           "scale_clean": scc, "fp32_max_abs_diff": worst_abs, "fp32_max_rel_frob": worst_rel,
           "bf16_factor_rel_err_mean": sum(rel_bf16) / len(rel_bf16), "bf16_factor_rel_err_max": max(rel_bf16),
           "norm_diff_over_noisy_mean": sum(rel_norm) / len(rel_norm)}
    (out / "k1_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    if not worst_rel <= 1e-5:
        raise SystemExit(f"K1 LoRA != dW_n - dW_c: max relative error {worst_rel:.3e}")


if __name__ == "__main__":
    main()
