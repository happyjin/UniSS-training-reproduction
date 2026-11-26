"""sst_hsmm_v1 prereg change 41: ES members of the drafter LoRA as PEFT dirs vLLM serves (hsc13 --drafters).

Space: the q/k/v/o LoRA of the 48 text layers (shapes of the SAP verifier adapter, r 16, lora_alpha 32 -> scale 2). A is ONE
fixed random matrix per module (kaiming_uniform a = sqrt(5), torch.Generator seed --seed), shared by every member; ES moves B
only (centre B = 0 = wave17). eps_k ~ N(0, I) over all B entries (generator seed --seed + 1000 + k, modules in sorted key order).
  members  --pairs P --sigma s: <out>/p<k>_pos (B = +s eps_k) and <out>/p<k>_neg (B = -s eps_k), k = 0..P-1
  step     --grad <json {"weights": {k: w_k}}> --sigma s: g = sum_k w_k eps_k; theta1 = s sqrt(D) g / |g| -> <out>/step_pos and the
           control <out>/step_neg (= -theta1)
Every dir: adapter_config.json + adapter_model.safetensors (keys base_model.model.thinker.model.<stem>.lora_A/B.weight) + es.json.
Refuses to overwrite.
    <python> -m experiments.sst_hsmm_v1.es_make members --out <dir> --pairs 8 --sigma 0.01
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "eval_outputs/sst_hsmm_v1/v/lora/a1.0000"


def shapes():
    f = safe_open(str(TEMPLATE / "adapter_model.safetensors"), "pt")
    out = {}
    for k in sorted(f.keys()):
        if k.endswith("lora_A.weight"):
            stem = k[: -len(".lora_A.weight")]
            a = tuple(f.get_slice(k).get_shape())
            b = tuple(f.get_slice(stem + ".lora_B.weight").get_shape())
            out[stem] = (a, b)
    return out


def fixed_A(sh, seed):
    g = torch.Generator().manual_seed(seed)
    A = {}
    for stem, (a, _) in sh.items():
        t = torch.empty(a)
        bound = 1.0 / math.sqrt(a[1])            # kaiming_uniform_(a=sqrt(5)): gain sqrt(2 / 6), bound = sqrt(3) * gain / sqrt(fan_in)
        t.uniform_(-bound, bound, generator=g)
        A[stem] = t
    return A


def eps(sh, seed):
    g = torch.Generator().manual_seed(seed)
    return {stem: torch.randn(b, generator=g) for stem, (_, b) in sh.items()}


def write(dirp: Path, A, B, meta):
    if dirp.exists():
        raise SystemExit(f"refuse to overwrite {dirp}")
    dirp.mkdir(parents=True)
    t = {}
    for stem in A:
        t[f"{stem}.lora_A.weight"] = A[stem].contiguous()
        t[f"{stem}.lora_B.weight"] = B[stem].contiguous()
    save_file(t, str(dirp / "adapter_model.safetensors"))
    cfg = json.load(open(TEMPLATE / "adapter_config.json"))
    cfg.pop("wave16", None)
    cfg["es"] = meta
    assert cfg["r"] == 16 and float(cfg["lora_alpha"]) == 32.0, cfg
    (dirp / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (dirp / "es.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("members", "step"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--grad", default="")
    a = ap.parse_args()
    out = Path(a.out)
    sh = shapes()
    A = fixed_A(sh, a.seed)
    D = sum(b[0] * b[1] for _, b in sh.values())
    if a.mode == "members":
        for k in range(a.pairs):
            e = eps(sh, a.seed + 1000 + k)
            for sgn, nm in ((1.0, "pos"), (-1.0, "neg")):
                B = {s: sgn * a.sigma * v for s, v in e.items()}
                write(out / f"p{k}_{nm}", A, B, {"pair": k, "sign": sgn, "sigma": a.sigma, "seed": a.seed, "D": D})
        dw = []
        for stem in list(sh)[:4]:                 # report the perturbation scale on a few modules
            e = eps(sh, a.seed + 1000)[stem]
            dW = 2.0 * a.sigma * e @ A[stem]
            dw.append((stem.split("thinker.model.")[1], float(dW.std())))
        print(json.dumps({"members": 2 * a.pairs, "D": D, "dW_std_examples": dw}))
    else:
        w = json.load(open(a.grad))["weights"]
        g = {s: torch.zeros(b) for s, (_, b) in sh.items()}
        for k, wk in w.items():
            e = eps(sh, a.seed + 1000 + int(k))
            for s in g:
                g[s] += float(wk) * e[s]
        nrm = math.sqrt(sum(float((v * v).sum()) for v in g.values()))
        c = a.sigma * math.sqrt(D) / nrm
        for sgn, nm in ((1.0, "step_pos"), (-1.0, "step_neg")):
            write(out / nm, A, {s: sgn * c * v for s, v in g.items()},
                  {"step": sgn, "sigma": a.sigma, "seed": a.seed, "D": D, "grad": a.grad, "weights": w})
        print(json.dumps({"D": D, "grad_norm": nrm, "scale": c}))


if __name__ == "__main__":
    main()
