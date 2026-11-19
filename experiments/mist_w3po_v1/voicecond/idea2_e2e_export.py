"""Idea 2 e2e test (prereg addendum 2026-11-19): export the Talker of an e2e checkpoint as a plain T1c-format model dir.

The LoRA of <ckpt> (r 16, alpha 32, q_proj / v_proj) is merged in fp32 into a COPY of T1c and saved in bf16 with exactly
T1c's tensor names (model.safetensors) plus T1c's config / tokenizer files, so si_infer / vLLM load it with --model <out>.
T1c itself is never written. Checks (written to <out>/../export.json): the merged dir loads with transformers; on a few
cvt DEV teacher sequences the semantic-code log-probs of the merged model match the PEFT model of training (bf16 base +
LoRA) and differ from T1c.
    venvs/vllm_omni_030/bin/python -m experiments.mist_w3po_v1.voicecond.idea2_e2e_export --ckpt <ckpt> --out <dir>/uniss
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
T1C = REPO / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss"
CV = REPO / "data/open_voicecond_v1/idea2/cvt"
SEMANTIC_0 = 155761


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-check", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert ck["lora"] is not None, "checkpoint has no LoRA (frozen_talker ablation?)"
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    # merge in fp32 on a copy
    m32 = get_peft_model(AutoModelForCausalLM.from_pretrained(str(T1C), torch_dtype=torch.float32), cfg)
    res = set_peft_model_state_dict(m32, ck["lora"])
    assert not res.unexpected_keys, res.unexpected_keys
    merged = m32.merge_and_unload()
    sd = merged.state_dict()
    with safe_open(str(T1C / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        meta = f.metadata()
    miss = [k for k in keys if k not in sd]
    assert not miss, miss[:5]
    tmp = out.parent / f".{out.name}.partial"
    tmp.mkdir(parents=True)
    save_file({k: sd[k].to(torch.bfloat16).contiguous() for k in keys}, str(tmp / "model.safetensors"),
              metadata=meta or {"format": "pt"})
    for p in T1C.iterdir():
        if p.name != "model.safetensors":
            shutil.copy2(p, tmp / p.name)
    tmp.rename(out)
    del m32, merged, sd
    # checks on cvt DEV sequences
    dev = torch.device(a.device)
    recs = [json.loads(l) for l in open(sorted(glob.glob(str(CV / "teacher_out/zh2en_dev/part_*.jsonl")))[0])][:a.n_check]
    recs += [json.loads(l) for l in open(sorted(glob.glob(str(CV / "teacher_out/en2zh_dev/part_*.jsonl")))[0])][:a.n_check]

    def logp(model, r):
        b = r["blocks"][0]
        ids = torch.tensor([b["train_ids"]], device=dev)
        lf, nc = b["label_from"], len(b["code_ids"])
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0, lf - 1: lf - 1 + nc, SEMANTIC_0:SEMANTIC_0 + 8192].float()
        return torch.log_softmax(lg, -1)

    def kl(p, q):
        return float((p.exp() * (p - q)).sum(-1).mean())
    mm = AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.bfloat16).to(dev).eval()
    lm = [logp(mm, r) for r in recs]
    del mm
    pm = get_peft_model(AutoModelForCausalLM.from_pretrained(str(T1C), torch_dtype=torch.bfloat16), cfg)
    set_peft_model_state_dict(pm, ck["lora"])
    pm = pm.to(dev).eval()
    lp = [logp(pm, r) for r in recs]
    with pm.disable_adapter():
        l0 = [logp(pm, r) for r in recs]
    rep = {"ckpt": str(a.ckpt), "ckpt_step": ck["step"], "ckpt_sha1": sha1(a.ckpt), "out": str(out),
           "model_sha1": sha1(out / "model.safetensors"), "n_tensors": len(keys), "dev_seqs": [r["id"] for r in recs],
           "kl_merged_vs_peft": [kl(x, y) for x, y in zip(lp, lm)], "kl_t1c_vs_peft": [kl(x, y) for x, y in zip(lp, l0)],
           "argmax_agree_merged_peft": [float((x.argmax(-1) == y.argmax(-1)).float().mean()) for x, y in zip(lp, lm)]}
    rep["ok"] = bool(max(rep["kl_merged_vs_peft"]) < 0.1 * max(1e-6, min(rep["kl_t1c_vs_peft"])) or
                     max(rep["kl_merged_vs_peft"]) < 1e-3)
    (out.parent / "export.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not rep["ok"]:
        raise SystemExit("merged model does not reproduce the PEFT model on dev sequences")


if __name__ == "__main__":
    main()
