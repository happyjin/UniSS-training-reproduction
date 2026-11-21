"""Robust K1 (reports/mist_w3po_v1/robust_k1_unmerged_prereg.zh-CN.md, section 3): si_thinker_stream (shared, NOT
modified) on the base Qwen3-Omni with ONE vLLM LoRA applied to every request (no weight merge).

A wrapper (as wave16_si_thinker_ncwi.py): before si_thinker_stream.main() runs, vllm.LLM.__init__ gets
hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]} -- vLLM would otherwise build vllm-omni's
outer Qwen3OmniMoeForConditionalGeneration, which "does not support LoRA yet" (wave 16); the Thinker class declares
SupportsLoRA and maps the full checkpoint itself (thinker.model. -> language_model.model., talker./code2wav. dropped) --
plus enable_lora=True, max_loras=1, max_lora_rank=<adapter r rounded up to a vLLM rank>, lora_target_modules
["qkv_proj", "o_proj"] (text-layer attention only, no MoE-expert LoRA); vllm.LLM.generate passes the same LoRARequest
for every prompt. Prints the adapter config and the number of prompts served.

    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.k1_si_thinker --lora <peft dir> \
        <si_thinker_stream arguments, --thinker = the base>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RANKS = (8, 16, 32, 64, 128)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--lora", required=True)
    a, rest = ap.parse_known_args()
    p = Path(a.lora).resolve()
    cfg = json.load(open(p / "adapter_config.json"))
    r = int(cfg["r"])
    max_rank = min(x for x in RANKS if x >= r)
    assert set(cfg["target_modules"]) <= {"q_proj", "k_proj", "v_proj", "o_proj"}, cfg["target_modules"]
    from vllm.lora.request import LoRARequest
    req = LoRARequest("k1", 1, str(p))
    print("k1 lora:", json.dumps({"path": str(p), "r": r, "lora_alpha": cfg["lora_alpha"],
                                  "scale": cfg["lora_alpha"] / r, "target_modules": cfg["target_modules"],
                                  "max_lora_rank": max_rank}), flush=True)
    import vllm
    orig_init, orig_gen = vllm.LLM.__init__, vllm.LLM.generate
    n = [0]

    def init(self, *args, **kw):
        assert "hf_overrides" not in kw and "enable_lora" not in kw, "si_thinker_stream set its own LoRA / overrides"
        kw.update(hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]},
                  enable_lora=True, max_loras=1, max_lora_rank=max_rank, lora_target_modules=["qkv_proj", "o_proj"])
        print("k1 lora: LLM kwargs", json.dumps({k: v for k, v in kw.items() if "lora" in k or k == "hf_overrides"}),
              flush=True)
        orig_init(self, *args, **kw)

    def gen(self, prompts, sampling_params=None, *args, **kw):
        assert "lora_request" not in kw and not args, "si_thinker_stream passed its own LoRA / positional args"
        ps = prompts if isinstance(prompts, list) else [prompts]
        n[0] += len(ps)
        out = orig_gen(self, ps, sampling_params, lora_request=[req] * len(ps), **kw)
        return out if isinstance(prompts, list) else out[:1]

    vllm.LLM.__init__ = init
    vllm.LLM.generate = gen
    sys.argv = [sys.argv[0]] + rest
    from experiments.uniss_qwen3_omni_v1.evaluation import si_thinker_stream as S
    S.main()
    print("k1 lora: prompts served with the adapter", n[0], flush=True)


if __name__ == "__main__":
    main()
