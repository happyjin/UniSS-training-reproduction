"""Robust wave 16 NCWI (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md, section 3): si_thinker_stream (shared,
NOT modified) on the base Qwen3-Omni with vLLM multi-LoRA, each request with the WiSE coefficient alpha* of its item's
b-hat bin.

A wrapper, not a copy (as wave9_si_thinker_anchor.py): before si_thinker_stream.main() runs, vllm.LLM.__init__ is patched to
add enable_lora=True, max_loras=8, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"] (the trained LoRA sits on
the text layers' attention only; no MoE-expert LoRA), and vllm.LLM.generate to pass one LoRARequest per prompt: the item is
found by the sha1 of the prompt audio's first 1 s (wave16_noise_est.py `key`, = wave9_common.head_key), its alpha* comes
from --assign (wave16_fit.py assign_<dir>.jsonl), alpha* = 0 -> no LoRA (the base), else the PEFT adapter
<lora-root>/a<alpha:.4f> (wave16_make_lora.py). --const-alpha A gives every item alpha A (the alpha-0.5 consistency check
against the merged unist_wise05). A prompt whose key is not in the table stops the run. Prints per-alpha prompt counts.

    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave16_si_thinker_ncwi --assign <assign_dir.jsonl> \
        --lora-root <dir> [--const-alpha 0.5] <si_thinker_stream arguments, --thinker = the base>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave9_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--lora-root", required=True)
    ap.add_argument("--const-alpha", type=float, default=None)
    a, rest = ap.parse_known_args()
    table = {}
    for r in map(json.loads, open(a.assign, encoding="utf-8")):
        al = round(float(a.const_alpha if a.const_alpha is not None else r["alpha"]), 4)
        assert table.setdefault(r["key"], al) == al, f"key collision with different alphas: {r['id']}"
    alphas = sorted({v for v in table.values() if v > 0})
    assert len(alphas) <= 8, alphas
    from vllm.lora.request import LoRARequest
    reqs = {}
    for k, al in enumerate(alphas, start=1):
        p = Path(a.lora_root) / f"a{al:.4f}"
        cfg = json.load(open(p / "adapter_config.json"))
        assert abs(cfg["wave16"]["wise_alpha"] - al) < 1e-9, (p, cfg["wave16"])
        reqs[al] = LoRARequest(f"ncwi_a{al:.4f}", k, str(p.resolve()))
    print("wave16 ncwi:", json.dumps({"items": len(table), "adapters": {str(x): v.lora_path for x, v in reqs.items()},
                                      "const_alpha": a.const_alpha}), flush=True)
    import vllm
    orig_init, orig_gen = vllm.LLM.__init__, vllm.LLM.generate
    st = Counter()

    def init(self, *args, **kw):
        kw.update(enable_lora=True, max_loras=8, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"])
        print("wave16 ncwi: LLM kwargs", json.dumps({k: v for k, v in kw.items() if "lora" in k}), flush=True)
        orig_init(self, *args, **kw)

    def gen(self, prompts, sampling_params=None, *args, **kw):
        assert "lora_request" not in kw and not args, "si_thinker_stream passed its own LoRA / positional args"
        ps = prompts if isinstance(prompts, list) else [prompts]
        lr = []
        for p in ps:
            key = C.head_key(p["multi_modal_data"]["audio"][0])
            if key not in table:
                raise SystemExit(f"wave16 ncwi: prompt audio key {key} not in {a.assign}")
            al = table[key]
            st[str(al)] += 1
            lr.append(reqs[al] if al > 0 else None)
        out = orig_gen(self, ps, sampling_params, lora_request=lr, **kw)
        return out if isinstance(prompts, list) else out[:1]

    vllm.LLM.__init__ = init
    vllm.LLM.generate = gen
    sys.argv = [sys.argv[0]] + rest
    from experiments.uniss_qwen3_omni_v1.evaluation import si_thinker_stream as S
    S.main()
    print("wave16 ncwi: prompts per alpha", json.dumps(dict(sorted(st.items()))), flush=True)


if __name__ == "__main__":
    main()
