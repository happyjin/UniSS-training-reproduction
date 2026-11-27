"""change 60 G0 for EGA (exhaustion-gated anticipation): whole-audio wave2_S1 dev decodes where the head is the BASE Thinker's own
whole-audio greedy translation (wave17, eval_outputs/sst_hsmm_v1/devoff = the G0 reference decode itself), trailing sentence-final
punctuation ("。！？.!?") and whitespace dropped, and the anticipation LoRA ($OFF_LORA, PEFT dir, served on wave17 like hsc25
--draft-lora) continues greedily from plain prompt + head (max 160 tokens). translation = head + tail. On complete audio a good
anticipation LoRA should end at once (tail = the dropped punctuation); G0 then measures what EGA's tail adds on complete input.
    OFF_LORA=<peft dir> (or OFF_MODEL=<train dir>/merged -> <train dir>/peft/a1.0000) <vllm python> -m experiments.sst_hsmm_v1.dev_offline_ega gen --direction d --shard k --num-shards n --out <jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
WAVE17 = str(REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05")
TDATA = REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data"
DEVOFF = str(REPO / "eval_outputs/sst_hsmm_v1/devoff/offline_{d}_*.jsonl")
PUNCT_END = set("。！？.!?")


def strip_end(s):
    s = s.rstrip()
    while s and (s[-1] in PUNCT_END or s[-1].isspace()):
        s = s[:-1]
    return s


def gen(a):
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    base = {r["id"]: r["translation"] for f in sorted(glob.glob(DEVOFF.format(d=a.direction)))
            for r in map(json.loads, open(f, encoding="utf-8"))}
    rows = [json.loads(l) for i, l in enumerate(open(TDATA / f"items_{a.direction}.jsonl", encoding="utf-8"))
            if l.strip() and i % a.num_shards == a.shard]
    rows = [r for r in rows if r["id"] in base]
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    tgt = "cmn" if a.direction == "en2zh" else "eng"
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": inference_instruction()[tgt]}]}]
    head = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=WAVE17, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=20261124, tensor_parallel_size=1, enable_lora=True, max_loras=1, max_lora_rank=16,
              lora_target_modules=["qkv_proj", "o_proj"],
              hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]})   # as hsc22 (LoRA needs the Thinker class)
    lora = os.environ.get("OFF_LORA") or str(Path(os.environ["OFF_MODEL"]).parent / "peft/a1.0000")   # pv4_g0.sh passes OFF_MODEL=<TR>/merged
    assert (Path(lora) / "adapter_config.json").exists(), lora
    lreq = LoRARequest("draft_lora", 1, str(Path(lora).resolve()))
    prompts, heads = [], []
    for r in rows:
        x, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, r["audio"]
        h = strip_end(base[r["id"]])
        heads.append(h)
        prompts.append({"prompt": head + h, "multi_modal_data": {"audio": (x.mean(axis=1), 16000)}})
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[151645]), use_tqdm=False,
                       lora_request=lreq)
    with open(a.out + ".tmp", "x", encoding="utf-8") as fh:
        for r, h, o in zip(rows, heads, res):
            fh.write(json.dumps({"id": r["id"], "translation": (h + o.outputs[0].text).strip(), "head": h,
                                 "tail": o.outputs[0].text}, ensure_ascii=False) + "\n")
    Path(a.out + ".tmp").rename(a.out)
    print("done", len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen",))
    ap.add_argument("--direction", choices=("en2zh", "zh2en"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    gen(ap.parse_args())


if __name__ == "__main__":
    main()
