"""sst_hsmm_v1 probe (no gate decision; diagnosis of the 2026-11-22 P2 final-read runaway). Replays the LAST read of the
first N items per direction from p2_finalstop_bug_20261122 (prompt = head + text committed before the final read, whole
audio), greedy, WAIT NOT masked, stop on WAIT or EOS, max 160: which token ends the turn and how long the text is.
    SST_GATE=.. SST_TOKMS=.. <vllm_omni_030 python> -m experiments.sst_hsmm_v1.probe_final_stop --n 40 --out <json>
"""
import argparse, json, sys
from pathlib import Path
import soundfile as sf
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.sst_hsmm_v1.e2e_si_thinker import QWEN3, WAIT, THINKER_EOS

B = REPO / "eval_outputs/sst_hsmm_v1/p2_finalstop_bug_20261122"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    llm = LLM(model=str(REPO / "eval_outputs/sst_hsmm_v1/p1/merged"), gpu_memory_utilization=0.6, max_model_len=8192,
              limit_mm_per_prompt={"audio": 1}, seed=20261021)
    res = {}
    for d in ("en2zh", "zh2en"):
        tgt = "cmn" if d.endswith("zh") else "eng"
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": inference_instruction()[tgt]}]}]
        head = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        items = {json.loads(l)["id"]: json.loads(l) for l in open(B / f"data/items_{d}_0.jsonl", encoding="utf-8")}
        outs = [json.loads(l) for l in open(B / f"thinker_{d}/blocks_{d}_w1.jsonl", encoding="utf-8")][: a.n]
        prompts, rows = [], []
        for o in outs:
            it = items[o["id"]]
            x = sf.read(it["audio"], dtype="float32", always_2d=True)[0].mean(axis=1)[: int(it["dur_ms"]) * 16]
            committed = "".join(b["text"] for b in o["blocks"] if not b["final"])
            prompts.append({"prompt": head + committed, "multi_modal_data": {"audio": (x, 16000)}})
            rows.append((o, committed, it))
        sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[WAIT, THINKER_EOS], logprobs=3)
        gen = llm.generate(prompts, sp, use_tqdm=False)
        recs = []
        for (o, committed, it), g in zip(rows, gen):
            ids = list(g.outputs[0].token_ids)
            end = "WAIT" if ids and ids[-1] == WAIT else "EOS" if ids and ids[-1] == THINKER_EOS else "LEN"
            lp = g.outputs[0].logprobs[-1] if g.outputs[0].logprobs else {}
            recs.append({"id": o["id"], "end": end, "n": len(ids), "committed": committed, "text": g.outputs[0].text,
                         "bug_final": o["blocks"][-1]["text"][:80], "ref": it.get("translation"),
                         "last_top": {str(k): round(v.logprob, 3) for k, v in lp.items()}})
        res[d] = {"ends": {e: sum(r["end"] == e for r in recs) for e in ("WAIT", "EOS", "LEN")}, "items": recs}
        print(d, res[d]["ends"], flush=True)
    json.dump(res, open(a.out, "x", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
