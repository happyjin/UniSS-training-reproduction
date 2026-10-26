"""E2E-v0's text side (docs/multi_speaker_si_plan_v2.zh-CN.md): the multi-speaker Thinker (training/
pretrain_thinker_ms.py, merged) interprets each piece of a meeting mixture as it arrives.

Every --window s the Thinker is given the audio heard so far and its own reply so far, under the multi-speaker
instruction, and continues it greedily until the wait marker <|box_end|> (at the last read the marker is barred
and the reply ends). Whatever it says is committed at once -- the model was trained on when to speak, so there is
no agreement step. The reply carries the speakers: "[S<k>@<t>]" opens each change of speaker (k by order of first
appearance in the piece, t the second where that speech starts), and every block is written out with the label it
falls under and that start.

Output, one line per piece: {"id", "raw" (the reply with labels), "translation" (labels removed),
"blocks": [{"write_ms", "spk", "anchor_ms", "text", "final"}]}.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.e2e_ms_stream --thinker <merged> \
        --items-file data/ami_multispk_si_v1/e2e_v0/items_test.jsonl --out <dir>/blocks_ms_w1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
THINKER_EOS = 151645
WAIT_MARK = 151649   # <|box_end|>
LABEL = re.compile(r"\[S(\d+)@(\d+(?:\.\d+)?)\]")


def split_blocks(text: str, write_ms: int, final: bool, cur: dict) -> list[dict]:
    """The text one read added, cut at its labels; cur = {"spk", "anchor_ms"} carries the open label across reads."""
    out, pos = [], 0
    for m in list(LABEL.finditer(text)) + [None]:
        seg = text[pos: m.start() if m else len(text)]
        if seg.strip():
            out.append({"write_ms": write_ms, "spk": cur["spk"], "anchor_ms": cur["anchor_ms"], "text": seg.strip(),
                        "final": False})
        if m:
            cur["spk"], cur["anchor_ms"] = f"S{m.group(1)}", round(float(m.group(2)) * 1000)
            pos = m.end()
    if final and out:
        out[-1]["final"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinker", required=True, help="merged multi-speaker Thinker (a full Qwen3-Omni copy)")
    ap.add_argument("--items-file", required=True, help="jsonl of {id, audio, dur_ms}")
    ap.add_argument("--out", required=True, help="output jsonl (refuses an existing file)")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--wait-bias", type=float, default=0.0, help="added to the wait marker's logit at every read")
    ap.add_argument("--inst", choices=["ms", "sd"], default="ms", help="sd: the single-speaker E1 prompt")
    ap.add_argument("--direction", default="en2zh")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    from experiments.uniss_qwen3_omni_v1.training.ms_prompt import ms_instruction

    items = [json.loads(l) for l in open(args.items_file, encoding="utf-8")]
    if args.limit and args.limit < len(items):
        items = [items[round(i * (len(items) - 1) / (args.limit - 1))] for i in range(args.limit)]
    for it in items:   # an item may be a span of a longer recording (start / end in ms, as RealSI's sentences are)
        x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, it["audio"]
        it["wav"] = x.mean(axis=1)[int(it.get("start", 0)) * 16: int(it["end"]) * 16 if it.get("end") else None]
        it["dur_ms"] = len(it["wav"]) // 16
        it.update(reply="", blocks=[], cur={"spk": "S1", "anchor_ms": 0}, done=False)
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    ins = (ms_instruction(args.direction) if args.inst == "ms"
           else inference_instruction()["cmn" if args.direction.endswith("zh") else "eng"])
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=args.thinker, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"audio": 1}, seed=args.seed)
    read = SamplingParams(temperature=0.0, max_tokens=200, stop_token_ids=[THINKER_EOS, WAIT_MARK],
                          logit_bias={WAIT_MARK: args.wait_bias} if args.wait_bias else None)
    last = SamplingParams(temperature=0.0, max_tokens=400, stop_token_ids=[THINKER_EOS], logit_bias={WAIT_MARK: -100.0})
    win = round(args.window * 1000)
    t0, r = time.time(), 0
    while any(not it["done"] for it in items):
        r += 1
        live = [it for it in items if not it["done"]]
        reqs, params = [], []
        for it in live:
            it["t"] = min(r * win, it["dur_ms"])
            it["final"] = it["t"] >= it["dur_ms"]
            reqs.append({"prompt": head + it["reply"], "multi_modal_data": {"audio": (it["wav"][: it["t"] * 16], 16000)}})
            params.append(last if it["final"] else read)
        outs = llm.generate(reqs, params, use_tqdm=False)
        for it, o in zip(live, outs):
            text = o.outputs[0].text.replace("<|box_end|>", "")
            it["reply"] += text
            it["blocks"] += split_blocks(text, it["t"], it["final"], it["cur"])
            if it["final"]:
                it["done"] = True
                if it["blocks"] and not it["blocks"][-1]["final"]:
                    it["blocks"][-1]["final"] = True
        print(f"read {r}: {len(live)} pieces, {time.time() - t0:.0f}s", flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it["id"], "raw": it["reply"], "translation": LABEL.sub("", it["reply"]).strip(),
                                 "blocks": it["blocks"]}, ensure_ascii=False) + "\n")
    print(f"done {len(items)} pieces in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
