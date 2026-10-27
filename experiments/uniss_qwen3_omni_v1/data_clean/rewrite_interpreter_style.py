"""Fluency plan v3 (step 3): the Thinker's whole-sentence translations rewritten as a simultaneous interpreter would say
them -- in the speaker's order, concise, without fillers or repeats, nothing added or left out. The rewriter is the base
Qwen3-Omni (it hears the source audio and reads the translation), served by vLLM.

Input: the oracle dumps (si_thinker_stream --oracle --dump; the oracle of a read with nothing committed is the Thinker's
greedy whole-sentence translation) and the items (audio, reference translation). Output, one line per item:
{"id", "dir", "oracle", "rewrite", "ref"}.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.rewrite_interpreter_style --out <jsonl> [--limit n] \
        [--shard k --num-shards n]
"""

from __future__ import annotations

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
DUMPS = "data/si_bayes_lr_v1/dump_*/dumps/*.jsonl,data/si_belief_v1/shard_*/dumps/*.jsonl"
PROMPT = {
    "en2zh": ("下面是这段英文语音的中文译文：\n{t}\n\n请把它改写成同声传译员会说的话：尽量按照说话人说话的先后顺序来表达，语句简洁，"
              "去掉口头禅、重复和自我修正，但不能遗漏或添加任何信息，也不能改变意思。只输出改写后的中文译文。"),
    "zh2en": ("Here is an English translation of this Chinese speech:\n{t}\n\nRewrite it the way a simultaneous interpreter "
              "would say it: follow the speaker's order as closely as possible, be concise, drop fillers, repetitions and "
              "self-corrections, but do not leave out or add any information and do not change the meaning. Output only the "
              "rewritten English translation."),
}


def oracles() -> dict:
    """item id -> (direction, the Thinker's greedy whole-sentence translation)."""
    out = {}
    for f in sorted({f for pat in DUMPS.split(",") for f in glob.glob(str(REPO / pat))}):
        d = "en2zh" if "_en2zh_" in f else "zh2en"
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("n_committed") == 0 and "oracle_units" in r and r["id"] not in out:
                ou = r["oracle_units"]
                out[r["id"]] = (d, "".join(ou) if d == "en2zh" else " ".join(ou))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="this many items per direction (stable hash order)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    items = {}
    for d in ("en2zh", "zh2en"):
        for line in open(REPO / f"data/unist_simul_thinker_v1/items_{d}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            items[r["id"]] = r
    orc = oracles()
    todo = []
    for d in ("en2zh", "zh2en"):
        ids = sorted((i for i, (dd, _) in orc.items() if dd == d and i in items), key=lambda i: zlib.crc32(i.encode()))
        todo += [(i, d) for i in (ids[: args.limit] if args.limit else ids)]
    todo = todo[args.shard:: args.num_shards]

    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    llm = LLM(model=QWEN3, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1})
    params = SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[151645])
    prompts = []
    for i, d in todo:
        x, sr = sf.read(items[i]["audio"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                             {"type": "text", "text": PROMPT[d].format(t=orc[i][1])}]}]
        head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        prompts.append({"prompt": head, "multi_modal_data": {"audio": (x.astype(np.float32), 16000)}})
    res = llm.generate(prompts, params)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for (i, d), r in zip(todo, res):
            fh.write(json.dumps({"id": i, "dir": d, "oracle": orc[i][1], "rewrite": r.outputs[0].text.strip(),
                                 "ref": items[i]["translation"]}, ensure_ascii=False) + "\n")
    print("rewrote", len(todo), flush=True)


if __name__ == "__main__":
    main()
