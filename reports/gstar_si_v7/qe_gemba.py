"""Paper v7 (prereg_v7 §5; v6 review M09 / T19): reference-free quality estimate of the AliMeeting English references,
which are machine translations (tencent Hy-MT2-30B-A3B) of the human Mandarin transcripts. No CometKiwi model is on
the server (gated on the hub) and no new environment may be built, so the estimate is GEMBA-DA (Kocmi & Federmann,
2023): an LLM other than the translator -- the original Qwen3-Omni-30B-A3B-Instruct Thinker, text only, greedy --
scores each (source, translation) pair 0-100. Every non-hesitation utterance of the AliMeeting test reference windows
is scored once. Output: one json line per utterance {"record_id", "score", "raw"}; the Thinker is split over the
visible cards (--max-gib each), as the v7 decoders.

    CUDA_VISIBLE_DEVICES=1,2 python reports/gstar_si_v7/qe_gemba.py --out <qe.jsonl> [--batch 48] [--limit 0]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

PROMPT = ("Score the following translation from Chinese to English on a continuous scale from 0 to 100, where a score "
          "of zero means \"no meaning preserved\" and score of one hundred means \"perfect meaning and grammar\".\n\n"
          "Chinese source: \"{src}\"\nEnglish translation: \"{tgt}\"\nScore:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/alimeeting_si_v1/windows_test_ref.jsonl")
    ap.add_argument("--model", default="pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-gib", type=int, default=34)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    seen, items = set(), []
    for l in open(a.windows, encoding="utf-8"):
        for u in json.loads(l)["utts"]:
            if u["hesitation"] or u["record_id"] in seen:
                continue
            seen.add(u["record_id"])
            items.append((u["record_id"], u["zh"], u.get("ref", u.get("en", ""))))
    if a.limit:
        items = items[: a.limit]
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    tok.padding_side = "left"
    tmpl = json.load(open(Path(a.model) / "chat_template.json"))["chat_template"]   # the tokenizer itself carries none
    mem = {i: f"{a.max_gib}GiB" for i in range(torch.cuda.device_count())}
    mem["cpu"] = "400GiB"
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto",
                                                                max_memory=mem)
    thinker = omni.thinker.eval()
    del omni
    torch.cuda.empty_cache()
    dev = thinker.get_input_embeddings().weight.device
    t0 = time.time()
    with open(out, "x", encoding="utf-8") as fh, torch.no_grad():
        for b0 in range(0, len(items), a.batch):
            bt = items[b0: b0 + a.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(src=s, tgt=t)}], tokenize=False,
                                             add_generation_prompt=True, chat_template=tmpl) for _, s, t in bt]
            enc = tok(texts, return_tensors="pt", padding=True).to(dev)
            gen = thinker.generate(**enc, max_new_tokens=8, do_sample=False)
            for (rid, _, _), g in zip(bt, gen[:, enc["input_ids"].shape[1]:]):
                raw = tok.decode(g, skip_special_tokens=True)
                m = re.search(r"\d+(\.\d+)?", raw)
                fh.write(json.dumps({"record_id": rid, "score": float(m.group()) if m else None, "raw": raw},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{b0 + len(bt)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    print("QE_DONE", out, flush=True)


if __name__ == "__main__":
    main()
