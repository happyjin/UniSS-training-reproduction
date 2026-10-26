"""When each word of RealSI's reference translation could first be said: word-level text for si_infer.

RealSI hands its translation over a clause at a time, and a clause's text
exists only once the whole clause has been heard -- half the first clauses
run past 3 s, and a third of the sentences are a single clause, so speech
cannot start before the source ends. Here the reference translation is
released word by word instead, as our read/write data was timed (the omni
rule): a target word may be said once every source word it is aligned to has
been heard.

  words   (venv uniss-formal-align-v1) Qwen3-ForcedAligner on each sentence's
          source audio and transcript: source word times (English words,
          Chinese characters), as the a45 data was aligned.
  links   (venv align_omni_v1) OmniAlign between those source units (Chinese
          grouped into jieba words) and the translation's words (jieba words
          into Chinese, words into English), layer 7, threshold 1e-3, as
          data_clean/omni_align.py; then each target word's support (the
          latest end among its links; unlinked words wait for the next
          linked one) and where it starts in the translation.

    python realsi_align.py words --direction zh2en --out <dir>
    python realsi_align.py links --direction zh2en --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ALIGNER = "/opt/dlami/nvme/jasonleeeli/projects/multi_speaker_pipeline_share/multi_speaker_pipeline_share/model_weights/Qwen3-ForcedAligner-0.6B"
EN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")


def target_words(text: str, zh: bool) -> list[str]:
    if not zh:
        return EN_WORD.findall(text)
    import jieba
    return [w for w in jieba.cut(text) if re.search(r"[一-鿿A-Za-z0-9]", w)]


def cmd_words(args) -> None:
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items

    out = Path(args.out) / f"words_{args.direction}.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    model = Qwen3ForcedAligner.from_pretrained(ALIGNER, dtype=torch.bfloat16, device_map="cuda:0")
    lang = "Chinese" if args.direction.startswith("zh") else "English"
    items, cache, recs = realsi_items(args.direction), {}, []
    for s in range(0, len(items), args.batch):
        batch = items[s: s + args.batch]
        audio = []
        for it in batch:
            if it["wav"] not in cache:
                x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
                cache = {it["wav"]: x.mean(axis=1)}
            audio.append((cache[it["wav"]][it["start"] * 16: it["end"] * 16], 16000))
        res = model.align(audio=audio, text=[it["src_text"] for it in batch], language=[lang] * len(batch))
        for it, words in zip(batch, res):
            recs.append({"id": it["id"], "source_words": [{"text": str(w.text), "start_ms": int(round(w.start_time * 1000)),
                                                           "end_ms": int(round(w.end_time * 1000))} for w in words]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    print(f"{args.direction}: {len(recs)} sentences -> {out}", flush=True)


def cmd_links(args) -> None:
    import jieba
    import torch
    import transformers
    from experiments.uniss_qwen3_omni_v1.data_clean.omni_align import MODEL, encode, hidden, source_units
    from experiments.uniss_qwen3_omni_v1.data_clean.build_si_train_index import char_starts
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items

    out = Path(args.out) / f"align_{args.direction}.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    jieba.setLogLevel(60)
    device = torch.device("cuda:0")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    src_zh, tgt_zh = args.direction.startswith("zh"), args.direction.endswith("zh")
    words = {json.loads(l)["id"]: json.loads(l)["source_words"]
             for l in open(Path(args.out) / f"words_{args.direction}.jsonl", encoding="utf-8")}
    recs, unmatched = [], 0
    for it in realsi_items(args.direction):
        sw = words[it["id"]]
        su, last = source_units(sw, src_zh)
        tw = target_words(it["trg_text"], tgt_zh)
        starts = char_starts(it["trg_text"], tw)
        if not su or not tw or starts is None:
            unmatched += 1
            continue
        (sid, sown), (tid, town) = encode(tok, su), encode(tok, tw)
        a, b = hidden(model, tok, [sid], 7, device)[0], hidden(model, tok, [tid], 7, device)[0]
        sim = a @ b.T
        st, ts = torch.softmax(sim, dim=-1), torch.softmax(sim, dim=-2)
        support = [None] * len(tw)
        for i, j in torch.nonzero((st > 1e-3) & (ts > 1e-3), as_tuple=False).tolist():
            if i < len(sown) and j < len(town):
                t, u = town[j], last[sown[i]]
                support[t] = max(support[t] or 0, sw[u]["end_ms"])
        nxt = it["end"] - it["start"]
        for j in range(len(support) - 1, -1, -1):
            if support[j] is None:
                support[j] = nxt
            nxt = support[j]
        recs.append({"id": it["id"], "target_words": tw, "word_start": starts, "support_ms": support})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    print(f"{args.direction}: {len(recs)} sentences, {unmatched} skipped -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("words", "links"))
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    (cmd_words if args.cmd == "words" else cmd_links)(args)


if __name__ == "__main__":
    main()
