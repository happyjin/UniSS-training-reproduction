"""Fluency plan v3 (step 3), prefix-constrained start oracle: at a sentence's first reads (nothing committed yet) the
draft often begins in the source's order while the whole-sentence oracle reorders, so the oracle-prefix rows teach the
Thinker to wait (36% of start reads agree on no unit, 17% mid-sentence). Here the base Thinker, hearing the whole
sentence, is made to continue from the draft's first L units (L = all of it, half, one unit); if that whole translation
is as close to the human reference as the free oracle (sentence chrF >= the oracle's - slack, length within 0.8-1.25),
saying those L units now is safe and becomes the read's target.

Training data only: the oracle dumps of UniST items (data/si_bayes_lr_v1, data/si_belief_v1) with the references in
data/unist_simul_thinker_v1. One line per start read: {"id", "dir", "t", "draft", "lcp", "best_L", "target", "full",
"chrf_oracle", "chrf_full"}; best_L = lcp when no longer prefix passes.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_start_oracle --out <jsonl> [--per-dir n] [--shard k --num-shards n]
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


def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dir", type=int, default=25000, help="start reads per direction whose draft goes past the lcp")
    ap.add_argument("--slack", type=float, default=1.0)
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
    reads = {"en2zh": [], "zh2en": []}
    for f in sorted({f for pat in DUMPS.split(",") for f in glob.glob(str(REPO / pat))}):
        d = "en2zh" if "_en2zh_" in f else "zh2en"
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("n_committed") != 0 or "oracle_units" not in r or r["id"] not in items:
                continue
            L0 = lcp(r["draft_units"], r["oracle_units"])
            if len(r["draft_units"]) > L0:
                reads[d].append({"id": r["id"], "dir": d, "t": r["t"], "draft": r["draft_units"], "oracle": r["oracle_units"], "lcp": L0})
    todo = []
    for d, rs in reads.items():
        rs.sort(key=lambda r: zlib.crc32(f"{r['id']}|{r['t']}".encode()))
        todo += rs[: args.per_dir]
    todo = todo[args.shard:: args.num_shards]

    from sacrebleu.metrics import CHRF
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    chrf = CHRF()
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    instr = inference_instruction()
    heads = {}
    for d, tgt in (("en2zh", "cmn"), ("zh2en", "eng")):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instr[tgt]}]}]
        heads[d] = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    join = lambda us, d: "".join(us) if d == "en2zh" else " ".join(us)
    llm = LLM(model=QWEN3, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1})
    params = SamplingParams(temperature=0.0, max_tokens=200, stop_token_ids=[151645])
    audio_cache = {}

    def audio(i):
        if i not in audio_cache:
            x, sr = sf.read(items[i]["audio"], dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            audio_cache[i] = x.astype(np.float32)
        return audio_cache[i]

    reqs, owners = [], []
    for k, r in enumerate(todo):
        n = len(r["draft"])
        cands = sorted({n, (n + 1) // 2, 1} - set(range(0, r["lcp"] + 1)), reverse=True)
        r["cands"] = cands
        for L in cands:
            reqs.append({"prompt": heads[r["dir"]] + join(r["draft"][:L], r["dir"]),
                         "multi_modal_data": {"audio": (audio(r["id"]), 16000)}})
            owners.append((k, L))
    res = llm.generate(reqs, params)
    fulls = {}
    for (k, L), o in zip(owners, res):
        r = todo[k]
        cont = o.outputs[0].text
        sep = "" if r["dir"] == "en2zh" or cont[:1].isspace() or cont[:1] in ",.;:!?)'" else " "
        fulls[(k, L)] = join(r["draft"][:L], r["dir"]) + sep + cont
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for k, r in enumerate(todo):
            ref = items[r["id"]]["translation"]
            orc = join(r["oracle"], r["dir"])
            c0 = chrf.sentence_score(orc, [ref]).score
            n0 = max(1, len(orc.replace(" ", "")) if r["dir"] == "en2zh" else len(orc.split()))
            best, best_full, best_c = r["lcp"], None, None
            for L in r["cands"]:   # longest first
                full = fulls[(k, L)]
                n1 = len(full.replace(" ", "")) if r["dir"] == "en2zh" else len(full.split())
                c1 = chrf.sentence_score(full, [ref]).score
                if c1 >= c0 - args.slack and 0.8 <= n1 / n0 <= 1.25:
                    best, best_full, best_c = L, full, c1
                    break
            fh.write(json.dumps({"id": r["id"], "dir": r["dir"], "t": r["t"], "draft": join(r["draft"], r["dir"]),
                                 "oracle": orc, "lcp": r["lcp"], "best_L": best,
                                 "target": join(r["draft"][:best], r["dir"]), "full": best_full,
                                 "chrf_oracle": round(c0, 2), "chrf_full": None if best_c is None else round(best_c, 2),
                                 "ref": ref}, ensure_ascii=False) + "\n")
    print("start reads", len(todo), flush=True)


if __name__ == "__main__":
    main()
