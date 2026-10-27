"""Fluency plan v3, recipe 2 (Hibiki's contextual alignment / BS-SiMT, with the Thinker's own likelihood): for each
training item, the Thinker's whole-sentence translation y (the oracle of the dumps' first read) is force-decoded after
hearing only the first t seconds, for every read time t; unit j may be said once its log-probability has come within
delta of its whole-audio value and stays there. Delays are made monotone (a unit never before the one it follows), so
a read at t commits y[:K(t)] -- always a prefix of the oracle, the text the Thinker itself would say with the whole
sentence heard.

Training data only (UniST items of data/unist_simul_thinker_v1 with the oracle dumps); one line per item:
{"id", "dir", "oracle", "units", "reads_ms", "delay_ms": {delta: [per unit]}}.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_align_oracle --out <jsonl> [--per-dir n] [--shard k --num-shards n]
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
WINDOW_MS = {"en2zh": 500, "zh2en": 1000}
DELTAS = (0.5, 1.0, 2.0)


def unit_spans(text: str, zh: bool) -> list[tuple[int, int]]:
    """Character spans of the units (Chinese: non-space characters; English: whitespace words)."""
    import re
    return [(m.start(), m.end()) for m in re.finditer(r"\S" if zh else r"\S+", text)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dir", type=int, default=6000)
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
    orc = {}
    for f in sorted({f for pat in DUMPS.split(",") for f in glob.glob(str(REPO / pat))}):
        d = "en2zh" if "_en2zh_" in f else "zh2en"
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("n_committed") == 0 and "oracle_units" in r and r["id"] in items and r["id"] not in orc:
                ou = r["oracle_units"]
                orc[r["id"]] = (d, "".join(ou) if d == "en2zh" else " ".join(ou))
    todo = []
    for d in ("en2zh", "zh2en"):
        ids = sorted((i for i, (dd, t) in orc.items() if dd == d and t.strip()), key=lambda i: zlib.crc32(i.encode()))
        todo += [(i, d) for i in ids[: args.per_dir]]
    todo = todo[args.shard:: args.num_shards]

    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams

    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    tok = processor.tokenizer
    instr = inference_instruction()
    heads = {}
    for d, tgt in (("en2zh", "cmn"), ("zh2en", "eng")):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instr[tgt]}]}]
        heads[d] = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=QWEN3, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1})
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    reqs, owners, meta = [], [], []
    for n_it, (i, d) in enumerate(todo):
        x, sr = sf.read(items[i]["audio"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        x = x.astype(np.float32)
        dur = len(x) // 16
        y = orc[i][1]
        enc = tok(y, add_special_tokens=False, return_offsets_mapping=True)
        reads = list(range(WINDOW_MS[d], dur, WINDOW_MS[d])) + [dur]
        meta.append({"id": i, "dir": d, "y": y, "ids": enc["input_ids"], "offsets": enc["offset_mapping"], "reads": reads})
        for t in reads:
            reqs.append({"prompt": heads[d] + y, "multi_modal_data": {"audio": (x[: t * 16], 16000)}})
            owners.append((n_it, t))
    res = llm.generate(reqs, params)
    lp = {}
    for (n_it, t), o in zip(owners, res):
        n = len(meta[n_it]["ids"])
        pl = o.prompt_logprobs[-n:]
        ids = o.prompt_token_ids[-n:]
        if ids != meta[n_it]["ids"]:   # the reply did not tokenise alone the same way: skip this item
            lp.setdefault(n_it, {})["bad"] = True
            continue
        lp.setdefault(n_it, {})[t] = [float(next(iter(p.values())).logprob) if p else 0.0 for p in pl]
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out, "x", encoding="utf-8") as fh:
        for n_it, m in enumerate(meta):
            L = lp.get(n_it, {})
            if L.get("bad") or any(t not in L for t in m["reads"]):
                continue
            zh = m["dir"] == "en2zh"
            spans = unit_spans(m["y"], zh)
            # tokens of each unit (by character offsets)
            tok_unit = [next((u for u, (a, b) in enumerate(spans) if a < e and s < b), None) for s, e in m["offsets"]]
            M = np.array([L[t] for t in m["reads"]])          # [reads, tokens]
            final = M[-1]
            delays = {}
            for delta in DELTAS:
                ok = M >= final[None, :] - delta
                # earliest read from which the token stays within delta of its final log-probability
                first = np.full(M.shape[1], len(m["reads"]) - 1)
                for j in range(M.shape[1]):
                    bad = np.flatnonzero(~ok[:, j])
                    first[j] = 0 if not len(bad) else min(bad[-1] + 1, len(m["reads"]) - 1)
                ud = [0] * len(spans)
                for j, u in enumerate(tok_unit):
                    if u is not None:
                        ud[u] = max(ud[u], int(first[j]))
                for u in range(1, len(ud)):
                    ud[u] = max(ud[u], ud[u - 1])      # monotone
                delays[str(delta)] = [m["reads"][k] for k in ud]
            fh.write(json.dumps({"id": m["id"], "dir": m["dir"], "oracle": m["y"], "units": [m["y"][a:b] for a, b in spans],
                                 "reads_ms": m["reads"], "delay_ms": delays}, ensure_ascii=False) + "\n")
            kept += 1
    print("items", len(meta), "kept", kept, flush=True)


if __name__ == "__main__":
    main()
