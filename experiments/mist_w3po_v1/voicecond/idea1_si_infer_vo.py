"""si_infer with the 32 BiCodec global (voice) tokens OVERRIDDEN by precomputed ids (Idea 1 ZNE, plan
reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md §2.9 / §2.10 E1.2). Everything else is
experiments.uniss_qwen3_omni_v1.evaluation.si_infer, unchanged (same flags, same item order, same seeds): only
BiCodecTokenizer.encode_wav_to_tokens is replaced, at class level, by a lookup

    key = Path(wav_path).stem  ->  {"key", "ids": [32 ints], "sha1": sha1 of the prompt file the ids were computed from}

The prompt file si_infer passes must have exactly that sha1 (so the override is tied to the very prompt the baseline
used); a missing key or a sha1 mismatch raises (no silent fallback to the raw prompt's tokens). The returned tensor has
shape (1, 32): si_infer keeps flat(...)[:32], and the decoder (front.decode) receives the same 32 ids.

    python -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override <tokens.jsonl> <si_infer args...>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    i = argv.index("--voice-override")
    table_path = argv[i + 1]
    del argv[i:i + 2]
    table = {}
    for line in open(table_path, encoding="utf-8"):
        r = json.loads(line)
        assert len(r["ids"]) == 32 and all(0 <= int(v) < 4096 for v in r["ids"]), r["key"]
        assert r["key"] not in table, f"duplicate key {r['key']}"
        table[r["key"]] = r
    import torch
    from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
    stats = {"used": 0}

    def encode_override(self, wav_path):
        key = Path(str(wav_path)).stem
        r = table.get(key)
        if r is None:
            raise RuntimeError(f"voice override: no tokens for {wav_path}")
        sha = hashlib.sha1(Path(str(wav_path)).read_bytes()).hexdigest()
        if sha != r["sha1"]:
            raise RuntimeError(f"voice override: prompt {wav_path} sha1 {sha} != {r['sha1']} (tokens computed from another file)")
        stats["used"] += 1
        return torch.tensor([[int(v) for v in r["ids"]]], dtype=torch.long)

    BiCodecTokenizer.encode_wav_to_tokens = encode_override
    print(f"[idea1 voice override] {len(table)} entries from {table_path}", flush=True)
    sys.argv = [sys.argv[0]] + argv
    from experiments.uniss_qwen3_omni_v1.evaluation import si_infer
    si_infer.main()
    print(f"[idea1 voice override] used {stats['used']} times", flush=True)


if __name__ == "__main__":
    main()
