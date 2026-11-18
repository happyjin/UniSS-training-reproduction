"""OpenBroadcast-SI v1 hook (reports/mist_w3po_v1/voicecond_bench.zh-CN.md §hook): run si_infer UNCHANGED, except that
BiCodec's voice-prompt encoding (BiCodecTokenizer.encode_wav_to_tokens, whose first 32 ids si_infer uses as the talker
item's global voice tokens -- LM condition and decoder d-vector) is answered from a sidecar for the listed items.

Sidecar (json): {<talker item id>: [32 global token ids] | "<path to a 16 kHz wav to encode instead>"}. The item id is
the voice prompt file's stem (gstar_talker_v6 prep writes talker_in_all/voice_{causal,fallback}/<item id>.wav, the ids of
talker_in/items.jsonl, e.g. "obz001_on_0003__S2"). Items not in the sidecar keep the original encoding. Item order,
shards, sampling seeds and every other flag are the caller's (the T1c command of the baseline), so the run differs from
the baseline only in the overridden voice tokens. At exit it prints "[voice-override] ..." counts to the talker log.

    <venvs/vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.bench_talker_override --voice-override <json> \\
        <si_infer arguments>
(normally inserted by scripts/voicecond_bench_orch.py --voice-override <json>)
"""
from __future__ import annotations

import atexit
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if "--voice-override" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--voice-override")
    vo, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    O = json.load(open(vo, encoding="utf-8"))
    for k, v in O.items():
        if not isinstance(v, str):
            assert len(v) == 32 and all(isinstance(t, int) and 0 <= t < 4096 for t in v), f"bad tokens for {k}"
    import torch
    from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
    orig = BiCodecTokenizer.encode_wav_to_tokens
    n = {"ids": 0, "wav": 0, "kept": 0}

    def encode(self, wav_path):
        v = O.get(Path(str(wav_path)).stem)
        if v is None:
            n["kept"] += 1
            return orig(self, wav_path)
        if isinstance(v, str):
            n["wav"] += 1
            return orig(self, v)
        n["ids"] += 1
        return torch.tensor([v], dtype=torch.long)

    BiCodecTokenizer.encode_wav_to_tokens = encode
    atexit.register(lambda: print(f"[voice-override] {vo}: tokens {n['ids']}, wav {n['wav']}, original {n['kept']}",
                                  flush=True))
    sys.argv = ["si_infer"] + argv
    runpy.run_module("experiments.uniss_qwen3_omni_v1.evaluation.si_infer", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
