"""P1-b from P1-a records written elsewhere: BiCodec re-encode, source voice.

p1_native_talker.py makes p1b inline, right after each utterance's speech.
When P1-a comes from an engine that runs outside the BiCodec environment
(p1_vllm_omni.py), the same step runs here over its records instead: the
speech is re-encoded by BiCodec, its 32 global tokens replaced by the
source speaker's, and decoded again. The code is p1_native_talker's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1a", required=True, help="a p1a/<direction> directory with p1_*.jsonl")
    ap.add_argument("--parquet", required=True, help="the test parquet, for each row's bicodec_global")
    ap.add_argument("--output", required=True, help="the p1b/<direction> directory to write")
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from uniss import UniSSTokenizer

    rows = [json.loads(l) for p in sorted(Path(args.p1a).glob("p1_*.jsonl")) for l in open(p, encoding="utf-8")]
    voice = {r["id"]: r["bicodec_global"] for r in pq.read_table(args.parquet, columns=["id", "bicodec_global"]).to_pylist()}
    out = Path(args.output)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    part = out / "p1_swap.jsonl"
    done = {json.loads(l)["id"] for l in part.read_text(encoding="utf-8").splitlines() if l.strip()} if part.exists() else set()
    device = torch.device(args.device)
    vocoder = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)

    with part.open("a", encoding="utf-8") as fh:
        for k, ra in enumerate(r for r in rows if r["id"] not in done):
            b_path, n_sem, err = None, 0, None
            t1 = time.time()
            if ra.get("audio_path"):
                try:
                    _, bic = vocoder.tokenize(ra["audio_path"])
                    bic = [int(x) for x in (bic.flatten().tolist() if hasattr(bic, "flatten") else bic)]
                    semantic = bic[32:]
                    n_sem = len(semantic)
                    if semantic:
                        tokens = torch.tensor([*voice[ra["id"]], *semantic], dtype=torch.long, device=device)
                        b_path = str((out / "wav" / Path(ra["audio_path"]).name).resolve())
                        vocoder.save_audio(vocoder.decode(tokens), b_path, sample_rate=16_000)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:300]
            rb = {**ra, "system": ra["system"].replace("p1a_native_preset_voice", "p1b_bicodec_source_voice"),
                  "speaker": "source(bicodec_global)", "codes": n_sem, "audio_path": b_path,
                  "error": err or (None if b_path else "no audio"), "swap_seconds": round(time.time() - t1, 3)}
            rb.pop("audio_path_24k", None)
            fh.write(json.dumps(rb, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(f"  {k}  {rb['swap_seconds']:.2f}s swap", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
