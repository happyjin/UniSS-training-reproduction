"""Why stage 2 translates worse: the adapter, or the harder prompt?

Stage 2 came in 3.05 Text-BLEU below stage 1 on Zh->En, and two things
changed at once. The Thinker gained a rank-32 LoRA, and the prompt went
from S2TT ("translate it") to S2ST ("transcribe it, translate it, and
speak the translation"), whose reply has to carry a transcript before the
translation. Either could cost quality; the full-test number cannot say
which.

So all four cells, same utterances, same greedy decode:

    adapter off + S2TT prompt   = stage 1's translator exactly
    adapter off + S2ST prompt   = the prompt's cost alone
    adapter on  + S2TT prompt   = the adapter's cost alone
    adapter on  + S2ST prompt   = stage 2

``lora.adapters_disabled`` makes "off" exact rather than approximate: the
same weights, the adapters bypassed. Text only -- no Talker, no vocoder --
because this asks about translation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.uniss_omni_s2st_v1.evaluation.cascade_s2st import translate
from experiments.uniss_omni_s2st_v1.evaluation.stage2_generate import speak, split_reply
from experiments.uniss_omni_s2st_v1.runtime.omni_processor import load_audio_text_processor
from experiments.uniss_omni_s2st_v1.training import stage2_tasks
from experiments.uniss_streaming_p2st_traj_v1.training import lora


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pretrained_models/Qwen2.5-Omni-3B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dev-parquet", nargs="+", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen2_5OmniForConditionalGeneration

    wanted = None
    if args.subset:
        wanted = {
            json.loads(line)["id"]
            for line in Path(args.subset).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    rows = []
    for path in args.dev_parquet:
        table = pq.read_table(
            path, columns=["id", "src_lang", "tgt_lang", "translation", "source_audio_path"]
        )
        rows.extend(r for r in table.to_pylist() if wanted is None or r["id"] in wanted)
    rows.sort(key=lambda r: (r["id"], r["tgt_lang"]))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"{len(rows)} utterances", flush=True)

    processor = load_audio_text_processor(args.model)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    # The Talker and the vocoder play no part in a text ablation.
    for attr in ("token2wav", "talker"):
        if hasattr(model, attr):
            delattr(model, attr)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layers = lora.inject(
        model.thinker, rank=args.lora_rank, alpha=args.lora_alpha,
        targets=stage2_tasks.THINKER_TARGETS,
    )
    lora.load_state_dict(layers, state["thinker_lora"])
    for layer in layers:
        layer.to(device=model.device, dtype=torch.bfloat16)
    model.eval()

    def both_prompts(row):
        s2tt = translate(model, processor, row["source_audio_path"],
                         row["tgt_lang"], args.max_new_tokens)
        reply, _ = speak(model, processor, row["source_audio_path"],
                         row["src_lang"], row["tgt_lang"],
                         max_new_tokens=args.max_new_tokens)
        return s2tt, split_reply(reply)[1]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out / f"ablation_{args.shard}.jsonl").open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows):
            with lora.adapters_disabled(layers):
                off_s2tt, off_s2st = both_prompts(row)
            on_s2tt, on_s2st = both_prompts(row)
            handle.write(json.dumps({
                "id": row["id"], "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
                "mode": "tts", "translation_ref": row["translation"],
                "off_s2tt": off_s2tt, "off_s2st": off_s2st,
                "on_s2tt": on_s2tt, "on_s2st": on_s2st,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 50 == 0:
                print(f"  {position}/{len(rows)}", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
