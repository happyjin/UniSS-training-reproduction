"""Phase C gate: is the assembly wired correctly, before anything is trained?

Four questions, none of which training can answer and all of which it can
hide. A miswired assembly still produces a loss curve that falls.

  1. Do the dimensions line up end to end -- 2048 through the projection
     to 896, and the codec head masked to BiCodec's 8,192?
  2. Is the Talker causal and correctly offset? Changing the code fed at
     input index t+1 must leave every logit at index <= t *exactly*
     unchanged and must move index t+1. Exact, not small.
  3. Does Qwen3's feature extractor pad the mel to its full window the way
     Qwen2.5's did? That cost stage 2 a 17x slowdown before it was found.
  4. What does a real utterance cost in memory, at the lengths CVSS-T
     actually contains rather than the 15 seconds the model card quotes?
"""

from __future__ import annotations

import argparse
import json
import time

import soundfile as sf
import torch

from experiments.uniss_omni_s2st_v1.evaluation.stage0_forward_check import (
    causality_probe,
)
from experiments.uniss_omni_s2st_v1.modeling.stage0_assembly import build_text_stream
from experiments.uniss_qwen3_omni_v1.modeling.qwen3_assembly import build_assembly

INSTRUCTION = {
    "eng": "Listen to the Chinese speech and translate it into English. "
           "Reply with the English translation only.",
    "cmn": "Listen to the English speech and translate it into Chinese. "
           "Reply with the Chinese translation only.",
}


@torch.no_grad()
def reply_states(a, audio_path, tgt_lang, reply_text):
    """The states the Talker would read, for a gold reply."""
    waveform, _ = sf.read(audio_path, dtype="float32")
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": INSTRUCTION[tgt_lang]},
    ]}]
    text = a.processor.apply_chat_template(
        [conversation], add_generation_prompt=True, tokenize=False
    )
    t0 = time.perf_counter()
    inputs = a.processor(text=text, audio=[waveform], sampling_rate=16_000,
                         return_tensors="pt")
    extract_s = time.perf_counter() - t0
    reply_ids = a.tokenizer(reply_text, add_special_tokens=False)["input_ids"]
    ids = torch.cat([
        inputs["input_ids"][0],
        torch.tensor(reply_ids, dtype=torch.long),
    ]).unsqueeze(0).to(a.device)
    kwargs = {"input_ids": ids, "output_hidden_states": True, "return_dict": True}
    for k in ("input_features", "feature_attention_mask"):
        if k in inputs:
            kwargs[k] = inputs[k].to(a.device)
    out = a.thinker(**kwargs)
    hidden = out.hidden_states[-1][:, -len(reply_ids):, :]
    embeds = a.thinker.get_input_embeddings()(
        torch.tensor([reply_ids], device=a.device)
    )
    feat_shape = tuple(inputs["input_features"].shape) if "input_features" in inputs else None
    return hidden, embeds, {"extract_s": extract_s, "feature_shape": feat_shape,
                            "audio_s": len(waveform) / 16_000,
                            "prompt_tokens": int(inputs["input_ids"].shape[1]),
                            "reply_tokens": len(reply_ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--talker-checkpoint", required=True)
    ap.add_argument("--dev-parquet",
                    default="data/cvss_t/tokenized/cvss_t_zh_en_v1/cvss_t_zh_en_test.parquet")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    report = {}
    a = build_assembly(talker_checkpoint=args.talker_checkpoint, device=args.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    after_load = torch.cuda.memory_allocated(a.device) / 1024**3
    print(f"装配完成，权重占 {after_load:.1f} GiB", flush=True)

    # 1. dimensions
    proj = a.talker.thinker_to_talker_proj.weight
    dims = {
        "thinker_hidden": int(a.thinker.get_input_embeddings().weight.shape[1]),
        "proj": list(proj.shape),
        "talker_hidden": int(a.talker.config.hidden_size),
        "codec_vocab": int(a.talker.config.vocab_size),
        "valid_codes": int(a.output_mask.sum()),
        "global_embed": list(a.global_embed.weight.shape),
    }
    report["dims"] = dims
    print("维度:", json.dumps(dims), flush=True)
    # 8,192 codes plus eos. The Talker has to be able to stop, and
    # valid_output_mask allows exactly that one reserved row.
    assert dims["valid_codes"] == 8193, dims["valid_codes"]
    assert dims["proj"] == [896, 2048] and dims["thinker_hidden"] == 2048

    rows = pq.read_table(args.dev_parquet, columns=[
        "id", "translation", "tgt_lang", "bicodec_global", "target_bicodec",
        "source_audio_path", "source_audio_duration_seconds",
    ]).to_pylist()
    rows.sort(key=lambda r: -r["source_audio_duration_seconds"])
    probe_rows = [rows[0]] + rows[len(rows)//2 : len(rows)//2 + args.rows - 1]

    report["utterances"] = []
    for row in probe_rows:
        hidden, embeds, info = reply_states(
            a, row["source_audio_path"], row["tgt_lang"], row["translation"])
        codes = list(row["target_bicodec"])[:200]
        prefix = len(row["bicodec_global"])
        steps = len(codes) + prefix
        eos_e, pad_e = a.markers.embeds(hidden.dtype, a.device)
        stream = build_text_stream(hidden.float(), embeds.float(), steps=steps,
                                   eos_embed=eos_e.float(), pad_embed=pad_e.float()
                                   ).to(dtype=hidden.dtype)
        # Causality is a property of the Talker's mask, which the speaker
        # prefix does not participate in: it is 32 fixed vectors prepended
        # to the embeddings. Probing the code-only stream keeps this the
        # same test stage 0 ran, and the prefix is checked separately by
        # shape below.
        codec_ids = torch.tensor(
            [[a.talker.codec_bos_token] + codes[:-1]], device=a.device)
        code_stream = stream[:, prefix : prefix + codec_ids.shape[1], :]
        mask = torch.ones(codec_ids.shape, dtype=torch.long, device=a.device)
        pos = min(40, len(codes) - 2)
        probe = causality_probe(a.talker, codec_ids, code_stream, mask, a.device,
                                position=pos, code_size=8192)

        # The prefix path, checked for shape rather than causality.
        gl = a.global_embed(torch.tensor([row["bicodec_global"]], device=a.device))
        prefixed = torch.cat(
            [gl, a.talker.get_input_embeddings()(codec_ids)], dim=1)
        lm_in = a.talker.thinker_to_talker_proj(
            prefixed + stream[:, : prefixed.shape[1], :])
        info["prefix_shapes"] = {
            "global_embed_out": list(gl.shape),
            "prefixed": list(prefixed.shape),
            "talker_input": list(lm_in.shape),
        }
        assert gl.shape[1] == prefix == 32, gl.shape
        assert lm_in.shape[-1] == a.talker.config.hidden_size

        info.update({"id": row["id"], "probe": probe})
        report["utterances"].append(info)
        print(f"  {row['id'][:34]}  音频 {info['audio_s']:.1f}s  "
              f"特征 {info['feature_shape']}  抽取 {info['extract_s']*1000:.0f}ms  "
              f"因果 泄漏={probe['max_delta_at_or_before']:.3e} "
              f"移动={probe['delta_at_perturbed']:.3e}", flush=True)

    report["peak_gib"] = torch.cuda.max_memory_allocated(a.device) / 1024**3
    leaks = [u["probe"]["max_delta_at_or_before"] for u in report["utterances"]]
    moves = [u["probe"]["delta_at_perturbed"] for u in report["utterances"]]
    report["causal_ok"] = max(leaks) == 0.0 and min(moves) > 0.0
    print(f"\n峰值显存 {report['peak_gib']:.1f} GiB")
    print(f"因果性: 最大泄漏 {max(leaks):.3e}，最小移动 {min(moves):.3e} -> "
          f"{'PASS' if report['causal_ok'] else 'FAIL'}")
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("PHASE_C_CHECK_DONE")


if __name__ == "__main__":
    main()
