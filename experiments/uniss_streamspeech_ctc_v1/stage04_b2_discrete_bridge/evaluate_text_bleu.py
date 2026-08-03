#!/usr/bin/env python3
"""Greedy Phase3 text probe driven by the trained B2 audio bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE03 = Path(__file__).resolve().parents[1] / "stage03_multitask_encoder"
STAGE02 = Path(__file__).resolve().parents[1] / "stage02_ctc_probe"
for path in (ROOT, STAGE03, STAGE02, Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bridge import replace_embedding_span
from bridge_data import B2BridgeAudioDataset
from model import FrozenEncoderB2Bridge
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.sample_builders import build_performance_sample
from evaluation.uniss_outputs import parse_with_tokenizer


PERFORMANCE_SUFFIX_TOKENS = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--ctc-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--endpoint-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--codebook-model", type=Path, required=True)
    parser.add_argument("--phase3-model", type=Path, required=True)
    parser.add_argument("--direction-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--direction-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_bridge(args, qwen, eng_vocab, cmn_vocab, device):
    qwen_glm_embeddings = qwen.get_input_embeddings().weight[
        c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
    ].detach().float().cpu()
    model = FrozenEncoderB2Bridge.from_checkpoints(
        endpoint_checkpoint=args.endpoint_checkpoint,
        historical_stage_b_checkpoint=args.historical_stage_b_checkpoint,
        codebook_model=args.codebook_model,
        qwen_glm_embeddings=qwen_glm_embeddings,
        eng_vocab_size=eng_vocab,
        cmn_vocab_size=cmn_vocab,
    )
    checkpoint = torch.load(args.bridge_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def build_prompt_embeddings(qwen, text_encoder, record, bridge_output, device):
    speech_length = int(bridge_output.token_lengths[0])
    sample = build_performance_sample(
        source_glm=[0] * speech_length,
        bicodec_global=record["bicodec_global"],
        tgt_lang=str(record["tgt_lang"]),
        translation=str(record["translation"]),
        target_bicodec=record["target_bicodec"],
        text_encoder=text_encoder,
        source_id=str(record["id"]),
    )
    ids = torch.tensor(sample.prompt_ids, dtype=torch.long, device=device)
    embeddings = qwen.get_input_embeddings()(ids)
    span_start = len(sample.prompt_ids) - PERFORMANCE_SUFFIX_TOKENS - speech_length
    embeddings = replace_embedding_span(
        embeddings,
        bridge_output.qwen_speech_embeddings[0],
        span_start=span_start,
        speech_length=speech_length,
    )
    return embeddings.unsqueeze(0)


@torch.inference_mode()
def greedy_translation(qwen, prompt_embeddings, tokenizer, max_tokens):
    generated = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = qwen(inputs_embeds=prompt_embeddings, use_cache=True)
    cache = output.past_key_values
    logits = output.logits[:, -1].float()
    while len(generated) < max_tokens:
        logits[:, c.VOCAB_SIZE :] = -torch.inf
        token = int(logits.argmax(dim=-1)[0])
        generated.append(token)
        if token in {c.TOKEN_END_CONTENT, c.TOKEN_EOS}:
            break
        next_ids = torch.tensor([[token]], device=logits.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = qwen(input_ids=next_ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1].float()
    parsed = parse_with_tokenizer(generated, mode="performance", tokenizer=tokenizer)
    return generated, parsed["generated_translation"] or ""


def main():
    args = parse_args()
    if args.direction_offset < 0 or args.max_samples <= 0:
        raise ValueError("direction-offset must be non-negative and max-samples positive")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.phase3_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    qwen.requires_grad_(False)
    eng_vocab = spm.SentencePieceProcessor(
        model_file=str(args.ctc_tokenizer_dir / "ctc_eng.model")
    ).vocab_size()
    cmn_vocab = spm.SentencePieceProcessor(
        model_file=str(args.ctc_tokenizer_dir / "ctc_cmn.model")
    ).vocab_size()
    bridge, checkpoint = load_bridge(args, qwen, eng_vocab, cmn_vocab, device)
    dataset = B2BridgeAudioDataset(
        args.dataset_index, "valid", args.source_manifest, args.source_offsets
    )
    selected = []
    matching = 0
    for index in range(len(dataset)):
        target = dataset._target_row(index)
        direction = 0 if str(target["direction"]) == "eng->cmn" else 1
        if direction != args.direction_id:
            continue
        if matching < args.direction_offset:
            matching += 1
            continue
        matching += 1
        selected.append(index)
        if len(selected) >= args.max_samples:
            break
    rows = []
    for index in selected:
        value = dataset[index]
        waveform = value["waveform"].unsqueeze(0).to(device)
        lengths = torch.tensor([waveform.shape[1]], device=device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            bridge_output = bridge(waveform, lengths)
            prompt = build_prompt_embeddings(
                qwen, text_encoder, value["phase3_record"], bridge_output, device
            )
            token_ids, translation = greedy_translation(
                qwen, prompt, tokenizer, args.max_text_tokens
            )
        rows.append(
            {
                "id": value["id"],
                "direction": "eng->cmn" if args.direction_id == 0 else "cmn->eng",
                "translation_ref": value["phase3_record"]["translation"],
                "generated_translation": translation,
                "generated_token_ids": token_ids,
                "generated_end_content": c.TOKEN_END_CONTENT in token_ids,
                "generation_seconds": time.perf_counter() - started,
            }
        )
    payload = {
        "schema_version": "uniss_streamspeech_stage04_b2_text_probe_v1",
        "bridge_checkpoint": str(args.bridge_checkpoint),
        "bridge_step": checkpoint["step"],
        "bridge_validation_phase3_nll": checkpoint["validation_phase3_nll"],
        "direction_id": args.direction_id,
        "direction_offset": args.direction_offset,
        "samples": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"samples": len(rows), "nonempty": sum(bool(row["generated_translation"]) for row in rows)}))


if __name__ == "__main__":
    main()

