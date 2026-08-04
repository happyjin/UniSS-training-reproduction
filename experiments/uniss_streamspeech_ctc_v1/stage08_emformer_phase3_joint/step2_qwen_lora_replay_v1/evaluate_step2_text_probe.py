#!/usr/bin/env python3
"""Fixed direction-partitioned text probe for a Stage08 Step2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TREE = Path(__file__).resolve().parents[2]
STAGE02 = TREE / "stage02_ctc_probe"
STAGE03 = TREE / "stage03_multitask_encoder"
STAGE03_AR = STAGE03 / "ar_s2tt_v1"
STAGE04 = TREE / "stage04_b2_discrete_bridge"
STAGE07 = TREE / "stage07_end_to_end_eval"
STEP1 = Path(__file__).resolve().parents[1] / "step1_frozen_qwen"
STEP2 = Path(__file__).resolve().parent
for path in (STEP2, STEP1, STAGE04, STAGE03_AR, STAGE03, STAGE02, ROOT, STAGE07):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.checkpoint_io import (
    load_step2_lora_into_qwen,
)
from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.bridge_data import (
    B2BridgeAudioDataset,
)
from experiments.uniss_streamspeech_ctc_v1.stage07_end_to_end_eval.evaluate_text_probe import (
    build_prompt_embeddings,
    greedy_translation,
    select_indices,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.checkpoint_io import (
    load_step1_inference_into_model,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.model import (
    JointEmformerB1,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.lora import (
    inject_lora,
)
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--ctc-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--stage03b-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--stage04-checkpoint", type=Path, required=True)
    parser.add_argument("--stage06-initialize-checkpoint", type=Path, required=True)
    parser.add_argument("--step1-megatron-checkpoint", type=Path, required=True)
    parser.add_argument("--step2-megatron-checkpoint", type=Path, required=True)
    parser.add_argument("--codebook-model", type=Path, required=True)
    parser.add_argument("--phase3-model", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--direction-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--direction-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_model(args, qwen, eng_vocab: int, cmn_vocab: int, device: torch.device):
    qwen_glm_embeddings = qwen.get_input_embeddings().weight[
        c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
    ].detach().float().cpu()
    model, initialization = JointEmformerB1.from_checkpoints(
        stage03b_checkpoint=args.stage03b_checkpoint,
        historical_stage_b_checkpoint=args.historical_stage_b_checkpoint,
        stage04_checkpoint=args.stage04_checkpoint,
        stage06_checkpoint=args.stage06_initialize_checkpoint,
        codebook_model=args.codebook_model,
        qwen_glm_embeddings=qwen_glm_embeddings,
        eng_vocab_size=eng_vocab,
        cmn_vocab_size=cmn_vocab,
        unfreeze_encoder_layers=args.unfreeze_encoder_layers,
    )
    step1 = load_step1_inference_into_model(
        model,
        args.step1_megatron_checkpoint,
        unfreeze_encoder_layers=args.unfreeze_encoder_layers,
    )
    model.requires_grad_(False)
    return model.to(device).eval(), {
        "initialization": initialization.__dict__,
        "step1": step1,
    }


def main() -> None:
    args = parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output_json}")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.phase3_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    qwen.requires_grad_(False)
    targets = [value.strip() for value in args.lora_targets.split(",") if value.strip()]
    injection = inject_lora(
        qwen,
        target_modules=targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    step2 = load_step2_lora_into_qwen(qwen, args.step2_megatron_checkpoint)
    qwen.requires_grad_(False)
    qwen.eval()
    eng_vocab = spm.SentencePieceProcessor(
        model_file=str(args.ctc_tokenizer_dir / "ctc_eng.model")
    ).vocab_size()
    cmn_vocab = spm.SentencePieceProcessor(
        model_file=str(args.ctc_tokenizer_dir / "ctc_cmn.model")
    ).vocab_size()
    model, provenance = load_model(args, qwen, eng_vocab, cmn_vocab, device)
    dataset = B2BridgeAudioDataset(
        args.dataset_index, "valid", args.source_manifest, args.source_offsets
    )
    selected = select_indices(
        dataset, args.direction_id, args.direction_offset, args.max_samples
    )
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    for index in selected:
        value = dataset[index]
        waveform = value["waveform"].unsqueeze(0).to(device)
        lengths = torch.tensor([waveform.shape[1]], device=device)
        source_seconds = waveform.shape[1] / 16000.0
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            bridge_output = model.encode_to_b1(waveform, lengths)
            prompt, speech_tokens = build_prompt_embeddings(
                qwen, text_encoder, value["phase3_record"], bridge_output, device
            )
        torch.cuda.synchronize()
        bridge_seconds = time.perf_counter() - started
        token_ids, translation, qwen_first, qwen_total = greedy_translation(
            qwen, prompt, tokenizer, args.max_text_tokens
        )
        total_seconds = bridge_seconds + qwen_total
        rows.append(
            {
                "id": value["id"],
                "direction": "eng->cmn" if args.direction_id == 0 else "cmn->eng",
                "translation_ref": value["phase3_record"]["translation"],
                "generated_translation": translation,
                "generated_token_ids": token_ids,
                "generated_end_content": c.TOKEN_END_CONTENT in token_ids,
                "source_seconds": source_seconds,
                "speech_tokens": speech_tokens,
                "bridge_seconds": bridge_seconds,
                "qwen_first_token_seconds": qwen_first,
                "generation_seconds": qwen_total,
                "total_seconds": total_seconds,
                "compute_rtf_source": total_seconds / max(source_seconds, 1e-6),
            }
        )
    payload = {
        "schema_version": "uniss_streamspeech_stage08_step2_text_probe_v1",
        "research_only": True,
        "checkpoint": {
            **provenance,
            "step2": step2,
            "iteration": step2["iteration"],
            "lora": injection.__dict__,
        },
        "direction_id": args.direction_id,
        "direction_offset": args.direction_offset,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "samples": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": len(rows),
                "nonempty": sum(bool(row["generated_translation"]) for row in rows),
                "iteration": step2["iteration"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
