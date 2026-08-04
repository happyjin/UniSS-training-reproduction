"""Load the frozen Stage09 bundle from Megatron Stage08 checkpoints."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = Path(__file__).resolve().parents[1]
for path in (
    ROOT,
    TREE / "stage02_ctc_probe",
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder/ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
    TREE / "stage07_end_to_end_eval",
    TREE / "stage08_emformer_phase3_joint/step1_frozen_qwen",
    TREE / "stage08_emformer_phase3_joint/step2_qwen_lora_replay_v1",
):
    sys.path.insert(0, str(path))

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.checkpoint_io import (
    load_step1_trainable_into_model,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step1_frozen_qwen.model import (
    JointEmformerB1,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.checkpoint_io import (
    load_step2_lora_into_qwen,
)
from experiments.uniss_streamspeech_ctc_v1.stage08_emformer_phase3_joint.step2_qwen_lora_replay_v1.lora import (
    inject_lora,
)
from training import constants_uniss as c

from .config import Stage09Config


@dataclass(frozen=True)
class Stage09Provenance:
    research_only: bool
    step1: dict[str, object]
    step2: dict[str, object]
    lora: dict[str, object]
    joint_initialization: dict[str, object]


@dataclass
class Stage09Bundle:
    joint: JointEmformerB1
    qwen: object
    tokenizer: object
    processors: dict[str, spm.SentencePieceProcessor]
    provenance: Stage09Provenance
    device: torch.device


def load_stage09_bundle(config: Stage09Config) -> Stage09Bundle:
    config.validate()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.phase3_model, local_files_only=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        config.phase3_model,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    qwen.requires_grad_(False)
    qwen_glm_embeddings = qwen.get_input_embeddings().weight[
        c.GLM_SEMANTIC_OFFSET : c.GLM_SEMANTIC_OFFSET + c.GLM_SEMANTIC_SIZE
    ].detach().float().cpu()
    processors = {
        language: spm.SentencePieceProcessor(
            model_file=str(config.tokenizer_dir / f"ctc_{language}.model")
        )
        for language in ("eng", "cmn")
    }
    joint, initialization = JointEmformerB1.from_checkpoints(
        stage03b_checkpoint=config.stage03b_checkpoint,
        historical_stage_b_checkpoint=config.historical_stage_b_checkpoint,
        stage04_checkpoint=config.stage04_checkpoint,
        stage06_checkpoint=config.stage06_checkpoint,
        codebook_model=config.codebook_model,
        qwen_glm_embeddings=qwen_glm_embeddings,
        eng_vocab_size=processors["eng"].vocab_size(),
        cmn_vocab_size=processors["cmn"].vocab_size(),
        unfreeze_encoder_layers=4,
    )
    step1 = load_step1_trainable_into_model(joint, config.step1_checkpoint)
    joint.requires_grad_(False)
    joint.to(device).eval()
    lora = inject_lora(
        qwen,
        target_modules=("q_proj", "v_proj"),
        rank=8,
        alpha=16.0,
        dropout=0.05,
    )
    step2 = load_step2_lora_into_qwen(qwen, config.step2_checkpoint)
    qwen.requires_grad_(False)
    qwen.eval()
    provenance = Stage09Provenance(
        research_only=True,
        step1=step1,
        step2=step2,
        lora=asdict(lora),
        joint_initialization=asdict(initialization),
    )
    return Stage09Bundle(joint, qwen, tokenizer, processors, provenance, device)
