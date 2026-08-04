"""Immutable paths for the isolated Stage09 research runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Stage09Config:
    dataset_index: Path = ROOT / "data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json"
    source_manifest: Path = ROOT / "data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl"
    tokenizer_dir: Path = ROOT / "data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers"
    stage03b_checkpoint: Path = ROOT / "checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt"
    historical_stage_b_checkpoint: Path = ROOT / "checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt"
    stage04_checkpoint: Path = ROOT / "checkpoints/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1/best.pt"
    stage06_checkpoint: Path = ROOT / "checkpoints/uniss_streamspeech_ctc_v1/stage06_b1_megatron_v2/iter_0000600"
    step1_checkpoint: Path = ROOT / "checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_p3w2_zhen1p25_v1/iter_0000350"
    step2_checkpoint: Path = ROOT / "checkpoints/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_r8_replay30_research_v1/iter_0000100"
    codebook_model: Path = ROOT / "pretrained_models/UniSS/glm4_tokenizer"
    phase3_model: Path = ROOT / "checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf"
    device: str = "cuda:0"
    confirmations: int = 2
    lagging_k: int = 0

    @property
    def source_offsets(self) -> Path:
        return Path(f"{self.source_manifest}.offsets.bin")

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, Path) and not value.exists():
                raise FileNotFoundError(f"missing Stage09 {name}: {value}")
        if self.confirmations <= 0:
            raise ValueError("confirmations must be positive")
        if self.lagging_k < 0:
            raise ValueError("lagging_k must be non-negative")
