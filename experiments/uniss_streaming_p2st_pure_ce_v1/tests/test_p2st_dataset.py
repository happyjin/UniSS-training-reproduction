"""End to end through the real pool, the real audio and the real collator.

The point of these is that nothing is mocked on the path that matters: the
rows come from a pool built by ``build_p2st_pools``, the waveforms come off
disk, and the batch comes out of the base experiment's own
``collate_e2e_family``.  A shape that only holds for synthetic rows would not
tell us whether the trainer will accept this pool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.uniss_streaming_p2st_pure_ce_v1.training.p2st_dataset import (
    P2STPackedFamilyDataset,
    collate_p2st_family,
    p2st_packed_task_to_runtime_item,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    causal_glm_token_count,
)

POOL_ROOT = Path("data/processed/uniss_streaming_p2st_pure_ce_v1/p2st_pool_v5_20260902T135453Z")
MANIFEST = POOL_ROOT / "POOL_MANIFEST.json"


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip(f"p2st pool not built at {MANIFEST}")
    return json.loads(MANIFEST.read_text())


@pytest.mark.parametrize(
    "family", [FAMILY_P2ST_ASR, FAMILY_P2ST_MT, FAMILY_P2ST_TTS]
)
def test_dataset_opens_every_family_from_the_manifest(manifest, family):
    dataset = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=family, load_audio=False
    )
    assert len(dataset) == int(manifest["families"][family]["rows"])
    item = dataset[0]
    assert item["family"] == family
    assert int(item["tokens"].shape[-1]) == int(manifest["seq_length"])
    assert item["teacher_bindings"] == []
    assert item["teacher_posteriors"] == []


def test_asr_rows_load_a_cut_prefix_of_the_real_audio(manifest):
    dataset = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_ASR, load_audio=True
    )
    item = dataset[0]
    rows = item["acoustic_rows"]
    assert rows
    for row in rows:
        cut = int(row["source_pcm_end"])
        assert int(row["waveform_length"]) == cut
        assert int(row["waveform"].numel()) == cut
        assert causal_glm_token_count(cut) == int(row["source_glm_length"])
        # The cut is a strict prefix, never the whole file, or there would be
        # nothing prefix-to-prefix about it.
        assert cut <= 16_000 * 600


def test_the_cut_really_is_a_prefix_of_the_full_file(manifest):
    """Read the file independently and compare the leading samples."""
    import soundfile as sf

    dataset = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_ASR, load_audio=True
    )
    item = dataset[0]
    checked = 0
    for row in item["acoustic_rows"][:8]:
        full, rate = sf.read(str(row["source_audio"]), dtype="float32")
        if full.ndim == 2:
            full = full[:, 0]
        assert int(rate) == 16_000
        cut = int(row["source_pcm_end"])
        assert cut <= len(full)
        expected = torch.as_tensor(full[:cut], dtype=torch.float32)
        assert torch.equal(row["waveform"], expected)
        checked += 1
    assert checked > 0


def test_a_row_with_a_wrong_cut_is_rejected_in_the_dataloader(manifest):
    """A bad pool must fail here, with the sample id, not inside the objective."""
    dataset = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_ASR, load_audio=False
    )
    with dataset.path.open("rb") as handle:
        handle.seek(int(dataset.offsets[0]))
        raw = json.loads(handle.readline())
    raw["acoustic_rows"][0]["source_pcm_end"] += 1280
    with pytest.raises(ValueError, match="does not yield the promised GLM length"):
        p2st_packed_task_to_runtime_item(
            raw, seq_length=int(manifest["seq_length"]), load_audio=True
        )
    raw["acoustic_rows"][0]["source_pcm_end"] = 0
    with pytest.raises(ValueError, match="no audio cut"):
        p2st_packed_task_to_runtime_item(
            raw, seq_length=int(manifest["seq_length"]), load_audio=True
        )


def test_collate_produces_the_tensors_the_trainer_reads(manifest):
    dataset = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_ASR, load_audio=True
    )
    batch = collate_p2st_family([dataset[0], dataset[1]])
    assert batch["family"] == FAMILY_P2ST_ASR
    seq = int(manifest["seq_length"])
    for key in ("tokens", "labels", "loss_kinds", "loss_mask", "position_ids"):
        assert tuple(batch[key].shape) == (2, seq), key
    # These five are what pretrain_e2e_megatron reads to run the frontend and
    # splice its output into the embedding.
    for key in (
        "waveform",
        "waveform_lengths",
        "glm_ids",
        "glm_positions",
        "glm_lengths",
        "acoustic_batch",
    ):
        assert key in batch, key
    rows = int(batch["waveform"].shape[0])
    assert rows == int(batch["waveform_lengths"].shape[0]) == rows
    for index in range(rows):
        length = int(batch["waveform_lengths"][index])
        glm = int(batch["glm_lengths"][index])
        assert causal_glm_token_count(length) == glm
        assert torch.isfinite(batch["waveform"][index, :length]).all()


def test_text_families_collate_without_a_waveform(manifest):
    dataset = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_TTS, load_audio=True
    )
    batch = collate_p2st_family([dataset[0], dataset[1]])
    assert batch["family"] == FAMILY_P2ST_TTS
    assert "waveform" not in batch


def test_collate_refuses_to_mix_families(manifest):
    asr = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_ASR, load_audio=False
    )
    tts = P2STPackedFamilyDataset.from_pool_manifest(
        MANIFEST, family=FAMILY_P2ST_TTS, load_audio=False
    )
    with pytest.raises(ValueError, match="cannot mix p2st task families"):
        collate_p2st_family([asr[0], tts[0]])


def test_supervised_positions_carry_only_this_pool_s_loss_kinds(manifest):
    from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (  # noqa: E501
        LOSS_ASR,
        LOSS_BOUNDARY,
        LOSS_EOS,
        LOSS_MT,
        LOSS_NONE,
        LOSS_SEMANTIC,
    )

    allowed = {
        FAMILY_P2ST_ASR: {LOSS_NONE, LOSS_ASR, LOSS_BOUNDARY, LOSS_EOS},
        FAMILY_P2ST_MT: {LOSS_NONE, LOSS_MT, LOSS_BOUNDARY, LOSS_EOS},
        FAMILY_P2ST_TTS: {LOSS_NONE, LOSS_SEMANTIC, LOSS_BOUNDARY, LOSS_EOS},
    }
    for family, kinds in allowed.items():
        dataset = P2STPackedFamilyDataset.from_pool_manifest(
            MANIFEST, family=family, load_audio=False
        )
        seen = set(int(v) for v in dataset[0]["loss_kinds"].reshape(-1).tolist())
        assert seen <= kinds, (family, sorted(seen - kinds))
        assert seen & (kinds - {LOSS_NONE}), family
