from experiments.uniss_phasea_event_constrained_grpo_long_v2.data.build_action_packs import (
    action_sample,
    finalize_pack,
)
from training import constants_uniss as c


def row(action):
    return {
        "sample_id": "x",
        "chunk_end_ms": 320,
        "tgt_lang": "eng",
        "speaker_global": list(range(32)),
        "causal_source_glm": [1, 2, 3],
        "natural_action_target": action,
        "target_text_delta_ids": [100] if action == "WRITE" else [],
    }


def test_action_mask_predicts_wait_or_write_token():
    for action, token in (("READ", c.TOKEN_WAIT_READ), ("WRITE", c.TOKEN_WRITE_GENERATE)):
        sample = action_sample(row(action))
        position = sample["action_mask"].index(1.0)
        assert sample["labels"][position] == token


def test_finalize_pack_preserves_separate_masks():
    sample = action_sample(row("WRITE"))
    pack = {
        key: list(sample[key])
        for key in (
            "tokens",
            "labels",
            "position_ids",
            "response_mask",
            "action_mask",
            "replay_mask",
            "family_ids",
        )
    }
    pack["sample_boundaries"] = [[0, len(sample["tokens"])]]
    pack["identities"] = [sample["identity"]]
    result = finalize_pack(pack, 128)
    assert result["action_tokens"] == 1
    assert result["response_tokens"] > 0
    assert len(result["tokens"]) == 128

