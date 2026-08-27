from experiments.uniss_phasea_rl_trainlong_eval_v1.evaluation.build_protocol import (
    select_records,
)


def _row(index: int, direction: str, duration: int, split: str = "train") -> dict:
    src, tgt = direction.split("->")
    return {
        "episode_id": f"{split}_{index}_{src}_{tgt}",
        "direction": direction,
        "src_lang": src,
        "tgt_lang": tgt,
        "source_audio": __file__,
        "source_duration_ms": duration,
        "teacher_transcription": "source",
        "teacher_translation": "target",
        "component_count": 1,
        "components": [{"sample_id": f"{split}_component_{index}"}],
        "component_ids_sha256": f"components_{split}_{index}",
        "source_audio_sha256": f"audio_{split}_{index}",
    }


def test_selection_uses_longest_formal_rows_without_validation_overlap() -> None:
    train = [
        _row(0, "cmn->eng", 10),
        _row(1, "cmn->eng", 20),
        _row(2, "eng->cmn", 15),
        _row(3, "eng->cmn", 25),
    ]
    valid = [_row(9, "cmn->eng", 30, split="valid")]
    rollout = {
        "episodes": 4,
        "summaries": [{"episode_id": row["episode_id"]} for row in train],
    }
    selected = select_records(train, valid, rollout, per_direction=1)
    assert [row["episode_id"] for row in selected] == [
        "train_1_cmn_eng",
        "train_3_eng_cmn",
    ]
    assert all(row["rl_train_seen"] for row in selected)
