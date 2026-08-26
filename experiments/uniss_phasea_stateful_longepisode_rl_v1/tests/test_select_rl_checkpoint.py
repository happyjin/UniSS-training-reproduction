from pathlib import Path

from experiments.uniss_phasea_stateful_longepisode_rl_v1.evaluation.select_rl_checkpoint import (
    parse_validation_metrics,
    select_checkpoint,
)


def validation_line(iteration: int, total: float, ratio: float) -> str:
    return (
        f" validation loss at iteration {iteration} | "
        f"loss/total value: {total:.6E} | loss/policy value: {-total:.6E} | "
        "loss/reference_kl value: 1.000000E-03 | "
        "loss/phase3_replay value: 2.000000E+00 | "
        f"diagnostic/ratio_mean value: {ratio:.6E} | "
        "diagnostic/ratio_clipped_fraction value: 1.000000E-02 |"
    )


def test_parse_ignores_final_validation_suffix() -> None:
    text = validation_line(8, -0.1, 1.0) + "\n" + validation_line(8, -0.5, 1.0).replace(
        " |", " on validation set |", 1
    )
    rows = parse_validation_metrics(text)
    assert len(rows) == 1
    assert rows[0]["metrics"]["loss/total"] == -0.1


def test_selection_records_bands_without_blocking(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        validation_line(8, -0.1, 1.0) + "\n" + validation_line(16, -0.2, 1.4)
    )
    root = tmp_path / "checkpoints"
    for iteration in (8, 16):
        path = root / f"iter_{iteration:07d}"
        path.mkdir(parents=True)
        (path / ".metadata").write_text("ok")
    result = select_checkpoint(log, root)
    assert result["selected_iteration"] == 16
    assert result["selected_quality_annotations"] == [
        "ratio_outside_recording_band"
    ]
