import csv

import pytest

from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.training_audit import (
    parse_gpu,
    parse_training,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.write_report import (
    _f,
    _listening_table,
    _pct,
)


def test_training_audit_parses_megatron_metrics(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "[2026-08-25 22:00:00.000000] iteration        5/    2510 | "
        "loss/asr_ce: 1.500000E+00 | grpo/active: 1.000000E+00 | "
        "number of skipped iterations:   0 | number of nan iterations:   0 |\n"
        "[2026-08-25 22:01:00.000000] iteration       10/    2510 | "
        "loss/asr_ce: 1.000000E+00 | grpo/active: 1.000000E+00 | "
        "number of skipped iterations:   1 | number of nan iterations:   0 |\n",
        encoding="utf-8",
    )
    result = parse_training(log)
    assert result["last_step"] == 10
    assert result["target_steps"] == 2510
    assert result["skipped_iterations"] == 1
    assert result["nan_iterations"] == 0
    assert result["last_metrics"]["loss/asr_ce"] == pytest.approx(1.0)


def test_gpu_audit_filters_to_assigned_active_devices(tmp_path):
    path = tmp_path / "gpu.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "index",
                "memory_used_mib",
                "utilization_gpu_percent",
                "power_draw_w",
                "power_limit_w",
            ]
        )
        writer.writerow(["t0", 0, 500, 0, 80, 700])
        writer.writerow(["t1", 0, 50000, 90, 400, 700])
        writer.writerow(["t1", 1, 60000, 100, 500, 700])
        writer.writerow(["t1", 2, 70000, 100, 600, 700])
    result = parse_gpu(path, {0, 1})
    assert result["active_observations"] == 2
    assert result["utility_mean_percent"] == pytest.approx(95.0)
    assert result["power_mean_w"] == pytest.approx(450.0)
    assert result["memory_max_mib"] == pytest.approx(60000.0)


def test_report_helpers_render_listening_metrics():
    payload = {
        "a1_sft_full_recovery1": {
            "chunks_ms": {
                "320": {
                    "weighted_asr_error_rate": 0.25,
                    "first_audio_source_ms": {"p50": 640.0},
                    "prefinal_audio_rate": 0.75,
                    "healthy_audio_rate": 1.0,
                    "runtime_rtf": 1.5,
                }
            }
        }
    }
    # The production renderer requires all four fixed arms.
    for arm in (
        "a2_g4_full_recovery1",
        "a3_g8_full_recovery1",
        "a4_g8_seed2_full_recovery1",
    ):
        payload[arm] = payload["a1_sft_full_recovery1"]
    lines = _listening_table(payload, "试听")
    text = "\n".join(lines)
    assert "25.00%" in text
    assert "640.0 ms" in text
    assert _f(None) == "—"
    assert _pct(0.125) == "12.50%"
