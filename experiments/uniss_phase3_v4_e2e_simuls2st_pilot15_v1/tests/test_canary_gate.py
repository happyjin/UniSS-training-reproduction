from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.build_task_pools import (
    BUILD_SCHEMA,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.canary_gate import (
    EXPECTED_RUNS,
    REQUIRED_FAMILY_COUNTS,
    _write_new_json,
    build_preflight,
    finalize_canaries,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    TASK_FAMILIES,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _build_report(tmp_path: Path, split: str) -> Path:
    quality = _write_json(tmp_path / f"{split}_QUALITY_GATE.json", {"status": "passed"})
    families = {}
    for family in TASK_FAMILIES:
        data = tmp_path / f"{split}_{family}.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        index = tmp_path / f"{split}_{family}.offsets.bin"
        index.write_bytes(b"12345678")
        counts = {"supervised_tokens": 17}
        counts.update({name: 3 for name in REQUIRED_FAMILY_COUNTS[family]})
        families[family] = {
            "family": family,
            "records": 1,
            "bytes": data.stat().st_size,
            "path": str(data),
            "counts": counts,
            "index": {"binary_path": str(index)},
        }
    return _write_json(
        tmp_path / f"{split}_BUILD_COMPLETE.json",
        {
            "schema_version": BUILD_SCHEMA,
            "status": "passed",
            "split": split,
            "seq_length": 18_000,
            "records": 2,
            "quality_gate": str(quality),
            "families": families,
        },
    )


def _teacher_audit(tmp_path: Path, name: str) -> Path:
    output = tmp_path / f"{name}.jsonl"
    output.write_text("{}\n{}\n", encoding="utf-8")
    return _write_json(
        tmp_path / f"{name}_AUDIT.json",
        {
            "status": "passed",
            "selection_start": 0,
            "selection_stop": 2,
            "output": str(output),
            "output_bytes": output.stat().st_size,
        },
    )


def _preflight_inputs(tmp_path: Path) -> dict[str, object]:
    return {
        "data_run_id": "data-v1",
        "task_pool_run_id": "pool-v1",
        "teacher_run_id": "teacher-v1",
        "train_report": _build_report(tmp_path, "train"),
        "valid_report": _build_report(tmp_path, "valid"),
        "v1_train_audit": _teacher_audit(tmp_path, "v1_train"),
        "phase3_train_audit": _teacher_audit(tmp_path, "phase3_train"),
        "v1_valid_audit": _teacher_audit(tmp_path, "v1_valid"),
        "phase3_valid_audit": _teacher_audit(tmp_path, "phase3_valid"),
        "gold_gate": _write_json(
            tmp_path / "GOLD_TRAJECTORY_GATE.json",
            {"status": "passed", "formal_training_authorized": False},
        ),
    }


def test_canary_preflight_requires_all_active_family_counts(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    value = build_preflight(**inputs)
    assert value["status"] == "passed"
    assert value["formal_training_authorized"] is False
    train = json.loads(Path(inputs["train_report"]).read_text(encoding="utf-8"))
    train["families"]["streaming_asr_event"]["counts"][
        "teacher:v1_asr:positions"
    ] = 0
    _write_json(tmp_path / "bad_train.json", train)
    inputs["train_report"] = tmp_path / "bad_train.json"
    with pytest.raises(ValueError, match="v1_asr"):
        build_preflight(**inputs)


def test_canary_preflight_cannot_authorize_formal_training(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    _write_json(
        Path(inputs["gold_gate"]),
        {"status": "passed", "formal_training_authorized": True},
    )
    with pytest.raises(ValueError, match="unauthorized"):
        build_preflight(**inputs)


def test_canary_reports_refuse_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    _write_new_json(output, {"status": "passed"})
    with pytest.raises(FileExistsError):
        _write_new_json(output, {"status": "changed"})


def test_canary_finalizer_requires_six_clean_checkpointed_runs(tmp_path: Path) -> None:
    inputs = _preflight_inputs(tmp_path)
    preflight = _write_json(tmp_path / "PREFLIGHT.json", build_preflight(**inputs))
    results = []
    for name, family, train_iters in EXPECTED_RUNS:
        log = tmp_path / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("training completed with finite losses\n", encoding="utf-8")
        Path(f"{log}.command").write_text("torchrun --e2e-smoke\n", encoding="utf-8")
        gpu = tmp_path / "logs" / f"{name}.gpu.csv"
        gpu.write_text(
            "timestamp,index,memory_used_mib,utilization_gpu_percent,power_draw_w,power_limit_w\n"
            "now,0,100,99,500,700\n",
            encoding="utf-8",
        )
        save = tmp_path / "checkpoints" / name
        save.mkdir(parents=True, exist_ok=True)
        (save / "latest_checkpointed_iteration.txt").write_text(
            str(train_iters), encoding="utf-8"
        )
        results.append(
            {
                "name": name,
                "family": family,
                "train_iters": train_iters,
                "exit_code": 0,
                "log": str(log),
                "gpu_csv": str(gpu),
                "save_dir": str(save),
                "tensorboard_dir": str(tmp_path / "runs" / name),
            }
        )
    result_path = tmp_path / "RUN_RESULTS.jsonl"
    result_path.write_text(
        "".join(json.dumps(value) + "\n" for value in results), encoding="utf-8"
    )
    report = finalize_canaries(preflight, result_path)
    assert report["status"] == "passed"
    assert report["formal_training_authorized"] is False
    assert len(report["runs"]) == 6
    results[0]["log"] = str(tmp_path / "logs" / "fatal.log")
    Path(results[0]["log"]).write_text("RuntimeError: broken\n", encoding="utf-8")
    Path(f"{results[0]['log']}.command").write_text("torchrun\n", encoding="utf-8")
    result_path.write_text(
        "".join(json.dumps(value) + "\n" for value in results), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="fatal pattern"):
        finalize_canaries(preflight, result_path)
