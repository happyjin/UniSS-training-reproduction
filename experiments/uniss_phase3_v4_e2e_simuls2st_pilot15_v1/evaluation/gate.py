"""Pure metric aggregation and fail-closed Stage-B authorization gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


GATE_SCHEMA = "uniss_phase3_v4_e2e_free_running_gate_v1"
SELECTION_SCHEMA = "uniss_phase3_v4_e2e_free_running_selection_v1"
WORKER_SCHEMA = "uniss_phase3_v4_e2e_free_running_worker_v1"

ASR_LIMITS = {"cmn": 0.210112, "eng": 0.353399}
MT_RETENTION_LIMIT = 0.95
TARGET_COVERAGE_LIMIT = 0.98
SEMANTIC_COVERAGE_LIMIT = 1.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_units(value: str, language: str) -> list[str]:
    normalized = " ".join(str(value).strip().split()).lower()
    if language == "cmn":
        return list("".join(normalized.split()))
    return normalized.split()


def lcs_length(left: Sequence[object], right: Sequence[object]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for first in left:
        current = [0]
        for column, second in enumerate(right, 1):
            current.append(
                previous[column - 1] + 1
                if first == second
                else max(previous[column], current[-1])
            )
        previous = current
    return previous[-1]


def incremental_text_metrics(
    predictions: Sequence[str], reference: str, language: str
) -> dict[str, object]:
    previous: list[str] = []
    rollbacks = 0
    nonempty = 0
    for prediction in predictions:
        current = text_units(prediction, language)
        nonempty += int(bool(current))
        if previous and current[: len(previous)] != previous:
            rollbacks += 1
        previous = current
    reference_units = text_units(reference, language)
    coverage = lcs_length(reference_units, previous) / max(1, len(reference_units))
    return {
        "events": len(predictions),
        "nonempty_events": nonempty,
        "rollback_events": rollbacks,
        "final_hypothesis": predictions[-1] if predictions else "",
        "reference_units": len(reference_units),
        "hypothesis_units": len(previous),
        "coverage": coverage,
    }


def generated_runs(loss_kinds: Sequence[int], none_kind: int = 0) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, kind in enumerate((*loss_kinds, none_kind)):
        active = int(kind) != int(none_kind)
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _weighted_asr(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counters: dict[str, Counter[str]] = {}
    for sample in samples:
        language = str(sample["src_lang"])
        counter = counters.setdefault(language, Counter())
        asr = sample["e_asr"]
        if not isinstance(asr, Mapping):
            raise TypeError("E-ASR worker row is malformed")
        counter["errors"] += int(asr["errors"])
        counter["units"] += int(asr["reference_units"])
        counter["samples"] += 1
        counter["source_rollbacks"] += int(asr.get("source_rollbacks", 0))
        counter["empty_samples"] += int(not str(asr.get("hypothesis", "")).strip())
    output: dict[str, object] = {}
    for language, values in sorted(counters.items()):
        output[language] = {
            **dict(values),
            "error_rate": values["errors"] / max(1, values["units"]),
            "limit": ASR_LIMITS.get(language),
        }
    return output


def _corpus_scores(samples: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    import sacrebleu

    by_direction: dict[str, list[tuple[str, str]]] = {}
    for sample in samples:
        direction = f"{sample['src_lang']}->{sample['tgt_lang']}"
        value = sample[key]
        if not isinstance(value, Mapping):
            raise TypeError(f"{key} worker row is malformed")
        by_direction.setdefault(direction, []).append(
            (str(value["final_hypothesis"]), str(sample["translation_reference"]))
        )
    output: dict[str, object] = {}
    for direction, pairs in sorted(by_direction.items()):
        hypotheses = [pair[0] for pair in pairs]
        references = [[pair[1] for pair in pairs]]
        tokenize = "zh" if direction.endswith("->cmn") else "13a"
        output[direction] = {
            "samples": len(pairs),
            "bleu": sacrebleu.corpus_bleu(
                hypotheses, references, tokenize=tokenize
            ).score,
            "chrf": sacrebleu.corpus_chrf(hypotheses, references).score,
        }
    return output


def _mt_summary(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    candidate = _corpus_scores(samples, "e_mt_gold")
    baseline = _corpus_scores(samples, "phase3_mt_gold")
    directions: dict[str, object] = {}
    for direction in sorted(candidate):
        current = candidate[direction]
        anchor = baseline[direction]
        if not isinstance(current, Mapping) or not isinstance(anchor, Mapping):
            raise TypeError("MT corpus score is malformed")
        directions[direction] = {
            "samples": int(current["samples"]),
            "candidate_bleu": float(current["bleu"]),
            "phase3_bleu": float(anchor["bleu"]),
            "bleu_retention": float(current["bleu"]) / max(1e-9, float(anchor["bleu"])),
            "candidate_chrf": float(current["chrf"]),
            "phase3_chrf": float(anchor["chrf"]),
            "chrf_retention": float(current["chrf"]) / max(1e-9, float(anchor["chrf"])),
        }
    coverage = [float(sample["e_mt_gold"]["coverage"]) for sample in samples]  # type: ignore[index]
    rollback = sum(int(sample["e_mt_gold"]["rollback_events"]) for sample in samples)  # type: ignore[index]
    return {
        "directions": directions,
        "target_coverage_min": min(coverage) if coverage else 0.0,
        "target_coverage_mean": sum(coverage) / max(1, len(coverage)),
        "target_rollback_events": rollback,
    }


def _s2s_summary(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = [sample["e_s2s_free"] for sample in samples if sample.get("e_s2s_free")]
    if not values:
        return {"samples": 0}
    semantic = [float(value["semantic_coverage"]) for value in values]  # type: ignore[index]
    return {
        "samples": len(values),
        "semantic_coverage_min": min(semantic),
        "semantic_coverage_mean": sum(semantic) / len(semantic),
        "invalid_semantic_tokens": sum(
            int(value["invalid_semantic_tokens"]) for value in values  # type: ignore[index]
        ),
        "target_text_before_source_eos": sum(
            bool(value["target_text_before_source_eos"]) for value in values  # type: ignore[index]
        ),
        "target_semantic_before_source_eos": sum(
            bool(value["target_semantic_before_source_eos"]) for value in values  # type: ignore[index]
        ),
        "non_silent_pcm": sum(
            bool(value["audio"]["non_silent"]) for value in values  # type: ignore[index]
        ),
        "malformed_segments": sum(
            int(value["malformed_segments"]) for value in values  # type: ignore[index]
        ),
        "source_rollback_events": sum(
            int(value.get("source_rollback_events", 0)) for value in values  # type: ignore[union-attr]
        ),
        "target_rollback_events": sum(
            int(value.get("target_rollback_events", 0)) for value in values  # type: ignore[union-attr]
        ),
    }


def build_gate(
    *,
    canary_report: str | Path,
    selection: str | Path,
    worker_reports: Sequence[str | Path],
    candidate_checkpoint: str | Path,
    candidate_hf: str | Path,
    v1_initialization: str | Path,
) -> dict[str, object]:
    canary = json.loads(Path(canary_report).read_text(encoding="utf-8"))
    if canary.get("status") != "passed" or bool(canary.get("formal_training_authorized")):
        raise ValueError("free-running gate requires a passed, unauthorized canary report")
    selection_value = json.loads(Path(selection).read_text(encoding="utf-8"))
    if selection_value.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unexpected free-running selection schema")
    workers = [json.loads(Path(path).read_text(encoding="utf-8")) for path in worker_reports]
    if not workers or any(value.get("schema_version") != WORKER_SCHEMA for value in workers):
        raise ValueError("free-running worker report schema differs")
    expected_workers = int(workers[0]["num_workers"])
    indices = sorted(int(value["worker_index"]) for value in workers)
    if len(workers) != expected_workers or indices != list(range(expected_workers)):
        raise ValueError("free-running worker set is incomplete")
    samples = [sample for worker in workers for sample in worker["samples"]]
    selected_ids = {str(value["sample_id"]) for value in selection_value["records"]}
    observed_ids = {str(value["sample_id"]) for value in samples}
    if len(samples) != len(selected_ids) or observed_ids != selected_ids:
        raise ValueError("free-running samples do not exactly cover the fixed selection")

    asr = _weighted_asr(samples)
    mt = _mt_summary(samples)
    s2s = _s2s_summary(samples)
    checks: dict[str, bool] = {}
    for language, limit in ASR_LIMITS.items():
        value = asr.get(language)
        checks[f"e_asr_{language}_retained"] = bool(
            isinstance(value, Mapping)
            and int(value["samples"]) > 0
            and float(value["error_rate"]) <= limit
        )
    checks["e_asr_source_rollback_zero"] = all(
        int(value["source_rollbacks"]) == 0  # type: ignore[index]
        for value in asr.values()
    )
    directions = mt["directions"]
    checks["e_mt_all_directions_present"] = bool(directions) and len(directions) == 2
    checks["e_mt_bleu_retention"] = bool(directions) and all(
        float(value["bleu_retention"]) >= MT_RETENTION_LIMIT
        for value in directions.values()  # type: ignore[union-attr]
    )
    checks["e_mt_chrf_retention"] = bool(directions) and all(
        float(value["chrf_retention"]) >= MT_RETENTION_LIMIT
        for value in directions.values()  # type: ignore[union-attr]
    )
    checks["e_mt_target_rollback_zero"] = int(mt["target_rollback_events"]) == 0
    checks["e_mt_target_coverage"] = float(mt["target_coverage_min"]) >= TARGET_COVERAGE_LIMIT
    s2s_count = int(s2s.get("samples", 0))
    checks["e_s2s_present"] = s2s_count >= 4
    checks["e_s2s_semantic_coverage"] = (
        s2s_count > 0
        and float(s2s.get("semantic_coverage_min", 0.0)) >= SEMANTIC_COVERAGE_LIMIT
    )
    checks["e_s2s_semantic_tokens_valid"] = int(s2s.get("invalid_semantic_tokens", 1)) == 0
    checks["e_s2s_pre_eos_target_text"] = int(
        s2s.get("target_text_before_source_eos", 0)
    ) == s2s_count
    checks["e_s2s_pre_eos_target_semantic"] = int(
        s2s.get("target_semantic_before_source_eos", 0)
    ) == s2s_count
    checks["e_s2s_non_silent_pcm"] = int(s2s.get("non_silent_pcm", 0)) == s2s_count
    checks["e_s2s_rollback_zero"] = (
        int(s2s.get("source_rollback_events", 1)) == 0
        and int(s2s.get("target_rollback_events", 1)) == 0
    )
    checks["e_s2s_structure_valid"] = int(s2s.get("malformed_segments", 1)) == 0
    passed = all(checks.values())
    return {
        "schema_version": GATE_SCHEMA,
        "status": "passed" if passed else "failed",
        "formal_training_authorized": passed,
        "canary_report": {
            "path": str(Path(canary_report).resolve()),
            "sha256": sha256_file(canary_report),
        },
        "selection": {
            "path": str(Path(selection).resolve()),
            "sha256": sha256_file(selection),
            "samples": len(samples),
        },
        "candidate": {
            "checkpoint": str(Path(candidate_checkpoint).resolve()),
            "hf_model": str(Path(candidate_hf).resolve()),
        },
        "formal_initialization": str(Path(v1_initialization).resolve()),
        "checks": checks,
        "metrics": {"e_asr": asr, "e_mt_gold": mt, "e_s2s_free": s2s},
        "worker_reports": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in worker_reports
        ],
        "authorization_rule": (
            "A passed canary checkpoint is used only to validate the implementation. "
            "The formal run must restart from the immutable V1 compound checkpoint."
        ),
    }


def write_new_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


__all__ = [
    "ASR_LIMITS",
    "GATE_SCHEMA",
    "SELECTION_SCHEMA",
    "WORKER_SCHEMA",
    "build_gate",
    "generated_runs",
    "incremental_text_metrics",
    "lcs_length",
    "sha256_file",
    "text_units",
    "write_new_json",
]
