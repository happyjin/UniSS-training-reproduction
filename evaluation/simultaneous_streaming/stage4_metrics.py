"""Stage4 policy, token-latency, playback-latency, and continuity metrics."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Mapping, Sequence


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def token_latency_metrics(row: Mapping[str, object]) -> dict[str, float | None]:
    source_length = max(1, int(row["source_glm_length"]))
    reference_length = max(1, int(row["reference_target_text_length"]))
    positions: list[float] = []
    emission_ms: list[float] = []
    traces = row["event_trace"]
    assert isinstance(traces, list)
    for event in traces:
        if not isinstance(event, Mapping) or event.get("action") != "write":
            continue
        text_ids = event.get("generated_text_ids") or []
        positions.extend([float(event["source_glm_end"])] * len(text_ids))
        emission_ms.extend([float(event["source_end_ms"])] * len(text_ids))
    target_length = len(positions)
    if target_length == 0:
        return {
            "al_glm_tokens_proxy": None,
            "ap_proxy": None,
            "dal_glm_tokens_proxy": None,
            "laal_glm_tokens_proxy": None,
            "atd_ms_proxy": None,
            "first_write_ms_proxy": None,
        }
    ratio = target_length / source_length
    adaptive_ratio = max(target_length, reference_length) / source_length
    tau = next(
        (index + 1 for index, value in enumerate(positions) if value >= source_length),
        target_length,
    )
    al_lags = [positions[index] - index / ratio for index in range(tau)]
    laal_lags = [positions[index] - index / adaptive_ratio for index in range(tau)]
    dal_positions: list[float] = []
    for index, value in enumerate(positions):
        if index == 0:
            dal_positions.append(value)
        else:
            dal_positions.append(max(value, dal_positions[-1] + 1.0 / ratio))
    dal_lags = [dal_positions[index] - index / ratio for index in range(target_length)]
    source_duration_ms = float(row["source_duration_ms_proxy"])
    ideal_ms = [source_duration_ms * (index + 1) / target_length for index in range(target_length)]
    return {
        "al_glm_tokens_proxy": statistics.fmean(al_lags),
        "ap_proxy": sum(positions) / (source_length * target_length),
        "dal_glm_tokens_proxy": statistics.fmean(dal_lags),
        "laal_glm_tokens_proxy": statistics.fmean(laal_lags),
        "atd_ms_proxy": statistics.fmean(
            max(0.0, actual - ideal) for actual, ideal in zip(emission_ms, ideal_ms)
        ),
        "first_write_ms_proxy": next(
            (
                float(event["source_end_ms"])
                for event in traces
                if isinstance(event, Mapping) and event.get("action") == "write"
            ),
            None,
        ),
    }


def policy_metrics(row: Mapping[str, object]) -> dict[str, float | int | bool]:
    traces = row["event_trace"]
    assert isinstance(traces, list)
    confusion: Counter[tuple[str, str]] = Counter()
    for event in traces:
        assert isinstance(event, Mapping)
        confusion[(str(event["reference_action"]), str(event["action"]))] += 1
    total = sum(confusion.values())
    binary_correct = confusion[("wait", "wait")] + confusion[("write", "write")]

    def cls(name: str) -> tuple[float, float, float]:
        tp = confusion[(name, name)]
        fp = sum(value for (reference, prediction), value in confusion.items() if prediction == name and reference != name)
        fn = sum(value for (reference, prediction), value in confusion.items() if reference == name and prediction != name)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        return precision, recall, f1

    wait_precision, wait_recall, wait_f1 = cls("wait")
    write_precision, write_recall, write_f1 = cls("write")
    reference_first = next(
        (
            float(event["source_end_ms"])
            for event in traces
            if isinstance(event, Mapping) and event.get("reference_action") == "write"
        ),
        None,
    )
    predicted_first = next(
        (
            float(event["source_end_ms"])
            for event in traces
            if isinstance(event, Mapping) and event.get("action") == "write"
        ),
        None,
    )
    consecutive_wait = 0
    max_consecutive_wait = 0
    for event in traces:
        assert isinstance(event, Mapping)
        if event.get("action") == "wait":
            consecutive_wait += 1
            max_consecutive_wait = max(max_consecutive_wait, consecutive_wait)
        else:
            consecutive_wait = 0
    return {
        "events": total,
        "binary_accuracy": safe_div(binary_correct, total),
        "macro_f1": (wait_f1 + write_f1) / 2,
        "wait_precision": wait_precision,
        "wait_recall": wait_recall,
        "wait_f1": wait_f1,
        "write_precision": write_precision,
        "write_recall": write_recall,
        "write_f1": write_f1,
        "premature_write_given_wait": safe_div(
            confusion[("wait", "write")],
            confusion[("wait", "wait")] + confusion[("wait", "write")],
        ),
        "unnecessary_wait_given_write": safe_div(
            confusion[("write", "wait")],
            confusion[("write", "wait")] + confusion[("write", "write")],
        ),
        "final_flush_success": bool(traces and isinstance(traces[-1], Mapping) and traces[-1].get("action") == "write"),
        "forced_actions": int(row.get("forced_action_count", 0)),
        "structural_recoveries": int(row.get("structural_recovery_count", 0)),
        "max_prompt_tokens": int(row.get("max_prompt_tokens", 0)),
        "training_context_exceeded": bool(row.get("training_context_exceeded", False)),
        "invalid_action_events": sum(
            isinstance(event, Mapping) and event.get("raw_action") == "other"
            for event in traces
        ),
        "wait_events": sum(
            isinstance(event, Mapping) and event.get("action") == "wait" for event in traces
        ),
        "write_events": sum(
            isinstance(event, Mapping) and event.get("action") == "write" for event in traces
        ),
        "max_consecutive_wait_events": max_consecutive_wait,
        "reference_first_write_ms": reference_first,
        "predicted_first_write_ms": predicted_first,
        "first_write_delta_ms": (
            None
            if reference_first is None or predicted_first is None
            else predicted_first - reference_first
        ),
        "append_only_revision_events": 0,
    }


def event_compute_metrics(row: Mapping[str, object]) -> dict[str, float | None]:
    traces = row["event_trace"]
    assert isinstance(traces, list)

    def values(name: str) -> list[float]:
        output = []
        for event in traces:
            if not isinstance(event, Mapping):
                continue
            value = event.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                output.append(float(value))
        return output

    action_request = values("action_request_seconds")
    action_queue = values("action_queue_seconds")
    action_ttft = values("action_ttft_seconds")
    write_request = values("write_request_seconds")
    write_queue = values("write_queue_seconds")
    write_ttft = values("write_ttft_seconds")
    codec = values("codec_seconds")
    chunk_act = []
    for event in traces:
        if not isinstance(event, Mapping):
            continue
        chunk_act.append(
            float(event.get("action_request_seconds") or 0.0)
            + float(event.get("write_request_seconds") or 0.0)
            + float(event.get("codec_seconds") or 0.0)
        )
    return {
        "action_request_seconds_mean": mean(action_request),
        "action_request_seconds_p95": percentile(action_request, 0.95),
        "action_queue_seconds_mean": mean(action_queue),
        "action_ttft_seconds_mean": mean(action_ttft),
        "action_ttft_seconds_p95": percentile(action_ttft, 0.95),
        "write_request_seconds_mean": mean(write_request),
        "write_request_seconds_p95": percentile(write_request, 0.95),
        "write_queue_seconds_mean": mean(write_queue),
        "write_ttft_seconds_mean": mean(write_ttft),
        "write_ttft_seconds_p95": percentile(write_ttft, 0.95),
        "codec_event_seconds_mean": mean(codec),
        "codec_event_seconds_p95": percentile(codec, 0.95),
        "chunk_act_seconds_mean": mean(chunk_act),
        "chunk_act_seconds_p95": percentile(chunk_act, 0.95),
    }


def playback_metrics(row: Mapping[str, object], sample_rate: int = 16000) -> dict[str, float | int | None]:
    traces = row["event_trace"]
    assert isinstance(traces, list)
    source_duration = float(row["source_duration_ms_proxy"]) / 1000.0
    nca_playback_end = 0.0
    ca_playback_end = 0.0
    compute_cursor = 0.0
    first_nca: float | None = None
    first_ca: float | None = None
    gaps_nca: list[float] = []
    gaps_ca: list[float] = []
    qwen_seconds = 0.0
    codec_seconds = 0.0
    chunks = 0
    audio_seconds = 0.0
    for event in traces:
        assert isinstance(event, Mapping)
        source_ready = float(event["source_end_ms"]) / 1000.0
        action_seconds = float(event.get("action_request_seconds") or 0.0)
        write_seconds = float(event.get("write_request_seconds") or 0.0)
        event_codec_seconds = float(event.get("codec_seconds") or 0.0)
        compute = action_seconds + write_seconds + event_codec_seconds
        qwen_seconds += action_seconds + write_seconds
        codec_seconds += event_codec_seconds
        compute_cursor = max(compute_cursor, source_ready) + compute
        samples = int(event.get("audio_samples") or 0)
        if samples <= 0:
            continue
        duration = samples / sample_rate
        audio_seconds += duration
        chunks += 1
        if first_nca is None:
            first_nca = source_ready
            first_ca = compute_cursor
            nca_start = source_ready
            ca_start = compute_cursor
        else:
            gaps_nca.append(max(0.0, source_ready - nca_playback_end))
            gaps_ca.append(max(0.0, compute_cursor - ca_playback_end))
            nca_start = max(source_ready, nca_playback_end)
            ca_start = max(compute_cursor, ca_playback_end)
        nca_playback_end = nca_start + duration
        ca_playback_end = ca_start + duration
    return {
        "num_audio_chunks": chunks,
        "generated_audio_seconds": audio_seconds,
        "start_offset_nca_ms": None if first_nca is None else first_nca * 1000.0,
        "start_offset_ca_ms": None if first_ca is None else first_ca * 1000.0,
        "end_offset_nca_ms": None if not chunks else (nca_playback_end - source_duration) * 1000.0,
        "end_offset_ca_ms": None if not chunks else (ca_playback_end - source_duration) * 1000.0,
        "playback_gap_count_nca": sum(value > 0 for value in gaps_nca),
        "playback_gap_sum_nca_ms": sum(gaps_nca) * 1000.0,
        "playback_gap_mean_nca_ms": mean([value * 1000.0 for value in gaps_nca if value > 0]),
        "playback_gap_count_ca": sum(value > 0 for value in gaps_ca),
        "playback_gap_sum_ca_ms": sum(gaps_ca) * 1000.0,
        "playback_gap_mean_ca_ms": mean([value * 1000.0 for value in gaps_ca if value > 0]),
        "qwen_compute_seconds": qwen_seconds,
        "codec_compute_seconds": codec_seconds,
        "rtf_generated_audio": safe_div(qwen_seconds + codec_seconds, audio_seconds),
        "rtf_source_audio": safe_div(qwen_seconds + codec_seconds, source_duration),
    }


def per_sample_metrics(row: Mapping[str, object]) -> dict[str, object]:
    generated_semantic = [int(value) for value in row.get("semantic_values") or []]
    reference_semantic = [int(value) for value in row.get("reference_semantic_values") or []]
    generated_bigrams = Counter(zip(generated_semantic, generated_semantic[1:]))
    reference_bigrams = Counter(zip(reference_semantic, reference_semantic[1:]))
    bigram_overlap = sum((generated_bigrams & reference_bigrams).values())
    generated_unigrams = Counter(generated_semantic)
    reference_unigrams = Counter(reference_semantic)
    unigram_overlap = sum((generated_unigrams & reference_unigrams).values())

    def overlap_f1(overlap: int, generated_count: int, reference_count: int) -> float:
        precision = safe_div(overlap, generated_count)
        recall = safe_div(overlap, reference_count)
        return safe_div(2 * precision * recall, precision + recall)

    max_run = 0
    current_run = 0
    previous: int | None = None
    repeated_tokens = 0
    for value in generated_semantic:
        if value == previous:
            current_run += 1
            repeated_tokens += 1
        else:
            current_run = 1
            previous = value
        max_run = max(max_run, current_run)
    streaming_audio = row.get("streaming_audio") or {}
    audio_metrics = {
        key: value
        for key, value in streaming_audio.items()  # type: ignore[union-attr]
        if key != "boundaries" and isinstance(value, (int, float))
    }
    return {
        **policy_metrics(row),
        **token_latency_metrics(row),
        **playback_metrics(row),
        **event_compute_metrics(row),
        **audio_metrics,
        "generated_text_tokens": len(row.get("generated_text_ids") or []),
        "reference_text_tokens": int(row.get("reference_target_text_length") or 0),
        "text_length_ratio": safe_div(
            len(row.get("generated_text_ids") or []),
            int(row.get("reference_target_text_length") or 0),
        ),
        "generated_semantic_tokens": len(generated_semantic),
        "reference_semantic_tokens": len(reference_semantic),
        "semantic_length_ratio": safe_div(len(generated_semantic), len(reference_semantic)),
        "semantic_aligned_token_accuracy": safe_div(
            sum(left == right for left, right in zip(generated_semantic, reference_semantic)),
            max(len(generated_semantic), len(reference_semantic)),
        ),
        "semantic_unigram_f1": overlap_f1(
            unigram_overlap, len(generated_semantic), len(reference_semantic)
        ),
        "semantic_bigram_f1": overlap_f1(
            bigram_overlap,
            max(0, len(generated_semantic) - 1),
            max(0, len(reference_semantic) - 1),
        ),
        "semantic_max_identical_run": max_run,
        "semantic_adjacent_repeat_rate": safe_div(repeated_tokens, len(generated_semantic)),
        "nonempty_text": bool(row.get("generated_translation")),
        "nonempty_semantic": bool(generated_semantic),
    }


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    enriched = [(row, per_sample_metrics(row)) for row in rows]
    groups: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = defaultdict(list)
    for row, metrics in enriched:
        groups[f"{row['src_lang']}->{row['tgt_lang']}"].append((row, metrics))

    def aggregate_group(values: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]) -> dict[str, object]:
        numeric: dict[str, list[float]] = defaultdict(list)
        for _, metrics in values:
            for key, value in metrics.items():
                if isinstance(value, bool):
                    numeric[key].append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric[key].append(float(value))
        return {
            "samples": len(values),
            "means": {key: statistics.fmean(items) for key, items in sorted(numeric.items())},
            "p50": {key: percentile(items, 0.50) for key, items in sorted(numeric.items())},
            "p95": {key: percentile(items, 0.95) for key, items in sorted(numeric.items())},
        }

    return {
        "schema_version": "simul_uniss_stage4_streaming_metrics_v1",
        "overall": aggregate_group(enriched),
        "directions": {key: aggregate_group(values) for key, values in sorted(groups.items())},
    }
