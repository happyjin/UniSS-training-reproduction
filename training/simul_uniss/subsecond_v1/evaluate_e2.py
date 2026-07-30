"""Evaluate Stage-B E2 frontend and fixed-wait-k latency on UniST audio."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torchaudio

from training.simul_uniss.jsonl_index import load_index
from training.simul_uniss.subsecond_v1.model import CausalAudioStudentV2, StageBModelConfig
from training.simul_uniss.subsecond_v1.streaming import (
    CausalStudentStreamingSession,
    TokenEmission,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _first_reference_match(
    emissions: list[TokenEmission], reference: list[int], stability_threshold: float | None = None
) -> TokenEmission | None:
    if not reference:
        return None
    for emission in emissions:
        if emission.token_id != reference[0]:
            continue
        if stability_threshold is None or emission.stability_probability >= stability_threshold:
            return emission
    return None


def _first_at_or_after(
    emission: TokenEmission | None, *, wait_k: int, chunk_ms: int
) -> tuple[float | None, float | None]:
    if emission is None:
        return None, None
    scheduled = float(wait_k * chunk_ms)
    nca = max(scheduled, emission.nca_ms)
    ca = max(nca, emission.ca_ms)
    return nca, ca


def _duration_bucket(duration_ms: float) -> str:
    if duration_ms <= 4000:
        return "<=4s"
    if duration_ms <= 8000:
        return "4-8s"
    return ">8s"


def evaluate_record(
    model: CausalAudioStudentV2,
    item: dict[str, Any],
    *,
    chunk_ms: int,
    wait_ks: list[int],
    stability_threshold: float,
) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(item["source_audio"])
    waveform = waveform[:1]
    if sample_rate != model.config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, model.config.sample_rate)
    waveform = waveform.squeeze(0)
    chunk_samples = round(chunk_ms * model.config.sample_rate / 1000)
    session = CausalStudentStreamingSession(model)
    if waveform.numel() == 0:
        session.feed(waveform, final=True)
    else:
        for start in range(0, waveform.numel(), chunk_samples):
            end = min(waveform.numel(), start + chunk_samples)
            session.feed(waveform[start:end], final=end == waveform.numel())

    reference = [int(value) for value in item["source_glm"]]
    predicted = [event.token_id for event in session.glm_emissions]
    first_predicted = session.glm_emissions[0] if session.glm_emissions else None
    first_correct = _first_reference_match(session.glm_emissions, reference)
    first_stable = _first_reference_match(
        session.glm_emissions, reference, stability_threshold=stability_threshold
    )
    result: dict[str, Any] = {
        "id": str(item["id"]),
        "src_lang": str(item["src_lang"]),
        "tgt_lang": str(item["tgt_lang"]),
        "direction": f"{item['src_lang']}->{item['tgt_lang']}",
        "duration_ms": float(waveform.numel() * 1000 / model.config.sample_rate),
        "duration_bucket": _duration_bucket(waveform.numel() * 1000 / model.config.sample_rate),
        "reference_glm_tokens": len(reference),
        "predicted_glm_tokens": len(predicted),
        "glm_edit_distance": _edit_distance(predicted, reference),
        "glm_token_agreement": max(
            0.0, 1.0 - _edit_distance(predicted, reference) / max(1, len(reference))
        ),
        "first_predicted_glm_nca_ms": None if first_predicted is None else first_predicted.nca_ms,
        "first_predicted_glm_ca_ms": None if first_predicted is None else first_predicted.ca_ms,
        "first_reference_match_nca_ms": None if first_correct is None else first_correct.nca_ms,
        "first_reference_match_ca_ms": None if first_correct is None else first_correct.ca_ms,
        "first_stable_reference_match_nca_ms": None if first_stable is None else first_stable.nca_ms,
        "first_stable_reference_match_ca_ms": None if first_stable is None else first_stable.ca_ms,
        "first_predicted_glm_token": None if first_predicted is None else first_predicted.token_id,
        "reference_first_glm_token": None if not reference else reference[0],
        "active_rtf": session.active_rtf,
        "active_compute_ms": session.active_compute_seconds * 1000.0,
        "final_backlog_ms": session.final_backlog_ms,
        "realtime_violation": session.final_backlog_ms > chunk_ms,
        "chunk_act_mean_ms": statistics.fmean(
            event.compute_ms for event in session.chunk_events
        ),
        "chunk_act_p95_ms": _percentile(
            (event.compute_ms for event in session.chunk_events), 0.95
        ),
        "chunks": len(session.chunk_events),
        "output_frames": session.summary()["output_frames"],
    }
    for wait_k in wait_ks:
        for label, emission in (
            ("predicted", first_predicted),
            ("correct", first_correct),
            ("stable", first_stable),
        ):
            nca, ca = _first_at_or_after(emission, wait_k=wait_k, chunk_ms=chunk_ms)
            result[f"wait_k{wait_k}_{label}_first_write_nca_ms"] = nca
            result[f"wait_k{wait_k}_{label}_first_write_ca_ms"] = ca
    return result


def _metric_summary(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [float(row[name]) for row in records if row.get(name) is not None]
    return {
        "coverage": len(values) / max(1, len(records)),
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "under_1000_fraction_all": sum(value < 1000.0 for value in values)
        / max(1, len(records)),
    }


def aggregate_records(
    records: list[dict[str, Any]], *, chunk_ms: int, right_context_ms: int, wait_ks: list[int]
) -> dict[str, Any]:
    metric_names = [
        "first_predicted_glm_nca_ms",
        "first_predicted_glm_ca_ms",
        "first_reference_match_nca_ms",
        "first_reference_match_ca_ms",
        "first_stable_reference_match_nca_ms",
        "first_stable_reference_match_ca_ms",
        "active_rtf",
        "final_backlog_ms",
        "chunk_act_mean_ms",
        "chunk_act_p95_ms",
        "glm_token_agreement",
    ]
    for wait_k in wait_ks:
        for label in ("predicted", "correct", "stable"):
            metric_names.extend(
                (
                    f"wait_k{wait_k}_{label}_first_write_nca_ms",
                    f"wait_k{wait_k}_{label}_first_write_ca_ms",
                )
            )
    directions = sorted({str(row["direction"]) for row in records})
    buckets = ("<=4s", "4-8s", ">8s")
    selected = [
        "first_predicted_glm_nca_ms",
        "first_stable_reference_match_nca_ms",
        "active_rtf",
        "glm_token_agreement",
    ]
    selected.extend(f"wait_k{k}_stable_first_write_ca_ms" for k in wait_ks)
    return {
        "schema_version": "simul_uniss_subsecond_e2_summary_v1",
        "records": len(records),
        "chunk_ms": chunk_ms,
        "right_context_ms": right_context_ms,
        "wait_ks": wait_ks,
        "metrics": {name: _metric_summary(records, name) for name in metric_names},
        "realtime_violation_fraction": sum(bool(row["realtime_violation"]) for row in records)
        / max(1, len(records)),
        "by_direction": {
            direction: {
                "records": sum(row["direction"] == direction for row in records),
                "metrics": {
                    name: _metric_summary(
                        [row for row in records if row["direction"] == direction], name
                    )
                    for name in selected
                },
            }
            for direction in directions
        },
        "by_duration": {
            bucket: {
                "records": sum(row["duration_bucket"] == bucket for row in records),
                "metrics": {
                    name: _metric_summary(
                        [row for row in records if row["duration_bucket"] == bucket], name
                    )
                    for name in selected
                },
            }
            for bucket in buckets
        },
    }


def _report(summaries: list[dict[str, Any]], checkpoint: Path, manifest: Path) -> str:
    lines = [
        "# Stage B E2 真流式延迟评估",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- manifest: `{manifest}`",
        "- 口径：PCM 增量输入、causal log-Mel、Emformer cache；NCA 只计音频时间，CA 加入实测计算排队。",
        "- 当前 E2 只评估 causal frontend / First WRITE 下界，不等于 Qwen + BiCodec 端到端首个可播放翻译音频。",
        "",
        "## 汇总",
        "",
        "| chunk/right | records | First GLM NCA p50/p95 | stable首token覆盖 | wait-k2 stable First WRITE CA p50/p95 | <1s比例 | active RTF p50 | GLM agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary["metrics"]
        first = metrics["first_predicted_glm_nca_ms"]
        stable = metrics["first_stable_reference_match_nca_ms"]
        wait = metrics["wait_k2_stable_first_write_ca_ms"]
        rtf = metrics["active_rtf"]
        agreement = metrics["glm_token_agreement"]
        lines.append(
            f"| {summary['chunk_ms']}/{summary['right_context_ms']} ms | {summary['records']} | "
            f"{first['p50']:.1f}/{first['p95']:.1f} ms | {stable['coverage']:.1%} | "
            f"{wait['p50'] if wait['p50'] is not None else float('nan'):.1f}/"
            f"{wait['p95'] if wait['p95'] is not None else float('nan'):.1f} ms | "
            f"{wait['under_1000_fraction_all']:.1%} | {rtf['p50']:.4f} | {agreement['mean']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 判定规则",
            "",
            "- `First predicted GLM` 很低只证明模型较早发出 token，可能是错误 token。",
            "- 方案是否满足 `<1 s`，至少应查看 `wait-k stable First WRITE CA` 的覆盖率和比例。",
            "- Stage B 独立质量门要求最终 GLM token agreement ≥90%；未通过时不能据此声称端到端同传已满足1秒且质量合格。",
            "- 真正端到端 `<1 s` 还必须在 E4/E5 接入 Qwen micro-WRITE、Streaming BiCodec、网络和播放器缓冲后重新测量 Useful First Audio CA。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--configs", default="160:80")
    parser.add_argument("--wait-k", default="2,3")
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--stability-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("gloo")
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = load_index(manifest)
    if offsets is None or not offsets:
        raise ValueError(f"missing or empty index for {manifest}")
    total = min(len(offsets), args.limit_records or len(offsets))
    wait_ks = [int(value) for value in args.wait_k.split(",") if value]
    configs = []
    for value in args.configs.split(","):
        chunk_ms, right_ms = (int(part) for part in value.split(":"))
        if chunk_ms % 40 or right_ms % 40:
            raise ValueError("chunk and right context must be multiples of 40 ms")
        configs.append((chunk_ms, right_ms))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model_state = checkpoint["model"]
    base_config = dict(checkpoint["model_config"])
    del checkpoint
    summaries: list[dict[str, Any]] = []
    for chunk_ms, right_ms in configs:
        config_name = f"chunk{chunk_ms}_right{right_ms}"
        model_config = StageBModelConfig.from_dict(
            {
                **base_config,
                "segment_frames": chunk_ms // 40,
                "right_context_frames": right_ms // 40,
            }
        )
        model = CausalAudioStudentV2(model_config).to(device).eval()
        model.load_state_dict(model_state, strict=True)
        rank_path = output_dir / config_name / f"records.rank{rank:02d}.jsonl"
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        processed = 0
        with manifest.open("rb") as handle, rank_path.open("w", encoding="utf-8") as output:
            for index in range(rank, total, world_size):
                handle.seek(offsets[index])
                item = json.loads(handle.readline())
                record = evaluate_record(
                    model,
                    item,
                    chunk_ms=chunk_ms,
                    wait_ks=wait_ks,
                    stability_threshold=args.stability_threshold,
                )
                record["record_index"] = index
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
                if processed % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "config": config_name,
                                "processed": processed,
                                "rank_total": (max(0, total - 1 - rank) // world_size + 1),
                            }
                        ),
                        flush=True,
                    )
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            records: list[dict[str, Any]] = []
            for shard in sorted((output_dir / config_name).glob("records.rank*.jsonl")):
                with shard.open("r", encoding="utf-8") as handle:
                    records.extend(json.loads(line) for line in handle if line.strip())
            records.sort(key=lambda row: int(row["record_index"]))
            summary = aggregate_records(
                records,
                chunk_ms=chunk_ms,
                right_context_ms=right_ms,
                wait_ks=wait_ks,
            )
            summary["checkpoint"] = str(checkpoint_path)
            summary["manifest"] = str(manifest)
            _atomic_json(output_dir / config_name / "summary.json", summary)
            summaries.append(summary)
            print(json.dumps({"config": config_name, "summary": summary}, sort_keys=True), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        _atomic_json(output_dir / "summary.json", {"configs": summaries})
        (output_dir / "REPORT.md").write_text(
            _report(summaries, checkpoint_path, manifest), encoding="utf-8"
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
