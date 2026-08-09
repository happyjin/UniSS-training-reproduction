#!/usr/bin/env python3
"""Step 0 - split Stage09->Stage11 streaming wall clock into pipeline stages.

The recommendation document makes every later step conditional on this measurement:
if Qwen autoregressive decoding dominates, the NAR CTC speech head is the right first
change; if BiCodec decoding dominates, an incremental codec comes first; if Qwen prefill
dominates, the lambda-shaped KV cache comes first.

Two passes are run over the same samples. The baseline pass is completely unpatched and
produces the honest wall clock and RTF. The instrumented pass installs the timing probes
and produces the attribution. Reporting both makes the instrumentation overhead visible
instead of hiding it inside the numbers that drive the decision.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "experiments/uniss_streamspeech_ctc_v1"
for _path in (
    ROOT,
    TREE / "stage02_ctc_probe",
    TREE / "stage03_multitask_encoder",
    TREE / "stage03_multitask_encoder/ar_s2tt_v1",
    TREE / "stage04_b2_discrete_bridge",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import sacrebleu  # noqa: E402

from experiments.simul_s2st_route_v1.common.instrumentation import CallTreeTimer  # noqa: E402
from experiments.simul_s2st_route_v1.step0_rtf_decomposition.probe import (  # noqa: E402
    LABEL_DESCRIPTIONS,
    TOP_LEVEL_LABELS,
    install_bundle_probes,
    install_pipeline_probes,
    install_session_probes,
)
from experiments.uniss_streamspeech_ctc_v1.stage04_b2_discrete_bridge.bridge_data import (  # noqa: E402
    B2BridgeAudioDataset,
)
from experiments.uniss_streamspeech_ctc_v1.stage09_online_runtime.config import (  # noqa: E402
    Stage09Config,
)
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.config import (  # noqa: E402
    Stage11Config,
)
from experiments.uniss_streamspeech_ctc_v1.stage11_streaming_audio.engine import (  # noqa: E402
    Stage11Engine,
)

SCHEMA_VERSION = "simul_s2st_route_v1_step0_rtf_decomposition_v1"
SAMPLE_RATE = 16_000
INGRESS_MS = 160
DIRECTIONS = ("eng->cmn", "cmn->eng")


@dataclass
class SampleRun:
    sample_id: str
    direction: str
    source_seconds: float
    target_seconds: float
    wall_seconds: float
    compute_rtf: float
    first_write_ms: float | None
    first_audio_nca_ms: float | None
    first_audio_ca_ms: float | None
    valid_audio_writes: int
    rejected_writes: int
    fallback_used: bool
    fallback_reason: str | None
    translation: str
    reference: str
    generated_semantic_tokens: int
    stage09_events: int
    write_events: int
    wait_events: int
    instrumented: bool
    timer: dict[str, object] | None = None
    error: str | None = None
    session_push_seconds: float = 0.0
    session_setup_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        value = dict(self.__dict__)
        return value


@dataclass
class Selection:
    index: int
    sample_id: str
    direction: str
    speaker_tokens: list[int]
    reference: str
    waveform: np.ndarray
    source_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.source_seconds = len(self.waveform) / SAMPLE_RATE


def select_samples(
    config: Stage09Config,
    *,
    per_direction: int,
    max_source_seconds: float,
    min_source_seconds: float,
    scan_limit: int,
) -> list[Selection]:
    dataset = B2BridgeAudioDataset(
        config.dataset_index, "valid", config.source_manifest, config.source_offsets
    )
    wanted = {direction: per_direction for direction in DIRECTIONS}
    chosen: list[Selection] = []
    for index in range(min(len(dataset), scan_limit)):
        if not any(wanted.values()):
            break
        direction = str(dataset._target_row(index)["direction"])
        if wanted.get(direction, 0) <= 0:
            continue
        row = dataset[index]
        waveform = row["waveform"].numpy().astype(np.float32)
        seconds = len(waveform) / SAMPLE_RATE
        if not min_source_seconds <= seconds <= max_source_seconds:
            continue
        record = row["phase3_record"]
        chosen.append(
            Selection(
                index=index,
                sample_id=str(row["id"]),
                direction=direction,
                speaker_tokens=[int(value) for value in record["bicodec_global"]],
                reference=str(record["translation"]),
                waveform=waveform,
            )
        )
        wanted[direction] -= 1
    missing = {key: value for key, value in wanted.items() if value > 0}
    if missing:
        raise RuntimeError(
            f"could not fill the requested sample budget within {scan_limit} rows: {missing}"
        )
    chosen.sort(key=lambda item: (item.direction, item.index))
    return chosen


def run_one(
    engine: Stage11Engine,
    selection: Selection,
    *,
    request_dir: Path,
    timer: CallTreeTimer | None,
) -> SampleRun:
    if request_dir.exists():
        shutil.rmtree(request_dir)
    instrumented = timer is not None

    def span(label: str):
        return nullcontext() if timer is None else timer.span(label)

    started = time.perf_counter()
    session_patcher = None
    try:
        with span("session_setup"):
            session = engine.new_session(
                direction=selection.direction,
                speaker_tokens=selection.speaker_tokens,
                request_dir=request_dir,
            )
        if timer is not None:
            session_patcher = install_session_probes(timer, session)
        result = None
        chunk = INGRESS_MS * SAMPLE_RATE // 1000
        for start in range(0, len(selection.waveform), chunk):
            end = min(len(selection.waveform), start + chunk)
            with span("session_push"):
                for update in session.push(
                    selection.waveform[start:end], final=end == len(selection.waveform)
                ):
                    if update.result is not None:
                        result = update.result
        if result is None:
            raise RuntimeError("Stage11 session did not finalize")
        write_events = sum(event.policy_action == "WRITE" for event in result.events)
        run = SampleRun(
            sample_id=selection.sample_id,
            direction=selection.direction,
            source_seconds=float(result.source_seconds),
            target_seconds=float(result.target_seconds),
            wall_seconds=float(result.wall_seconds),
            compute_rtf=float(result.wall_seconds) / max(float(result.source_seconds), 1e-6),
            first_write_ms=result.first_write_ms,
            first_audio_nca_ms=result.first_audio_nca_ms,
            first_audio_ca_ms=result.first_audio_ca_ms,
            valid_audio_writes=int(result.valid_audio_writes),
            rejected_writes=int(result.rejected_writes),
            fallback_used=bool(result.fallback_used),
            fallback_reason=result.fallback_reason,
            translation=str(result.translation),
            reference=selection.reference,
            generated_semantic_tokens=sum(event.semantic_tokens for event in result.events),
            stage09_events=len(result.events),
            write_events=write_events,
            wait_events=len(result.events) - write_events,
            instrumented=instrumented,
        )
    except Exception:  # a broken sample must not discard the rest of the measurement
        elapsed = time.perf_counter() - started
        run = SampleRun(
            sample_id=selection.sample_id,
            direction=selection.direction,
            source_seconds=selection.source_seconds,
            target_seconds=0.0,
            wall_seconds=elapsed,
            compute_rtf=elapsed / max(selection.source_seconds, 1e-6),
            first_write_ms=None,
            first_audio_nca_ms=None,
            first_audio_ca_ms=None,
            valid_audio_writes=0,
            rejected_writes=0,
            fallback_used=False,
            fallback_reason=None,
            translation="",
            reference=selection.reference,
            generated_semantic_tokens=0,
            stage09_events=0,
            write_events=0,
            wait_events=0,
            instrumented=instrumented,
            error=traceback.format_exc(limit=6),
        )
    finally:
        if session_patcher is not None:
            session_patcher.close()
    if timer is not None:
        run.timer = timer.to_dict()
        for stat in timer.stats():
            if stat.path == "session_push":
                run.session_push_seconds = stat.inclusive_seconds
            elif stat.path == "session_setup":
                run.session_setup_seconds = stat.inclusive_seconds
    return run


def bucket_totals(timer: CallTreeTimer) -> dict[str, float]:
    """Inclusive seconds for the direct children of ``session_push`` plus the remainder."""

    stats = {stat.path: stat for stat in timer.stats()}
    push = stats.get("session_push")
    total = push.inclusive_seconds if push else 0.0
    buckets: dict[str, float] = {}
    accounted = 0.0
    for label in TOP_LEVEL_LABELS:
        stat = stats.get(f"session_push/{label}")
        seconds = stat.inclusive_seconds if stat else 0.0
        buckets[label] = seconds
        accounted += seconds
    buckets["session_push_other"] = max(0.0, total - accounted)
    buckets["session_setup"] = (
        stats["session_setup"].inclusive_seconds if "session_setup" in stats else 0.0
    )
    return buckets


def summarize(rows: list[SampleRun]) -> dict[str, object]:
    usable = [row for row in rows if row.error is None]
    if not usable:
        return {"samples": 0}
    return {
        "samples": len(usable),
        "failed": len(rows) - len(usable),
        "source_seconds_total": sum(row.source_seconds for row in usable),
        "wall_seconds_total": sum(row.wall_seconds for row in usable),
        "compute_rtf_pooled": sum(row.wall_seconds for row in usable)
        / max(sum(row.source_seconds for row in usable), 1e-6),
        "compute_rtf_mean": statistics.fmean(row.compute_rtf for row in usable),
        "first_write_ms_mean": _mean_optional(row.first_write_ms for row in usable),
        "first_audio_nca_ms_mean": _mean_optional(row.first_audio_nca_ms for row in usable),
        "first_audio_ca_ms_mean": _mean_optional(row.first_audio_ca_ms for row in usable),
        "fallback_rate": sum(row.fallback_used for row in usable) / len(usable),
        "valid_audio_writes_total": sum(row.valid_audio_writes for row in usable),
        "rejected_writes_total": sum(row.rejected_writes for row in usable),
        "generated_semantic_tokens_total": sum(row.generated_semantic_tokens for row in usable),
        "stage09_events_total": sum(row.stage09_events for row in usable),
        "write_events_total": sum(row.write_events for row in usable),
    }


def _mean_optional(values) -> float | None:
    collected = [float(value) for value in values if value is not None]
    return statistics.fmean(collected) if collected else None


def direction_bleu(rows: list[SampleRun]) -> dict[str, object]:
    scores: dict[str, object] = {}
    for direction in DIRECTIONS:
        subset = [row for row in rows if row.direction == direction and row.error is None]
        if not subset:
            continue
        hypotheses = [row.translation for row in subset]
        references = [[row.reference for row in subset]]
        tokenize = "zh" if direction == "eng->cmn" else "13a"
        scores[direction] = {
            "samples": len(subset),
            "text_bleu": sacrebleu.corpus_bleu(
                hypotheses, references, tokenize=tokenize
            ).score,
            "chrf": sacrebleu.corpus_chrf(hypotheses, references).score,
        }
    return scores


def render_markdown(payload: dict[str, object]) -> str:
    baseline = payload["baseline"]
    instrumented = payload["instrumented"]
    buckets = payload["attribution"]["buckets"]
    tree = payload["attribution"]["tree"]
    total = float(payload["attribution"]["session_push_seconds"]) or 1e-9

    lines = [
        "# Step 0 — streaming S2ST wall-clock decomposition",
        "",
        f"> Run `{payload['run_name']}` · {payload['generated_at']} · research only.",
        "> Baseline pass is unpatched; the instrumented pass adds CUDA synchronisation at every",
        "> span boundary, so treat the baseline for RTF and the instrumented pass for shares.",
        "",
        "## 1. Headline",
        "",
        "| Pass | Samples | Source s | Wall s | Compute RTF (pooled) | First audio NCA | First audio CA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in (("baseline", baseline), ("instrumented", instrumented)):
        summary = block["summary"]
        if not summary.get("samples"):
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {summary['samples']} | {summary['source_seconds_total']:.1f} | "
            f"{summary['wall_seconds_total']:.1f} | {summary['compute_rtf_pooled']:.2f} | "
            f"{_ms(summary['first_audio_nca_ms_mean'])} | {_ms(summary['first_audio_ca_ms_mean'])} |"
        )
    overhead = payload["attribution"]["instrumentation_overhead_ratio"]
    lines += [
        "",
        f"Instrumentation overhead on total wall clock: **{overhead * 100:.1f}%**.",
        "",
        "## 2. Where the wall clock goes",
        "",
        "| Bucket | Seconds | Share of streaming wall | RTF contribution | What it is |",
        "|---|---:|---:|---:|---|",
    ]
    source_seconds = float(payload["attribution"]["source_seconds"]) or 1e-9
    for label, seconds in sorted(buckets.items(), key=lambda item: -item[1]):
        if seconds <= 0.0:
            continue
        lines.append(
            f"| `{label}` | {seconds:.2f} | {seconds / total * 100:.1f}% | "
            f"{seconds / source_seconds:.3f} | {LABEL_DESCRIPTIONS.get(label, '')} |"
        )
    lines += [
        "",
        "## 3. Full call tree",
        "",
        "| Path | Calls | Inclusive s | Exclusive s | Share (inclusive) |",
        "|---|---:|---:|---:|---:|",
    ]
    for node in tree:
        indent = "&nbsp;" * 4 * int(node["depth"])
        lines.append(
            f"| {indent}`{node['label']}` | {node['calls']} | "
            f"{node['inclusive_seconds']:.2f} | {node['exclusive_seconds']:.2f} | "
            f"{node['inclusive_seconds'] / total * 100:.1f}% |"
        )
    lines += [
        "",
        "## 4. Per-sample (baseline pass)",
        "",
        "| Sample | Direction | Source s | Wall s | RTF | NCA | CA | Writes ok/rej | Fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in baseline["samples"]:
        lines.append(
            f"| `{row['sample_id']}` | {row['direction']} | {row['source_seconds']:.2f} | "
            f"{row['wall_seconds']:.2f} | {row['compute_rtf']:.2f} | "
            f"{_ms(row['first_audio_nca_ms'])} | {_ms(row['first_audio_ca_ms'])} | "
            f"{row['valid_audio_writes']}/{row['rejected_writes']} | "
            f"{'yes' if row['fallback_used'] else 'no'} |"
        )
    bleu = payload["baseline"]["bleu"]
    if bleu:
        lines += [
            "",
            "## 5. Text quality on the same samples",
            "",
            "| Direction | Samples | Text-BLEU | chrF |",
            "|---|---:|---:|---:|",
        ]
        for direction, block in bleu.items():
            lines.append(
                f"| {direction} | {block['samples']} | {block['text_bleu']:.2f} | "
                f"{block['chrf']:.2f} |"
            )
    lines += ["", "## 6. Configuration", "", "```json", json.dumps(payload["config"], indent=2), "```", ""]
    return "\n".join(lines)


def _ms(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.0f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--samples-per-direction", type=int, default=4)
    parser.add_argument("--min-source-seconds", type=float, default=3.0)
    parser.add_argument("--max-source-seconds", type=float, default=12.0)
    parser.add_argument("--scan-limit", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-write-tokens", type=int, default=384)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "eval_outputs/simul_s2st_route_v1/step0_rtf_decomposition",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="run only the instrumented pass (halves runtime, loses the honest RTF)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.output_json, args.output_md):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite Step 0 report: {output}")

    stage09 = Stage09Config(device=args.device)
    stage11 = Stage11Config(max_write_tokens=args.max_write_tokens)
    selections = select_samples(
        stage09,
        per_direction=args.samples_per_direction,
        max_source_seconds=args.max_source_seconds,
        min_source_seconds=args.min_source_seconds,
        scan_limit=args.scan_limit,
    )
    print(
        json.dumps(
            {
                "stage": "selected",
                "samples": [
                    {
                        "id": item.sample_id,
                        "direction": item.direction,
                        "seconds": round(item.source_seconds, 2),
                    }
                    for item in selections
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    engine = Stage11Engine(stage09, stage11)
    engine.load()
    device = engine.bundle.device
    artifact_root = args.artifact_root / args.run_name

    baseline_rows: list[SampleRun] = []
    if not args.skip_baseline:
        for selection in selections:
            request_dir = artifact_root / "baseline" / selection.direction.replace("->", "_to_")
            row = run_one(
                engine,
                selection,
                request_dir=request_dir / selection.sample_id,
                timer=None,
            )
            baseline_rows.append(row)
            print(
                json.dumps(
                    {
                        "stage": "baseline",
                        "id": row.sample_id,
                        "direction": row.direction,
                        "rtf": round(row.compute_rtf, 3),
                        "error": bool(row.error),
                    }
                ),
                flush=True,
            )

    aggregate = CallTreeTimer(device=device, synchronize=True)
    instrumented_rows: list[SampleRun] = []
    patcher = install_pipeline_probes(aggregate)
    try:
        install_bundle_probes(patcher, engine.bundle)
        for selection in selections:
            timer = CallTreeTimer(device=device, synchronize=True)
            patcher.timer = timer
            request_dir = artifact_root / "instrumented" / selection.direction.replace("->", "_to_")
            row = run_one(
                engine,
                selection,
                request_dir=request_dir / selection.sample_id,
                timer=timer,
            )
            instrumented_rows.append(row)
            aggregate.merge(timer)
            print(
                json.dumps(
                    {
                        "stage": "instrumented",
                        "id": row.sample_id,
                        "direction": row.direction,
                        "rtf": round(row.compute_rtf, 3),
                        "error": bool(row.error),
                    }
                ),
                flush=True,
            )
    finally:
        patcher.close()

    buckets = bucket_totals(aggregate)
    stats = {stat.path: stat for stat in aggregate.stats()}
    push_seconds = stats["session_push"].inclusive_seconds if "session_push" in stats else 0.0
    baseline_summary = summarize(baseline_rows) if baseline_rows else {"samples": 0}
    instrumented_summary = summarize(instrumented_rows)
    overhead = 0.0
    if baseline_summary.get("samples") and instrumented_summary.get("samples"):
        overhead = (
            float(instrumented_summary["wall_seconds_total"])
            / max(float(baseline_summary["wall_seconds_total"]), 1e-6)
            - 1.0
        )
    source_seconds = float(
        instrumented_summary.get("source_seconds_total")
        or baseline_summary.get("source_seconds_total")
        or 0.0
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "run_name": args.run_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "device": args.device,
            "samples_per_direction": args.samples_per_direction,
            "min_source_seconds": args.min_source_seconds,
            "max_source_seconds": args.max_source_seconds,
            "ingress_ms": INGRESS_MS,
            "max_write_tokens": args.max_write_tokens,
            "stage09": {
                key: str(value) for key, value in stage09.__dict__.items()
            },
            "stage11": {key: str(value) for key, value in stage11.__dict__.items()},
        },
        "selection": [
            {
                "id": item.sample_id,
                "dataset_index": item.index,
                "direction": item.direction,
                "source_seconds": item.source_seconds,
            }
            for item in selections
        ],
        "baseline": {
            "summary": baseline_summary,
            "bleu": direction_bleu(baseline_rows) if baseline_rows else {},
            "samples": [row.to_dict() for row in baseline_rows],
        },
        "instrumented": {
            "summary": instrumented_summary,
            "bleu": direction_bleu(instrumented_rows),
            "samples": [row.to_dict() for row in instrumented_rows],
        },
        "attribution": {
            "session_push_seconds": push_seconds,
            "source_seconds": source_seconds,
            "instrumentation_overhead_ratio": overhead,
            "buckets": buckets,
            "tree": [stat.to_dict() for stat in aggregate.stats()],
            "descriptions": LABEL_DESCRIPTIONS,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")

    ranked = sorted(
        ((label, seconds) for label, seconds in buckets.items() if label != "session_setup"),
        key=lambda item: -item[1],
    )
    print(
        json.dumps(
            {
                "stage": "done",
                "baseline_rtf": baseline_summary.get("compute_rtf_pooled"),
                "instrumented_rtf": instrumented_summary.get("compute_rtf_pooled"),
                "overhead_ratio": round(overhead, 4),
                "top_buckets": [
                    {
                        "label": label,
                        "seconds": round(seconds, 2),
                        "share": round(seconds / max(push_seconds, 1e-9), 4),
                    }
                    for label, seconds in ranked[:5]
                ],
                "report": str(args.output_md),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
