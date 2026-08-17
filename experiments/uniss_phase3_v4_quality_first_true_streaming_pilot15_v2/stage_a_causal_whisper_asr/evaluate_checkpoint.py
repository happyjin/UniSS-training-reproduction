#!/usr/bin/env python3
"""Free-running cached-runtime diagnosis for a trained Stage A v2 checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as v1,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    append_only_commit_audit,
    hidden_metrics,
    load_trained_objective,
    make_cached_frontend,
    run_cached_frontend,
    token_metrics,
)
from training import constants_uniss as c


@torch.inference_mode()
def acoustic_runtime_outputs(
    objective,
    frontend,
    waveform: torch.Tensor,
    source_glm: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    device = next(objective.parameters()).device
    values = waveform.numpy()
    recomputed = frontend.forward_recomputed_reference(values)
    cached = run_cached_frontend(frontend, values)
    hidden_parity = hidden_metrics(recomputed.pre_vq_hidden.cpu(), cached.hidden)
    token_parity = token_metrics(recomputed.token_ids.cpu(), cached.tokens)

    hidden = cached.hidden[0].to(device)
    reference_hidden = recomputed.pre_vq_hidden[0].to(device)
    if len(hidden) + 1 == len(source_glm):
        deficit = v1.terminal_codec_extension_deficit_samples(
            int(waveform.numel()), len(hidden), len(source_glm)
        )
        if deficit is None:
            raise ValueError("unaudited terminal cached-token extension")
        hidden = torch.cat((hidden, hidden[-1:]), dim=0)
        reference_hidden = torch.cat((reference_hidden, reference_hidden[-1:]), dim=0)
    if len(hidden) != len(source_glm):
        raise ValueError(f"cached GLM length mismatch: {len(hidden)} vs {len(source_glm)}")

    codes = objective._nearest_codes(hidden)
    reference_codes = objective._nearest_codes(reference_hidden)
    residual = objective.bridge_projection(objective.bridge_norm(hidden))
    reference_residual = objective.bridge_projection(
        objective.bridge_norm(reference_hidden)
    )
    residual_parity = hidden_metrics(reference_residual.cpu(), residual.cpu())

    batch = waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([waveform.numel()], dtype=torch.long, device=device)
    output = objective.frontend(batch, lengths, chunk_ms=160)
    ctc_logits = objective.ctc_head(output.frame_hidden)[
        0, : int(output.frame_lengths[0])
    ]
    raw_ctc = ctc_logits.float().argmax(dim=-1).tolist()
    collapsed = v1.collapse_ctc(raw_ctc, objective.ctc_blank_id)
    ctc = {
        "input_frames": len(raw_ctc),
        "raw_nonblank_frames": sum(value != objective.ctc_blank_id for value in raw_ctc),
        "collapsed_nonblank_tokens": len(collapsed),
        "blank_ratio": sum(value == objective.ctc_blank_id for value in raw_ctc)
        / max(1, len(raw_ctc)),
        "text": bytes(value for value in collapsed if 0 <= value < 256).decode(
            "utf-8", errors="replace"
        ),
    }
    parity = {
        "hidden": hidden_parity,
        "tokens": token_parity,
        "bridge_residual": residual_parity,
        "semantic_codes_exact": bool(torch.equal(reference_codes, codes)),
    }
    reference = {
        "codes": reference_codes,
        "residual": reference_residual,
    }
    return codes, residual, ctc, {"parity": parity, "reference": reference}


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(v1.summarize_rows(rows))
    rollbacks = [row["committed_rollback"] for row in rows]
    runtime = [row["cached_recomputed_parity"] for row in rows]
    summary.update(
        {
            "committed_rollback_count": sum(int(value["rollback_count"]) for value in rollbacks),
            "committed_rollback_rows": sum(not bool(value["append_only"]) for value in rollbacks),
            "committed_append_only_rate": sum(bool(value["append_only"]) for value in rollbacks)
            / max(1, len(rollbacks)),
            "cached_recomputed_hidden_pass_rate": sum(
                bool(value["hidden"]["allclose"]) for value in runtime
            )
            / max(1, len(runtime)),
            "cached_recomputed_token_exact_rate": sum(
                bool(value["tokens"]["exact"]) for value in runtime
            )
            / max(1, len(runtime)),
            "cached_recomputed_bridge_pass_rate": sum(
                bool(value["bridge_residual"]["allclose"]) for value in runtime
            )
            / max(1, len(runtime)),
            "cached_recomputed_free_generation_exact_rate": sum(
                bool(value["free_generation_exact"]) for value in runtime
            )
            / max(1, len(runtime)),
            "evaluations_by_language": dict(Counter(str(row["language"]) for row in rows)),
        }
    )
    return summary


def markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Stage A v2 cached-runtime free-running diagnosis",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- HF Qwen: `{payload['hf_model']}`",
        f"- Evaluations: {summary['samples']}",
        f"- Committed rollback count: **{summary['committed_rollback_count']}**",
        f"- Append-only rows: **{summary['committed_append_only_rate']:.4f}**",
        f"- Cached/recomputed token parity: **{summary['cached_recomputed_token_exact_rate']:.4f}**",
        f"- Cached/recomputed free-generation parity: **{summary['cached_recomputed_free_generation_exact_rate']:.4f}**",
        f"- Streaming WER/CER: **{summary['ar_error_rate_by_task'].get('streaming_asr', 0.0):.4f}**",
        f"- Causal-full WER/CER: **{summary['ar_error_rate_by_task'].get('causal_full_asr', 0.0):.4f}**",
        "",
        "| task | language | sample | CTC blank | AR text | metric | error rate | rollback |",
        "|---|---|---|---:|---|---|---:|---:|",
    ]
    for row in payload["samples"]:
        lines.append(
            "| {task} | {language} | {sample_id} | {blank:.4f} | {text} | "
            "{metric} | {error:.4f} | {rollback} |".format(
                task=row["task"],
                language=row["language"],
                sample_id=row["sample_id"],
                blank=row["ctc"]["blank_ratio"],
                text=str(row["ar_free_running"]["text"]).replace("|", "\\|"),
                metric=row["ar_free_running"]["metric"],
                error=row["ar_free_running"]["error_rate"],
                rollback=row["committed_rollback"]["rollback_count"],
            )
        )
    lines.extend(
        [
            "",
            "Rollback is measured on the persistent accepted-token ledger: each new WRITE event may only append; every earlier committed token and its hash must remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--valid-packs", type=Path, required=True)
    parser.add_argument("--max-samples-per-task", type=int, default=2)
    parser.add_argument("--max-event-tokens", type=int, default=96)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-acoustics-per-pack", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite Stage A v2 diagnosis")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid Stage A v2 diagnosis partition")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    qwen.requires_grad_(False)
    objective, checkpoint = load_trained_objective(
        args.checkpoint,
        args.whispervq_model,
        device,
        dtype=torch.float32,
    )
    frontend = make_cached_frontend(objective, device)
    selected = list(
        v1.iter_selected(
            args.valid_packs,
            args.max_samples_per_task,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            max_acoustics_per_pack=args.max_acoustics_per_pack,
        )
    )
    rows: list[dict[str, Any]] = []
    for sample in selected:
        waveform = v1.load_waveform(str(sample["source_audio"]))
        codes, residual, ctc, runtime = acoustic_runtime_outputs(
            objective, frontend, waveform, sample["source_glm"]
        )
        base_ids = codes.long() + c.GLM_SEMANTIC_OFFSET
        speech_embeddings = qwen.get_input_embeddings()(base_ids) + residual.to(
            qwen.get_input_embeddings().weight.dtype
        )
        reference_codes = runtime["reference"]["codes"]
        reference_residual = runtime["reference"]["residual"]
        reference_embeddings = qwen.get_input_embeddings()(
            reference_codes.long() + c.GLM_SEMANTIC_OFFSET
        ) + reference_residual.to(qwen.get_input_embeddings().weight.dtype)
        conceptual = sample["conceptual"]
        flags = sample["generated_flags"]
        glm_map = {
            int(position): index for index, position in enumerate(sample["glm_positions"])
        }
        teacher = v1.teacher_forced_accuracy(
            qwen,
            conceptual,
            flags,
            glm_map,
            speech_embeddings,
            len(tokenizer),
        )
        free = v1.free_running_asr(
            qwen,
            tokenizer,
            conceptual,
            flags,
            glm_map,
            speech_embeddings,
            language=str(sample["language"]),
            max_event_tokens=args.max_event_tokens,
        )
        reference_free = v1.free_running_asr(
            qwen,
            tokenizer,
            conceptual,
            flags,
            glm_map,
            reference_embeddings,
            language=str(sample["language"]),
            max_event_tokens=args.max_event_tokens,
        )
        runtime["parity"]["free_generation_exact"] = (
            free["generated_tokens"] == reference_free["generated_tokens"]
        )
        metric, errors, units = v1.error_counts(
            str(sample["reference"]), str(free["text"]), str(sample["language"])
        )
        free.update(
            {
                "metric": metric,
                "errors": errors,
                "reference_units": units,
                "error_rate": errors / max(1, units),
            }
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "language": sample["language"],
                "reference": sample["reference"],
                "source_audio": sample["source_audio"],
                "chunk_ms": 160,
                "ctc": ctc,
                "ar_teacher_forced": teacher,
                "ar_free_running": free,
                "committed_rollback": append_only_commit_audit(free["events"]),
                "cached_recomputed_parity": runtime["parity"],
            }
        )
    payload = {
        "schema_version": "uniss_quality_first_stage_a_checkpoint_diagnosis_v2",
        "checkpoint": str(checkpoint),
        "hf_model": str(args.hf_model.resolve()),
        "valid_packs": str(args.valid_packs.resolve()),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "max_acoustics_per_pack": args.max_acoustics_per_pack,
        "runtime": {
            "pcm_block_ms": 160,
            "right_context_ms": 0,
            "frontend_dtype": "torch.float32",
            "qwen_dtype": "torch.bfloat16",
        },
        "summary": summarize_rows(rows),
        "samples": rows,
    }
    v1.atomic_json(args.output_json.resolve(), payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
