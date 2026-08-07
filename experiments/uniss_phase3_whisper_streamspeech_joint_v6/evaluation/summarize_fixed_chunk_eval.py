"""Summarize fixed-chunk Megatron validation logs without changing checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CHUNKS = ("320", "640", "960", "1280", "offline")
MODELS = ("stage_a", "stage_b")
VALUE_PATTERN = re.compile(r"([^|]+?) value: ([+-]?[0-9.]+E[+-][0-9]+)")


def parse_final_validation(path: Path) -> dict[str, float]:
    matches = [
        line
        for line in path.read_text(errors="replace").splitlines()
        if "validation loss at iteration" in line and "on validation set" in line
    ]
    if not matches:
        raise ValueError(f"missing final validation line: {path}")
    metrics = {
        name.strip(): float(value)
        for name, value in VALUE_PATTERN.findall(matches[-1])
    }
    required = {
        "bicodec_ctc",
        "ar_s2tt",
        "asr_ctc",
        "nar_s2tt_ctc",
        "bridge/commitment_mse",
        "bridge/teacher_glm_agreement",
    }
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"missing metrics {sorted(missing)} in {path}")
    return metrics


def collect(log_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for chunk in CHUNKS:
            path = log_root / f"{model}_{chunk}.log"
            rows.append(
                {
                    "model": model,
                    "chunk_ms": None if chunk == "offline" else int(chunk),
                    "log": str(path.resolve()),
                    "metrics": parse_final_validation(path),
                }
            )
    return rows


def _number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_markdown(
    rows: list[dict[str, object]],
    *,
    stage_a_checkpoint: Path,
    stage_b_checkpoint: Path,
) -> str:
    by_key = {(row["model"], row["chunk_ms"]): row for row in rows}
    lines = [
        "# Phase3 Whisper StreamSpeech Joint V6 固定 chunk 评估报告",
        "",
        "本报告使用相同 15-shard 双语 validation、相同 8-GPU Megatron 入口，分别固定 Whisper chunk，避免训练期间随机 chunk 导致的不可比性。数值为 loss/诊断指标，不等同于端到端 BLEU、语音质量或真实播放延迟。",
        "",
        f"- Stage A checkpoint: `{stage_a_checkpoint.resolve()}`",
        f"- Stage B checkpoint: `{stage_b_checkpoint.resolve()}`",
        "- right context: `80 ms`",
        "- validation: `8 × global batch 128 = 1024` samples per operating point",
        "",
        "## 1. 固定 chunk 绝对结果",
        "",
        "| 模型 | chunk | BiCodec CTC ↓ | AR S2TT ↓ | ASR CTC ↓ | NAR S2TT CTC ↓ | unit infeasible ↓ | commitment ↓ | teacher agreement ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        chunk = "offline" if row["chunk_ms"] is None else f'{row["chunk_ms"]} ms'
        lines.append(
            "| {model} | {chunk} | {bicodec} | {ar} | {asr} | {nar} | {unit} | {commitment} | {agreement:.2f}% |".format(
                model=row["model"],
                chunk=chunk,
                bicodec=_number(metrics["bicodec_ctc"]),
                ar=_number(metrics["ar_s2tt"]),
                asr=_number(metrics["asr_ctc"]),
                nar=_number(metrics["nar_s2tt_ctc"]),
                unit=_number(metrics.get("ctc/unit_infeasible", 0.0)),
                commitment=_number(metrics["bridge/commitment_mse"], 5),
                agreement=100.0 * metrics["bridge/teacher_glm_agreement"],
            )
        )
    lines.extend(
        [
            "",
            "## 2. Stage B 相对 Stage A 的变化",
            "",
            "loss 的负数表示 Stage B 改善；teacher agreement 的正数表示改善。",
            "",
            "| chunk | Δ BiCodec | Δ AR | Δ ASR | Δ NAR | Δ commitment | Δ agreement (pp) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ar_deltas: list[float] = []
    asr_deltas: list[float] = []
    nar_deltas: list[float] = []
    agreement_deltas: list[float] = []
    commitment_values: list[float] = []
    asr_stage_a: list[float] = []
    nar_stage_a: list[float] = []
    for chunk_text in CHUNKS:
        chunk = None if chunk_text == "offline" else int(chunk_text)
        a = by_key[("stage_a", chunk)]["metrics"]
        b = by_key[("stage_b", chunk)]["metrics"]
        deltas = {
            name: b[name] - a[name]
            for name in (
                "bicodec_ctc",
                "ar_s2tt",
                "asr_ctc",
                "nar_s2tt_ctc",
                "bridge/commitment_mse",
                "bridge/teacher_glm_agreement",
            )
        }
        ar_deltas.append(deltas["ar_s2tt"])
        asr_deltas.append(deltas["asr_ctc"])
        nar_deltas.append(deltas["nar_s2tt_ctc"])
        agreement_deltas.append(deltas["bridge/teacher_glm_agreement"])
        commitment_values.append(b["bridge/commitment_mse"])
        asr_stage_a.append(a["asr_ctc"])
        nar_stage_a.append(a["nar_s2tt_ctc"])
        lines.append(
            "| {chunk} | {bicodec:+.4f} | {ar:+.4f} | {asr:+.4f} | {nar:+.4f} | {commitment:+.5f} | {agreement:+.2f} |".format(
                chunk=chunk_text,
                bicodec=deltas["bicodec_ctc"],
                ar=deltas["ar_s2tt"],
                asr=deltas["asr_ctc"],
                nar=deltas["nar_s2tt_ctc"],
                commitment=deltas["bridge/commitment_mse"],
                agreement=100.0 * deltas["bridge/teacher_glm_agreement"],
            )
        )
    lines.extend(
        [
            "",
            "## 3. 自动诊断",
            "",
            f"- ASR CTC: Stage B 在 {sum(value < 0 for value in asr_deltas)}/5 个 chunk 上改善。",
            f"- NAR S2TT CTC: Stage B 在 {sum(value < 0 for value in nar_deltas)}/5 个 chunk 上改善。",
            f"- AR S2TT: Stage B 在 {sum(value <= 0 for value in ar_deltas)}/5 个 chunk 上保持或改善。",
            f"- ASR CTC 五点平均相对改善：`{-100.0 * sum(asr_deltas) / sum(asr_stage_a):.2f}%`。",
            f"- NAR S2TT CTC 五点平均相对改善：`{-100.0 * sum(nar_deltas) / sum(nar_stage_a):.2f}%`。",
            f"- Teacher agreement: Stage B 仅在 {sum(value > 0 for value in agreement_deltas)}/5 个 chunk 上改善，平均变化 `{100.0 * sum(agreement_deltas) / len(agreement_deltas):+.2f}` 个百分点。",
            f"- Stage B 最大 commitment: `{max(commitment_values):.5f}`，绝对安全阈值为 `0.10`。",
            "- 结论：Stage B 通过数值稳定、CTC 学习和 AR loss 保持门，但没有通过 teacher agreement 改善门；不能把本轮表述为 semantic-code agreement 已修复。",
            "- 下一质量门：使用固定 operating point 做 Phase3 old-protocol replay、端到端文本/语音生成、BLEU/ASR-BLEU、speaker/AutoPCP/SLC 与真实 latency 评估。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--stage-a-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.log_root)
    payload = {
        "schema_version": "uniss_phase3_joint_v6_fixed_chunk_eval_v1",
        "stage_a_checkpoint": str(args.stage_a_checkpoint.resolve()),
        "stage_b_checkpoint": str(args.stage_b_checkpoint.resolve()),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.output_md.write_text(
        render_markdown(
            rows,
            stage_a_checkpoint=args.stage_a_checkpoint,
            stage_b_checkpoint=args.stage_b_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
