"""Summarize legacy and attention-mask-v2 English Speech-BLEU results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def english_groups(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(name): dict(value)
        for name, value in dict(payload["groups"]).items()
        if str(name).endswith(":cmn->eng")
    }


def length_ratio(metric: dict[str, object]) -> float:
    return float(metric["sys_len"]) / float(metric["ref_len"])


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    incomplete: list[str] = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, relative_root = line.split("\t", 1)
        run_root = args.repo_root / relative_root
        corrected_root = run_root / "metrics_whisper_attention_mask_v2"
        required = [
            corrected_root / "COMPLETE",
            corrected_root / "verification.json",
            corrected_root / "speech_bleu_eng.json",
            run_root / "metrics" / "speech_bleu.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            incomplete.append(label)
            if args.allow_incomplete:
                continue
            raise FileNotFoundError(f"{label} is incomplete; missing: {missing}")

        verification = read_json(corrected_root / "verification.json")
        if not verification.get("complete"):
            raise RuntimeError(f"{label} failed verification: {verification}")
        legacy = english_groups(read_json(run_root / "metrics" / "speech_bleu.json"))
        corrected = english_groups(read_json(corrected_root / "speech_bleu_eng.json"))
        if set(legacy) != set(corrected):
            raise RuntimeError(
                f"{label} group mismatch: legacy={sorted(legacy)} corrected={sorted(corrected)}"
            )
        for group in sorted(corrected):
            old_metric = legacy[group]
            new_metric = corrected[group]
            rows.append(
                {
                    "run": label,
                    "group": group,
                    "sample_count": int(new_metric["sample_count"]),
                    "legacy_speech_bleu": float(old_metric["score"]),
                    "corrected_speech_bleu": float(new_metric["score"]),
                    "delta": float(new_metric["score"]) - float(old_metric["score"]),
                    "legacy_sys_ref_length_ratio": length_ratio(old_metric),
                    "corrected_sys_ref_length_ratio": length_ratio(new_metric),
                    "verification": verification,
                }
            )

    summary = {
        "manifest_count": sum(
            1 for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
        ),
        "completed_run_count": len({str(row["run"]) for row in rows}),
        "incomplete_runs": incomplete,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "| Run | Mode | N | Legacy Speech-BLEU | Corrected Speech-BLEU | Delta | Legacy length ratio | Corrected length ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {group} | {sample_count} | {legacy_speech_bleu:.4f} | "
            "{corrected_speech_bleu:.4f} | {delta:+.4f} | "
            "{legacy_sys_ref_length_ratio:.3f} | {corrected_sys_ref_length_ratio:.3f} |".format(
                **row
            )
        )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
