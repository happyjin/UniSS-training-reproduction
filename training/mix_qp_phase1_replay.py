"""Mix Quality/Performance JSONL with Phase1 replay at 2:1.

Emits two QP rows and one Phase1 row, repeating until the QP file ends.
Extra Phase1 rows are left unread so the ratio stays 2:1 even though each
UniST row produces four Phase1 samples and only two QP samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row must be an object at {path}:{line_no}")
            yield row


def mix_qp_replay(qp_path: Path, phase1_path: Path, output_path: Path) -> dict[str, int]:
    qp_iter = _iter_jsonl(qp_path)
    phase1_iter = _iter_jsonl(phase1_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    counts = {"qp": 0, "phase1": 0, "total": 0}
    with tmp_path.open("w", encoding="utf-8") as handle:
        while True:
            emitted_qp = 0
            for _ in range(2):
                try:
                    row = next(qp_iter)
                except StopIteration:
                    row = None
                if row is None:
                    break
                row = dict(row)
                row["mix_group"] = "qp"
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                counts["qp"] += 1
                counts["total"] += 1
                emitted_qp += 1
            if emitted_qp == 0:
                break
            if emitted_qp != 2:
                raise RuntimeError(
                    f"{qp_path} ended mid-pair after {counts['qp']} QP rows"
                )
            try:
                row = next(phase1_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{phase1_path} ran out before QP finished at total={counts['total']}"
                ) from exc
            row = dict(row)
            row["mix_group"] = "phase1"
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            counts["phase1"] += 1
            counts["total"] += 1
    if counts["qp"] != 2 * counts["phase1"]:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ratio mismatch: {counts}")
    tmp_path.replace(output_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qp", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=None)
    args = parser.parse_args()
    counts = mix_qp_replay(args.qp, args.phase1, args.output)
    if args.expected_total is not None and counts["total"] != args.expected_total:
        raise SystemExit(
            f"expected {args.expected_total} mixed rows, wrote {counts['total']}"
        )
    print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
