"""Assemble resumable formal Stage-A A4--A8 chunks without rewriting parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from array import array
from pathlib import Path
from typing import Iterable

from training.simul_uniss.jsonl_index import write_index


SCHEMA = "simul_uniss_subsecond_formal_stage_a_complete_v2"


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


def _markers(root: Path, name: str, expected_parts: int) -> list[dict[str, object]]:
    paths = sorted(root.rglob(name))
    if len(paths) != expected_parts:
        raise ValueError(f"expected {expected_parts} {name} markers under {root}, found {len(paths)}")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(value.get("status") != "complete" for value in values):
        raise ValueError(f"not every {name} marker is complete")
    return values


def _concatenate(paths: Iterable[Path], output: Path, *, accepted_only: bool = False) -> dict[str, object]:
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}"
    offsets = array("Q")
    offset = 0
    records = 0
    try:
        with temporary.open("wb") as target:
            for path in paths:
                with path.open("rb") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        if accepted_only:
                            value = json.loads(line)
                            if not bool(value.get("formal_a68_pass")):
                                continue
                        offsets.append(offset)
                        target.write(line)
                        offset += len(line)
                        records += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "records": records,
        "bytes": output.stat().st_size,
        "index": write_index(output, offsets),
    }


def _split_accepted(path: Path, output_dir: Path, validation_modulus: int) -> dict[str, object]:
    outputs = {
        "train": output_dir / "formal_train_manifest.jsonl",
        "valid": output_dir / "formal_valid_manifest.jsonl",
    }
    temporaries = {name: output_dir / f".{value.name}.tmp.{os.getpid()}" for name, value in outputs.items()}
    handles = {name: value.open("wb") for name, value in temporaries.items()}
    offsets = {name: array("Q") for name in outputs}
    positions = {name: 0 for name in outputs}
    counts = {name: 0 for name in outputs}
    try:
        with path.open("rb") as source:
            for line in source:
                value = json.loads(line)
                digest = int(hashlib.sha256(str(value.get("id", "")).encode()).hexdigest()[:16], 16)
                split = "valid" if digest % validation_modulus == 0 else "train"
                offsets[split].append(positions[split])
                handles[split].write(line)
                positions[split] += len(line)
                counts[split] += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for name, temporary in temporaries.items():
            os.replace(temporary, outputs[name])
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
    return {
        name: {
            "path": str(outputs[name]),
            "records": counts[name],
            "bytes": outputs[name].stat().st_size,
            "index": write_index(outputs[name], offsets[name]),
        }
        for name in outputs
    }


def assemble(args: argparse.Namespace) -> dict[str, object]:
    a45_root = Path(args.a45_root).resolve()
    a68_root = Path(args.a68_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / "STAGE_A_COMPLETE.json"
    if complete.is_file():
        value = json.loads(complete.read_text(encoding="utf-8"))
        print(json.dumps({"status": "already_complete", **value}, sort_keys=True))
        return value
    a45_markers = _markers(a45_root, "STAGE_A_A45_COMPLETE.json", args.expected_parts)
    a68_markers = _markers(a68_root, "STAGE_A_A68_COMPLETE.json", args.expected_parts)
    a68_marker_counts = [
        value.get("counts", {}).get("records")
        if isinstance(value.get("counts"), dict)
        else None
        for value in a68_markers
    ]
    expected_a68_records = (
        sum(int(value) for value in a68_marker_counts)
        if all(value is not None for value in a68_marker_counts)
        else None
    )
    a45_inputs = [Path(str(value["output_manifest"])) for value in a45_markers]
    a68_inputs = [Path(str(value["output_manifest"])) for value in a68_markers]
    for path in [*a45_inputs, *a68_inputs]:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    a45 = _concatenate(a45_inputs, output_dir / "a45_manifest.jsonl")
    formal = _concatenate(a68_inputs, output_dir / "formal_manifest.jsonl")
    accepted = _concatenate(
        a68_inputs,
        output_dir / "formal_accepted_manifest.jsonl",
        accepted_only=True,
    )
    splits = _split_accepted(
        Path(str(accepted["path"])), output_dir, args.validation_modulus
    )
    if int(a45["records"]) != args.expected_records:
        raise ValueError(f"A4/A5 record count {a45['records']} != expected {args.expected_records}")
    # A6/A8 intentionally reads only A4/A5-passing rows.  It must therefore
    # equal the audited sum in its own part markers, not the pre-filter A45
    # input count.
    if expected_a68_records is not None and int(formal["records"]) != expected_a68_records:
        raise ValueError(
            f"A6/A8 record count {formal['records']} != marker total {expected_a68_records}"
        )
    marker = {
        "schema_version": SCHEMA,
        "status": "complete",
        "scope": "formal_stage_a_a4_a8_v2",
        "warning": (
            "UniST contains BiCodec tokens rather than original waveforms; A4/A5 align reconstructed "
            "audio and preserve released-vs-reconstructed WhisperVQ compatibility as a separate metric."
        ),
        "expected_parts": args.expected_parts,
        "expected_records": args.expected_records,
        "expected_a68_records": expected_a68_records,
        "a45": a45,
        "formal_all": formal,
        "formal_accepted": accepted,
        "formal_splits": splits,
        "validation_modulus": args.validation_modulus,
        "formal_acceptance_rate": int(accepted["records"]) / max(1, int(formal["records"])),
        "a45_part_markers": [str(value["output_manifest"]) for value in a45_markers],
        "a68_part_markers": [str(value["output_manifest"]) for value in a68_markers],
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(complete, marker)
    print(json.dumps(marker, sort_keys=True))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a45-root", required=True)
    parser.add_argument("--a68-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-parts", type=int, default=30)
    parser.add_argument("--expected-records", type=int, default=1_500_000)
    parser.add_argument("--validation-modulus", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    assemble(parse_args())


if __name__ == "__main__":
    main()
