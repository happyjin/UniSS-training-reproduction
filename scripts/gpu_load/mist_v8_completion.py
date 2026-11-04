"""Read-only completion checks for Claude's MIST v8 plan.

An idle GPU or a Claude task-notification marked 'completed' is insufficient:
the agent can be waiting on background work. Verify the deliverables themselves.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import shlex
import subprocess
import time


def identity(path):
    stat = path.stat()
    return str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=96)
def jsonl_index(file_identity):
    """Cache only file summaries, not large transcripts or decoded text."""
    path = Path(file_identity[0])
    ids, meetings, timed = set(), {}, set()
    duplicates = 0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["id"]
            duplicates += key in ids
            ids.add(key)
            if "meeting" in row:
                meetings[key] = row["meeting"]
            reads = row.get("read_ms")
            if (isinstance(reads, list) and reads
                    and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in reads)):
                timed.add(key)
    return ids, meetings, timed, duplicates


def coverage(expected, output, reasons, evidence, *, require_timing=False):
    if not expected:
        reasons.append(f"empty expected window set: {output}")
        return
    if not output.is_file():
        reasons.append(f"final decoded output missing: {output}")
        return
    stamp = identity(output)
    evidence.append(stamp)
    got, _, timed, duplicates = jsonl_index(stamp)
    if duplicates or not expected.issubset(got):
        reasons.append(f"decoded windows incomplete: {output}: {len(expected & got)}/{len(expected)}, duplicates={duplicates}")
    if require_timing and not expected.issubset(timed):
        reasons.append(f"B6 per-read wall-clock measurements incomplete: {output}")


def statistics(path, reasons, evidence):
    if not path.is_file():
        reasons.append("MIST v8 final statistical tests missing")
        return
    evidence.append(identity(path))
    data = json.loads(path.read_text())
    missing = []
    for dataset in ("en2zh", "zh2en", "ali"):
        for test in ("E-ov1", "E-ov2", "E-lat", "E-pol"):
            key = f"{test}-{dataset}"
            result = data.get(key)
            if not isinstance(result, dict):
                missing.append(key)
                continue
            numbers = [result.get(k) for k in ("diff", "p", "p_holm", "meetings")]
            interval = result.get("ci")
            valid = (all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers)
                     and 0 <= result["p"] <= 1 and 0 <= result["p_holm"] <= 1 and result["meetings"] > 0
                     and isinstance(interval, list) and len(interval) == 2
                     and all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval)
                     and isinstance(result.get("holds"), bool))
            # A negative/null-hypothesis result is complete; significance is not a gate.
            if not valid:
                missing.append(key)
    if missing:
        reasons.append("MIST v8 statistical tests incomplete: " + ", ".join(missing))


def git_delivery(item, root, reasons, evidence):
    directory = root / item["directory"]
    label = item["directory"]
    def git(*args):
        return subprocess.run(["git", "-C", str(directory), *args],
                              capture_output=True, text=True, timeout=15)
    head = git("rev-parse", "HEAD")
    dirty = git("diff", "--name-only", "-z", "HEAD")
    ignored = tuple(item.get("ignore_dirty_prefixes", []))
    changed = [name for name in dirty.stdout.split("\0") if name and not name.startswith(ignored)]
    if head.returncode or dirty.returncode or changed:
        reasons.append(f"tracked files not finalized: {label}")
    if item.get("remote_ref"):
        delivered = git("merge-base", "--is-ancestor", "HEAD", item["remote_ref"])
        pushed = delivered.returncode == 0
    else:
        delivered = git("for-each-ref", "--contains=HEAD", "--format=%(refname)", "refs/remotes/")
        pushed = delivered.returncode == 0 and bool(delivered.stdout.strip())
    if not pushed:
        reasons.append(f"latest commit has no matching local push record: {label}")
    evidence.append(("git", label, head.stdout.strip(), delivered.stdout.strip(), pushed))


def claude_activity(config, root, reasons, evidence):
    paths = {root / name for name in config["session_files"]}
    for pattern in config.get("session_globs", []):
        # Absolute patterns are restricted to the configured directory.
        pattern = root / pattern
        paths.update(pattern.parent.glob(pattern.name))
    latest = 0.0
    for path in sorted(paths):
        if not path.is_file():
            reasons.append(f"Claude session missing: {path.name}")
            continue
        stamp = identity(path)
        evidence.append(("claude_session", *stamp))
        latest = max(latest, stamp[-1] / 1e9)
    quiet = config.get("quiet_seconds", 600)
    if time.time() - latest < quiet:
        reasons.append(f"Claude session/subagents are still active (require {quiet} seconds quiet)")


def extra_checks(config, root, reasons, evidence):
    statistics(root / config["statistics"], reasons, evidence)
    claude_activity(config, root, reasons, evidence)
    for item in config["git_delivery"]:
        git_delivery(item, root, reasons, evidence)
    # Avoid scanning large decode files until queues and final documents are ready.
    if reasons:
        return
    manifests = {key: root / value for key, value in config["windows"].items()}
    windows = {}
    for key, path in manifests.items():
        stamp = identity(path)
        evidence.append(stamp)
        windows[key] = jsonl_index(stamp)
    subset_path = root / config["subset_manifest"]
    evidence.append(identity(subset_path))
    subsets = json.loads(subset_path.read_text())
    runs = root / config["runs"]
    names = set(config["expected_runs"]) | {p.name for p in runs.glob("on_*") if p.is_dir()}
    for name in sorted(names):
        dataset = name.split("_")[1]
        expected, meetings, _, _ = windows[dataset]
        if "lag2" in name:
            selected = set(subsets["ali" if dataset == "ali" else "ami"])
            expected = {key for key in expected if meetings[key] in selected}
        coverage(expected, runs / name / "hyp.jsonl", reasons, evidence)
    # Read B6's manifest as data. Never execute shell commands from it.
    manifest = root / config["b6_manifest"]
    evidence.append(identity(manifest))
    jobs = json.loads(manifest.read_text())
    if not jobs:
        reasons.append("B6 job manifest is empty")
    for job in jobs:
        argv = shlex.split(job["cmd"])
        if "--windows" not in argv or "--out" not in argv or "--time-reads" not in argv:
            reasons.append(f"unrecognized B6 measurement job: {job['id']}")
            continue
        source = root / argv[argv.index("--windows") + 1]
        output = root / argv[argv.index("--out") + 1]
        stamp = identity(source)
        evidence.append(stamp)
        expected = jsonl_index(stamp)[0]
        coverage(expected, output, reasons, evidence, require_timing=True)
