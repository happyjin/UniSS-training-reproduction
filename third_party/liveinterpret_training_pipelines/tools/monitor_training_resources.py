#!/usr/bin/env python3
"""Poll host, process-tree, and GPU resources during training only.

The monitor is intentionally external to the training stack.  It follows the
training log's ordered ``Train:`` and ``Val:/Eval:`` progress markers and writes
resource samples only while the latest marker is ``Train:``.  Evaluation is
therefore excluded without changing the model, trainer, dataset, or template.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable


PHASE_RE = re.compile(
    rb"Train:|Val:|Eval:|Evaluation:|Predict:|"
    rb"['\"]train_runtime['\"]\s*:|End time of running main"
)
STEP_RE = re.compile(rb"['\"]global_step/max_steps['\"]\s*:\s*['\"](\d+)/(\d+)")
GPU_FIELDS = [
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "pcie.link.gen.current",
    "pcie.link.width.current",
]
HOST_FIELDS = [
    "timestamp_utc",
    "unix_s",
    "elapsed_s",
    "sample_interval_s",
    "root_pid",
    "phase",
    "global_step",
    "max_steps",
    "process_count",
    "thread_count",
    "host_cpu_busy_pct",
    "host_cpu_iowait_pct",
    "affinity_cpu_busy_pct",
    "process_cpu_cores",
    "rss_gib",
    "read_mib_s",
    "write_mib_s",
    "voluntary_ctx_switches_s",
    "involuntary_ctx_switches_s",
    "load1",
    "memory_available_gib",
]
GPU_CSV_FIELDS = [
    "timestamp_utc",
    "unix_s",
    "elapsed_s",
    "sample_interval_s",
    "root_pid",
    "phase",
    "global_step",
    "max_steps",
    "gpu_index",
    "gpu_uuid",
    "gpu_util_pct",
    "memory_util_pct",
    "memory_used_mib",
    "memory_total_mib",
    "power_w",
    "temperature_c",
    "pstate",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "pcie_gen",
    "pcie_width",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def parse_cpu_list(value: str | None) -> set[int] | None:
    if not value:
        return None
    cpus: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [v for v in values if math.isfinite(v)]
    if not data:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(data),
        "mean": round(statistics.fmean(data), 4),
        "p50": round(percentile(data, 0.50) or 0.0, 4),
        "p95": round(percentile(data, 0.95) or 0.0, 4),
        "max": round(max(data), 4),
    }


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


class PhaseTracker:
    def __init__(self, log_path: Path, initialize_from_tail: bool):
        self.log_path = log_path
        self.offset = 0
        self.phase = "waiting"
        self.global_step: int | None = None
        self.max_steps: int | None = None
        self.events: list[dict[str, Any]] = []
        if initialize_from_tail and log_path.exists():
            size = log_path.stat().st_size
            self.offset = max(0, size - 1024 * 1024)
            self.update()

    def _set_phase(self, phase: str, marker: str) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.events.append(
            {"timestamp_utc": utc_now(), "unix_s": time.time(), "phase": phase, "marker": marker}
        )

    def update(self) -> str:
        try:
            size = self.log_path.stat().st_size
        except FileNotFoundError:
            return self.phase
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return self.phase
        with self.log_path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        for match in STEP_RE.finditer(chunk):
            self.global_step = int(match.group(1))
            self.max_steps = int(match.group(2))
        for match in PHASE_RE.finditer(chunk):
            marker = match.group(0).decode("utf-8", errors="replace")
            if marker == "Train:":
                self._set_phase("train", marker)
            elif marker.startswith(("Val:", "Eval:", "Evaluation:", "Predict:")):
                self._set_phase("eval", marker)
            else:
                self._set_phase("done", marker)
        return self.phase


def read_cpu_snapshot() -> dict[int | str, tuple[int, int, int]]:
    result: dict[int | str, tuple[int, int, int]] = {}
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("cpu"):
                break
            parts = line.split()
            name = parts[0]
            if name != "cpu" and not name[3:].isdigit():
                continue
            values = [int(v) for v in parts[1:]]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            key: int | str = "all" if name == "cpu" else int(name[3:])
            iowait = values[4] if len(values) > 4 else 0
            result[key] = (total, idle, iowait)
    return result


def cpu_busy_pct(
    previous: dict[int | str, tuple[int, int, int]] | None,
    current: dict[int | str, tuple[int, int, int]],
    cpus: set[int] | None,
) -> float | None:
    if previous is None:
        return None
    keys: Iterable[int | str] = ["all"] if cpus is None else cpus
    total_delta = 0
    idle_delta = 0
    for key in keys:
        if key not in previous or key not in current:
            continue
        total_delta += current[key][0] - previous[key][0]
        idle_delta += current[key][1] - previous[key][1]
    if total_delta <= 0:
        return None
    return 100.0 * (total_delta - idle_delta) / total_delta


def cpu_iowait_pct(
    previous: dict[int | str, tuple[int, int, int]] | None,
    current: dict[int | str, tuple[int, int, int]],
) -> float | None:
    if previous is None or "all" not in previous or "all" not in current:
        return None
    total_delta = current["all"][0] - previous["all"][0]
    iowait_delta = current["all"][2] - previous["all"][2]
    if total_delta <= 0:
        return None
    return 100.0 * iowait_delta / total_delta


def read_proc_stat(pid: int) -> tuple[int, int] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        end = text.rfind(")")
        fields = text[end + 2 :].split()
        return int(fields[1]), int(fields[11]) + int(fields[12])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def process_tree(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        item = read_proc_stat(pid)
        if item is not None:
            children[item[0]].append(pid)
    result: list[int] = []
    queue: deque[int] = deque([root_pid])
    seen: set[int] = set()
    while queue:
        pid = queue.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        if Path(f"/proc/{pid}").exists():
            result.append(pid)
            queue.extend(children.get(pid, ()))
    return result


def read_status(pid: int) -> tuple[int, int, int, int]:
    rss_kib = threads = voluntary = involuntary = 0
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
                elif line.startswith("voluntary_ctxt_switches:"):
                    voluntary = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    involuntary = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return rss_kib, threads, voluntary, involuntary


def read_io(pid: int) -> tuple[int, int]:
    read_bytes = write_bytes = 0
    try:
        with open(f"/proc/{pid}/io", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):
                    read_bytes = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    write_bytes = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return read_bytes, write_bytes


def process_snapshot(root_pid: int) -> dict[str, Any]:
    pids = process_tree(root_pid)
    per_pid: dict[int, dict[str, int]] = {}
    rss_kib = threads = voluntary = involuntary = 0
    for pid in pids:
        stat_item = read_proc_stat(pid)
        if stat_item is None:
            continue
        status = read_status(pid)
        io = read_io(pid)
        per_pid[pid] = {
            "ticks": stat_item[1],
            "read_bytes": io[0],
            "write_bytes": io[1],
            "voluntary": status[2],
            "involuntary": status[3],
        }
        rss_kib += status[0]
        threads += status[1]
        voluntary += status[2]
        involuntary += status[3]
    return {
        "pids": per_pid,
        "process_count": len(per_pid),
        "rss_kib": rss_kib,
        "threads": threads,
        "voluntary": voluntary,
        "involuntary": involuntary,
    }


def positive_pid_delta(current: dict[int, dict[str, int]], previous: dict[int, dict[str, int]], key: str) -> int:
    total = 0
    for pid, values in current.items():
        if pid in previous:
            total += max(0, values[key] - previous[pid][key])
    return total


def mem_available_gib() -> float:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def load1() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


def gpu_snapshot(gpu_ids: str) -> list[list[str]]:
    cmd = [
        "nvidia-smi",
        "-i",
        gpu_ids,
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    return [[part.strip() for part in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]


class ResourceMonitor:
    def __init__(
        self,
        root_pid: int,
        log_path: Path,
        output_dir: Path,
        gpu_ids: str,
        cpu_list: set[int] | None,
        interval: float,
        initialize_phase_from_tail: bool,
        command: list[str] | None,
        summary_only: bool = False,
        summary_refresh_seconds: float = 10.0,
        summary_window_samples: int = 300,
    ):
        self.root_pid = root_pid
        self.log_path = log_path
        self.output_dir = output_dir
        self.gpu_ids = gpu_ids
        self.cpu_list = cpu_list
        self.interval = interval
        self.command = command
        self.summary_only = summary_only
        self.summary_refresh_seconds = summary_refresh_seconds
        self.summary_window_samples = summary_window_samples
        self.phase = PhaseTracker(log_path, initialize_phase_from_tail)
        self.start_wall = time.time()
        self.start_mono = time.monotonic()
        self.previous_mono: float | None = None
        self.previous_cpu: dict[int | str, tuple[int, int, int]] | None = None
        self.previous_proc: dict[str, Any] | None = None
        gpu_count = max(1, len([item for item in gpu_ids.split(",") if item.strip()]))
        self.host_rows = deque(maxlen=summary_window_samples) if summary_only else []
        self.gpu_rows = deque(maxlen=summary_window_samples * gpu_count) if summary_only else []
        self.total_train_samples = 0
        self.total_sampled_train_seconds = 0.0
        self.last_summary_mono = 0.0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.host_handle = None
        self.gpu_handle = None
        self.host_writer = None
        self.gpu_writer = None
        if not summary_only:
            self.host_handle = (self.output_dir / "host.csv").open("w", newline="", encoding="utf-8")
            self.gpu_handle = (self.output_dir / "gpu.csv").open("w", newline="", encoding="utf-8")
            self.host_writer = csv.DictWriter(self.host_handle, fieldnames=HOST_FIELDS)
            self.gpu_writer = csv.DictWriter(self.gpu_handle, fieldnames=GPU_CSV_FIELDS)
            self.host_writer.writeheader()
            self.gpu_writer.writeheader()
        else:
            self._write_rolling_summary(exit_code=None)

    def _sample_counters(self) -> tuple[float, dict[int | str, tuple[int, int, int]], dict[str, Any]]:
        now = time.monotonic()
        cpu = read_cpu_snapshot()
        proc = process_snapshot(self.root_pid)
        return now, cpu, proc

    def _write_training_sample(
        self,
        now_mono: float,
        cpu: dict[int | str, tuple[int, int, int]],
        proc: dict[str, Any],
    ) -> None:
        if self.previous_mono is None or self.previous_cpu is None or self.previous_proc is None:
            return
        seconds = max(1e-6, now_mono - self.previous_mono)
        current_pids = proc["pids"]
        previous_pids = self.previous_proc["pids"]
        ticks = positive_pid_delta(current_pids, previous_pids, "ticks")
        read_bytes = positive_pid_delta(current_pids, previous_pids, "read_bytes")
        write_bytes = positive_pid_delta(current_pids, previous_pids, "write_bytes")
        voluntary = positive_pid_delta(current_pids, previous_pids, "voluntary")
        involuntary = positive_pid_delta(current_pids, previous_pids, "involuntary")
        unix_s = time.time()
        elapsed = now_mono - self.start_mono
        row = {
            "timestamp_utc": utc_now(),
            "unix_s": f"{unix_s:.3f}",
            "elapsed_s": f"{elapsed:.3f}",
            "sample_interval_s": f"{seconds:.4f}",
            "root_pid": self.root_pid,
            "phase": "train",
            "global_step": "" if self.phase.global_step is None else self.phase.global_step,
            "max_steps": "" if self.phase.max_steps is None else self.phase.max_steps,
            "process_count": proc["process_count"],
            "thread_count": proc["threads"],
            "host_cpu_busy_pct": _fmt(cpu_busy_pct(self.previous_cpu, cpu, None)),
            "host_cpu_iowait_pct": _fmt(cpu_iowait_pct(self.previous_cpu, cpu)),
            "affinity_cpu_busy_pct": _fmt(cpu_busy_pct(self.previous_cpu, cpu, self.cpu_list)),
            "process_cpu_cores": f"{ticks / os.sysconf('SC_CLK_TCK') / seconds:.4f}",
            "rss_gib": f"{proc['rss_kib'] / 1024 / 1024:.4f}",
            "read_mib_s": f"{read_bytes / seconds / 1024 / 1024:.4f}",
            "write_mib_s": f"{write_bytes / seconds / 1024 / 1024:.4f}",
            "voluntary_ctx_switches_s": f"{voluntary / seconds:.3f}",
            "involuntary_ctx_switches_s": f"{involuntary / seconds:.3f}",
            "load1": f"{load1():.3f}",
            "memory_available_gib": f"{mem_available_gib():.3f}",
        }
        if self.host_writer is not None and self.host_handle is not None:
            self.host_writer.writerow(row)
            self.host_handle.flush()
        self.host_rows.append(row)
        self.total_train_samples += 1
        self.total_sampled_train_seconds += seconds

        try:
            gpu_values = gpu_snapshot(self.gpu_ids)
        except RuntimeError as exc:
            print(f"[resource-monitor] warning: {exc}", file=sys.stderr)
            return
        for values in gpu_values:
            values += [""] * (len(GPU_FIELDS) - len(values))
            gpu_row = {
                "timestamp_utc": row["timestamp_utc"],
                "unix_s": row["unix_s"],
                "elapsed_s": row["elapsed_s"],
                "sample_interval_s": row["sample_interval_s"],
                "root_pid": self.root_pid,
                "phase": "train",
                "global_step": row["global_step"],
                "max_steps": row["max_steps"],
                "gpu_index": values[0],
                "gpu_uuid": values[1],
                "gpu_util_pct": values[2],
                "memory_util_pct": values[3],
                "memory_used_mib": values[4],
                "memory_total_mib": values[5],
                "power_w": values[6],
                "temperature_c": values[7],
                "pstate": values[8],
                "sm_clock_mhz": values[9],
                "memory_clock_mhz": values[10],
                "pcie_gen": values[11],
                "pcie_width": values[12],
            }
            if self.gpu_writer is not None:
                self.gpu_writer.writerow(gpu_row)
            self.gpu_rows.append(gpu_row)
        if self.gpu_handle is not None:
            self.gpu_handle.flush()

    def _write_rolling_summary(self, exit_code: int | None) -> None:
        if not self.summary_only:
            return
        summary = build_summary(
            list(self.host_rows),
            list(self.gpu_rows),
            metadata={
                "rolling_summary": True,
                "updated_utc": utc_now(),
                "phase": self.phase.phase,
                "global_step": self.phase.global_step,
                "max_steps": self.phase.max_steps,
                "root_pid": self.root_pid,
                "gpu_ids": self.gpu_ids,
                "interval_s": self.interval,
                "window_samples": self.summary_window_samples,
                "total_train_samples": self.total_train_samples,
                "total_sampled_train_seconds": round(self.total_sampled_train_seconds, 3),
                "exit_code": exit_code,
                "evaluation_samples_included": 0,
            },
        )
        final_path = self.output_dir / "hardware.log"
        temp_path = self.output_dir / ".hardware.log.tmp"
        write_summary_text(temp_path, summary)
        os.replace(temp_path, final_path)
        self.last_summary_mono = time.monotonic()

    def run(self, alive: Callable[[], bool]) -> None:
        last_phase = self.phase.phase
        next_sample = time.monotonic()
        while alive():
            phase = self.phase.update()
            now_mono, cpu, proc = self._sample_counters()
            entered_train = phase == "train" and last_phase != "train"
            if phase == "train" and not entered_train:
                self._write_training_sample(now_mono, cpu, proc)
            self.previous_mono = now_mono
            self.previous_cpu = cpu
            self.previous_proc = proc
            if self.summary_only and (
                phase != last_phase
                or time.monotonic() - self.last_summary_mono >= self.summary_refresh_seconds
            ):
                self._write_rolling_summary(exit_code=None)
            last_phase = phase
            next_sample += self.interval
            time.sleep(max(0.0, next_sample - time.monotonic()))
        self.phase.update()

    def close(self, exit_code: int | None) -> None:
        if self.summary_only:
            self._write_rolling_summary(exit_code)
            return
        assert self.host_handle is not None and self.gpu_handle is not None
        self.host_handle.close()
        self.gpu_handle.close()
        with (self.output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp_utc", "unix_s", "phase", "marker"])
            writer.writeheader()
            writer.writerows(self.phase.events)
        summary = build_summary(
            self.host_rows,
            self.gpu_rows,
            metadata={
                "root_pid": self.root_pid,
                "gpu_ids": self.gpu_ids,
                "cpu_list": sorted(self.cpu_list) if self.cpu_list is not None else None,
                "interval_s": self.interval,
                "log_path": str(self.log_path),
                "output_dir": str(self.output_dir),
                "command": self.command,
                "start_unix_s": self.start_wall,
                "end_unix_s": time.time(),
                "exit_code": exit_code,
                "evaluation_samples_included": 0,
            },
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_summary_text(self.output_dir / "summary.txt", summary)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def numeric_column(rows: list[dict[str, Any]], name: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = safe_float(row.get(name))
        if value is not None:
            result.append(value)
    return result


def build_summary(
    host_rows: list[dict[str, Any]], gpu_rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    host_metrics = {}
    for name in [
        "host_cpu_busy_pct",
        "host_cpu_iowait_pct",
        "affinity_cpu_busy_pct",
        "process_cpu_cores",
        "rss_gib",
        "read_mib_s",
        "write_mib_s",
        "voluntary_ctx_switches_s",
        "involuntary_ctx_switches_s",
        "load1",
        "memory_available_gib",
    ]:
        host_metrics[name] = stats(numeric_column(host_rows, name))

    gpu_metrics: dict[str, Any] = {}
    gpu_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gpu_rows:
        gpu_groups[str(row["gpu_index"])].append(row)
    for gpu, rows in sorted(gpu_groups.items(), key=lambda item: int(item[0])):
        util = numeric_column(rows, "gpu_util_pct")
        gpu_metrics[gpu] = {
            "gpu_util_pct": stats(util),
            "gpu_zero_util_pct": round(100.0 * sum(v == 0 for v in util) / len(util), 4) if util else None,
            "memory_util_pct": stats(numeric_column(rows, "memory_util_pct")),
            "memory_used_mib": stats(numeric_column(rows, "memory_used_mib")),
            "power_w": stats(numeric_column(rows, "power_w")),
            "temperature_c": stats(numeric_column(rows, "temperature_c")),
            "sm_clock_mhz": stats(numeric_column(rows, "sm_clock_mhz")),
        }
    return {
        "metadata": metadata,
        "train_only_sample_count": len(host_rows),
        "sampled_train_seconds": round(sum(numeric_column(host_rows, "sample_interval_s")), 3),
        "host": host_metrics,
        "gpus": gpu_metrics,
    }


def write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    metadata = summary.get("metadata", {})
    lines = [
        "Training-only resource summary",
    ]
    if metadata.get("rolling_summary"):
        lines.extend([
            f"updated_utc: {metadata.get('updated_utc')}",
            f"phase: {metadata.get('phase')}",
            f"global_step: {metadata.get('global_step')}/{metadata.get('max_steps')}",
            f"gpu_ids: {metadata.get('gpu_ids')}",
            f"scope: last {metadata.get('window_samples')} training samples",
            f"window_samples_present: {summary['train_only_sample_count']}",
            f"total_train_samples: {metadata.get('total_train_samples')}",
            f"total_sampled_train_seconds: {metadata.get('total_sampled_train_seconds')}",
            f"exit_code: {metadata.get('exit_code')}",
        ])
    else:
        lines.extend([
            f"samples: {summary['train_only_sample_count']}",
            f"sampled_train_seconds: {summary['sampled_train_seconds']}",
        ])
    lines.extend([
        "evaluation_samples_included: 0",
        "",
    ])
    for name, metric in summary["host"].items():
        lines.append(
            f"host.{name}: mean={metric['mean']} p50={metric['p50']} "
            f"p95={metric['p95']} max={metric['max']}"
        )
    for gpu, metrics in summary["gpus"].items():
        lines.append("")
        lines.append(f"gpu.{gpu}.zero_util_pct: {metrics['gpu_zero_util_pct']}")
        for name, metric in metrics.items():
            if not isinstance(metric, dict):
                continue
            lines.append(
                f"gpu.{gpu}.{name}: mean={metric['mean']} p50={metric['p50']} "
                f"p95={metric['p95']} max={metric['max']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_existing(output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.setdefault("interval_s", 1.0)
    metadata["evaluation_samples_included"] = 0
    summary = build_summary(load_rows(output_dir / "host.csv"), load_rows(output_dir / "gpu.csv"), metadata)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_summary_text(output_dir / "summary.txt", summary)
    print((output_dir / "summary.txt").read_text(encoding="utf-8"), end="")


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("resource_monitor") / f"{stamp}_{os.getpid()}"


def ensure_output_available(output_dir: Path, summary_only: bool = False) -> None:
    protected = (["hardware.log"] if summary_only else
                 ["host.csv", "gpu.csv", "events.csv", "summary.json", "summary.txt", "metadata.json"])
    existing = [str(output_dir / name) for name in protected if (output_dir / name).exists()]
    if existing:
        raise SystemExit(
            "refusing to overwrite an existing monitor run; choose a new --output-dir: "
            + ", ".join(existing)
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu-ids", required=True, help="Physical GPU IDs, for example 0,1")
    parser.add_argument(
        "--cpu-list",
        help="CPU affinity list to measure separately, for example 24-47,120-143. Does not change affinity.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV and summaries")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Write only a bounded rolling hardware.log; do not retain per-sample CSV/JSON files.",
    )
    parser.add_argument(
        "--summary-refresh-seconds", type=float, default=10.0,
        help="Refresh interval for --summary-only hardware.log (default: 10).",
    )
    parser.add_argument(
        "--summary-window-samples", type=int, default=300,
        help="Number of recent training samples summarized by --summary-only (default: 300).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("run", help="Launch and monitor a training command")
    add_common_args(run_parser)
    run_parser.add_argument("--log-file", type=Path, default=None, help="Combined stdout/stderr log")
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")

    attach_parser = subparsers.add_parser("attach", help="Attach to an existing training process")
    add_common_args(attach_parser)
    attach_parser.add_argument("--pid", type=int, required=True, help="Root PID of the training process tree")
    attach_parser.add_argument("--log-file", type=Path, required=True, help="Existing live training log")

    summary_parser = subparsers.add_parser("summarize", help="Rebuild a summary from existing CSV files")
    summary_parser.add_argument("output_dir", type=Path)
    return parser


def run_command(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("run mode requires a command after --")
    output_dir = args.output_dir or default_output_dir()
    ensure_output_available(output_dir, args.summary_only)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or output_dir / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise SystemExit(f"refusing to overwrite existing --log-file: {log_path}")
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None

    def tee_output() -> None:
        for chunk in iter(lambda: os.read(process.stdout.fileno(), 65536), b""):
            log_handle.write(chunk)
            log_handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    tee_thread = threading.Thread(target=tee_output, name="training-log-tee", daemon=True)
    tee_thread.start()

    monitor = ResourceMonitor(
        root_pid=process.pid,
        log_path=log_path,
        output_dir=output_dir,
        gpu_ids=args.gpu_ids,
        cpu_list=parse_cpu_list(args.cpu_list),
        interval=args.interval,
        initialize_phase_from_tail=False,
        command=command,
        summary_only=args.summary_only,
        summary_refresh_seconds=args.summary_refresh_seconds,
        summary_window_samples=args.summary_window_samples,
    )
    metadata = {
        "mode": "run",
        "root_pid": process.pid,
        "gpu_ids": args.gpu_ids,
        "cpu_list": args.cpu_list,
        "interval_s": args.interval,
        "log_path": str(log_path),
        "command": command,
    }
    if not args.summary_only:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    interrupted = False

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    old_int = signal.signal(signal.SIGINT, forward_signal)
    old_term = signal.signal(signal.SIGTERM, forward_signal)
    try:
        monitor.run(lambda: process.poll() is None)
        exit_code = process.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        tee_thread.join(timeout=10)
        log_handle.close()
        exit_code = process.poll()
        monitor.close(exit_code)
    print(f"[resource-monitor] outputs: {output_dir}", file=sys.stderr)
    if interrupted and (exit_code is None or exit_code == 0):
        return 130
    return int(exit_code or 0)


def attach(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or default_output_dir()
    ensure_output_available(output_dir, args.summary_only)
    if not Path(f"/proc/{args.pid}").exists():
        raise SystemExit(f"PID does not exist: {args.pid}")
    monitor = ResourceMonitor(
        root_pid=args.pid,
        log_path=args.log_file,
        output_dir=output_dir,
        gpu_ids=args.gpu_ids,
        cpu_list=parse_cpu_list(args.cpu_list),
        interval=args.interval,
        initialize_phase_from_tail=True,
        command=None,
        summary_only=args.summary_only,
        summary_refresh_seconds=args.summary_refresh_seconds,
        summary_window_samples=args.summary_window_samples,
    )
    metadata = {
        "mode": "attach",
        "root_pid": args.pid,
        "gpu_ids": args.gpu_ids,
        "cpu_list": args.cpu_list,
        "interval_s": args.interval,
        "log_path": str(args.log_file),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        monitor.run(lambda: Path(f"/proc/{args.pid}").exists())
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close(None)
    print(f"[resource-monitor] outputs: {output_dir}", file=sys.stderr)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.mode in {"run", "attach"} and args.interval < 0.5:
        raise SystemExit("--interval must be at least 0.5 seconds")
    if args.mode in {"run", "attach"} and args.summary_only:
        if args.summary_refresh_seconds <= 0:
            raise SystemExit("--summary-refresh-seconds must be positive")
        if args.summary_window_samples <= 0:
            raise SystemExit("--summary-window-samples must be positive")
    if args.mode == "run":
        return run_command(args)
    if args.mode == "attach":
        return attach(args)
    summarize_existing(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
