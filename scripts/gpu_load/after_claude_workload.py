#!/usr/bin/env python3
"""Wait for configured Claude experiment artifacts, then supervise a training workload.

Experiment queues, papers and Claude sessions are read only. State and logs belong
to this supervisor. Completion requires artifacts AND drained queues AND idle GPUs.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import train_mist_workload as workload

REPO = workload.REPO
STOP = threading.Event()
LINE_COUNTS = {}


def read_json(path):
    return json.loads(path.read_text())


def line_count(path):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in LINE_COUNTS:
        with path.open("rb") as source:
            LINE_COUNTS[key] = sum(bool(line.strip()) for line in source)
    return LINE_COUNTS[key]


def experiment_gate(config, root=REPO):
    reasons, evidence = [], []
    for relative in config["queues"]:
        directory = root / relative
        state = read_json(directory / "state.json")
        if not state or any(v not in ("wait", "run", "done", "fail", "cancel") for v in state.values()):
            reasons.append(f"unrecognized queue state: {relative}")
        pending = [k for k, v in state.items() if v in ("wait", "run")]
        if pending:
            reasons.append(f"{relative}: {len(pending)} waiting/running jobs")
        for path in (directory / "add").glob("*.json"):
            unseen = [j["id"] for j in read_json(path) if j["id"] not in state]
            if unseen:
                reasons.append(f"queue has unconsumed jobs: {path.name}")
        evidence.append(state)
    for paper in config["papers"]:
        directory = root / paper["directory"]
        report = directory / paper["report"]
        pdf = directory / "main.pdf"
        if not report.is_file():
            reasons.append(f"final report missing: {report.relative_to(root)}")
            continue
        text = report.read_text()
        if len(text) < 600 or any(s in text for s in ("<!-- BODY -->", "<!-- CONT_SUMMARY -->")):
            reasons.append(f"final report is still a draft: {paper['directory']}")
        committed = subprocess.run(["git", "-C", str(directory), "show", f"HEAD:{paper['report']}"],
                                   capture_output=True, text=True, timeout=15)
        if committed.returncode or committed.stdout != text:
            reasons.append(f"final report not committed: {paper['directory']}")
        dirty = subprocess.run(["git", "-C", str(directory), "diff", "--name-only", "HEAD", "--",
                                "*.tex", "*.bib", "main.pdf"], capture_output=True, text=True, timeout=15)
        if dirty.returncode or dirty.stdout.strip():
            reasons.append(f"paper sources/PDF not finalized: {paper['directory']}")
        sources = list(directory.rglob("*.tex")) + list(directory.rglob("*.bib"))
        latest_source = max((p.stat().st_mtime for p in sources), default=0)
        if not pdf.is_file() or pdf.stat().st_size == 0:
            reasons.append(f"PDF missing: {paper['directory']}")
        elif pdf.stat().st_mtime < max(config["armed_at"], latest_source):
            reasons.append(f"PDF predates this request or latest sources: {paper['directory']}")
        else:
            evidence.append((str(pdf), pdf.stat().st_mtime_ns, hashlib.sha256(text.encode()).hexdigest()))
    metric = root / config["qabc_metrics"] if config.get("qabc_metrics") else None
    if metric is not None and not metric.is_file():
        reasons.append("QABC continuous-clock final metrics missing")
    elif metric is not None:
        data = read_json(metric)
        for direction in ("en2zh", "zh2en"):
            result = data.get("dirs", {}).get(direction, {})
            if "missing" not in result or result["missing"] or not result.get("points"):
                reasons.append(f"QABC final metrics incomplete: {direction}")
        evidence.append((str(metric), metric.stat().st_mtime_ns))
    # Avoid reading the speech result files while experiments are still active.
    if not reasons and config.get("qabc_output"):
        task_files = sorted((root / config["qabc_output"]).glob("cont_tasks*.jsonl"))
        if not task_files:
            reasons.append("QABC speech task manifests missing")
        for manifest in task_files:
            for row in manifest.read_text().splitlines():
                task = json.loads(row)
                output = Path(task["out"])
                output = root / output
                variants = [output] + [Path(f"{output}-r{i}") for i in range(1, 10)]
                expected = line_count(root / task["blocks"])
                complete = False
                for folder in variants:
                    summary, asr = folder / "asr_speech.summary.json", folder / "asr_speech.jsonl"
                    if summary.is_file() and asr.is_file():
                        info = read_json(summary)
                        count = sum(info.get(k, 0) for k in ("transcribed", "empty", "skipped_existing"))
                        if expected > 0 and count >= expected and line_count(asr) >= expected:
                            complete = True
                            break
                if not complete:
                    reasons.append(f"QABC speech/ASR incomplete: {task['key']}")
            evidence.append((str(manifest), manifest.stat().st_mtime_ns))
    if config.get("mist_v8"):
        from mist_v8_completion import extra_checks
        extra_checks(config["mist_v8"], root, reasons, evidence)
    if not config.get("launch_enabled", True):
        reasons.append("monitor-only mode: workload launch is disabled")
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return reasons, digest


def process_info(pid):
    base = Path("/proc") / str(pid)
    fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "start": fields[19], "cwd": os.readlink(base / "cwd"),
            "argv": (base / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")}


def holder_parent(pid, selected):
    """Recognize only this repository's existing target_gpu_util.py holder."""
    for _ in range(8):
        try:
            info = process_info(pid)
        except (OSError, ValueError, IndexError):
            return None
        argv = info["argv"]
        if (info["cwd"] == str(REPO) and argv and Path(argv[0]).name.startswith("python")
                and any(Path(x).name == "target_gpu_util.py" for x in argv[1:])):
            script = next(x for x in argv[1:] if Path(x).name == "target_gpu_util.py")
            if (REPO / script).resolve() != REPO / "scripts/gpu_load/target_gpu_util.py":
                return None
            if "--devices" not in argv:
                return None
            cards = workload.devices(argv[argv.index("--devices") + 1])
            return info if set(cards).issubset(selected) else None
        pid = info["ppid"]
        if pid <= 1:
            return None
    return None


def signal_same_process(info, sig):
    try:
        if process_info(info["pid"])["start"] == info["start"]:
            os.kill(info["pid"], sig)
    except (OSError, ValueError, IndexError):
        pass


def stop_training(process):
    if process is None or process.poll() is not None:
        return
    # Capture descendant identities before termination; workers have their own
    # process groups. PID start times prevent signals reaching a reused PID.
    found = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                info = process_info(int(path.name))
                found[info["pid"]] = info
            except (OSError, ValueError, IndexError):
                pass
    owned = {process.pid}
    while True:
        expanded = owned | {p for p, v in found.items() if v["ppid"] in owned}
        if expanded == owned:
            break
        owned = expanded
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    for pid in owned - {process.pid}:
        signal_same_process(found[pid], signal.SIGTERM)


def recover(tuning, text):
    """Bounded parameter repairs; never rewrite experimental code or install dependencies."""
    if "another GPU process appeared" in text or "GPU became busy" in text or "GPUs busy:" in text:
        return "wait"
    if "out of memory" in text.lower() or "training OOM" in text or "less than 1 GiB free" in text:
        if tuning["micro_batch_size"] == 1 and tuning["min_free_mib"] >= 24576:
            return "failed"
        tuning["micro_batch_size"] = max(1, tuning["micro_batch_size"] // 2)
        tuning["min_free_mib"] = min(24576, tuning["min_free_mib"] + 4096)
        return "retry"
    if "non-finite" in text or "nonfinite" in text:
        if tuning["lr"] <= 1e-6:
            return "failed"
        tuning["lr"] = max(1e-6, tuning["lr"] / 10)
        return "retry"
    if "DataLoader" in text or "stalled" in text:
        if tuning["num_workers"]:
            tuning["num_workers"] = 0
            return "retry"
    return "failed"


def tail(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 262144))
        return stream.read().decode(errors="replace")


def save_status(directory, **fields):
    path = directory / "status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated_utc": time.strftime("%FT%TZ", time.gmtime()),
                               "supervisor_pid": os.getpid(), **fields}, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def supervise(config, directory):
    tuning = {"micro_batch_size": 4, "min_free_mib": 8192, "lr": 1e-4,
              "num_workers": 2, "memory_window_steps": 64}
    child, log_stream, run_log = None, None, None
    stable_since, stable_digest, idle_since = None, None, None
    last_reason, failures, attempts, smoke_done, adjustments = None, 0, 0, False, 0
    started, low_history = 0.0, deque()
    try:
        while not STOP.is_set() and not (directory / "STOP").exists():
            now = time.monotonic()
            try:
                reasons, digest = experiment_gate(config)
                if child is not None:
                    if reasons or digest != stable_digest:
                        stop_training(child)
                        log_stream.close()
                        child, smoke_done, stable_since = None, False, None
                        workload.log("experiment completion state changed; released training workload")
                        continue
                    code = child.poll()
                    text = tail(run_log)
                    stalled = time.time() - run_log.stat().st_mtime > config["stall_seconds"]
                    if code is None and stalled:
                        stop_training(child)
                        code, text = 1, text + "\nstalled"
                    if code is not None:
                        log_stream.close()
                        child = None
                        if code == 0 and not smoke_done:
                            smoke_done, idle_since = True, None
                            workload.log("single-GPU validation passed; preparing all selected GPUs")
                            continue
                        action = recover(tuning, text)
                        failures += action != "wait"
                        workload.log(f"training exited rc={code}; action={action}; tuning={tuning}")
                        if action == "failed" or failures > config["max_restarts"]:
                            save_status(directory, phase="failed", reason="automatic repairs exhausted or unknown error",
                                        log=str(run_log), tuning=tuning)
                            return 1
                        stable_since, idle_since, smoke_done = None, None, False
                        STOP.wait(60)
                        continue
                    if smoke_done:
                        cards = workload.gpu_snapshot()
                        low_history.append((now, {g: cards[g]["used"] for g in config["devices"]}))
                        while low_history and now - low_history[0][0] > 600:
                            low_history.popleft()
                        if (adjustments < 2 and now - started > 1200 and low_history
                                and now - low_history[0][0] >= 540):
                            low = [g for g in config["devices"]
                                   if max(v[g] for _, v in low_history) < min(cards[g]["total"] * .95,
                                      cards[g]["total"] - tuning["min_free_mib"]) - 4096]
                            if low:
                                stop_training(child)
                                log_stream.close()
                                child, idle_since = None, None
                                adjustments += 1
                                tuning["memory_window_steps"] = 16 if adjustments == 1 else 8
                                workload.log(f"memory below target on {low}; recalibrating with {tuning}")
                                continue
                    save_status(directory, phase="running" if smoke_done else "validating", training_pid=child.pid,
                                log=str(run_log), tuning=tuning, attempts=attempts)
                else:
                    if reasons or stable_digest != digest:
                        stable_since, idle_since = None, None
                        stable_digest = digest
                    if not reasons:
                        stable_since = now if stable_since is None else stable_since
                        if now - stable_since < config["completion_stable_seconds"]:
                            reasons = [f"final artifacts and queues must remain unchanged for {config['completion_stable_seconds']} seconds"]
                    if not reasons:
                        cards = workload.gpu_snapshot()
                        if set(config["devices"]) - cards.keys():
                            raise RuntimeError("selected GPUs missing")
                        pids = set().union(*(cards[g]["pids"] for g in config["devices"]))
                        if pids and config.get("replace_repo_holder"):
                            holders = [holder_parent(pid, set(config["devices"])) for pid in pids]
                            if all(holders):
                                for info in {h["pid"]: h for h in holders}.values():
                                    signal_same_process(info, signal.SIGTERM)
                                    workload.log(f"replacing this repository's old matrix holder PID={info['pid']}")
                        busy = [g for g in config["devices"] if cards[g]["pids"] or cards[g]["used"] > 1000]
                        if busy:
                            idle_since = None
                            reasons = [f"waiting for GPUs to be idle: {busy}"]
                        else:
                            idle_since = now if idle_since is None else idle_since
                            if now - idle_since < config["idle_seconds"]:
                                reasons = [f"waiting for all selected GPUs to remain idle for {config['idle_seconds']} seconds"]
                    if not reasons:
                        attempts += 1
                        run_log = directory / f"training_{time.time_ns()}.log"
                        selected = config["devices"] if smoke_done else config["devices"][:1]
                        command = [config["python"], "-B", "-u", str(REPO / "scripts/gpu_load/train_mist_workload.py"),
                                   "--devices", ",".join(map(str, selected)), "--idle-seconds", "0", "--max-used-mib", "0",
                                   "--log-interval", "1" if not smoke_done else "5", "--grad-accum",
                                   str(math.ceil(16 / tuning["micro_batch_size"]))]
                        for key, value in tuning.items():
                            command += ["--" + key.replace("_", "-"), str(value)]
                        if not smoke_done:
                            command += ["--steps", "12"]
                        log_stream = run_log.open("x")
                        child = subprocess.Popen(command, cwd=REPO, stdout=log_stream, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, start_new_session=True)
                        started, low_history = now, deque()
                        workload.log(f"started {'full training' if smoke_done else '12-step validation'} PID={child.pid}; log={run_log}")
                        continue
                    if reasons != last_reason:
                        workload.log("waiting: " + "; ".join(reasons))
                        last_reason = reasons
                    save_status(directory, phase="waiting", reasons=reasons, tuning=tuning, attempts=attempts)
            except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
                # Missing/partial files or NVML errors cannot be treated as completion.
                stop_training(child)
                child, stable_since, idle_since = None, None, None
                if log_stream and not log_stream.closed:
                    log_stream.close()
                workload.log(f"monitor check failed; waiting: {error}")
                save_status(directory, phase="waiting", reasons=[str(error)], tuning=tuning)
            STOP.wait(config["poll_seconds"])
    finally:
        stop_training(child)
        if log_stream and not log_stream.closed:
            log_stream.close()
    save_status(directory, phase="stopped", tuning=tuning)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    if args.check_once:
        reasons, digest = experiment_gate(config)
        print(json.dumps({"ready": not reasons, "reasons": reasons, "fingerprint": digest}, ensure_ascii=False, indent=2))
        return 0
    if args.state_dir is None:
        parser.error("--state-dir is required")
    directory = args.state_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: STOP.set())
        workload.log(f"armed supervisor PID={os.getpid()}; GPUs={config['devices']}")
        return supervise(config, directory)


if __name__ == "__main__":
    sys.exit(main())
