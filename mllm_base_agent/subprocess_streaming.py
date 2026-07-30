"""Shared helpers for running a benchmark task as a streamed, timeout-guarded subprocess.

Extracted from ``mllm_base_agent/dual_agent/procthor/run_benchmark.py`` so that
AI2-THOR/ProcTHOR, dual-agent/single-agent benchmark runners can all share the same
"stream child stdout to cluster log + disk, enforce a wall-clock timeout, emit a
runtime_status.json heartbeat" behavior instead of drifting out of sync.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def write_task_runtime_status(task_output_dir: Path, **fields: Any) -> None:
    """Write an atomic, human-readable status file while a child task is running."""
    try:
        status_path = task_output_dir / "runtime_status.json"
        payload: Dict[str, Any] = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **fields,
        }
        temp_path = status_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, status_path)
    except Exception as e:
        print(f"  ⚠️  Failed to write runtime status: {e}", flush=True)


def run_task_subprocess_streaming(
    cmd: List[str],
    cwd: Path,
    task_id: str,
    task_output_dir: Path,
    timeout_seconds: Optional[int],
    heartbeat_seconds: int = 60,
) -> Tuple[int, str, float]:
    """Run a task while streaming child output to both the cluster log and disk.

    Unlike a plain ``subprocess.run(cmd, timeout=...)``, this:
    - Streams stdout/stderr line-by-line to both the parent process's stdout
      (visible in cluster job logs) and a per-task ``live.log`` file on disk,
      so a still-running task's progress is visible without waiting for it
      to finish.
    - Periodically writes ``runtime_status.json`` (heartbeat) with pid/elapsed
      time, so external tooling can detect a hung task even if it produces no
      output for a long time.
    - Enforces ``timeout_seconds`` (if given) by killing the child's entire
      process group (``start_new_session=True`` + ``os.killpg``), not just the
      immediate child, so grandchild processes (e.g. a Unity subprocess
      launched by ai2thor) are also reliably terminated.
    """
    live_log_path = task_output_dir / "live.log"
    started_at = time.time()
    print(f"  ▶️  {task_id} started; live log: {live_log_path}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    write_task_runtime_status(
        task_output_dir,
        task_id=task_id,
        state="running",
        pid=process.pid,
        started_at=datetime.now().isoformat(timespec="seconds"),
        command=cmd,
    )

    output_parts: List[str] = []
    last_output_at = started_at
    timed_out = False
    assert process.stdout is not None
    with open(live_log_path, "w", encoding="utf-8", buffering=1) as live_log:
        while process.poll() is None:
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if line:
                    output_parts.append(line)
                    live_log.write(line)
                    print(f"[{task_id}] {line}", end="", flush=True)
                    last_output_at = time.time()
            now = time.time()
            elapsed = now - started_at
            if timeout_seconds and elapsed >= timeout_seconds:
                timed_out = True
                message = f"Task timed out after {timeout_seconds}s; terminating process group.\n"
                output_parts.append(message)
                live_log.write(message)
                print(f"[{task_id}] {message}", end="", flush=True)
                try:
                    os.killpg(process.pid, 15)
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                break
            if now - last_output_at >= heartbeat_seconds:
                message = f"heartbeat: still running, elapsed={elapsed / 60:.1f} min, pid={process.pid}\n"
                live_log.write(message)
                print(f"[{task_id}] {message}", end="", flush=True)
                write_task_runtime_status(
                    task_output_dir,
                    task_id=task_id,
                    state="running",
                    pid=process.pid,
                    elapsed_seconds=round(elapsed, 1),
                    last_child_output_seconds_ago=round(now - last_output_at, 1),
                )
                last_output_at = now

        remainder = process.stdout.read()
        if remainder:
            output_parts.append(remainder)
            live_log.write(remainder)
            for line in remainder.splitlines(keepends=True):
                print(f"[{task_id}] {line}", end="", flush=True)

    return_code = 124 if timed_out else process.wait()
    duration = time.time() - started_at
    write_task_runtime_status(
        task_output_dir,
        task_id=task_id,
        state="timed_out" if timed_out else "finished",
        pid=process.pid,
        return_code=return_code,
        duration_seconds=round(duration, 1),
        live_log=str(live_log_path),
    )
    return return_code, "".join(output_parts), duration
