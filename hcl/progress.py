from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Iterator


_PROGRESS_LOCK = Lock()


def progress(message: str) -> None:
    """Print one immediately visible progress line unless explicitly disabled."""
    if os.environ.get("HCL_PROGRESS", "1").lower() in {"0", "false", "no", "off"}:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    with _PROGRESS_LOCK:
        print(f"[HCL {timestamp}] {message}", file=sys.stderr, flush=True)


def report_item(index: int, total: int) -> bool:
    """Report first/last items and every HCL_PROGRESS_EVERY items."""
    try:
        every = max(int(os.environ.get("HCL_PROGRESS_EVERY", "1")), 1)
    except ValueError:
        every = 1
    return index == 1 or index == total or index % every == 0


@contextmanager
def progress_heartbeat(message: str) -> Iterator[None]:
    """Emit periodic progress while one blocking operation is still running."""
    try:
        interval = float(os.environ.get("HCL_HEARTBEAT_SECONDS", "30"))
    except ValueError:
        interval = 30.0
    if interval <= 0 or os.environ.get("HCL_PROGRESS", "1").lower() in {"0", "false", "no", "off"}:
        yield
        return

    stopped = Event()
    started = perf_counter()

    def emit() -> None:
        while not stopped.wait(interval):
            progress(f"{message} still_running seconds={perf_counter() - started:.0f}")

    worker = Thread(target=emit, name="hcl-progress-heartbeat", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=interval)
