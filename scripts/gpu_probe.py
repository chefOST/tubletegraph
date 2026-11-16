"""Optional GPU-only samples for a requested hardware-sizing pilot."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time


def parse_sample(output: str) -> tuple[float, float, float]:
    rows = output.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("Expected one allocated GPU")
    values = [float(value.strip()) for value in rows[0].split(",")]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("GPU probe returned invalid values")
    utilization, used, total = values
    if not 0 <= utilization <= 100 or not 0 <= used <= total or total <= 0:
        raise ValueError("GPU probe returned values outside their expected ranges")
    return utilization, used, total


def summarize_samples(samples: list[dict]) -> dict:
    if not samples:
        return {"samples": 0}
    utilization = sorted(sample["gpu_utilization_pct"] for sample in samples)

    def percentile(fraction):
        position = (len(utilization) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(utilization) - 1)
        return utilization[lower] + (utilization[upper] - utilization[lower]) * (position - lower)

    return {
        "samples": len(samples),
        "gpu_utilization_mean_pct": sum(utilization) / len(utilization),
        "gpu_utilization_p50_pct": percentile(0.5),
        "gpu_utilization_p95_pct": percentile(0.95),
        "active_sample_fraction": sum(value > 0 for value in utilization) / len(utilization),
        "busy_sample_fraction": sum(value >= 90 for value in utilization) / len(utilization),
        "peak_memory_used_mib": max(sample["memory_used_mib"] for sample in samples),
        "memory_total_mib": max(sample["memory_total_mib"] for sample in samples),
    }


class GpuProbe:
    """Keep the open CSV on local SSD; copy it to output storage only after close."""

    def __init__(self, output_path: Path, *, interval_seconds: float = 5.0):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("GPU probe interval must be finite and positive")
        self.output_path = Path(output_path)
        self.interval_seconds = interval_seconds
        self.samples = []
        self.stage_seconds = {}
        self.current_stage = "setup"
        self.errors = 0
        self.last_error = None
        self.stop = threading.Event()

    @contextmanager
    def stage(self, name: str):
        if name not in {"initialization", "native_inference"}:
            raise ValueError("Unknown GPU probe stage")
        self.current_stage = name
        started = time.monotonic()
        try:
            yield
        finally:
            self.stage_seconds[name] = time.monotonic() - started
            self.current_stage = "between_stages"

    def _collect(self):
        with self.local_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "elapsed_seconds", "stage", "gpu_utilization_pct",
                "memory_used_mib", "memory_total_mib",
            ])
            writer.writeheader()
            while not self.stop.is_set():
                try:
                    stage = self.current_stage
                    response = subprocess.run(
                        ["nvidia-smi", "--id=0",
                         "--query-gpu=utilization.gpu,memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
                        check=True, capture_output=True, text=True, timeout=10,
                    )
                    utilization, used, total = parse_sample(response.stdout)
                    sample = {"elapsed_seconds": time.monotonic() - self.started,
                              "stage": stage, "gpu_utilization_pct": utilization,
                              "memory_used_mib": used, "memory_total_mib": total}
                    self.samples.append(sample)
                    writer.writerow(sample)
                    handle.flush()
                except Exception as error:
                    self.errors += 1
                    self.last_error = f"{type(error).__name__}: {error}"
                self.stop.wait(self.interval_seconds)

    def __enter__(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="tubelet-gpu-probe-")
        self.local_path = Path(self.scratch.name) / "gpu.csv"
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            self.errors += 1
            self.last_error = "GPU probe did not stop; open sample file was not copied"
            return
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.local_path, self.output_path)
        except OSError as error:
            self.errors += 1
            self.last_error = f"{type(error).__name__}: {error}"
        finally:
            self.scratch.cleanup()

    def summary(self) -> dict:
        samples = list(self.samples)
        return {
            "sample_interval_seconds": self.interval_seconds,
            "busy_sample_threshold_pct": 90,
            **summarize_samples(samples),
            "stages": {
                name: {"elapsed_seconds": elapsed,
                       **summarize_samples([sample for sample in samples if sample["stage"] == name])}
                for name, elapsed in self.stage_seconds.items()
            },
            "collection_errors": self.errors,
            **({"last_collection_error": self.last_error} if self.last_error else {}),
        }
