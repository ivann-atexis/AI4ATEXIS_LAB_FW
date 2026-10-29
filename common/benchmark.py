from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common.config import ROOT, model_directory
from common.runtime import ModelSession
from common.types import InputData

LOGGER = logging.getLogger(__name__)


class DeviceMetrics:
    def __init__(self, model: Any) -> None:
        self.torch = sys.modules.get("torch")
        self.device = getattr(model, "device", None)
        if self.torch is not None and isinstance(model, self.torch.nn.Module):
            if self.device is None:
                parameter = next(model.parameters(), None)
                self.device = parameter.device if parameter is not None else None
        else:
            self.torch = None
        self.cuda = self.torch is not None and getattr(self.device, "type", None) == "cuda"
        self.mps = self.torch is not None and getattr(self.device, "type", None) == "mps"

    def synchronize(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
        elif self.mps:
            self.torch.mps.synchronize()

    def reset_peak(self) -> None:
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self) -> dict[str, Any]:
        if not self.cuda:
            return {
                "backend": "unavailable",
                "reason": "Pics allocateur CUDA disponibles uniquement pour un modele PyTorch CUDA.",
            }
        return {
            "backend": "torch.cuda",
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
        }


def gpu_snapshot() -> dict[str, Any]:
    fields = ["uuid", "name", "driver_version", "utilization.gpu", "memory.used", "memory.total"]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("Mesures nvidia-smi indisponibles : %s", error)
        return {"available": False, "reason": str(error)}
    if result.returncode:
        reason = result.stderr.strip() or f"nvidia-smi termine avec le code {result.returncode}."
        LOGGER.warning("Mesures nvidia-smi indisponibles : %s", reason)
        return {"available": False, "reason": reason}
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if any(len(row) != len(fields) for row in rows):
        raise ValueError("Sortie nvidia-smi inattendue.")
    return {
        "available": True,
        "units": {"utilization.gpu": "%", "memory.used": "MiB", "memory.total": "MiB"},
        "devices": [dict(zip(fields, row, strict=True)) for row in rows],
    }


def source_hash(name: str) -> str:
    digest = hashlib.sha256()
    files = [
        path
        for directory in (ROOT / "common", model_directory(name))
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".txt"}
    ]
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def record_input(input_data: InputData) -> InputData:
    data = dict(input_data) if isinstance(input_data, dict) else input_data
    if isinstance(data, dict) and "image" in data:
        from PIL.Image import Image

        image = data["image"]
        if isinstance(image, Image):
            data["image"] = {
                "mode": image.mode,
                "size": list(image.size),
                "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
    return data


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark(
    session: ModelSession,
    input_data: InputData,
    runs: int,
    warmup: int,
    load_seconds: float,
) -> dict[str, Any]:
    if runs < 1 or warmup < 0:
        raise ValueError("runs doit etre >= 1 et warmup >= 0.")
    metrics = DeviceMetrics(session.model)
    before = gpu_snapshot()
    for _ in range(warmup):
        session.infer(input_data)
    metrics.synchronize()
    metrics.reset_peak()
    samples = []
    for index in range(runs):
        metrics.synchronize()
        start = time.perf_counter()
        result = session.infer(input_data)
        metrics.synchronize()
        elapsed = time.perf_counter() - start
        samples.append({"seconds": elapsed, "result": result.to_dict()})
        LOGGER.info("Iteration %d/%d : %.3f s.", index + 1, runs, elapsed)
    latencies = [sample["seconds"] for sample in samples]
    elapsed = sum(latencies)
    token_counts = [sample["result"]["usage"]["output_tokens"] for sample in samples]
    known_tokens = all(isinstance(count, int) for count in token_counts)
    output_tokens = sum(token_counts) if known_tokens else None
    return {
        "schema_version": 1,
        "model": session.name,
        "config": session.config,
        "input": record_input(input_data),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "runtime": os.getenv("MODEL_FORGE_RUNTIME"),
            "base_image": os.getenv("MODEL_FORGE_BASE_IMAGE"),
            "device": str(metrics.device) if metrics.device is not None else None,
            "dtype": str(session.model.dtype) if hasattr(session.model, "dtype") else None,
            "source_sha256": source_hash(session.name),
            "packages": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
            },
        },
        "measurement": {
            "scope": "end_to_end_inference",
            "load_seconds": load_seconds,
            "warmup": warmup,
            "runs": runs,
            "latency_seconds": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": percentile(latencies, 0.95),
                "min": min(latencies),
                "max": max(latencies),
            },
            "requests_per_second": runs / elapsed,
            "generated_tokens_per_second": (
                output_tokens / elapsed if output_tokens is not None and output_tokens > 0 else None
            ),
            "memory": metrics.memory(),
            "gpu_before": before,
            "gpu_after": gpu_snapshot(),
        },
        "samples": samples,
    }
