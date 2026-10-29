from __future__ import annotations

import random
from typing import Any


def resolve_device(requested: str = "auto") -> Any:
    import torch

    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demande mais indisponible. Verifier le runtime NVIDIA.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"GPU inexistant : {device}.")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS demande mais indisponible.")
    elif device.type != "cpu":
        raise ValueError(f"Device non gere par ce chargeur PyTorch : {device}.")
    return device


def resolve_dtype(requested: str, device: Any) -> Any:
    import torch

    if requested == "auto":
        if device.type == "cuda":
            with torch.cuda.device(device):
                requested = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            requested = "float16" if device.type == "mps" else "float32"
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested not in dtypes:
        raise ValueError(f"dtype non pris en charge : {requested}.")
    if requested == "bfloat16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Ce GPU ne prend pas en charge bfloat16.")
    return dtypes[requested]


def seed_torch(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
