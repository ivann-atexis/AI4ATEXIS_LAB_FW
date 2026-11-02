from __future__ import annotations

from typing import Any

from common.device import resolve_device, resolve_dtype, seed_torch
from common.hub import download_snapshot
from common.transformers import require_transformers
from common.types import Config


def load_model(config: Config) -> tuple[Any, Any]:
    require_transformers()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = resolve_device(config["device"])
    dtype = resolve_dtype(config["dtype"], device)
    seed_torch(config["seed"])
    snapshot = str(download_snapshot(config))
    loading_options = {
        "trust_remote_code": config.get("trust_remote_code", False),
        "local_files_only": True,
    }
    processor = AutoProcessor.from_pretrained(snapshot, **loading_options)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        dtype=dtype,
        attn_implementation=config.get("attention_implementation", "sdpa"),
        **loading_options,
    ).to(device).eval()
    return model, processor
