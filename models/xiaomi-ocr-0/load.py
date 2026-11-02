from __future__ import annotations

from copy import deepcopy
from typing import Any

from common.device import resolve_device, resolve_dtype, seed_torch
from common.hub import download_snapshot
from common.transformers import require_transformers
from common.types import Config
from .runtime import XiaomiOCRRuntime


def _layout_components() -> tuple[dict[str, Any], Any]:
    from minicpmvocr.config import DEFAULT_CONFIG
    from minicpmvocr.layouts import create_layout_detector

    return DEFAULT_CONFIG, create_layout_detector


def load_model(config: Config) -> tuple[XiaomiOCRRuntime, Any]:
    require_transformers()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = resolve_device(config["device"])
    dtype = resolve_dtype(config["dtype"], device)
    seed_torch(config["seed"])
    model_snapshot = str(download_snapshot(config))
    loading_options = {
        "trust_remote_code": config.get("trust_remote_code", False),
        "local_files_only": True,
    }
    processor = AutoProcessor.from_pretrained(model_snapshot, **loading_options)
    model = AutoModelForImageTextToText.from_pretrained(
        model_snapshot,
        dtype=dtype,
        **loading_options,
    ).to(device).eval()

    default_config, create_layout_detector = _layout_components()
    layout_config = config["layout"]
    snapshot_config: Config = {
        "model_id": layout_config["model_id"],
        "revision": layout_config["revision"],
        "trust_remote_code": False,
        "local_files_only": config.get("local_files_only", False),
    }
    layout_model_dir = download_snapshot(snapshot_config)
    sdk_layout = deepcopy(default_config["layout"])
    sdk_layout.update(
        {
            "enabled": True,
            "provider": layout_config["provider"],
            "model_dir": str(layout_model_dir),
            "device": layout_config["device"],
            "threshold": layout_config["threshold"],
            "nms": layout_config["nms"],
        }
    )
    detector = create_layout_detector(sdk_layout)
    detector.start()
    return XiaomiOCRRuntime(model=model, detector=detector), processor
