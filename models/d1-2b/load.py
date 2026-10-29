from typing import Any

from common.device import resolve_device, resolve_dtype, seed_torch
from common.hub import download_snapshot
from common.transformers import require_transformers
from common.types import Config


def load_model(config: Config) -> tuple[Any, None]:
    if config.get("generation"):
        raise ValueError("d1 utilise system_one(), sans parametres de generation.")
    require_transformers()
    from transformers import AutoModel

    device = resolve_device(config["device"])
    dtype = resolve_dtype(config["dtype"], device)
    seed_torch(config["seed"])
    snapshot = str(download_snapshot(config))
    model = AutoModel.from_pretrained(
        snapshot,
        dtype=dtype,
        trust_remote_code=config.get("trust_remote_code", False),
        local_files_only=True,
    )
    return model.to(device).eval(), None
