from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from common.types import Config

LOGGER = logging.getLogger(__name__)


def download_snapshot(config: Config) -> Path:
    from huggingface_hub import snapshot_download

    revision = config.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("Un snapshot HF exige une revision SHA complete (40 caracteres).")
    if config.get("trust_remote_code", False):
        LOGGER.warning(
            "Code distant autorise pour %s, revision %s. Verifier sa provenance et sa licence.",
            config["model_id"],
            config["revision"],
        )
    offline = config.get("local_files_only", False) or os.getenv(
        "HF_HUB_OFFLINE", "0"
    ).upper() in {"1", "TRUE", "YES", "ON"}
    snapshot = Path(
        snapshot_download(
            repo_id=config["model_id"],
            revision=config["revision"],
            local_files_only=offline,
        )
    )
    config["resolved_revision"] = snapshot.name
    return snapshot
