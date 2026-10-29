from __future__ import annotations

import re
from pathlib import Path

import yaml

from common.types import Config

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
NAME_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")


def model_directory(name: str) -> Path:
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Nom de modele invalide : {name!r}.")
    directory = MODELS_DIR / name
    if not directory.is_dir():
        raise ValueError(f"Modele inconnu : {name}. Disponibles : {', '.join(list_models())}.")
    return directory


def list_models() -> list[str]:
    return sorted(
        path.name
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "config.yaml").is_file()
    )


def load_config(
    name: str, path: Path | None = None, overrides: Config | None = None
) -> Config:
    directory = model_directory(name)
    config_path = path if path is not None else directory / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("La configuration YAML doit etre un objet.")
    if not all(isinstance(key, str) for key in config):
        raise ValueError("Les cles de configuration doivent etre des chaines.")
    if overrides:
        config.update(overrides)
    if not isinstance(config.get("model_id"), str) or not config["model_id"].strip():
        raise ValueError("La configuration doit definir model_id.")
    for key in ("trust_remote_code", "local_files_only"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} doit etre un booleen.")
    seed = config.setdefault("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed doit etre un entier entre 0 et 2**32 - 1.")
    config.setdefault("device", "auto")
    config.setdefault("dtype", "auto")
    if not isinstance(config["device"], str) or not isinstance(config["dtype"], str):
        raise ValueError("device et dtype doivent etre des chaines.")
    generation = config.setdefault("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation doit etre un objet.")
    return config
