from typing import Any

from common.transformers import load_causal_model
from common.types import Config


def load_model(config: Config) -> tuple[Any, Any]:
    return load_causal_model(config)
