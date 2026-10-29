from typing import Any

from common.transformers import infer_causal
from common.types import Config, InferenceResult, InputData


def infer(
    model: Any, processor: Any, input_data: InputData, config: Config
) -> InferenceResult:
    return infer_causal(model, processor, input_data, config)
