from collections.abc import Generator
from threading import Event
from typing import Any

from common.transformers import infer_causal, stream_causal
from common.types import Config, InferenceResult, InputData, StreamChunk


def infer(
    model: Any, processor: Any, input_data: InputData, config: Config
) -> InferenceResult:
    return infer_causal(model, processor, input_data, config)


def stream(
    model: Any, processor: Any, input_data: InputData, config: Config, cancel: Event
) -> Generator[StreamChunk, None, None]:
    yield from stream_causal(model, processor, input_data, config, cancel)
