from collections.abc import Generator
from threading import Event
from typing import Any

from common.transformers import infer_causal, stream_causal
from common.types import Config, InferenceResult, InputData, StreamChunk

from .tools import finish_tools, stream_tools

def infer(
    model: Any, processor: Any, input_data: InputData, config: Config
) -> InferenceResult:
    return finish_tools(infer_causal(model, processor, input_data, config), input_data)


def stream(
    model: Any, processor: Any, input_data: InputData, config: Config, cancel: Event
) -> Generator[StreamChunk, None, None]:
    yield from stream_tools(stream_causal(model, processor, input_data, config, cancel), input_data)
