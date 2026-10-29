from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
Config: TypeAlias = dict[str, Any]
InputData: TypeAlias = str | dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    output: JSONValue
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("Les compteurs de tokens doivent etre des entiers >= 0.")
        json.dumps(self.output, allow_nan=False)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "output": self.output,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }


@dataclass(frozen=True)
class AppOptions:
    input_data: InputData
    host: str = "127.0.0.1"
    port: int = 7860
