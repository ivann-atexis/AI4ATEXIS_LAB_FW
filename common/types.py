from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, TypeAlias

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
    finish_reason: Literal["stop", "length"] | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("Les compteurs de tokens doivent etre des entiers >= 0.")
        if self.finish_reason not in (None, "stop", "length"):
            raise ValueError("finish_reason doit valoir stop, length ou None.")
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
class StreamChunk:
    text: str = ""
    result: InferenceResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Le delta de streaming doit etre un texte.")
        if self.result is not None:
            if not isinstance(self.result, InferenceResult):
                raise TypeError("Le resultat final doit etre un InferenceResult.")
            if self.text:
                raise ValueError("Le chunk final ne doit pas contenir de delta.")


@dataclass(frozen=True)
class AppOptions:
    input_data: InputData
    host: str = "127.0.0.1"
    port: int = 7860
