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
class ToolCall:
    id: str
    name: str
    arguments: dict[str, JSONValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False, allow_nan=False),
            },
        }


@dataclass(frozen=True)
class InferenceResult:
    output: JSONValue
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("Les compteurs de tokens doivent etre des entiers >= 0.")
        if self.finish_reason not in (None, "stop", "length", "tool_calls"):
            raise ValueError("finish_reason invalide.")
        if bool(self.tool_calls) != (self.finish_reason == "tool_calls"):
            raise ValueError("tool_calls et finish_reason doivent etre coherents.")
        for call in self.tool_calls:
            if not isinstance(call, ToolCall):
                raise TypeError("ToolCall attendu.")
            json.dumps(call.to_dict(), allow_nan=False)
        json.dumps(self.output, allow_nan=False)

    def to_dict(self) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "output": self.output,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }
        if self.tool_calls:
            result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return result


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
