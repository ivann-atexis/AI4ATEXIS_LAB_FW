from __future__ import annotations

import json
import re
from collections.abc import Generator
from dataclasses import replace
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from referencing.exceptions import Unresolvable

from common.tool_calling import tool_schemas
from common.types import InferenceResult, InputData, StreamChunk, ToolCall

OPEN = "<tool_call>"
BLOCK = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_-]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAMETER = re.compile(r"<parameter=([^<>\r\n]+)>(.*?)</parameter>", re.DOTALL)
PROTECTED = re.compile(r"```.*?(?:```|$)|<think>.*?(?:</think>|$)", re.DOTALL)


def finish_tools(result: InferenceResult, data: InputData) -> InferenceResult:
    if not isinstance(data, dict) or not data.get("tools") or data.get("tool_choice") == "none":
        return result
    if not isinstance(result.output, str):
        raise TypeError("Une sortie texte est attendue du backend LFM3.")
    schemas = tool_schemas(data["tools"])
    text = result.output
    calls = []
    content = []
    position = 0
    blocks = list(BLOCK.finditer(text))
    protected = [
        match.span() for match in PROTECTED.finditer(text)
        if not any(block.start() <= match.start() < block.end() for block in blocks)
    ]

    def is_protected(index: int) -> bool:
        return any(start <= index < end for start, end in protected)

    def malformed(start: int, end: int) -> bool:
        return any(
            not is_protected(start + match.start())
            for match in re.finditer(r"</?tool_call>", text[start:end])
        )

    for block in blocks:
        if is_protected(block.start()):
            continue
        prefix = text[position:block.start()]
        if malformed(position, block.start()):
            raise RuntimeError("Appel d'outil LFM3 incomplet ou mal forme.")
        content.append(prefix)
        name, body = block.groups()
        if name not in schemas:
            raise RuntimeError(f"LFM3 a appele un outil inconnu : {name}.")
        schema = schemas[name]
        validator = Draft202012Validator(schema)
        arguments: dict[str, Any] = {}
        consumed = 0
        for parameter in PARAMETER.finditer(body):
            if body[consumed:parameter.start()].strip():
                raise RuntimeError("Arguments LFM3 mal formes.")
            key, raw = parameter.groups()
            if key in arguments:
                raise RuntimeError("Argument d'outil LFM3 duplique.")
            raw = raw.removeprefix("\n").removesuffix("\n")
            property_schema = schema.get("properties", {}).get(key, {})
            value: Any = raw
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = raw
            try:
                property_validator = validator.evolve(schema=property_schema)
                if isinstance(decoded, str) and property_validator.is_valid(raw):
                    value = raw
                elif property_validator.is_valid(decoded):
                    value = decoded
                elif not property_validator.is_valid(raw):
                    raise RuntimeError(f"Argument LFM3 invalide : {name}.{key}.")
            except Unresolvable as error:
                raise RuntimeError(f"Reference de schema non resolue : {name}.{key}.") from error
            arguments[key] = value
            consumed = parameter.end()
        try:
            valid = validator.is_valid(arguments)
        except Unresolvable as error:
            raise RuntimeError(f"Reference de schema non resolue : {name}.") from error
        if body[consumed:].strip() or not valid:
            raise RuntimeError(f"Arguments LFM3 non conformes au schema : {name}.")
        try:
            json.dumps(arguments, allow_nan=False)
        except ValueError as error:
            raise RuntimeError("Arguments LFM3 non finis.") from error
        calls.append(ToolCall("call_" + uuid4().hex, name, arguments))
        position = block.end()
    tail = text[position:]
    if malformed(position, len(text)) or (
        tail.rstrip().endswith(("<tool_", "<tool_call")) and not is_protected(len(text) - 1)
    ):
        raise RuntimeError("Appel d'outil LFM3 tronque ou mal forme ; augmenter max_tokens.")
    content.append(tail)
    choice = data.get("tool_choice", "auto")
    if (choice == "required" or isinstance(choice, dict)) and not calls:
        raise RuntimeError("LFM3 n'a pas produit l'appel d'outil demande.")
    if isinstance(choice, dict) and any(call.name != choice["function"]["name"] for call in calls):
        raise RuntimeError("LFM3 n'a pas respecte l'outil impose.")
    if data.get("parallel_tool_calls") is False and len(calls) > 1:
        raise RuntimeError("LFM3 a produit plusieurs appels malgre parallel_tool_calls=false.")
    return replace(
        result,
        output="".join(content),
        tool_calls=tuple(calls),
        finish_reason="tool_calls" if calls else result.finish_reason,
    )


def stream_tools(
    chunks: Generator[StreamChunk, None, None], data: InputData
) -> Generator[StreamChunk, None, None]:
    active = isinstance(data, dict) and bool(data.get("tools")) and data.get("tool_choice") != "none"
    if not active:
        yield from chunks
        return
    pending = ""
    emitted = ""
    hidden = False
    try:
        for chunk in chunks:
            if chunk.result is not None:
                result = finish_tools(chunk.result, data)
                if not isinstance(result.output, str) or not result.output.startswith(emitted):
                    raise RuntimeError("Le texte final LFM3 differe du flux emis.")
                tail = result.output[len(emitted):]
                if tail:
                    yield StreamChunk(text=tail)
                yield StreamChunk(result=result)
            elif not hidden and chunk.text:
                pending += chunk.text
                index = pending.find(OPEN)
                if index >= 0:
                    safe, pending, hidden = pending[:index], "", True
                else:
                    keep = max(
                        (size for size in range(1, len(OPEN)) if pending.endswith(OPEN[:size])),
                        default=0,
                    )
                    safe = pending[:-keep] if keep else pending
                    pending = pending[-keep:] if keep else ""
                if safe:
                    emitted += safe
                    yield StreamChunk(text=safe)
            else:
                yield StreamChunk()
    finally:
        chunks.close()
