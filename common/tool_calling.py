from __future__ import annotations

import copy
import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def json_object(text: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Argument JSON duplique : {key}.")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Les arguments d'un outil doivent etre un objet JSON.")
    json.dumps(value, allow_nan=False)
    return value


def tool_schemas(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schemas = {}
    if not isinstance(tools, list):
        raise ValueError("tools doit etre une liste.")

    def local_references(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef"} and (
                    not isinstance(item, str) or not item.startswith("#")
                ):
                    raise ValueError("Les schemas outils ne peuvent pas charger de references externes.")
                local_references(item)
        elif isinstance(value, list):
            for item in value:
                local_references(item)

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Seuls les outils function sont pris en charge.")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) or name in schemas:
            raise ValueError("Les noms d'outils doivent etre non vides et uniques.")
        schema = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(schema, dict):
            raise ValueError("parameters doit etre un schema JSON objet.")
        local_references(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ValueError(f"Schema JSON invalide pour l'outil {name}.") from error
        schemas[name] = schema
    return schemas


def tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    pending: set[str] = set()
    seen: set[str] = set()
    previous_tool: str | None = None
    for message in normalized:
        if not isinstance(message, dict) or set(message) - {
            "role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"
        }:
            raise ValueError("Champs de message non pris en charge.")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Role de message non pris en charge.")
        if role != "tool" and pending:
            raise ValueError("Les resultats des outils doivent suivre l'appel assistant.")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
            raise ValueError("tool_calls doit etre une liste d'appels.")
        if calls and role != "assistant":
            raise ValueError("Seul un message assistant peut contenir tool_calls.")
        if role == "tool":
            identifier = message.get("tool_call_id")
            if not isinstance(identifier, str) or not identifier or (
                identifier not in pending and identifier != previous_tool
            ):
                raise ValueError("tool_call_id ne correspond a aucun appel en attente.")
            pending.discard(identifier)
            previous_tool = identifier
        else:
            previous_tool = None
            if message.get("tool_call_id") is not None:
                raise ValueError("tool_call_id est reserve aux messages tool.")
        if message.get("content") is None:
            if not calls:
                raise ValueError("Un message sans appel d'outil exige un contenu.")
            message["content"] = ""
        if not isinstance(message["content"], (str, list)):
            raise ValueError("Le contenu doit etre du texte ou des parties de contenu.")
        if isinstance(message["content"], list):
            for part in message["content"]:
                if not isinstance(part, dict) or (
                    part.get("type") == "text" and not isinstance(part.get("text"), str)
                ) or part.get("type") not in {"text", "tool_reference"}:
                    raise ValueError("Partie de contenu non prise en charge.")
                if part["type"] == "tool_reference" and (
                    role != "tool" or not isinstance(part.get("name"), str)
                ):
                    raise ValueError("Reference d'outil invalide.")
        for call in calls:
            identifier = call.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("Chaque appel d'outil exige un id unique.")
            if call.get("type") != "function" or not isinstance(call.get("function"), dict):
                raise ValueError("Appel d'outil function attendu.")
            function = call["function"]
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise ValueError("Nom de fonction manquant.")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json_object(arguments)
            elif not isinstance(arguments, dict):
                raise ValueError("Les arguments doivent etre un objet JSON.")
            else:
                json.dumps(arguments, allow_nan=False)
            pending.add(identifier)
            seen.add(identifier)
    if pending:
        raise ValueError("Il manque des resultats d'outils dans l'historique.")
    return normalized
