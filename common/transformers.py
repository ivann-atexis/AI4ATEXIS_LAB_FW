from __future__ import annotations

from importlib.metadata import version
from typing import Any

from common.device import resolve_device, resolve_dtype, seed_torch
from common.hub import download_snapshot
from common.types import Config, InferenceResult, InputData


def require_transformers(minimum: str = "5.19.0") -> None:
    from packaging.version import Version

    installed = version("transformers")
    if Version(installed) < Version(minimum):
        raise RuntimeError(
            f"Transformers >= {minimum} requis, version installee : {installed}. "
            "Utiliser l'image du framework : docker compose build lab."
        )


def load_causal_model(config: Config) -> tuple[Any, Any]:
    require_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config["device"])
    dtype = resolve_dtype(config["dtype"], device)
    seed_torch(config["seed"])
    snapshot = str(download_snapshot(config))
    options = {
        "trust_remote_code": config.get("trust_remote_code", False),
        "local_files_only": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(snapshot, **options)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype, **options)
    return model.to(device).eval(), tokenizer


def chat_messages(input_data: InputData) -> list[dict[str, str]]:
    if isinstance(input_data, str):
        if not input_data.strip():
            raise ValueError("Le prompt ne doit pas etre vide.")
        return [{"role": "user", "content": input_data}]
    if not isinstance(input_data, dict):
        raise ValueError("Entree attendue : texte, {prompt: ...} ou {messages: [...]}.")
    if set(input_data) == {"prompt"}:
        if not isinstance(input_data["prompt"], str):
            raise ValueError("prompt doit etre un texte.")
        return chat_messages(input_data["prompt"])
    if set(input_data) != {"messages"}:
        raise ValueError("Utiliser uniquement prompt ou messages pour ce modele texte.")
    messages = input_data["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages doit etre une liste non vide.")
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise ValueError("Chaque message exige un role system/user/assistant et un texte.")
    return messages


def infer_causal(
    model: Any, tokenizer: Any, input_data: InputData, config: Config
) -> InferenceResult:
    import torch

    messages = chat_messages(input_data)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **config.get("chat_template", {}),
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, **config["generation"])
    sequences = getattr(outputs, "sequences", outputs)
    generated = sequences[0, input_tokens:]
    return InferenceResult(
        output=tokenizer.decode(generated, skip_special_tokens=True),
        input_tokens=input_tokens,
        output_tokens=generated.numel(),
    )
