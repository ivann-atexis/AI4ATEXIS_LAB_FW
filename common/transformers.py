from __future__ import annotations

import math
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from queue import Empty
from threading import Event
from typing import Any

from common.device import resolve_device, resolve_dtype, seed_torch
from common.hub import download_snapshot
from common.types import Config, InferenceResult, InputData, StreamChunk


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


def chat_messages(input_data: InputData, *, allow_tools: bool = False) -> list[dict[str, Any]]:
    if isinstance(input_data, str):
        if not input_data.strip():
            raise ValueError("Le prompt ne doit pas etre vide.")
        return [{"role": "user", "content": input_data}]
    if not isinstance(input_data, dict):
        raise ValueError("Entree attendue : texte, {prompt: ...} ou {messages: [...]}.")
    fields = set(input_data) - (
        {"generation", "tools", "tool_choice", "parallel_tool_calls"} if allow_tools else {"generation"}
    )
    if fields == {"prompt"}:
        if not isinstance(input_data["prompt"], str):
            raise ValueError("prompt doit etre un texte.")
        return chat_messages(input_data["prompt"])
    if fields != {"messages"}:
        raise ValueError("Utiliser prompt ou messages, avec generation facultatif.")
    messages = input_data["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages doit etre une liste non vide.")
    if allow_tools:
        from common.tool_calling import tool_messages

        return tool_messages(messages)
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise ValueError("Chaque message exige un role system/user/assistant et un texte.")
    return messages


def causal_generation(input_data: InputData, config: Config) -> Config:
    generation = dict(config["generation"])
    overrides = input_data.get("generation", {}) if isinstance(input_data, dict) else {}
    allowed = {"max_new_tokens", "temperature", "top_p", "do_sample", "stop_strings"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("Parametres generation non pris en charge pour cette requete.")
    if any(value is None for value in overrides.values()):
        raise ValueError("Les surcharges generation ne doivent pas contenir null.")
    generation.update(overrides)
    maximum = generation.get("max_new_tokens")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
    ):
        raise ValueError("max_new_tokens doit etre un entier >= 1.")
    for key, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = generation.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lower <= value <= upper
            or (key == "top_p" and value == 0)
        ):
            raise ValueError(f"{key} hors limites.")
    if "do_sample" in generation and not isinstance(generation["do_sample"], bool):
        raise ValueError("do_sample doit etre un booleen.")
    if "temperature" in overrides:
        generation["do_sample"] = overrides["temperature"] > 0
    elif "top_p" in overrides:
        generation["do_sample"] = True
    if generation.get("temperature") == 0:
        generation["do_sample"] = False
    if not generation.get("do_sample", False):
        for key in ("temperature", "top_p", "top_k"):
            generation.pop(key, None)
    stops = generation.get("stop_strings", [])
    if isinstance(stops, str):
        stops = [stops]
    if (
        not isinstance(stops, list)
        or len(stops) > 4
        or any(not isinstance(stop, str) or not stop for stop in stops)
    ):
        raise ValueError("stop_strings doit contenir au plus quatre textes non vides.")
    if stops:
        generation["stop_strings"] = stops
    else:
        generation.pop("stop_strings", None)
    return generation


def causal_inputs(model: Any, tokenizer: Any, input_data: InputData, config: Config) -> Any:
    tools_enabled = config.get("api", {}).get("tool_calling", False)
    messages = chat_messages(input_data, allow_tools=tools_enabled)
    options = dict(config.get("chat_template", {}))
    if tools_enabled and isinstance(input_data, dict):
        from common.tool_calling import tool_schemas

        tools = input_data.get("tools") or []
        schemas = tool_schemas(tools)
        choice = input_data.get("tool_choice", "auto")
        if isinstance(choice, dict):
            function = choice.get("function")
            if choice.get("type") != "function" or not isinstance(function, dict):
                raise ValueError("tool_choice invalide.")
            if function.get("name") not in schemas:
                raise ValueError("L'outil impose n'est pas fourni.")
        elif choice not in (None, "auto", "none", "required"):
            raise ValueError("tool_choice invalide.")
        if choice == "required" and not tools:
            raise ValueError("tool_choice=required exige des outils.")
        if choice != "none" and tools:
            options["tools"] = tools
            if choice == "required" or isinstance(choice, dict):
                instruction = (
                    f"You must call the provided tool {choice['function']['name']}, "
                    "and no other tool, before answering."
                    if isinstance(choice, dict)
                    else "You must call one of the provided tools before answering."
                )
                if messages[0]["role"] == "system":
                    messages[0]["content"] += "\n\n" + instruction
                else:
                    messages = [{"role": "system", "content": instruction}, *messages]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **options,
    ).to(model.device)
    return inputs


class StopTextBuffer:
    def __init__(self, stops: list[str]) -> None:
        self.stops = stops
        self.pending = ""
        self.stopped = False

    def push(self, text: str) -> str:
        if self.stopped:
            return ""
        self.pending += text
        positions = [self.pending.find(stop) for stop in self.stops]
        found = [position for position in positions if position >= 0]
        if found:
            text = self.pending[: min(found)]
            self.pending = ""
            self.stopped = True
            return text
        # Withhold only a suffix that could still become a stop string.
        keep = max(
            (
                size
                for stop in self.stops
                for size in range(1, len(stop))
                if self.pending.endswith(stop[:size])
            ),
            default=0,
        )
        text = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        return text

    def finish(self) -> str:
        text, self.pending = self.pending, ""
        return text


def causal_result(
    model: Any, tokenizer: Any, outputs: Any, input_tokens: int, generation: Config
) -> InferenceResult:
    sequences = getattr(outputs, "sequences", outputs)
    generated = sequences[0, input_tokens:]
    buffer = StopTextBuffer(generation.get("stop_strings", []))
    text = buffer.push(
        tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    ) + buffer.finish()
    eos = generation.get(
        "eos_token_id", getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    )
    eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
    ended = generated.numel() > 0 and generated[-1].item() in eos_ids
    maximum = generation.get("max_new_tokens")
    limit_reached = maximum is not None and generated.numel() >= maximum
    return InferenceResult(
        output=text,
        input_tokens=input_tokens,
        output_tokens=generated.numel(),
        finish_reason="length" if limit_reached and not ended and not buffer.stopped else "stop",
    )


def infer_causal(
    model: Any, tokenizer: Any, input_data: InputData, config: Config
) -> InferenceResult:
    import torch

    generation = causal_generation(input_data, config)
    inputs = causal_inputs(model, tokenizer, input_data, config)
    options = dict(generation)
    if options.get("stop_strings"):
        options["tokenizer"] = tokenizer
    with torch.inference_mode():
        outputs = model.generate(**inputs, **options)
    return causal_result(model, tokenizer, outputs, inputs["input_ids"].shape[-1], generation)


def stream_causal(
    model: Any, tokenizer: Any, input_data: InputData, config: Config, cancel: Event
) -> Generator[StreamChunk, None, None]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            return torch.full(
                (input_ids.shape[0],),
                cancel.is_set(),
                dtype=torch.bool,
                device=input_ids.device,
            )

    generation = causal_generation(input_data, config)
    if generation.get("num_beams", 1) != 1 or generation.get("num_return_sequences", 1) != 1:
        raise ValueError("Le streaming exige une seule sequence sans beam search.")
    inputs = causal_inputs(model, tokenizer, input_data, config)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        timeout=0.1,
    )
    options = dict(generation)
    options["stopping_criteria"] = StoppingCriteriaList(
        [*generation.get("stopping_criteria", []), Cancelled()]
    )
    if options.get("stop_strings"):
        options["tokenizer"] = tokenizer

    def generate() -> InferenceResult:
        with torch.inference_mode():
            outputs = model.generate(**inputs, **options, streamer=streamer)
        return causal_result(model, tokenizer, outputs, inputs["input_ids"].shape[-1], generation)

    buffer = StopTextBuffer(generation.get("stop_strings", []))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-stream") as executor:
        future = executor.submit(generate)
        try:
            while not cancel.is_set():
                try:
                    text = next(streamer)
                except Empty:
                    if future.done():
                        future.result()
                        raise RuntimeError("Le backend n'a pas termine son streamer.")
                    yield StreamChunk()
                    continue
                except StopIteration:
                    break
                delta = buffer.push(text)
                if delta:
                    yield StreamChunk(text=delta)
            if not cancel.is_set():
                result = future.result()
                tail = buffer.finish()
                if tail:
                    yield StreamChunk(text=tail)
                yield StreamChunk(result=result)
        finally:
            cancel.set()
