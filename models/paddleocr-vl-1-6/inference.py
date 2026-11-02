from __future__ import annotations

from typing import Any, Literal

from PIL import Image

from common.types import Config, InferenceResult, InputData

TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def _read_input(
    input_data: InputData, config: Config
) -> tuple[Image.Image, str, str]:
    if not isinstance(input_data, dict):
        raise ValueError("PaddleOCR-VL exige une image dans input_data['image'].")
    unknown = set(input_data) - {"image", "task", "mode", "state"}
    if unknown:
        raise ValueError(f"Champs inconnus : {', '.join(sorted(unknown))}.")
    image = input_data.get("image")
    if not isinstance(image, Image.Image):
        raise ValueError("Fournir une image PIL dans input_data['image'].")
    if "task" in input_data and "mode" in input_data:
        raise ValueError("Fournir task ou mode, pas les deux.")
    task = input_data.get(
        "task", input_data.get("mode", config.get("default_task", "ocr"))
    )
    if not isinstance(task, str) or task not in TASK_PROMPTS:
        raise ValueError(f"task doit etre l'un de : {', '.join(TASK_PROMPTS)}.")
    instruction = input_data.get("state", "")
    if not isinstance(instruction, str):
        raise ValueError("state doit etre une instruction texte facultative.")
    return image.convert("RGB"), task, instruction.strip()


def _finish_reason(
    model: Any, new_tokens: Any, count: int, maximum: int
) -> Literal["stop", "length"]:
    if count < maximum:
        return "stop"
    if count:
        eos_token_id = getattr(
            getattr(model, "generation_config", None), "eos_token_id", None
        )
        eos_ids: set[int] = set()
        if isinstance(eos_token_id, int) and not isinstance(eos_token_id, bool):
            eos_ids.add(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids.update(
                token
                for token in eos_token_id
                if isinstance(token, int) and not isinstance(token, bool)
            )
        if int(new_tokens[-1].item()) in eos_ids:
            return "stop"
    return "length"


def _positive_config_int(config: Config, key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} doit etre un entier >= 1.")
    return value


def infer(
    model: Any,
    processor: Any,
    input_data: InputData,
    config: Config,
) -> InferenceResult:
    import torch

    if processor is None:
        raise TypeError("PaddleOCR-VL exige son AutoProcessor Transformers.")
    image, task, instruction = _read_input(input_data, config)
    original_size = image.size
    max_pixels = _positive_config_int(
        config,
        "spotting_max_pixels" if task == "spotting" else "max_pixels",
        1605632 if task == "spotting" else 1280 * 28 * 28,
    )
    if task == "spotting":
        upscale_threshold = _positive_config_int(
            config, "spotting_upscale_threshold", 1500
        )
        if image.width < upscale_threshold and image.height < upscale_threshold:
            image = image.resize(
                (image.width * 2, image.height * 2),
                Image.Resampling.LANCZOS,
            )

    prompt = TASK_PROMPTS[task]
    if instruction:
        prompt = f"{prompt}\n\n{instruction}"
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": processor.image_processor.size.shortest_edge,
                    "longest_edge": max_pixels,
                }
            }
        },
    )
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise TypeError("Le processor doit fournir input_ids sous forme de tenseur 2D.")
    inputs = inputs.to(model.device)

    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation doit etre un objet.")
    max_new_tokens = generation.get("max_new_tokens", 512)
    do_sample = generation.get("do_sample", False)
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("generation.max_new_tokens doit etre un entier >= 1.")
    if not isinstance(do_sample, bool):
        raise ValueError("generation.do_sample doit etre un booleen.")

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if (
        not isinstance(sequences, torch.Tensor)
        or sequences.ndim != 2
        or sequences.shape[0] != 1
    ):
        raise TypeError("La generation doit retourner une sequence 2D unique.")
    new_tokens = sequences[0, input_ids.shape[-1] :]
    decoded = processor.batch_decode(
        new_tokens.unsqueeze(0),
        skip_special_tokens=True,
    )
    if not decoded or not isinstance(decoded[0], str):
        raise TypeError("Le processor n'a pas decode la sequence generee.")
    text = decoded[0].strip()
    output_tokens = int(new_tokens.numel())
    width, height = original_size
    block = {
        "order": 1,
        "label": task,
        "bbox": [0, 0, width, height],
        "content": text,
        "raw_content": text,
    }
    return InferenceResult(
        {
            "image_width": width,
            "image_height": height,
            "task": task,
            "crop_blocks": [block],
            "markdown": text,
        },
        input_tokens=int(input_ids.shape[-1]),
        output_tokens=output_tokens,
        finish_reason=_finish_reason(model, new_tokens, output_tokens, max_new_tokens),
    )
