from __future__ import annotations

import math
from typing import Any

from PIL import Image

from common.types import Config, InferenceResult, InputData
from .runtime import XiaomiOCRRuntime

MODES = ("regions", "page")


def _read_input(
    input_data: InputData, config: Config
) -> tuple[Image.Image, str, str]:
    if not isinstance(input_data, dict):
        raise ValueError("Xiaomi-OCR-0 exige une image dans input_data['image'].")
    unknown = set(input_data) - {"image", "state", "mode"}
    if unknown:
        raise ValueError(f"Champs OCR inconnus : {', '.join(sorted(unknown))}.")
    image = input_data.get("image")
    if not isinstance(image, Image.Image):
        raise ValueError("Fournir une image PIL dans input_data['image'].")
    state = input_data.get("state", "")
    if not isinstance(state, str):
        raise ValueError("state doit etre une instruction texte facultative.")
    mode = input_data.get("mode", config.get("default_mode", "regions"))
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"mode doit etre l'un de : {', '.join(MODES)}.")
    return image.convert("RGB"), state.strip(), mode


def _limit_image(image: Image.Image, config: Config) -> Image.Image:
    maximum = config.get("image_max_side", 2048)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("image_max_side doit etre un entier >= 1.")
    image = image.copy()
    image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    return image


def _prompt(mode: str, label: str, instruction: str) -> str:
    if mode == "page":
        prompt = (
            "Read the entire image and transcribe all visible text in Markdown. "
            "Preserve reading order, paragraphs, headings, lists, and tables where possible. "
            "Do not describe the image or invent missing text."
        )
    elif label.lower() in {"table", "table_region"}:
        prompt = (
            "Transcribe all text and structure in this table region. Return a Markdown table "
            "when the image contains tabular rows and columns. Do not invent missing cells."
        )
    elif any(term in label.lower() for term in ("formula", "equation")):
        prompt = "Transcribe the mathematical expression in this region using LaTeX."
    else:
        prompt = (
            "Transcribe all visible text in this image region in Markdown, preserving line "
            "breaks and reading order. Do not describe the image or invent missing text."
        )
    if instruction:
        prompt += f"\n\nAdditional user instruction: {instruction}"
    return prompt


def _generate(
    runtime: XiaomiOCRRuntime,
    processor: Any,
    image: Image.Image,
    prompt: str,
    config: Config,
) -> tuple[str, int, int]:
    import torch

    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _limit_image(image, config)},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise TypeError("Le processor Xiaomi doit fournir input_ids sous forme de tenseur 2D.")
    inputs = inputs.to(runtime.model.device)

    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation doit etre un objet.")
    max_new_tokens = generation.get("max_new_tokens", 4096)
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
        generated = runtime.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
        raise TypeError("La generation Xiaomi doit retourner une sequence 2D unique.")
    new_tokens = sequences[0, input_ids.shape[-1] :]
    decoded = processor.batch_decode(
        new_tokens.unsqueeze(0),
        skip_special_tokens=True,
    )
    if not decoded or not isinstance(decoded[0], str):
        raise TypeError("Le processor Xiaomi n'a pas decode la sequence generee.")
    return decoded[0].strip(), int(input_ids.shape[-1]), int(new_tokens.numel())


def _coordinates(block: dict[str, Any], width: int, height: int) -> list[int]:
    coordinates = block.get("bbox")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
        raise ValueError("Le detecteur doit retourner une bbox [x1, y1, x2, y2].")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in coordinates
    ):
        raise ValueError("Le detecteur a retourne une bbox non numerique.")
    result = [int(round(value)) for value in coordinates]
    x1, y1, x2, y2 = result
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError("Le detecteur a retourne une bbox vide ou hors de l'image.")
    return result


def _group_crops(
    crops: list[Any], width: int, height: int
) -> list[tuple[dict[str, Any], list[int]]]:
    grouped_indices: dict[int, int] = {}
    grouped_crops: list[tuple[dict[str, Any], list[int]]] = []
    for crop in crops:
        if not isinstance(crop, dict):
            raise TypeError("Chaque crop de mise en page doit etre un objet.")
        label = crop.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Chaque crop doit contenir un label texte.")
        bbox = _coordinates(crop, width, height)
        group_id = crop.get("group_id")
        if group_id is not None and (
            isinstance(group_id, bool) or not isinstance(group_id, int)
        ):
            raise ValueError("Le groupe de crops doit etre identifie par un entier.")
        if group_id is not None and group_id in grouped_indices:
            index = grouped_indices[group_id]
            first_crop, first_bbox = grouped_crops[index]
            if label != first_crop["label"]:
                raise ValueError("Les crops d'un meme groupe doivent partager le meme label.")
            grouped_crops[index] = (
                first_crop,
                [
                    min(first_bbox[0], bbox[0]),
                    min(first_bbox[1], bbox[1]),
                    max(first_bbox[2], bbox[2]),
                    max(first_bbox[3], bbox[3]),
                ],
            )
            continue
        if crop.get("image") is None:
            raise ValueError("Le crop de mise en page ne contient pas d'image.")
        if group_id is not None:
            grouped_indices[group_id] = len(grouped_crops)
        grouped_crops.append((crop, bbox))
    return grouped_crops


def _region_blocks(
    runtime: XiaomiOCRRuntime,
    processor: Any,
    image: Image.Image,
    instruction: str,
    config: Config,
) -> tuple[list[dict[str, Any]], str, int, int]:
    detections_by_image = runtime.detector.detect([image])
    if not isinstance(detections_by_image, list) or len(detections_by_image) != 1:
        raise TypeError("Le detecteur de mise en page doit retourner une liste pour une image.")
    detections = detections_by_image[0]
    if not isinstance(detections, list):
        raise TypeError("Les detections de mise en page doivent etre une liste.")
    crops = runtime.detector.crop(image, detections)
    if not isinstance(crops, list):
        raise TypeError("Le detecteur de mise en page doit retourner une liste de crops.")
    grouped_crops = _group_crops(crops, image.width, image.height)

    blocks: list[dict[str, Any]] = []
    markdown_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    for order, (crop, bbox) in enumerate(grouped_crops, start=1):
        label = crop["label"]
        crop_image = crop.get("image")
        if isinstance(crop_image, Image.Image):
            region_image = crop_image.convert("RGB")
        else:
            try:
                region_image = Image.fromarray(crop_image).convert("RGB")
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError("Chaque crop doit contenir une image PIL ou un tableau image.") from error
        if region_image.width < 1 or region_image.height < 1:
            raise ValueError("Le detecteur a retourne un crop vide.")
        text, input_count, output_count = _generate(
            runtime,
            processor,
            region_image,
            _prompt("regions", label, instruction),
            config,
        )
        score = crop.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("Le detecteur a retourne un score invalide.")
        blocks.append(
            {
                "order": order,
                "label": label,
                "bbox": bbox,
                "score": float(score),
                "content": text,
                "raw_content": text,
            }
        )
        if text:
            markdown_parts.append(text)
        input_tokens += input_count
        output_tokens += output_count
    return blocks, "\n\n".join(markdown_parts), input_tokens, output_tokens


def infer(
    model: XiaomiOCRRuntime,
    processor: Any,
    input_data: InputData,
    config: Config,
) -> InferenceResult:
    if processor is None:
        raise TypeError("Xiaomi-OCR-0 exige son AutoProcessor Transformers.")
    image, instruction, mode = _read_input(input_data, config)
    if mode == "page":
        markdown, input_tokens, output_tokens = _generate(
            model,
            processor,
            image,
            _prompt("page", "", instruction),
            config,
        )
        blocks = [
            {
                "order": 1,
                "label": "page",
                "bbox": [0, 0, image.width, image.height],
                "content": markdown,
                "raw_content": markdown,
            }
        ]
    else:
        blocks, markdown, input_tokens, output_tokens = _region_blocks(
            model,
            processor,
            image,
            instruction,
            config,
        )
    return InferenceResult(
        {
            "image_width": image.width,
            "image_height": image.height,
            "mode": mode,
            "crop_blocks": blocks,
            "markdown": markdown,
        },
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
