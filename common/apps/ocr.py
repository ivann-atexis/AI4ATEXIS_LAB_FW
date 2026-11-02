from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from common.runtime import ModelSession
from common.types import AppOptions

LOGGER = logging.getLogger(__name__)


def _bbox(block: dict[str, Any]) -> tuple[int, int, int, int]:
    coordinates = block.get("bbox")
    if not isinstance(coordinates, list) or len(coordinates) < 4:
        raise ValueError("La sortie OCR exige une bbox [x1, y1, x2, y2].")
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in coordinates[:4])
    except (TypeError, ValueError) as error:
        raise ValueError("La sortie OCR contient une bbox non numerique.") from error
    if x2 <= x1 or y2 <= y1:
        raise ValueError("La sortie OCR contient une bbox vide ou invalide.")
    return x1, y1, x2, y2


def _draw_regions(image: Image.Image, blocks: list[dict[str, Any]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for index, block in enumerate(blocks, start=1):
        x1, y1, x2, y2 = _bbox(block)
        label = str(block.get("label") or "region")
        draw.rectangle((x1, y1, x2, y2), outline="#ff3b30", width=3)
        draw.text((x1 + 3, max(0, y1 - 14)), f"{index} {label}", fill="#ff3b30", font=font)
    return annotated


def _selected_block(blocks: list[dict[str, Any]], position: Any) -> dict[str, Any]:
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        raise ValueError("Coordonnees de clic indisponibles pour cette image.")
    x, y = position
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (x, y)):
        raise ValueError("Coordonnees de clic invalides.")
    matches = []
    for index, block in enumerate(blocks, start=1):
        x1, y1, x2, y2 = _bbox(block)
        if x1 <= x <= x2 and y1 <= y <= y2:
            matches.append((index, (x2 - x1) * (y2 - y1), block))
    if not matches:
        return {"message": "Aucune region detectee a cet endroit."}
    index, _, block = min(matches, key=lambda match: match[1])
    return {
        "region": index,
        "label": block.get("label"),
        "bbox": block.get("bbox"),
        "content": block.get("content"),
        "raw_content": block.get("raw_content"),
    }


def create_ocr_app(
    session: ModelSession,
    options: AppOptions,
    *,
    title: str,
    description: str,
    modes: tuple[str, ...] = (),
    default_mode: str | None = None,
) -> gr.Blocks:
    if modes and default_mode not in modes:
        raise ValueError("default_mode doit faire partie des modes OCR proposes.")
    if not modes and default_mode is not None:
        raise ValueError("default_mode ne peut etre fourni sans modes OCR.")

    def analyze(
        image: Image.Image | None, mode: str | None = None
    ) -> tuple[Any, Any, Any, Any, Any, float | None]:
        if image is None:
            raise gr.Error("Charger une image avant de lancer l'OCR.")
        data: dict[str, Any] = {"image": image}
        if modes:
            if mode not in modes:
                raise gr.Error("Selectionner un mode OCR valide.")
            data["mode"] = mode
        started = perf_counter()
        try:
            result = session.infer(data)
        except ValueError as error:
            LOGGER.warning("Entree OCR invalide : %s", error)
            raise gr.Error(str(error)) from error
        elapsed = perf_counter() - started
        if not isinstance(result.output, dict):
            raise TypeError("L'adaptateur OCR doit retourner un objet structure.")
        blocks = result.output.get("crop_blocks")
        markdown = result.output.get("markdown")
        if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
            raise TypeError("La sortie OCR ne contient pas une liste de crop_blocks valide.")
        if not isinstance(markdown, str):
            raise TypeError("La sortie OCR ne contient pas le Markdown attendu.")
        annotated = _draw_regions(image, blocks)
        rows = [
            [
                index,
                str(block.get("label") or ""),
                ", ".join(str(value) for value in _bbox(block)),
                str(block.get("content") or block.get("raw_content") or ""),
            ]
            for index, block in enumerate(blocks, start=1)
        ]
        return annotated, rows, markdown, blocks, None, elapsed

    def select_region(blocks: list[dict[str, Any]], event: gr.SelectData) -> dict[str, Any]:
        try:
            return _selected_block(blocks, event.index)
        except ValueError as error:
            LOGGER.warning("Selection de region invalide : %s", error)
            raise gr.Error(str(error)) from error

    with gr.Blocks(title=title, analytics_enabled=False) as app:
        gr.Markdown(f"# {title}\n{description}")
        source = gr.Image(label="Image a analyser", type="pil")
        mode_selector = (
            gr.Dropdown(
                choices=list(modes),
                value=default_mode,
                label="Mode OCR",
            )
            if modes
            else None
        )
        analyze_button = gr.Button("Analyser", variant="primary")
        with gr.Row():
            annotated = gr.Image(
                label="Regions detectees - cliquer sur une boite",
                type="pil",
                interactive=False,
            )
            regions = gr.Dataframe(
                label="Regions OCR",
                headers=["#", "Type", "x1, y1, x2, y2", "Texte"],
                datatype=["number", "str", "str", "str"],
                type="array",
                interactive=False,
                wrap=True,
            )
        with gr.Row():
            selected = gr.JSON(label="Region selectionnee")
            duration = gr.Number(
                label="Temps d'analyse (s)",
                precision=3,
                interactive=False,
            )
        markdown = gr.Textbox(
            label="Markdown OCR",
            lines=14,
            interactive=False,
        )
        blocks = gr.State([])
        inputs = [source, mode_selector] if mode_selector is not None else source
        analyze_event = analyze_button.click(
            analyze,
            inputs=inputs,
            outputs=[annotated, regions, markdown, blocks, selected, duration],
            concurrency_limit=1,
        )
        annotated.select(select_region, inputs=blocks, outputs=selected, queue=False)
        reset_outputs = [source, annotated, regions, markdown, blocks, selected, duration]
        reset_values: list[Any] = [None, None, [], "", [], None, None]
        if mode_selector is not None:
            reset_outputs.insert(1, mode_selector)
            reset_values.insert(1, default_mode)
        gr.Button("Reinitialiser").click(
            lambda: tuple(reset_values),
            inputs=[],
            outputs=reset_outputs,
            cancels=[analyze_event],
            queue=False,
        )
    return app


def run_ocr_app(
    session: ModelSession,
    options: AppOptions,
    *,
    title: str,
    description: str,
    modes: tuple[str, ...] = (),
    default_mode: str | None = None,
) -> None:
    create_ocr_app(
        session,
        options,
        title=title,
        description=description,
        modes=modes,
        default_mode=default_mode,
    ).launch(server_name=options.host, server_port=options.port, share=False)
