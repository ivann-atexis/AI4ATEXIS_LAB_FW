from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
from paddleocr import PaddleOCRVL, TextDetection
from PIL import Image

SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1").rstrip("/")
if SERVER_URL.endswith("/chat/completions"):
    SERVER_URL = SERVER_URL.removesuffix("/chat/completions")
MODEL = os.getenv(
    "LLAMA_MODEL",
    str(Path.home() / "Documents/llama.cpp-tools-GB10/models/PaddleOCR-VL-1.6.Q8_0.gguf"),
)
LAYOUT_MODE = "PaddleOCR-VL · blocs de page"
TEXT_MODE = "TextDetection · lignes de texte"
_pipeline: Any = None
_text_detector: Any = None
_load_lock = threading.Lock()


def _get_pipeline() -> Any:
    global _pipeline
    with _load_lock:
        if _pipeline is None:
            _pipeline = PaddleOCRVL(
                device="cpu",
                vl_rec_backend="llama-cpp-server",
                vl_rec_server_url=SERVER_URL,
                vl_rec_api_model_name=MODEL,
            )
    return _pipeline


def _get_text_detector() -> Any:
    global _text_detector
    with _load_lock:
        if _text_detector is None:
            _text_detector = TextDetection(device="cpu")
    return _text_detector


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _result_image(result: Any, output_dir: Path) -> Image.Image:
    save_to_img = getattr(result, "save_to_img", None)
    if not callable(save_to_img):
        raise RuntimeError("Le résultat Paddle ne fournit pas save_to_img().")
    save_to_img(save_path=str(output_dir))
    for path in sorted(output_dir.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            with Image.open(path) as preview:
                return preview.convert("RGB").copy()
        except (OSError, ValueError):
            continue
    raise RuntimeError("Paddle n'a pas produit d'image de visualisation.")


def run_detection(image: Image.Image | None, mode: str):
    if image is None:
        raise gr.Error("Charge une image pour lancer le spotting.")
    image = image.convert("RGB")
    pipeline = _get_pipeline() if mode == LAYOUT_MODE else _get_text_detector()
    with tempfile.TemporaryDirectory(prefix="paddleocr-gradio-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.png"
        output_dir = temp_path / "output"
        output_dir.mkdir()
        image.save(input_path, format="PNG")

        results = iter(pipeline.predict(str(input_path)))
        result = next(results, None)
        if result is None:
            raise gr.Error("Paddle n'a retourné aucun résultat pour cette image.")

        result_data = getattr(result, "json", None)
        if callable(result_data):
            result_data = result_data()
        if not isinstance(result_data, (dict, list)):
            raise gr.Error("Le résultat Paddle ne contient pas de données JSON structurées.")
        try:
            preview = _result_image(result, output_dir)
        except (OSError, RuntimeError, ValueError) as error:
            raise gr.Error(f"Résultat calculé, mais visualisation indisponible: {error}") from error

    if mode == LAYOUT_MODE:
        status = "Boxes de mise en page PaddleOCR-VL; reconnaissance VLM servie par llama.cpp."
    else:
        status = "Polygones Paddle TextDetection autour des lignes de texte."
    return preview, status, _json_safe(result_data)


with gr.Blocks(title="PaddleOCR-VL Spotting") as demo:
    gr.Markdown("## PaddleOCR · Boxes")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Image")
            mode = gr.Radio(
                choices=[LAYOUT_MODE, TEXT_MODE],
                value=LAYOUT_MODE,
                label="Type de boxes",
            )
            run_button = gr.Button("Analyser", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Visualisation Paddle")
            status = gr.Textbox(label="État", interactive=False)
    boxes_output = gr.JSON(label="Résultat structuré et coordonnées")
    run_button.click(
        run_detection,
        inputs=[input_image, mode],
        outputs=[output_image, status, boxes_output],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
