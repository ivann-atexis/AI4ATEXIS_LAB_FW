from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PADDLE_PDX_PIPELINE_BENCHMARK", "True")

import gradio as gr
import numpy as np
from paddleocr import PaddleOCR, PaddleOCRVL
from paddlex.inference.utils.benchmark import benchmark
from PIL import Image

SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1").rstrip("/")
if SERVER_URL.endswith("/chat/completions"):
    SERVER_URL = SERVER_URL.removesuffix("/chat/completions")
MODEL = os.getenv(
    "LLAMA_MODEL",
    str(Path.home() / "Documents/llama.cpp-tools-GB10/models/PaddleOCR-VL-1.6.Q8_0.gguf"),
)
LAYOUT_MODE = "PaddleOCR-VL · blocs de page"
TEXT_MODE = "PaddleOCR · lignes + texte"
_pipeline: Any = None
_text_ocr: Any = None
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


def _get_text_ocr() -> Any:
    global _text_ocr
    with _load_lock:
        if _text_ocr is None:
            _text_ocr = PaddleOCR(
                device="cpu",
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
    return _text_ocr


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


def _add_line_detections(result_data: Any) -> dict[str, Any]:
    if not isinstance(result_data, dict):
        raise gr.Error("Le résultat OCR n'est pas un objet JSON.")
    ocr_data = result_data.get("res", result_data)
    if not isinstance(ocr_data, dict):
        raise gr.Error("Le résultat OCR ne contient pas les données de reconnaissance.")
    texts = ocr_data.get("rec_texts")
    scores = ocr_data.get("rec_scores")
    polygons = ocr_data.get("rec_polys")
    if not isinstance(texts, list) or not isinstance(polygons, list):
        raise gr.Error("Le résultat OCR ne contient pas textes et polygones alignés.")
    if len(texts) != len(polygons) or (
        isinstance(scores, list) and len(scores) != len(texts)
    ):
        raise gr.Error("Les textes reconnus et leurs polygones ne sont pas alignés.")

    detections = []
    for index, (text, polygon) in enumerate(zip(texts, polygons)):
        detection = {"text": text, "polygon": polygon}
        if isinstance(scores, list):
            detection["score"] = scores[index]
        detections.append(detection)
    return {**result_data, "detections": detections}


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


def _benchmark_stage_times() -> dict[str, float]:
    stage_times: dict[str, float] = {}
    for key, durations in benchmark.logs.items():
        parts = key.rsplit("@", 2)
        operation = parts[-2] if len(parts) == 3 else key
        stage_times[operation] = stage_times.get(operation, 0.0) + sum(durations)
    return stage_times


def _timing_rows(
    mode: str,
    model_load_ms: float,
    image_convert_ms: float,
    image_encode_ms: float,
    inference_ms: float,
    json_ms: float,
    preview_ms: float,
    total_ms: float,
    stage_times: dict[str, float],
) -> list[list[str | float | None]]:
    rows: list[list[str | float | None]] = []

    def add(label: str, duration_ms: float | None) -> None:
        seconds = round(duration_ms / 1000, 3) if duration_ms is not None else None
        rows.append([label, seconds])

    add("Chargement des modèles (à froid)", model_load_ms)
    add("Conversion RGB", image_convert_ms)
    add("Encodage PNG temporaire", image_encode_ms)
    if mode == LAYOUT_MODE:
        add(
            "Détection de mise en page",
            stage_times.get("LayoutAnalysisRunnerPredictor.apply"),
        )
        layout_prep_ops = (
            "paddleocr_vl_filter_overlap_boxes",
            "paddleocr_vl_crop_layout_regions",
            "paddleocr_vl_merge_adjacent_blocks",
            "paddleocr_vl_collect_block_vlm_inputs",
        )
        if any(operation in stage_times for operation in layout_prep_ops):
            add(
                "Préparation des blocs et crops",
                sum(stage_times.get(operation, 0.0) for operation in layout_prep_ops),
            )
        add(
            "Préparation des requêtes VLM",
            stage_times.get("doc_vlm_genai_prepare_requests"),
        )
        add(
            "Reconnaissance VLM (llama.cpp)",
            stage_times.get("paddleocr_vl_run_vl_recognition_batches"),
        )
        add(
            "Assemblage du résultat",
            stage_times.get("paddleocr_vl_assemble_parsing_results"),
        )
        add(
            "PaddleOCR-VL.predict (end-to-end)",
            stage_times.get("_PaddleOCRVLPipeline.predict", inference_ms),
        )
    else:
        add("PaddleOCR (détection + reconnaissance)", inference_ms)
    add("Extraction et conversion JSON", json_ms)
    add("Génération de l'aperçu", preview_ms)
    add("Total interface", total_ms)
    return rows


def run_detection(image: Image.Image | None, mode: str):
    if image is None:
        raise gr.Error("Charge une image pour lancer le spotting.")
    total_started = perf_counter()
    image_convert_started = perf_counter()
    image = image.convert("RGB")
    image_convert_ms = (perf_counter() - image_convert_started) * 1000
    model_load_started = perf_counter()
    pipeline = _get_pipeline() if mode == LAYOUT_MODE else _get_text_ocr()
    model_load_ms = (perf_counter() - model_load_started) * 1000
    with tempfile.TemporaryDirectory(prefix="paddleocr-gradio-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.png"
        output_dir = temp_path / "output"
        output_dir.mkdir()
        image_encode_started = perf_counter()
        image.save(input_path, format="PNG")
        image_encode_ms = (perf_counter() - image_encode_started) * 1000

        benchmark.reset()
        inference_started = perf_counter()
        results = iter(pipeline.predict(str(input_path)))
        result = next(results, None)
        inference_ms = (perf_counter() - inference_started) * 1000
        stage_times = _benchmark_stage_times()
        if result is None:
            raise gr.Error("Paddle n'a retourné aucun résultat pour cette image.")

        json_started = perf_counter()
        result_data = getattr(result, "json", None)
        if callable(result_data):
            result_data = result_data()
        if not isinstance(result_data, (dict, list)):
            raise gr.Error("Le résultat Paddle ne contient pas de données JSON structurées.")
        safe_result_data = _json_safe(result_data)
        if mode == TEXT_MODE:
            safe_result_data = _add_line_detections(safe_result_data)
        json_ms = (perf_counter() - json_started) * 1000
        preview_started = perf_counter()
        try:
            preview = _result_image(result, output_dir)
        except (OSError, RuntimeError, ValueError) as error:
            raise gr.Error(f"Résultat calculé, mais visualisation indisponible: {error}") from error
        preview_ms = (perf_counter() - preview_started) * 1000

    if mode == LAYOUT_MODE:
        status = "Boxes de mise en page PaddleOCR-VL; reconnaissance VLM servie par llama.cpp."
    else:
        status = "Texte reconnu et polygones des lignes par PaddleOCR sur CPU."
    total_ms = (perf_counter() - total_started) * 1000
    timings = _timing_rows(
        mode,
        model_load_ms,
        image_convert_ms,
        image_encode_ms,
        inference_ms,
        json_ms,
        preview_ms,
        total_ms,
        stage_times,
    )
    return preview, status, timings, safe_result_data


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
    timings_output = gr.Dataframe(
        headers=["Étape", "Durée (s)"],
        datatype=["str", "number"],
        label="Temps par étape (s)",
        interactive=False,
        wrap=True,
    )
    boxes_output = gr.JSON(label="Résultat structuré et coordonnées")
    run_button.click(
        run_detection,
        inputs=[input_image, mode],
        outputs=[output_image, status, timings_output, boxes_output],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
