import importlib

from common.apps.ocr import create_ocr_app as _create_ocr_app
from common.runtime import ModelSession
from common.types import AppOptions

TASK_PROMPTS = importlib.import_module(
    "models.paddleocr-vl-1-6.inference"
).TASK_PROMPTS


def create_ocr_app(session: ModelSession, options: AppOptions):
    return _create_ocr_app(
        session,
        options,
        title="PaddleOCR-VL-1.6",
        description=(
            "VLM OCR PaddleOCR-VL-1.6. Choisissez une tache de transcription "
            "et fournissez une image."
        ),
        modes=tuple(TASK_PROMPTS),
        default_mode="ocr",
    )


def run(session: ModelSession, options: AppOptions) -> None:
    create_ocr_app(session, options).launch(
        server_name=options.host,
        server_port=options.port,
        share=False,
    )
