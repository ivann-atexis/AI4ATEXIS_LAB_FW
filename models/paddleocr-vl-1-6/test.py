import json

from PIL import Image, ImageDraw

from common.runtime import ModelSession
from common.types import InferenceResult


def test(session: ModelSession) -> InferenceResult:
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 50), "AI4ATEXIS LAB", fill="black")
    draw.text((40, 120), "Total: 42.50 EUR", fill="black")
    result = session.infer({"image": image, "task": "ocr"})
    if not isinstance(result.output, dict):
        raise AssertionError("PaddleOCR-VL doit retourner un objet structure.")
    if not isinstance(result.output.get("markdown"), str):
        raise AssertionError("La sortie doit contenir le texte OCR.")
    if result.input_tokens is None or result.output_tokens is None:
        raise AssertionError("Les compteurs de tokens doivent etre renseignes.")
    json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
    return result
