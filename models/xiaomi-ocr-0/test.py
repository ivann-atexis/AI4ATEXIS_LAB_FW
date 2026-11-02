from PIL import Image, ImageDraw, ImageFont

from common.runtime import ModelSession
from common.types import InferenceResult


def test(session: ModelSession) -> InferenceResult:
    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.text((90, 90), "AI4ATEXIS OCR TEST", fill="black", font=font)
    draw.text((90, 240), "Invoice: A-1042", fill="black", font=font)
    draw.text((90, 390), "Total: 42.50 EUR", fill="black", font=font)
    draw.text((90, 540), "Due date: 2026-12-31", fill="black", font=font)

    result = session.infer({"image": image, "mode": "regions"})
    if not isinstance(result.output, dict):
        raise AssertionError("Xiaomi-OCR-0 doit retourner un objet structure.")
    blocks = result.output.get("crop_blocks")
    if not isinstance(blocks, list):
        raise AssertionError("Le detecteur doit retourner une liste de regions.")
    if not isinstance(result.output.get("markdown"), str):
        raise AssertionError("Xiaomi-OCR-0 doit retourner le texte OCR.")
    if result.input_tokens is None or result.output_tokens is None:
        raise AssertionError("Les compteurs de tokens Transformers doivent etre disponibles.")
    return result
