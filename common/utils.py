from __future__ import annotations

import base64
import binascii
import io
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.types import InputData, JSONValue

if TYPE_CHECKING:
    from PIL.Image import Image


def parse_input(text: str) -> InputData:
    if text.lstrip().startswith("{"):
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("L'entree JSON doit etre un objet.")
        return value
    return text


def with_image(input_data: InputData, image: Image) -> dict[str, Any]:
    data = {"state": input_data} if isinstance(input_data, str) else dict(input_data)
    data["image"] = image
    return data


def open_image(path: Path) -> Image:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def decode_image(encoded: str) -> Image:
    from PIL import Image

    if len(encoded) > 14_000_000:
        raise ValueError("Image trop volumineuse (limite : environ 10 Mo).")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("image_base64 doit contenir du base64 brut valide.") from error
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def write_json(path: Path, data: JSONValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
