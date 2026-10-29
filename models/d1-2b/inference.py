from typing import Any

from common.types import Config, InferenceResult, InputData


def infer(
    model: Any, processor: Any, input_data: InputData, config: Config
) -> InferenceResult:
    import torch

    if isinstance(input_data, str):
        data: dict[str, Any] = {"state": input_data}
    elif isinstance(input_data, dict):
        data = input_data
    else:
        raise ValueError("d1 attend un texte ou un objet {state, questions?, image?}.")
    if set(data) - {"state", "questions", "image"}:
        raise ValueError("Champs d1 acceptes : state, questions, image.")
    questions = data.get("questions", config.get("questions"))
    if not isinstance(questions, dict) or not questions:
        raise ValueError("d1 exige au moins une question nommee.")
    for name, question in questions.items():
        if (
            not isinstance(name, str)
            or not isinstance(question, dict)
            or not isinstance(question.get("type"), str)
        ):
            raise ValueError("Chaque question doit avoir un nom et un objet avec type.")
    state = data.get("state")
    image = data.get("image")
    if (state is None or state == "") and image is None:
        raise ValueError("Fournir un etat ou une image.")
    media = None
    if image is not None:
        from PIL.Image import Image

        if not isinstance(image, Image):
            raise ValueError("image doit etre une image chargee, pas un chemin ni une URL.")
        media = [image]
    with torch.inference_mode():
        result = model.system_one(state, questions, media=media)
    return InferenceResult(
        output=result["answers"],
        input_tokens=result["usage"]["input_tokens"],
        output_tokens=result["usage"]["output_tokens"],
    )
