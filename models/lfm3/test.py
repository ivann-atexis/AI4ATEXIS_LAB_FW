from common.runtime import ModelSession
from common.types import InferenceResult


def test(session: ModelSession) -> InferenceResult:
    result = session.infer(session.config["example_input"])
    if not isinstance(result.output, str) or not result.output.strip():
        raise AssertionError("LFM3 doit produire un texte non vide.")
    if result.output_tokens is None or result.output_tokens < 1:
        raise AssertionError("LFM3 doit compter les tokens produits.")
    return result
