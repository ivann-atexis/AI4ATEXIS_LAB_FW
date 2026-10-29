from common.runtime import ModelSession
from common.types import InferenceResult


def test(session: ModelSession) -> InferenceResult:
    result = session.infer(session.config["example_input"])
    if not isinstance(result.output, dict):
        raise AssertionError("d1 doit retourner des reponses structurees.")
    if set(result.output) != set(session.config["questions"]):
        raise AssertionError("d1 doit repondre a toutes les questions configurees.")
    if result.output_tokens != 0:
        raise AssertionError("d1 ne doit pas annoncer de tokens generes.")
    return result
