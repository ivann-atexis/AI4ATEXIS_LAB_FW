import json

from common.runtime import ModelSession
from common.types import InferenceResult


def test(session: ModelSession) -> InferenceResult:
    result = session.infer(session.config["example_input"])
    if result.output is None:
        raise AssertionError("Le smoke test attend une sortie non nulle.")
    json.dumps(result.to_dict(), allow_nan=False)
    return result
