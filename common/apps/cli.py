import json

from common.runtime import ModelSession
from common.types import AppOptions


def run(session: ModelSession, options: AppOptions) -> None:
    result = session.infer(options.input_data)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
