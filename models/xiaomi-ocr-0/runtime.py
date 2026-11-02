from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class XiaomiOCRRuntime:
    model: Any
    detector: Any
