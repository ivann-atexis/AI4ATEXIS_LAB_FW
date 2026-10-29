import os
from unittest import TestCase
from unittest.mock import Mock, patch

from common.config import load_config
from common.types import InferenceResult


class OfflineTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enterContext(
            patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "GRADIO_ANALYTICS_ENABLED": "False"},
            )
        )
        self.enterContext(
            patch("socket.create_connection", side_effect=AssertionError("Reseau interdit en test."))
        )


def fake_session(name: str = "lfm3", tokens: int | None = 2) -> Mock:
    session = Mock()
    session.name = name
    session.config = load_config(name)
    session.model = object()
    session.supports_streaming = name in {"lfm3", "template"}
    session.infer.return_value = InferenceResult(
        "answer", input_tokens=3, output_tokens=tokens, finish_reason="stop"
    )
    return session
