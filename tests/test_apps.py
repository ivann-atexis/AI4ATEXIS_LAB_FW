import base64
import importlib
import io
import json
import os
from unittest.mock import patch

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from fastapi.testclient import TestClient
from PIL import Image

from common.apps.api import create_app
from common.apps.gradio import create_chat_app, create_decision_app, create_generic_app, history_text
from common.types import AppOptions, InferenceResult
from tests.helpers import OfflineTestCase, fake_session


class AppTests(OfflineTestCase):
    def test_all_app_entrypoints_are_callable(self) -> None:
        for name in ("lfm3", "d1-2b", "template"):
            for app in ("cli", "gradio", "api"):
                with self.subTest(model=name, app=app):
                    module = importlib.import_module(f"models.{name}.apps.{app}.app")
                    self.assertTrue(callable(module.run))

    def test_api_text_contract_health_and_validation(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ready", "model": "lfm3"})
            response = client.post("/infer", json={"input_data": "hello"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {"output": "answer", "usage": {"input_tokens": 3, "output_tokens": 2}},
            )
            session.infer.assert_called_once_with("hello")
            self.assertEqual(client.post("/infer", json={}).status_code, 422)
            self.assertEqual(
                client.post("/infer", json={"input_data": "test", "unknown": 1}).status_code, 422
            )
            session.infer.side_effect = ValueError("invalid input")
            self.assertEqual(
                client.post("/infer", json={"input_data": "test"}).status_code, 422
            )

    def test_api_does_not_disguise_runtime_failure(self) -> None:
        session = fake_session()
        session.infer.side_effect = RuntimeError("GPU error")
        with TestClient(create_app(session), raise_server_exceptions=False) as client:
            self.assertEqual(
                client.post("/infer", json={"input_data": "test"}).status_code, 500
            )

    def test_api_accepts_uploaded_image_not_server_path(self) -> None:
        session = fake_session("d1-2b", tokens=0)
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        with TestClient(create_app(session)) as client:
            response = client.post(
                "/infer", json={"input_data": {"state": "describe"}, "image_base64": encoded}
            )
            self.assertEqual(response.status_code, 200)
            image = session.infer.call_args.args[0]["image"]
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (2, 2))
            self.assertEqual(
                client.post(
                    "/infer", json={"input_data": "test", "image_base64": "/etc/passwd"}
                ).status_code,
                422,
            )

    def test_gradio_history_supports_both_text_representations(self) -> None:
        self.assertEqual(history_text("hello"), "hello")
        self.assertEqual(history_text([{"type": "text", "text": "hello"}]), "hello")
        with self.assertRaises(ValueError):
            history_text([{"type": "image", "path": "/tmp/image"}])

    def test_gradio_builders_and_chat_callback(self) -> None:
        session = fake_session()
        with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"}):
            chat = create_chat_app(session, AppOptions("hello"))
            decision = create_decision_app(
                fake_session("d1-2b", tokens=0), AppOptions({"state": "fault"})
            )
            generic = create_generic_app(session, AppOptions({"prompt": "hello"}))
        try:
            answer = chat.fn(
                "second",
                [{"role": "user", "content": [{"type": "text", "text": "first"}]}],
            )
            self.assertEqual(answer, "answer")
            session.infer.assert_called_with(
                {
                    "messages": [
                        {"role": "user", "content": "first"},
                        {"role": "user", "content": "second"},
                    ]
                }
            )
            self.assertTrue(decision.config["dependencies"])
            self.assertTrue(generic.config["dependencies"])
            json.dumps(decision.config)
        finally:
            chat.close()
            decision.close()
            generic.close()
