import asyncio
import base64
from copy import deepcopy
from functools import partial
import importlib
import io
import json
import os
from typing import Any
from unittest.mock import patch

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from fastapi.testclient import TestClient
import gradio as gr
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
        app = create_app(session)
        self.assertEqual(app.title, "AI4ATEXIS_LAB_FW - lfm3")
        with TestClient(app) as client:
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
            self.assertEqual(chat.title, "AI4ATEXIS_LAB_FW - lfm3")
            self.assertEqual(decision.title, "AI4ATEXIS_LAB_FW - d1-2b")
            self.assertEqual(generic.title, "AI4ATEXIS_LAB_FW - lfm3")
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


class DecisionCriteriaTests(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = fake_session("d1-2b", tokens=0)
        self.original_questions = deepcopy(self.session.config["questions"])
        self.session.infer.return_value = InferenceResult(
            {"category": "maintenance"}, input_tokens=3, output_tokens=0
        )
        self.app = create_decision_app(self.session, AppOptions({"state": "fault"}))
        self.addCleanup(self.app.close)
        self.callback = next(
            fn for fn in self.app.fns.values() if getattr(fn.fn, "__name__", "") == "decide"
        )
        self.inputs = [deepcopy(component.value) for component in self.callback.inputs]
        self.set_input("Mode de questions", "Questions configurees")

    def set_input(self, label: str, value: object) -> None:
        index = next(
            index
            for index, component in enumerate(self.callback.inputs)
            if component.label == label
        )
        self.inputs[index] = value

    def set_criteria(self, name: str, rows: list[list[object]]) -> None:
        component = next(
            component
            for component in self.callback.inputs
            if component.label == f"Criteres - {name}"
        )
        value = deepcopy(component.value)
        value["data"] = rows
        self.set_input(component.label, value)

    def predict(self) -> dict[str, Any]:
        result = asyncio.run(self.app.process_api(self.callback, self.inputs))
        return result["data"][0].model_dump()

    def apply_image_example(self) -> None:
        callback = next(
            fn for fn in self.app.fns.values() if getattr(fn.fn, "__name__", "") == "image_example"
        )
        result = asyncio.run(self.app.process_api(callback, []))
        for component, value in zip(callback.outputs, result["data"], strict=True):
            self.set_input(component.label, value)

    def test_configured_mode_preserves_questions_and_output_contract(self) -> None:
        self.session.infer.assert_not_called()
        result = self.predict()
        self.session.infer.assert_called_once_with(
            {"state": "fault", "questions": self.original_questions}
        )
        self.assertEqual(
            result,
            {"output": {"category": "maintenance"}, "usage": {"input_tokens": 3, "output_tokens": 0}},
        )

    def test_response_time_measures_synchronized_inference_and_keeps_json_contract(self) -> None:
        events: list[str] = []
        ticks = iter((10.0, 11.23456))

        def clock() -> float:
            events.append("clock")
            return next(ticks)

        def infer(data: object) -> InferenceResult:
            events.append("infer")
            return self.session.infer.return_value

        self.session.infer.side_effect = infer
        with (
            patch("common.apps.gradio.perf_counter", side_effect=clock),
            patch(
                "common.apps.gradio.DeviceMetrics.synchronize",
                side_effect=lambda: events.append("sync"),
            ),
        ):
            result = asyncio.run(self.app.process_api(self.callback, self.inputs))
        self.assertEqual(events, ["sync", "clock", "infer", "sync", "clock"])
        self.assertEqual(result["data"][0].model_dump(), self.session.infer.return_value.to_dict())
        self.assertEqual(result["data"][1], 1.235)
        timer = self.callback.outputs[1]
        self.assertEqual(timer.label, "Temps de reponse (s)")
        self.assertFalse(timer.interactive)
        self.assertIsNone(timer.value)

    def test_invalid_input_is_not_timed(self) -> None:
        self.set_criteria("category", [])
        with (
            patch("common.apps.gradio.perf_counter") as clock,
            patch("common.apps.gradio.DeviceMetrics.synchronize") as synchronize,
        ):
            with self.assertRaises(gr.Error):
                self.predict()
        clock.assert_not_called()
        synchronize.assert_not_called()
        self.session.infer.assert_not_called()

    def test_failed_inference_does_not_return_a_successful_duration(self) -> None:
        self.session.infer.side_effect = RuntimeError("GPU error")
        with (
            patch("common.apps.gradio.perf_counter", return_value=10.0) as clock,
            patch("common.apps.gradio.DeviceMetrics.synchronize") as synchronize,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU error"):
                self.predict()
        clock.assert_called_once_with()
        synchronize.assert_called_once_with()

    def test_select_edit_and_add_choice_and_ordered_score_criteria(self) -> None:
        self.set_criteria(
            "category",
            [
                [False, "maintenance", "Equipment failure"],
                [True, " logistics ", " Delivery or missing parts "],
                [True, "quality", "Product defect"],
                [False, "", ""],
            ],
        )
        self.set_criteria("urgency", [[True, "critical"], [False, "medium"], [True, "low"]])
        self.predict()
        expected = deepcopy(self.original_questions)
        expected["category"]["criteria"] = {
            "logistics": "Delivery or missing parts",
            "quality": "Product defect",
        }
        expected["urgency"]["criteria"] = ["critical", "low"]
        self.session.infer.assert_called_once_with({"state": "fault", "questions": expected})
        self.assertEqual(self.session.config["questions"], self.original_questions)
        self.set_criteria(
            "category",
            [[True, name, description] for name, description in self.original_questions["category"]["criteria"].items()],
        )
        self.set_criteria("urgency", [[True, level] for level in ["low", "medium", "high"]])
        self.predict()
        self.assertEqual(self.session.infer.call_args.args[0]["questions"], self.original_questions)

    def test_add_buttons_create_editable_unselected_rows_without_inference(self) -> None:
        callbacks = [callback for callback in self.app.fns.values() if isinstance(callback.fn, partial)]
        self.assertEqual(len(callbacks), 3)
        for callback in callbacks:
            with self.subTest(table=callback.inputs[0].label):
                rows = deepcopy(callback.inputs[0].value)
                result = asyncio.run(self.app.process_api(callback, [rows]))
                updated = result["data"][0]["data"]
                width = len(rows["headers"])
                self.assertEqual(updated, [*rows["data"], [False, *([""] * (width - 1))]])
                result = asyncio.run(
                    self.app.process_api(callback, [{"headers": rows["headers"], "data": []}])
                )
                self.assertEqual(
                    result["data"][0]["data"],
                    [[False, *([""] * (width - 1))]],
                )
        self.session.infer.assert_not_called()

    def test_invalid_criteria_are_reported_before_inference(self) -> None:
        cases = [
            ("category", [[False, "maintenance", "Failure"]]),
            ("urgency", [[False, "low"]]),
            ("category", [[True, "", "Failure"]]),
            ("category", [[True, "maintenance", ""]]),
            ("category", [[True, "same", "First"], [True, " same ", "Second"]]),
            ("urgency", [[True, "low"], [True, "low"]]),
            ("urgency", [[True, " "]]),
            ("category", [[True, "maintenance"]]),
            ("category", [["yes", "maintenance", "Failure"]]),
        ]
        defaults = deepcopy(self.inputs)
        for name, rows in cases:
            with self.subTest(question=name, rows=rows):
                self.inputs = deepcopy(defaults)
                self.set_criteria(name, rows)
                with self.assertRaises(gr.Error):
                    self.predict()
                self.session.infer.assert_not_called()

    def test_form_ignores_json_unless_advanced_mode_is_selected(self) -> None:
        self.set_input("Questions (JSON)", "{invalid")
        self.predict()
        self.assertEqual(self.session.infer.call_args.args[0]["questions"], self.original_questions)

    def test_advanced_json_replaces_form_without_mutating_defaults(self) -> None:
        questions = {
            "custom": {
                "type": "choice",
                "instructions": "Choose a colour.",
                "criteria": {"blue": "Blue object", "red": "Red object"},
            }
        }
        self.set_input("Mode de questions", "JSON avance")
        self.set_input("Questions (JSON)", json.dumps(questions))
        self.set_criteria("category", [])
        self.predict()
        self.session.infer.assert_called_once_with({"state": "fault", "questions": questions})
        self.assertEqual(self.session.config["questions"], self.original_questions)

    def test_invalid_advanced_json_is_reported_before_inference(self) -> None:
        self.set_input("Mode de questions", "JSON avance")
        for specification in ("{invalid", "[]", "{}", "null"):
            with self.subTest(specification=specification):
                self.set_input("Questions (JSON)", specification)
                with self.assertRaises(gr.Error):
                    self.predict()
                self.session.infer.assert_not_called()

    def test_image_and_runtime_errors_keep_existing_behavior(self) -> None:
        image = Image.new("RGB", (2, 2))
        rows = [
            [[True, name, description] for name, description in self.original_questions["category"]["criteria"].items()],
            [[True, level] for level in self.original_questions["urgency"]["criteria"]],
        ]
        self.callback.fn("", "", image, "Questions configurees", "", "", [], *rows)
        data = self.session.infer.call_args.args[0]
        self.assertIs(data["image"], image)
        self.assertEqual(data["questions"], self.original_questions)
        self.session.infer.side_effect = ValueError("invalid input")
        with self.assertRaises(gr.Error):
            self.predict()
        self.session.infer.side_effect = RuntimeError("GPU error")
        with self.assertRaisesRegex(RuntimeError, "GPU error"):
            self.predict()

    def test_input_questions_initialize_controls_instead_of_config(self) -> None:
        questions = {
            "colour": {
                "type": "choice",
                "instructions": "Choose a colour.",
                "criteria": {"blue": "Blue object", "red": "Red object"},
            },
            "priority": {"type": "score", "criteria": ["normal", "critical"]},
        }
        initial = {"state": "custom state", "questions": questions}
        app = create_decision_app(self.session, AppOptions(initial))
        self.addCleanup(app.close)
        callback = next(
            fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "decide"
        )
        result = asyncio.run(
            app.process_api(
                callback,
                [
                    "Questions configurees"
                    if component.label == "Mode de questions"
                    else deepcopy(component.value)
                    for component in callback.inputs
                ],
            )
        )
        self.assertEqual(result["data"][0].model_dump()["output"], {"category": "maintenance"})
        self.session.infer.assert_called_once_with(initial)
        self.assertEqual(self.session.config["questions"], self.original_questions)
        self.assertEqual(initial["questions"], questions)

    def test_app_opens_with_prefilled_image_question_and_no_maintenance_text(self) -> None:
        app = create_decision_app(self.session, AppOptions(self.session.config["example_input"]))
        self.addCleanup(app.close)
        callback = next(
            fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "decide"
        )
        values = {component.label: component.value for component in callback.inputs}
        self.assertEqual(values["Mode de questions"], "Question personnalisee")
        self.assertEqual(values["Etat / document"], "")
        self.assertEqual(
            values["Theme / contexte (facultatif)"],
            "Reconnaissance d'animaux dans une image",
        )
        self.assertIn("chat", values["Question"])
        self.assertEqual(
            [row[1] for row in values["Criteres - personnalise"]["data"]],
            ["chat", "chien", "aucun", "les_deux"],
        )
        self.assertTrue(all(row[0] for row in values["Criteres - personnalise"]["data"]))
        visibility = next(
            fn for fn in app.fns.values()
            if getattr(fn.fn, "__name__", "") == "show_question_mode"
        )
        self.assertEqual([component.visible for component in visibility.outputs], [False, True, False])
        self.session.infer.assert_not_called()

    def test_reset_clears_all_custom_fields_media_and_result_without_inference(self) -> None:
        self.assertTrue(
            any(isinstance(block, gr.Button) and block.value == "reset" for block in self.app.blocks.values())
        )
        callback = next(
            fn for fn in self.app.fns.values() if getattr(fn.fn, "__name__", "") == "reset_form"
        )
        self.assertEqual(callback.inputs, [])
        for _ in range(2):
            result = asyncio.run(self.app.process_api(callback, []))
            values = {
                component.label: value
                for component, value in zip(callback.outputs, result["data"], strict=True)
            }
            self.assertEqual(
                set(values),
                {
                    "Mode de questions", "Theme / contexte (facultatif)", "Question",
                    "Criteres - personnalise", "Etat / document", "Image facultative",
                    "Reponses et usage", "Temps de reponse (s)",
                },
            )
            self.assertEqual(values["Mode de questions"], "Question personnalisee")
            for label in ("Theme / contexte (facultatif)", "Question", "Etat / document"):
                self.assertEqual(values[label], "")
            self.assertEqual(values["Criteres - personnalise"]["data"], [[False, "", ""]])
            self.assertIsNone(values["Image facultative"])
            self.assertIsNone(values["Reponses et usage"])
            self.assertIsNone(values["Temps de reponse (s)"])
            values["Criteres - personnalise"]["data"][0][1] = "changed"
        self.assertEqual(self.session.config["questions"], self.original_questions)
        self.session.infer.assert_not_called()
        self.apply_image_example()
        self.session.infer.assert_not_called()

    def test_reset_cancels_the_inference_event(self) -> None:
        analyze_id = next(index for index, callback in self.app.fns.items() if callback is self.callback)
        reset_id = next(
            index for index, callback in self.app.fns.items()
            if getattr(callback.fn, "__name__", "") == "reset_form"
        )
        reset = next(
            event for event in self.app.config["dependencies"] if event["id"] == reset_id
        )
        self.assertTrue(
            any(
                event["targets"] == reset["targets"] and analyze_id in event["cancels"]
                for event in self.app.config["dependencies"]
            )
        )

    def test_custom_theme_question_and_categories_replace_configured_questions(self) -> None:
        self.set_input("Mode de questions", "Question personnalisee")
        self.set_input("Theme / contexte (facultatif)", " Controle qualite ")
        self.set_input("Question", " Quelle est la couleur du produit ? ")
        self.set_criteria(
            "personnalise",
            [[True, " rouge ", " Produit rouge "], [True, "bleu", "Produit bleu"], [False, "", ""]],
        )
        self.set_criteria("category", [])
        self.set_criteria("urgency", [])
        self.set_input("Questions (JSON)", "{invalid")
        self.predict()
        self.session.infer.assert_called_once_with(
            {
                "state": "fault",
                "questions": {
                    "classification": {
                        "type": "choice",
                        "instructions": (
                            "Theme / contexte : Controle qualite\n\n"
                            "Quelle est la couleur du produit ?"
                        ),
                        "criteria": {"rouge": "Produit rouge", "bleu": "Produit bleu"},
                    }
                },
            }
        )
        self.assertEqual(self.session.config["questions"], self.original_questions)

    def test_custom_theme_is_optional(self) -> None:
        self.set_input("Mode de questions", "Question personnalisee")
        self.set_input("Question", "Is it red?")
        self.set_criteria("personnalise", [[True, "red", "A red product"]])
        for theme in (None, "", " "):
            with self.subTest(theme=theme):
                self.set_input("Theme / contexte (facultatif)", theme)
                self.predict()
                question = self.session.infer.call_args.args[0]["questions"]["classification"]
                self.assertEqual(question["instructions"], "Is it red?")

    def test_image_example_sets_editable_fields_without_inference(self) -> None:
        self.apply_image_example()
        values = {
            component.label: value
            for component, value in zip(self.callback.inputs, self.inputs, strict=True)
        }
        self.assertEqual(values["Mode de questions"], "Question personnalisee")
        self.assertEqual(values["Etat / document"], "")
        self.assertIn("image", values["Theme / contexte (facultatif)"])
        self.assertIn("chat", values["Question"])
        self.assertEqual(
            [row[1] for row in values["Criteres - personnalise"]["data"]],
            ["chat", "chien", "aucun", "les_deux"],
        )
        self.assertTrue(all(row[0] for row in values["Criteres - personnalise"]["data"]))
        self.session.infer.assert_not_called()
        self.assertEqual(self.session.config["questions"], self.original_questions)
        values["Criteres - personnalise"]["data"][0][1] = "changed"
        self.apply_image_example()
        value = next(
            value
            for component, value in zip(self.callback.inputs, self.inputs, strict=True)
            if component.label == "Criteres - personnalise"
        )
        self.assertEqual(value["data"][0][1], "chat")

    def test_image_example_supports_image_only_and_customized_question(self) -> None:
        self.apply_image_example()
        self.set_input("Question", "Quels animaux sont visibles ?")
        self.set_criteria(
            "personnalise",
            [
                [True, "chat", "Un chat uniquement"],
                [True, "chien", "Un chien uniquement"],
                [True, "aucun", "Ni chat ni chien"],
                [True, "les_deux", "Chat et chien presents ensemble"],
                [False, "oiseau", "Un oiseau"],
            ],
        )
        image = Image.new("RGB", (2, 2))
        values = [
            value["data"] if isinstance(component, gr.Dataframe) else value
            for component, value in zip(self.callback.inputs, self.inputs, strict=True)
        ]
        values[2] = image
        result = self.callback.fn(*values)
        data = self.session.infer.call_args.args[0]
        self.assertIs(data["image"], image)
        self.assertEqual(data["state"], "")
        self.assertEqual(set(data["questions"]), {"classification"})
        question = data["questions"]["classification"]
        self.assertEqual(
            question,
            {
                "type": "choice",
                "instructions": (
                    "Theme / contexte : Reconnaissance d'animaux dans une image\n\n"
                    "Quels animaux sont visibles ?"
                ),
                "criteria": {
                    "chat": "Un chat uniquement",
                    "chien": "Un chien uniquement",
                    "aucun": "Ni chat ni chien",
                    "les_deux": "Chat et chien presents ensemble",
                },
            },
        )
        self.assertEqual(result[0]["usage"], {"input_tokens": 3, "output_tokens": 0})
        self.assertGreaterEqual(result[1], 0)

    def test_invalid_custom_questions_are_reported_before_inference(self) -> None:
        self.set_input("Mode de questions", "Question personnalisee")
        self.set_criteria("personnalise", [[True, "red", "A red product"]])
        for question in (None, "", " "):
            with self.subTest(question=question):
                self.set_input("Question", question)
                with self.assertRaises(gr.Error):
                    self.predict()
        self.set_input("Question", "Choose a colour.")
        for rows in (
            [],
            [[False, "red", "A red product"]],
            [[True, "", "A red product"]],
            [[True, "red", ""]],
            [[True, "red", "Red"], [True, " red ", "Red again"]],
        ):
            with self.subTest(rows=rows):
                self.set_criteria("personnalise", rows)
                with self.assertRaises(gr.Error):
                    self.predict()
        self.session.infer.assert_not_called()

    def test_theme_alone_is_not_an_input_image_or_document(self) -> None:
        self.apply_image_example()
        with self.assertRaises(gr.Error):
            self.predict()
        self.session.infer.assert_not_called()

    def test_mode_switches_show_only_the_selected_question_editor(self) -> None:
        callback = next(
            fn for fn in self.app.fns.values()
            if getattr(fn.fn, "__name__", "") == "show_question_mode"
        )
        for mode, expected in (
            ("Questions configurees", [True, False, False]),
            ("Question personnalisee", [False, True, False]),
            ("JSON avance", [False, False, True]),
        ):
            with self.subTest(mode=mode):
                result = asyncio.run(self.app.process_api(callback, [mode]))
                self.assertEqual([update["visible"] for update in result["data"]], expected)
        self.session.infer.assert_not_called()
