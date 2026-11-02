from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from common.config import load_config
from tests.helpers import OfflineTestCase

load_module = importlib.import_module("models.paddleocr-vl-1-6.load")
inference_module = importlib.import_module("models.paddleocr-vl-1-6.inference")


class PaddleOCRVLLoadTests(OfflineTestCase):
    def test_load_uses_pinned_hugging_face_snapshot_and_sdpa(self) -> None:
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        processor = Mock()
        transformers = SimpleNamespace(
            AutoModelForImageTextToText=Mock(),
            AutoProcessor=Mock(),
        )
        transformers.AutoModelForImageTextToText.from_pretrained.return_value = model
        transformers.AutoProcessor.from_pretrained.return_value = processor
        config = load_config(
            "paddleocr-vl-1-6",
            overrides={"device": "cpu", "dtype": "float32"},
        )

        with (
            patch.dict("sys.modules", {"transformers": transformers}),
            patch.object(load_module, "require_transformers"),
            patch.object(
                load_module,
                "download_snapshot",
                return_value=Path("/cache/paddleocr-vl-1-6"),
            ) as download,
            patch.object(load_module, "resolve_device", return_value=torch.device("cpu")),
            patch.object(load_module, "resolve_dtype", return_value=torch.float32),
            patch.object(load_module, "seed_torch") as seed,
        ):
            loaded_model, loaded_processor = load_module.load_model(config)

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_processor, processor)
        download.assert_called_once_with(config)
        transformers.AutoProcessor.from_pretrained.assert_called_once_with(
            "/cache/paddleocr-vl-1-6",
            trust_remote_code=False,
            local_files_only=True,
        )
        transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(
            "/cache/paddleocr-vl-1-6",
            dtype=torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
            local_files_only=True,
        )
        seed.assert_called_once_with(42)


class PaddleOCRVLInferenceTests(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()

        class Batch(dict):
            def to(self, device):
                self["input_ids"] = self["input_ids"].to(device)
                return self

        self.model = Mock()
        self.model.device = torch.device("cpu")
        self.processor = Mock()
        self.processor.image_processor = SimpleNamespace(
            size=SimpleNamespace(shortest_edge=3136)
        )
        self.inputs = Batch(input_ids=torch.tensor([[10, 11, 12]]))
        self.processor.apply_chat_template.return_value = self.inputs
        self.model.generate.return_value = torch.tensor([[10, 11, 12, 44, 45]])
        self.processor.batch_decode.return_value = ["Recognized text"]
        self.config = load_config("paddleocr-vl-1-6")
        self.image = Image.new("RGB", (800, 600), "white")

    def test_ocr_task_returns_text_and_exact_token_counts(self) -> None:
        result = inference_module.infer(
            self.model,
            self.processor,
            {"image": self.image},
            self.config,
        )

        self.assertEqual(result.output["task"], "ocr")
        self.assertEqual(result.output["markdown"], "Recognized text")
        self.assertEqual(result.output["crop_blocks"][0]["bbox"], [0, 0, 800, 600])
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(
            self.processor.apply_chat_template.call_args.kwargs["processor_kwargs"],
            {
                "images_kwargs": {
                    "size": {"shortest_edge": 3136, "longest_edge": 1003520}
                }
            },
        )
        self.assertEqual(
            self.processor.apply_chat_template.call_args.args[0][0]["content"][1]["text"],
            "OCR:",
        )
        self.model.generate.assert_called_once()
        generation_args, generation_kwargs = self.model.generate.call_args
        self.assertEqual(generation_args, ())
        self.assertTrue(torch.equal(generation_kwargs.pop("input_ids"), self.inputs["input_ids"]))
        self.assertEqual(
            generation_kwargs,
            {"max_new_tokens": 512, "do_sample": False},
        )

    def test_task_override_and_instruction_are_forwarded(self) -> None:
        inference_module.infer(
            self.model,
            self.processor,
            {"image": self.image, "task": "table", "state": "Preserve cell order."},
            self.config,
        )

        self.assertEqual(
            self.processor.apply_chat_template.call_args.args[0][0]["content"][1]["text"],
            "Table Recognition:\n\nPreserve cell order.",
        )

    def test_gradio_mode_alias_is_accepted(self) -> None:
        inference_module.infer(
            self.model,
            self.processor,
            {"image": self.image, "mode": "table"},
            self.config,
        )

        self.assertEqual(
            self.processor.apply_chat_template.call_args.args[0][0]["content"][1]["text"],
            "Table Recognition:",
        )

    def test_spotting_upsizes_small_images_and_uses_its_pixel_budget(self) -> None:
        inference_module.infer(
            self.model,
            self.processor,
            {"image": self.image, "task": "spotting"},
            self.config,
        )

        image = self.processor.apply_chat_template.call_args.args[0][0]["content"][0]["image"]
        self.assertEqual(image.size, (1600, 1200))
        self.assertEqual(
            self.processor.apply_chat_template.call_args.kwargs["processor_kwargs"],
            {
                "images_kwargs": {
                    "size": {"shortest_edge": 3136, "longest_edge": 1605632}
                }
            },
        )

    def test_invalid_input_and_unknown_task_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "exige une image"):
            inference_module.infer(self.model, self.processor, "page", self.config)
        with self.assertRaisesRegex(ValueError, "task doit"):
            inference_module.infer(
                self.model,
                self.processor,
                {"image": self.image, "task": "unknown"},
                self.config,
            )

    def test_maximum_output_length_is_reported(self) -> None:
        config = dict(self.config)
        config["generation"] = {"max_new_tokens": 2, "do_sample": False}

        result = inference_module.infer(
            self.model,
            self.processor,
            {"image": self.image},
            config,
        )

        self.assertEqual(result.finish_reason, "length")

    def test_eos_at_output_limit_is_reported_as_stop(self) -> None:
        config = dict(self.config)
        config["generation"] = {"max_new_tokens": 2, "do_sample": False}
        self.model.generation_config = SimpleNamespace(eos_token_id=45)

        result = inference_module.infer(
            self.model,
            self.processor,
            {"image": self.image},
            config,
        )

        self.assertEqual(result.finish_reason, "stop")
