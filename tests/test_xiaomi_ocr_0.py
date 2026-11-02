from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
import torch
from PIL import Image

from common.config import load_config
from common.types import AppOptions, InferenceResult
from tests.helpers import OfflineTestCase, fake_session

load_module = importlib.import_module("models.xiaomi-ocr-0.load")
inference_module = importlib.import_module("models.xiaomi-ocr-0.inference")
gradio_module = importlib.import_module("models.xiaomi-ocr-0.apps.gradio.app")


class XiaomiOCR0LoadTests(OfflineTestCase):
    def test_load_uses_pinned_snapshots_and_transformers_auto_classes(self) -> None:
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        processor = Mock()
        detector = Mock()
        default_config = {
            "layout": {
                "provider": "default",
                "model_dir": "",
                "device": "cuda",
                "threshold": 0.3,
                "nms": True,
            }
        }
        transformers = SimpleNamespace(
            AutoModelForImageTextToText=Mock(),
            AutoProcessor=Mock(),
        )
        transformers.AutoModelForImageTextToText.from_pretrained.return_value = model
        transformers.AutoProcessor.from_pretrained.return_value = processor
        config = load_config("xiaomi-ocr-0", overrides={"device": "cpu", "dtype": "float32"})
        layout_factory = Mock(return_value=detector)

        with (
            patch.dict("sys.modules", {"transformers": transformers}),
            patch.object(load_module, "require_transformers"),
            patch.object(
                load_module,
                "download_snapshot",
                side_effect=[Path("/cache/xiaomi"), Path("/cache/layout")],
            ) as download,
            patch.object(
                load_module,
                "_layout_components",
                return_value=(default_config, layout_factory),
            ),
            patch.object(load_module, "resolve_device", return_value=torch.device("cpu")),
            patch.object(load_module, "resolve_dtype", return_value=torch.float32),
            patch.object(load_module, "seed_torch") as seed,
        ):
            runtime, loaded_processor = load_module.load_model(config)

        self.assertIs(runtime.model, model)
        self.assertIs(runtime.detector, detector)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(
            download.call_args_list[1].args[0],
            {
                "model_id": "PaddlePaddle/PP-DocLayoutV3_safetensors",
                "revision": "e4f489dc536556fc1be02b973ba06756e0d4a2ac",
                "trust_remote_code": False,
                "local_files_only": False,
            },
        )
        transformers.AutoProcessor.from_pretrained.assert_called_once_with(
            "/cache/xiaomi",
            trust_remote_code=False,
            local_files_only=True,
        )
        transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(
            "/cache/xiaomi",
            dtype=torch.float32,
            trust_remote_code=False,
            local_files_only=True,
        )
        layout_factory.assert_called_once()
        self.assertEqual(layout_factory.call_args.args[0]["model_dir"], "/cache/layout")
        self.assertEqual(
            layout_factory.call_args.args[0]["provider"],
            "ppdoclayout3_torch",
        )
        detector.start.assert_called_once_with()
        seed.assert_called_once_with(42)


class XiaomiOCR0InferenceTests(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()

        class Batch(dict):
            def to(self, device):
                self["input_ids"] = self["input_ids"].to(device)
                return self

        self.model = Mock()
        self.model.device = torch.device("cpu")
        self.processor = Mock()
        self.config = load_config("xiaomi-ocr-0")
        self.inputs = Batch(input_ids=torch.tensor([[10, 11, 12]]))
        self.processor.apply_chat_template.return_value = self.inputs
        self.model.generate.return_value = torch.tensor([[10, 11, 12, 44, 45]])
        self.processor.batch_decode.return_value = ["Recognized text"]
        self.detector = Mock()
        self.runtime = SimpleNamespace(model=self.model, detector=self.detector)
        self.image = Image.new("RGB", (100, 40), "white")

    def test_region_mode_returns_clickable_boxes_markdown_and_real_token_counts(self) -> None:
        self.detector.detect.return_value = [
            [{"label": "text", "bbox": [5, 6, 95, 36], "score": 0.94}]
        ]
        self.detector.crop.return_value = [
            {
                "label": "text",
                "bbox": [5, 6, 95, 36],
                "score": 0.94,
                "image": np.zeros((30, 90, 3), dtype=np.uint8),
            }
        ]

        result = inference_module.infer(
            self.runtime,
            self.processor,
            {"image": self.image},
            self.config,
        )

        self.assertEqual(
            result.output,
            {
                "image_width": 100,
                "image_height": 40,
                "mode": "regions",
                "crop_blocks": [
                    {
                        "order": 1,
                        "label": "text",
                        "bbox": [5, 6, 95, 36],
                        "score": 0.94,
                        "content": "Recognized text",
                        "raw_content": "Recognized text",
                    }
                ],
                "markdown": "Recognized text",
            },
        )
        self.assertEqual(result.input_tokens, 3)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(self.processor.apply_chat_template.call_args.kwargs["enable_thinking"], False)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 4096)
        self.detector.detect.assert_called_once_with([self.image])
        self.detector.crop.assert_called_once()

    def test_merged_layout_crops_are_transcribed_once_with_a_union_box(self) -> None:
        self.detector.detect.return_value = [[]]
        self.detector.crop.return_value = [
            {
                "label": "text",
                "bbox": [5, 6, 40, 20],
                "score": 0.94,
                "image": np.zeros((25, 35, 3), dtype=np.uint8),
                "group_id": 0,
            },
            {
                "label": "text",
                "bbox": [5, 20, 40, 36],
                "score": 0.92,
                "image": None,
                "group_id": 0,
            },
        ]

        result = inference_module.infer(
            self.runtime,
            self.processor,
            {"image": self.image},
            self.config,
        )

        self.assertEqual(len(result.output["crop_blocks"]), 1)
        self.assertEqual(result.output["crop_blocks"][0]["bbox"], [5, 6, 40, 36])
        self.assertEqual(result.output["crop_blocks"][0]["content"], "Recognized text")
        self.assertEqual(result.output["markdown"], "Recognized text")
        self.assertEqual(self.model.generate.call_count, 1)

    def test_missing_ungrouped_crop_image_returns_a_clear_error(self) -> None:
        self.detector.detect.return_value = [[]]
        self.detector.crop.return_value = [
            {
                "label": "text",
                "bbox": [5, 6, 40, 20],
                "score": 0.94,
                "image": None,
            }
        ]

        with self.assertRaisesRegex(ValueError, "ne contient pas d'image"):
            inference_module.infer(
                self.runtime,
                self.processor,
                {"image": self.image},
                self.config,
            )

        self.model.generate.assert_not_called()

    def test_page_mode_returns_a_full_image_selection_without_layout_detection(self) -> None:
        result = inference_module.infer(
            self.runtime,
            self.processor,
            {"image": self.image, "mode": "page"},
            self.config,
        )

        self.assertEqual(result.output["mode"], "page")
        self.assertEqual(result.output["crop_blocks"][0]["bbox"], [0, 0, 100, 40])
        self.assertEqual(result.output["markdown"], "Recognized text")
        self.detector.detect.assert_not_called()
        self.detector.crop.assert_not_called()

    def test_invalid_inputs_are_rejected_before_backend_calls(self) -> None:
        for data in (
            "not an image",
            {},
            {"image": "/tmp/image.png"},
            {"image": self.image, "mode": "unknown"},
            {"image": self.image, "unexpected": True},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                inference_module.infer(self.runtime, self.processor, data, self.config)
        self.model.generate.assert_not_called()
        self.detector.detect.assert_not_called()


class XiaomiOCR0AppTests(OfflineTestCase):
    def test_gradio_app_passes_image_and_mode_and_displays_regions(self) -> None:
        session = fake_session("xiaomi-ocr-0")
        session.infer.return_value = InferenceResult(
            {
                "crop_blocks": [
                    {
                        "label": "text",
                        "bbox": [10, 10, 90, 35],
                        "content": "recognized",
                    }
                ],
                "markdown": "recognized",
            }
        )
        app = gradio_module.create_ocr_app(session, AppOptions(""))
        self.addCleanup(app.close)
        analyze = next(
            fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "analyze"
        )
        image = Image.new("RGB", (100, 40), "white")
        annotated, rows, markdown, blocks, selected, elapsed = analyze(image, "regions")

        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(rows[0][1:], ["text", "10, 10, 90, 35", "recognized"])
        self.assertEqual(markdown, "recognized")
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(selected)
        self.assertGreaterEqual(elapsed, 0)
        session.infer.assert_called_once_with({"image": image, "mode": "regions"})
        self.assertTrue(app.config["dependencies"])

    def test_invalid_mode_is_reported_without_inference(self) -> None:
        session = fake_session("xiaomi-ocr-0")
        app = gradio_module.create_ocr_app(session, AppOptions(""))
        self.addCleanup(app.close)
        analyze = next(
            fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "analyze"
        )
        with self.assertRaises(gr.Error):
            analyze(Image.new("RGB", (100, 40)), "unknown")
        session.infer.assert_not_called()
