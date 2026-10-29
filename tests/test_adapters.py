import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from common.config import MODELS_DIR, load_config
from common.device import resolve_device, resolve_dtype
from common.transformers import chat_messages, infer_causal, load_causal_model, require_transformers
from common.types import InferenceResult
from tests.helpers import OfflineTestCase, fake_session


class AdapterTests(OfflineTestCase):
    def test_all_adapters_have_contract_and_apps(self) -> None:
        for name in ("lfm3", "d1-2b", "template"):
            for filename in (
                "config.yaml",
                "load.py",
                "inference.py",
                "test.py",
                "apps/cli/app.py",
                "apps/gradio/app.py",
                "apps/api/app.py",
            ):
                with self.subTest(model=name, file=filename):
                    self.assertTrue((MODELS_DIR / name / filename).is_file())
            self.assertTrue(callable(importlib.import_module(f"models.{name}.load").load_model))
            self.assertTrue(callable(importlib.import_module(f"models.{name}.inference").infer))

    def test_old_transformers_fails_explicitly(self) -> None:
        with patch("common.transformers.version", return_value="5.12.1"):
            with self.assertRaisesRegex(RuntimeError, "5.19.0"):
                require_transformers()

    def test_cpu_auto_and_explicit_cuda_failure(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = resolve_device("auto")
            self.assertEqual(device.type, "cpu")
            self.assertEqual(resolve_dtype("auto", device), torch.float32)
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_device("cuda")
        with self.assertRaises(ValueError):
            resolve_dtype("int4", torch.device("cpu"))

    def test_causal_loader_uses_config_and_local_snapshot(self) -> None:
        config = load_config("lfm3")
        config.update({"device": "cpu", "dtype": "float32", "seed": 123})
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        tokenizer = object()
        factory = SimpleNamespace(
            AutoModelForCausalLM=Mock(),
            AutoTokenizer=Mock(),
        )
        factory.AutoModelForCausalLM.from_pretrained.return_value = model
        factory.AutoTokenizer.from_pretrained.return_value = tokenizer
        with (
            patch.dict("sys.modules", {"transformers": factory}),
            patch("common.transformers.require_transformers"),
            patch("common.transformers.download_snapshot", return_value=Path("/snapshot")),
            patch("common.transformers.seed_torch") as seed,
        ):
            loaded_model, loaded_tokenizer = load_causal_model(config)
        self.assertIs(loaded_model, model)
        self.assertIs(loaded_tokenizer, tokenizer)
        seed.assert_called_once_with(123)
        factory.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "/snapshot",
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to.assert_called_once_with(torch.device("cpu"))
        model.eval.assert_called_once()

    def test_d1_loader_uses_automodel_not_causal_lm(self) -> None:
        loader = importlib.import_module("models.d1-2b.load")
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factory = SimpleNamespace(AutoModel=Mock())
        factory.AutoModel.from_pretrained.return_value = model
        config = load_config("d1-2b", overrides={"device": "cpu", "dtype": "float32"})
        with (
            patch.dict("sys.modules", {"transformers": factory}),
            patch.object(loader, "require_transformers"),
            patch.object(loader, "download_snapshot", return_value=Path("/d1-snapshot")),
            patch.object(loader, "seed_torch"),
        ):
            loaded, processor = loader.load_model(config)
        self.assertIs(loaded, model)
        self.assertIsNone(processor)
        factory.AutoModel.from_pretrained.assert_called_once_with(
            "/d1-snapshot",
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        config["generation"] = {"max_new_tokens": 1}
        with self.assertRaisesRegex(ValueError, "system_one"):
            loader.load_model(config)

    def test_causal_inference_decodes_only_new_tokens(self) -> None:
        class Batch(dict):
            def to(self, device):
                self.device = device
                return self

        inputs = Batch(input_ids=torch.tensor([[10, 11, 12]]))
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = inputs
        tokenizer.decode.return_value = "new tokens"
        model = SimpleNamespace(device=torch.device("cpu"), generate=Mock())
        sequences = torch.tensor([[10, 11, 12, 44, 45]])
        config = load_config("lfm3")
        for outputs in (sequences, SimpleNamespace(sequences=sequences)):
            with self.subTest(structured=not isinstance(outputs, torch.Tensor)):
                model.generate.return_value = outputs
                result = infer_causal(model, tokenizer, "hello", config)
                self.assertEqual(result.output, "new tokens")
                self.assertEqual(result.input_tokens, 3)
                self.assertEqual(result.output_tokens, 2)
                self.assertEqual(tokenizer.decode.call_args.args[0].tolist(), [44, 45])
        options = tokenizer.apply_chat_template.call_args.kwargs
        self.assertTrue(options["add_generation_prompt"])
        self.assertFalse(options["enable_thinking"])
        self.assertEqual(model.generate.call_args.kwargs["temperature"], 0.1)

    def test_chat_validation(self) -> None:
        self.assertEqual(chat_messages({"prompt": "hello"})[0]["content"], "hello")
        self.assertEqual(
            chat_messages({"messages": [{"role": "user", "content": "hi"}]})[0]["role"],
            "user",
        )
        for data in (
            "",
            {"image": "/etc/passwd"},
            {"messages": []},
            {"messages": [{}]},
            {"prompt": {"prompt": "not text"}},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                chat_messages(data)

    def test_d1_system_one_and_image(self) -> None:
        adapter = importlib.import_module("models.d1-2b.inference")
        config = load_config("d1-2b")
        answers = {name: {"value": 1} for name in config["questions"]}
        model = Mock()
        model.system_one.return_value = {
            "answers": answers,
            "usage": {"input_tokens": 17, "output_tokens": 0},
        }
        image = Image.new("RGB", (2, 2))
        result = adapter.infer(model, None, {"state": {"report": "fault"}, "image": image}, config)
        self.assertEqual(result.output, answers)
        self.assertEqual(result.output_tokens, 0)
        model.system_one.assert_called_once_with(
            {"report": "fault"}, config["questions"], media=[image]
        )
        model.generate.assert_not_called()
        result = adapter.infer(model, None, "fault", config)
        self.assertEqual(result.input_tokens, 17)

    def test_d1_rejects_empty_or_invalid_payload(self) -> None:
        adapter = importlib.import_module("models.d1-2b.inference")
        config = load_config("d1-2b")
        model = Mock()
        for data in (
            {},
            {"state": "fault", "questions": {}},
            {"state": "fault", "image": "/etc/passwd"},
            {"state": "fault", "unexpected": True},
            {"state": "fault", "questions": {"a": {"type": 42}}},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                adapter.infer(model, None, data, config)
        model.system_one.assert_not_called()

    def test_model_smoke_tests_have_model_specific_assertions(self) -> None:
        session = fake_session("lfm3")
        self.assertIsInstance(importlib.import_module("models.lfm3.test").test(session), InferenceResult)
        session = fake_session("d1-2b")
        session.infer.return_value = InferenceResult(
            {name: {} for name in session.config["questions"]}, 17, 0
        )
        self.assertEqual(importlib.import_module("models.d1-2b.test").test(session).output_tokens, 0)
