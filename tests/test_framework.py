import contextlib
import importlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import models

from common.benchmark import DeviceMetrics, benchmark, gpu_snapshot
from common.config import load_config, model_directory
from common.hub import download_snapshot
from common.runtime import ModelSession
from common.types import AppOptions, InferenceResult
from common.utils import write_json
from scripts import benchmark as benchmark_command
from scripts import download, run
from tests.helpers import OfflineTestCase, fake_session


class FrameworkTests(OfflineTestCase):
    def test_configuration_overrides_and_seed_validation(self) -> None:
        config = load_config("lfm3", overrides={"device": "cpu", "seed": 123})
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["seed"], 123)
        for seed in (-1, True, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                load_config("lfm3", overrides={"seed": seed})

    def test_names_cannot_escape_model_directory(self) -> None:
        for name in ("../lfm3", "/tmp", "a.b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                model_directory(name)

    def test_invalid_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for contents in ("[]", "model_id: 12", "model_id: org/name\nseed: false"):
                path.write_text(contents, encoding="utf-8")
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_config("lfm3", path)

    def test_hub_download_is_pinned_and_offline(self) -> None:
        config = load_config("lfm3")
        with patch(
            "huggingface_hub.snapshot_download", return_value="/snapshots/" + config["revision"]
        ) as snapshot:
            path = download_snapshot(config)
        snapshot.assert_called_once_with(
            repo_id=config["model_id"], revision=config["revision"], local_files_only=True
        )
        self.assertEqual(path.name, config["revision"])
        self.assertEqual(config["resolved_revision"], config["revision"])
        config["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "revision"):
            download_snapshot(config)

    def test_result_rejects_invalid_metrics_and_non_json(self) -> None:
        for tokens in (-1, True, 1.5):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                InferenceResult("text", output_tokens=tokens)
        with self.assertRaises(ValueError):
            InferenceResult(float("nan"))
        with self.assertRaises(TypeError):
            InferenceResult(object())

    def test_future_hyphenated_adapter_works_without_torch_or_hf(self) -> None:
        name = "future-custom-backend"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / name
            directory.mkdir()
            (directory / "config.yaml").write_text(
                "model_id: local/model\ndevice: custom\ndtype: int8\n", encoding="utf-8"
            )
            (directory / "load.py").write_text(
                "def load_model(config):\n    return object(), None\n", encoding="utf-8"
            )
            (directory / "inference.py").write_text(
                "from common.types import InferenceResult\n"
                "def infer(model, processor, input_data, config):\n"
                "    return InferenceResult({'received': input_data})\n",
                encoding="utf-8",
            )
            app = directory / "apps" / "cli"
            app.mkdir(parents=True)
            (app / "app.py").write_text("from common.apps.cli import run\n", encoding="utf-8")
            try:
                with (
                    patch("common.config.MODELS_DIR", Path(temporary)),
                    patch.object(models, "__path__", [temporary]),
                    patch.dict(sys.modules, {"torch": None, "transformers": None}),
                ):
                    importlib.invalidate_caches()
                    session = ModelSession(name, load_config(name))
                    self.assertEqual(
                        session.infer({"tensor": [1, 2]}).output, {"received": {"tensor": [1, 2]}}
                    )
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        session.run_app("cli", AppOptions("hello"))
                    self.assertEqual(json.loads(output.getvalue())["output"], {"received": "hello"})
            finally:
                for module in list(sys.modules):
                    if module == f"models.{name}" or module.startswith(f"models.{name}."):
                        del sys.modules[module]

    def test_session_rejects_nonstandard_adapter_result(self) -> None:
        loader = SimpleNamespace(load_model=Mock(return_value=(object(), None)))
        inference = SimpleNamespace(infer=Mock(return_value="not an InferenceResult"))
        with patch.dict(
            sys.modules,
            {"models.lfm3.load": loader, "models.lfm3.inference": inference},
        ):
            session = ModelSession("lfm3", load_config("lfm3"))
        with self.assertRaisesRegex(TypeError, "InferenceResult"):
            session.infer("hello")

    def test_benchmark_excludes_warmup_and_records_real_counts(self) -> None:
        session = fake_session(tokens=2)
        with (
            patch("common.benchmark.gpu_snapshot", return_value={"available": False}),
            patch("common.benchmark.time.perf_counter", side_effect=[10.0, 10.2, 11.0, 11.4]),
        ):
            report = benchmark(session, "hello", runs=2, warmup=3, load_seconds=9.0)
        self.assertEqual(session.infer.call_count, 5)
        self.assertEqual(len(report["samples"]), 2)
        measurement = report["measurement"]
        self.assertAlmostEqual(measurement["latency_seconds"]["mean"], 0.3)
        self.assertAlmostEqual(measurement["latency_seconds"]["p95"], 0.39)
        self.assertAlmostEqual(measurement["requests_per_second"], 2 / 0.6)
        self.assertAlmostEqual(measurement["generated_tokens_per_second"], 4 / 0.6)
        self.assertEqual(measurement["load_seconds"], 9.0)
        self.assertEqual(len(report["environment"]["source_sha256"]), 64)
        self.assertEqual(report["config"]["revision"], session.config["revision"])

    def test_decision_or_unknown_usage_has_no_generated_token_rate(self) -> None:
        for count in (0, None):
            session = fake_session("d1-2b", tokens=count)
            with (
                self.subTest(count=count),
                patch("common.benchmark.gpu_snapshot", return_value={"available": False}),
                patch("common.benchmark.time.perf_counter", side_effect=[1.0, 2.0]),
            ):
                report = benchmark(session, "state", runs=1, warmup=0, load_seconds=0.1)
            self.assertIsNone(report["measurement"]["generated_tokens_per_second"])

    def test_cuda_timing_synchronizes_and_reports_allocator_peaks(self) -> None:
        class FakeModule:
            pass

        model = FakeModule()
        model.device = SimpleNamespace(type="cuda")
        torch = SimpleNamespace(nn=SimpleNamespace(Module=FakeModule), cuda=Mock(), mps=Mock())
        torch.cuda.max_memory_allocated.return_value = 100
        torch.cuda.max_memory_reserved.return_value = 200
        with patch.dict(sys.modules, {"torch": torch}):
            metrics = DeviceMetrics(model)
            metrics.synchronize()
            metrics.reset_peak()
            memory = metrics.memory()
        torch.cuda.synchronize.assert_called_once_with(model.device)
        torch.cuda.reset_peak_memory_stats.assert_called_once_with(model.device)
        self.assertEqual(memory["peak_allocated_bytes"], 100)
        self.assertEqual(memory["peak_reserved_bytes"], 200)

    def test_gpu_snapshot_preserves_unavailable_gb10_memory(self) -> None:
        result = SimpleNamespace(
            returncode=0, stdout="GPU-1, NVIDIA GB10, 580.178.04, 25, [N/A], [N/A]\n"
        )
        with patch("common.benchmark.subprocess.run", return_value=result):
            snapshot = gpu_snapshot()
        self.assertEqual(snapshot["devices"][0]["memory.total"], "[N/A]")

    def test_atomic_json_write_does_not_replace_valid_report_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json(path, {"ok": True})
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_run_cli_and_listing_without_download(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run.main(["--list-models"]), 0)
        self.assertIn("d1-2b", output.getvalue())
        session = fake_session()
        with patch("scripts.run.ModelSession", return_value=session):
            self.assertEqual(run.main(["lfm3", "--prompt", "test", "--offline"]), 0)
        session.run_app.assert_called_once()
        self.assertEqual(session.run_app.call_args.args[1].input_data, "test")

    def test_invalid_cli_controls_fail_before_loading(self) -> None:
        with patch("scripts.run.ModelSession") as session:
            self.assertEqual(run.main(["lfm3", "--app", "api", "--prompt", "unused"]), 1)
            self.assertEqual(run.main(["lfm3", "--app", "gradio", "--prompt", "unused"]), 1)
            self.assertEqual(run.main(["lfm3", "--test", "--prompt", "unused"]), 1)
            session.assert_not_called()
        for command in (["--runs", "0"], ["--warmup", "-1"]):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    benchmark_command.main(["lfm3", *command])
                self.assertEqual(error.exception.code, 2)

    def test_api_does_not_require_an_example_input(self) -> None:
        config = load_config("lfm3")
        del config["example_input"]
        session = fake_session()
        with (
            patch("scripts.run.get_config", return_value=config),
            patch("scripts.run.ModelSession", return_value=session),
        ):
            self.assertEqual(run.main(["lfm3", "--app", "api"]), 0)
        self.assertEqual(session.run_app.call_args.args[0], "api")

    def test_benchmark_command_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            session = fake_session()
            with (
                patch("scripts.benchmark.ModelSession", return_value=session),
                patch("common.benchmark.gpu_snapshot", return_value={"available": False}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = benchmark_command.main(
                    ["lfm3", "--runs", "1", "--warmup", "0", "--output", str(path)]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(path.read_text())["measurement"]["runs"], 1)

    def test_download_command_does_not_load_model(self) -> None:
        def fake_download(config):
            config["resolved_revision"] = config["revision"]
            return Path("/snapshot")

        with (
            patch("scripts.download.download_snapshot", side_effect=fake_download),
            patch("common.runtime.ModelSession") as loader,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(download.main(["lfm3", "--offline"]), 0)
        loader.assert_not_called()
