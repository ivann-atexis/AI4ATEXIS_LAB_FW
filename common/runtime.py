from __future__ import annotations

import importlib
import logging
from threading import Lock

from common.config import model_directory
from common.types import AppOptions, Config, InferenceResult, InputData

LOGGER = logging.getLogger(__name__)


class ModelSession:
    def __init__(self, name: str, config: Config) -> None:
        model_directory(name)
        self.name = name
        self.config = config
        self._lock = Lock()
        loader = importlib.import_module(f"models.{name}.load")
        inference = importlib.import_module(f"models.{name}.inference")
        self._infer = inference.infer
        LOGGER.info("Chargement de %s (%s).", name, config["model_id"])
        self.model, self.processor = loader.load_model(config)
        LOGGER.info("Modele pret : %s.", name)

    def infer(self, input_data: InputData) -> InferenceResult:
        with self._lock:
            result = self._infer(self.model, self.processor, input_data, self.config)
        if not isinstance(result, InferenceResult):
            raise TypeError("infer() doit retourner common.types.InferenceResult.")
        return result

    def run_app(self, app: str, options: AppOptions) -> None:
        if app not in {"cli", "gradio", "api"}:
            raise ValueError(f"Application inconnue : {app}.")
        module = importlib.import_module(f"models.{self.name}.apps.{app}.app")
        module.run(self, options)
