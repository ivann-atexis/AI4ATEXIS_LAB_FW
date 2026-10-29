from __future__ import annotations

import importlib
import logging
from collections.abc import Generator
from threading import Event, Lock

from common.config import model_directory
from common.types import AppOptions, Config, InferenceResult, InputData, StreamChunk

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
        self._stream = getattr(inference, "stream", None)
        if self._stream is not None and not callable(self._stream):
            raise TypeError("Le hook stream doit etre une fonction.")
        LOGGER.info("Chargement de %s (%s).", name, config["model_id"])
        self.model, self.processor = loader.load_model(config)
        LOGGER.info("Modele pret : %s.", name)

    def infer(self, input_data: InputData) -> InferenceResult:
        with self._lock:
            result = self._infer(self.model, self.processor, input_data, self.config)
        if not isinstance(result, InferenceResult):
            raise TypeError("infer() doit retourner common.types.InferenceResult.")
        return result

    @property
    def supports_streaming(self) -> bool:
        return self._stream is not None

    def stream(
        self, input_data: InputData, cancel: Event | None = None
    ) -> Generator[StreamChunk, None, None]:
        if self._stream is None:
            raise ValueError("Cet adaptateur ne prend pas en charge le streaming.")
        cancellation = cancel if cancel is not None else Event()
        while not cancellation.is_set():
            if self._lock.acquire(timeout=0.1):
                break
            yield StreamChunk()
        else:
            return
        try:
            if cancellation.is_set():
                return
            chunks = self._stream(
                self.model, self.processor, input_data, self.config, cancellation
            )
            final_seen = False
            try:
                for chunk in chunks:
                    if cancellation.is_set():
                        return
                    if not isinstance(chunk, StreamChunk) or final_seen:
                        raise TypeError("Stream invalide : deltas puis un seul chunk final attendus.")
                    final_seen = chunk.result is not None
                    yield chunk
                if not final_seen and not cancellation.is_set():
                    raise RuntimeError("Le stream s'est termine sans resultat final.")
            finally:
                cancellation.set()
                chunks.close()
        finally:
            self._lock.release()

    def run_app(self, app: str, options: AppOptions) -> None:
        if app not in {"cli", "gradio", "api"}:
            raise ValueError(f"Application inconnue : {app}.")
        module = importlib.import_module(f"models.{self.name}.apps.{app}.app")
        module.run(self, options)
