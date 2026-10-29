from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from common.runtime import ModelSession
from common.types import AppOptions, InputData
from common.utils import decode_image, with_image

LOGGER = logging.getLogger(__name__)


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_data: str | dict[str, JsonValue]
    image_base64: str | None = Field(default=None, max_length=14_000_000)


def create_app(session: ModelSession) -> FastAPI:
    app = FastAPI(title=f"ModelForge - {session.name}")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "model": session.name}

    @app.post("/infer")
    def infer(payload: InferRequest) -> dict[str, JsonValue]:
        data: InputData = payload.input_data
        if payload.image_base64 is not None:
            try:
                data = with_image(data, decode_image(payload.image_base64))
            except (ValueError, OSError) as error:
                LOGGER.warning("Image API invalide : %s", error)
                raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            return session.infer(data).to_dict()
        except ValueError as error:
            LOGGER.warning("Entree API invalide : %s", error)
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def run(session: ModelSession, options: AppOptions) -> None:
    import uvicorn

    uvicorn.run(create_app(session), host=options.host, port=options.port)
