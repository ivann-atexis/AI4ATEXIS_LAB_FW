from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from threading import Event
from typing import Annotated, Any, Literal
from uuid import uuid4

import anyio
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import Send

from common.runtime import ModelSession
from common.types import InferenceResult, InputData

LOGGER = logging.getLogger(__name__)
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
StopString = Annotated[str, Field(min_length=1)]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextPart(APIModel):
    type: Literal["text"]
    text: str


class ChatMessage(APIModel):
    role: Literal["system", "user", "assistant"]
    content: str | Annotated[list[TextPart], Field(min_length=1)]


class StreamOptions(APIModel):
    include_usage: bool = Field(default=False, strict=True)


class ChatRequest(APIModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = Field(default=False, strict=True)
    stream_options: StreamOptions | None = None
    max_tokens: PositiveInt | None = None
    max_completion_tokens: PositiveInt | None = None
    temperature: float | None = Field(default=None, ge=0, le=2, strict=True)
    top_p: float | None = Field(default=None, gt=0, le=1, strict=True)
    stop: StopString | Annotated[list[StopString], Field(max_length=4)] | None = None
    n: int = Field(default=1, strict=True, ge=1, le=1)
    user: str | None = None

    @model_validator(mode="after")
    def validate_options(self) -> ChatRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Utiliser max_tokens ou max_completion_tokens, pas les deux.")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options exige stream=true.")
        return self

    def input_data(self) -> InputData:
        messages = [
            {
                "role": message.role,
                "content": (
                    message.content
                    if isinstance(message.content, str)
                    else "".join(part.text for part in message.content)
                ),
            }
            for message in self.messages
        ]
        generation: dict[str, Any] = {}
        maximum = self.max_completion_tokens or self.max_tokens
        if maximum is not None:
            generation["max_new_tokens"] = maximum
        for key in ("temperature", "top_p"):
            value = getattr(self, key)
            if value is not None:
                generation[key] = value
        if self.stop is not None:
            generation["stop_strings"] = [self.stop] if isinstance(self.stop, str) else self.stop
        data: dict[str, Any] = {"messages": messages}
        if generation:
            data["generation"] = generation
        return data


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class CompletionChoice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: TokenUsage | None = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "AI4ATEXIS_LAB_FW"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ErrorDetails(BaseModel):
    message: str
    type: str
    param: str | None
    code: str | None


class ErrorResponse(BaseModel):
    error: ErrorDetails


def error_body(
    message: str, *, server: bool = False, param: str | None = None, code: str | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "server_error" if server else "invalid_request_error",
            "param": param,
            "code": code,
        }
    }


class OpenAIError(HTTPException):
    def __init__(
        self, status: int, message: str, *, param: str | None = None, code: str | None = None
    ) -> None:
        self.body = error_body(message, server=status >= 500, param=param, code=code)
        super().__init__(
            status_code=status,
            detail=self.body,
        )


def token_usage(result: InferenceResult) -> TokenUsage | None:
    if result.input_tokens is None or result.output_tokens is None:
        return None
    return TokenUsage(
        prompt_tokens=result.input_tokens,
        completion_tokens=result.output_tokens,
        total_tokens=result.input_tokens + result.output_tokens,
    )


def completion_choice(result: InferenceResult) -> CompletionChoice:
    if not isinstance(result.output, str) or result.finish_reason is None:
        raise TypeError("Un adaptateur chat doit retourner un texte et un finish_reason.")
    return CompletionChoice(
        message=AssistantMessage(content=result.output), finish_reason=result.finish_reason
    )


def sse(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n\n"


async def chat_events(
    session: ModelSession, data: InputData, model: str, identifier: str, created: int,
    include_usage: bool,
) -> AsyncGenerator[str, None]:
    cancel = Event()
    chunks = session.stream(data, cancel)
    base = {
        "id": identifier,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }

    def frame(delta: dict[str, str], finish_reason: str | None = None) -> str:
        value = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if include_usage:
            value["usage"] = None
        return sse(value)

    try:
        yield frame({"role": "assistant", "content": ""})
        final = None
        while True:
            chunk = await run_in_threadpool(next, chunks, None)
            if chunk is None:
                break
            if chunk.result is not None:
                final = chunk.result
            elif chunk.text:
                yield frame({"content": chunk.text})
            else:
                yield ": keep-alive\n\n"
        if final is None:
            raise RuntimeError("Le stream ne contient pas de resultat final.")
        choice = completion_choice(final)
        yield frame({}, choice.finish_reason)
        if include_usage:
            usage = token_usage(final)
            yield sse({
                **base, "choices": [], "usage": usage.model_dump() if usage is not None else None
            })
        yield "data: [DONE]\n\n"
    except (ValueError, TypeError, RuntimeError, OSError):
        LOGGER.exception("Echec du stream OpenAI pour %s.", session.name)
        yield sse(error_body("Echec de l'inference. Consulter les logs du serveur.", server=True))
    finally:
        cancel.set()
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(chunks.close)


class ChatStreamingResponse(StreamingResponse):
    def __init__(self, events: AsyncGenerator[str, None]) -> None:
        self.events = events
        super().__init__(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            # ASGI 2.4 send failures do not close Starlette's iterator.
            with anyio.CancelScope(shield=True):
                await self.events.aclose()


def add_openai_routes(app: FastAPI, session: ModelSession) -> None:
    router = APIRouter(prefix="/v1", tags=["OpenAI-compatible"])
    created = int(time.time())
    model_id = session.config["model_id"]
    chat_enabled = session.config.get("api", {}).get("chat", False)

    @app.exception_handler(OpenAIError)
    async def openai_error(request: Request, error: OpenAIError) -> JSONResponse:
        LOGGER.warning("Erreur OpenAI HTTP %d : %s", error.status_code, error.body["error"]["code"])
        return JSONResponse(status_code=error.status_code, content=error.body)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> Any:
        if not request.url.path.startswith("/v1/"):
            return await request_validation_exception_handler(request, error)
        issue = error.errors()[0]
        param = ".".join(str(item) for item in issue["loc"] if item != "body") or None
        LOGGER.warning("Requete OpenAI invalide : %s.", param or "body")
        return JSONResponse(
            status_code=400,
            content=error_body(issue["msg"], param=param, code="invalid_request"),
        )

    @router.get("/models", response_model=ModelList)
    def models() -> ModelList:
        data = (
            [ModelInfo(id=model_id, created=created)]
            if chat_enabled else []
        )
        return ModelList(data=data)

    @router.post(
        "/chat/completions",
        response_model=ChatCompletion,
        response_model_exclude_none=True,
        responses={
            200: {
                "description": "Chat completion JSON ou flux SSE si stream=true.",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            400: {"model": ErrorResponse, "description": "Requete non prise en charge."},
            404: {"model": ErrorResponse, "description": "Modele inconnu dans cette session."},
            500: {"model": ErrorResponse, "description": "Erreur du backend."},
            "default": {"model": ErrorResponse, "description": "Erreur au format OpenAI."},
        },
    )
    async def chat_completions(payload: ChatRequest) -> ChatCompletion | StreamingResponse:
        if not chat_enabled:
            raise OpenAIError(400, "Ce modele n'expose pas de chat. Utiliser /infer.",
                              param="model", code="chat_not_supported")
        if payload.model not in {model_id, session.name}:
            raise OpenAIError(404, "Modele inconnu dans cette session.",
                              param="model", code="model_not_found")
        if payload.stream and not session.supports_streaming:
            raise OpenAIError(400, "Cet adaptateur ne prend pas en charge le streaming.",
                              param="stream", code="stream_not_supported")
        identifier = "chatcmpl-" + uuid4().hex
        timestamp = int(time.time())
        data = payload.input_data()
        if payload.stream:
            return ChatStreamingResponse(
                chat_events(
                    session, data, model_id, identifier, timestamp,
                    bool(payload.stream_options and payload.stream_options.include_usage),
                ),
            )
        try:
            result = await run_in_threadpool(session.infer, data)
            choice = completion_choice(result)
        except ValueError as error:
            raise OpenAIError(400, str(error), code="invalid_request") from error
        except (TypeError, RuntimeError, OSError) as error:
            LOGGER.exception("Echec de l'inference OpenAI pour %s.", session.name)
            raise OpenAIError(
                500, "Echec de l'inference. Consulter les logs du serveur.", code="inference_error"
            ) from error
        return ChatCompletion(
            id=identifier,
            created=timestamp,
            model=model_id,
            choices=[choice],
            usage=token_usage(result),
        )

    app.include_router(router)
