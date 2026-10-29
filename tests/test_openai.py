import asyncio
import copy
import json
from threading import Event

from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from common.apps.api import create_app
from common.apps.openai import chat_events
from common.types import InferenceResult, StreamChunk
from tests.helpers import OfflineTestCase, fake_session


def request_body(**options):
    return {
        "model": "LiquidAI/LFM3-1.7B",
        "messages": [{"role": "user", "content": "Bonjour"}],
        **options,
    }


def stream_chunks(text="Bonjour", tokens=3):
    yield StreamChunk(text=text[:3])
    yield StreamChunk(text=text[3:])
    yield StreamChunk(result=InferenceResult(text, 2, tokens, "stop"))


def events(response):
    return [
        json.loads(line[6:]) if line[6:] != "[DONE]" else "[DONE]"
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class OpenAITests(OfflineTestCase):
    def test_discovery_lists_only_the_loaded_chat_model(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        listing = response.json()
        self.assertEqual(listing["object"], "list")
        self.assertEqual(len(listing["data"]), 1)
        self.assertEqual(listing["data"][0]["id"], session.config["model_id"])
        self.assertEqual(listing["data"][0]["object"], "model")
        session.infer.assert_not_called()
        session.stream.assert_not_called()

    def test_decision_model_is_not_advertised_as_chat(self) -> None:
        session = fake_session("d1-2b")
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.get("/v1/models").json(), {"object": "list", "data": []})
            response = client.post(
                "/v1/chat/completions", json=request_body(model=session.config["model_id"])
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "chat_not_supported")
        session.infer.assert_not_called()

    def test_template_inherits_the_openai_routes(self) -> None:
        session = fake_session("template")
        with TestClient(create_app(session)) as client:
            self.assertEqual(
                client.get("/v1/models").json()["data"][0]["id"], session.config["model_id"]
            )
            response = client.post(
                "/v1/chat/completions", json=request_body(model="template")
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], session.config["model_id"])

    def test_completion_shape_and_history(self) -> None:
        session = fake_session()
        messages = [
            {"role": "system", "content": "Reponds en francais."},
            {"role": "user", "content": "Premier message"},
            {"role": "assistant", "content": "Premiere reponse"},
            {"role": "user", "content": [{"type": "text", "text": "Bon"}, {"type": "text", "text": "jour"}]},
        ]
        with TestClient(create_app(session)) as client:
            response = client.post("/v1/chat/completions", json=request_body(messages=messages))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertIsInstance(body["created"], int)
        self.assertEqual(
            body["choices"],
            [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        )
        self.assertEqual(
            body["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        )
        expected = [*messages[:-1], {"role": "user", "content": "Bonjour"}]
        session.infer.assert_called_once_with({"messages": expected})

    def test_parameters_are_per_request_and_max_token_aliases_work(self) -> None:
        session = fake_session()
        before = copy.deepcopy(session.config)
        session.infer.return_value = InferenceResult("short", 4, 2, "length")
        with TestClient(create_app(session)) as client:
            for key in ("max_tokens", "max_completion_tokens"):
                with self.subTest(key=key):
                    response = client.post(
                        "/v1/chat/completions",
                        json=request_body(**{key: 2}, temperature=0, top_p=0.8, stop=["END"]),
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")
                    self.assertEqual(
                        session.infer.call_args.args[0]["generation"],
                        {"max_new_tokens": 2, "temperature": 0.0, "top_p": 0.8, "stop_strings": ["END"]},
                    )
        self.assertEqual(session.config, before)

    def test_invalid_or_unsupported_options_are_openai_errors(self) -> None:
        session = fake_session()
        cases = [
            {"model": "unknown"},
            {"temperature": -1},
            {"temperature": True},
            {"top_p": 0},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"max_tokens": 2, "max_completion_tokens": 2},
            {"n": 2},
            {"response_format": {"type": "json_object"}},
            {"stream_options": {"include_usage": True}},
            {"stop": [""]},
            {"messages": []},
            {"messages": [{"role": "tool", "content": "result"}]},
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image"}}]}]},
        ]
        with TestClient(create_app(session)) as client:
            for options in cases:
                with self.subTest(options=options):
                    response = client.post("/v1/chat/completions", json=request_body(**options))
                    self.assertEqual(response.status_code, 404 if options.get("model") == "unknown" else 400)
                    self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
                    self.assertNotIn("detail", response.json())
        session.infer.assert_not_called()
        session.stream.assert_not_called()

    def test_inactive_tool_options_do_not_break_plain_chat(self) -> None:
        session = fake_session()
        session.stream.side_effect = lambda data, cancel: stream_chunks()
        with TestClient(create_app(session)) as client:
            for tools in (None, []):
                for streaming in (False, True):
                    with self.subTest(tools=tools, streaming=streaming):
                        response = client.post(
                            "/v1/chat/completions",
                            json=request_body(
                                tools=tools, tool_choice="none", parallel_tool_calls=True,
                                stream=streaming,
                            ),
                        )
                        self.assertEqual(response.status_code, 200)
                        if streaming:
                            self.assertEqual(events(response)[-1], "[DONE]")
                            data = session.stream.call_args.args[0]
                        else:
                            data = session.infer.call_args.args[0]
                        self.assertEqual(data, {"messages": request_body()["messages"]})

    def test_active_tools_are_refused_with_an_actionable_error(self) -> None:
        session = fake_session()
        session.config["api"]["tool_calling"] = False
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        with TestClient(create_app(session)) as client:
            for options, param in (
                ({"tools": tools}, "tools"),
                ({"tools": tools, "tool_choice": "required"}, "tools"),
                ({"tools": tools, "tool_choice": "auto"}, "tools"),
                ({"tools": tools, "tool_choice": {"type": "function", "function": {"name": "lookup"}}}, "tools"),
            ):
                with self.subTest(options=options):
                    response = client.post("/v1/chat/completions", json=request_body(**options))
                    self.assertEqual(response.status_code, 400)
                    error = response.json()["error"]
                    self.assertEqual(error["param"], param)
                    self.assertEqual(error["code"], "tools_not_supported")
                    self.assertIn("OpenWebUI", error["message"])
        session.infer.assert_not_called()
        session.stream.assert_not_called()

    def test_unknown_fields_remain_forbidden_and_are_named_in_the_message(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            response = client.post(
                "/v1/chat/completions", json=request_body(unknown_option=True)
            )
        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["param"], "unknown_option")
        self.assertIn("unknown_option", error["message"])
        session.infer.assert_not_called()

    def test_non_finite_sampling_is_rejected(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            response = client.post(
                "/v1/chat/completions",
                content=json.dumps(request_body(temperature=float("nan"))),
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(response.status_code, 400)
        session.infer.assert_not_called()

    def test_backend_failures_are_not_success_responses(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            for failure in (RuntimeError("GPU error"), TypeError("bad output"), ValueError("invalid input")):
                session.infer.side_effect = failure
                with self.subTest(failure=type(failure).__name__):
                    response = client.post("/v1/chat/completions", json=request_body())
                    self.assertEqual(response.status_code, 400 if isinstance(failure, ValueError) else 500)
                    self.assertIn("error", response.json())

    def test_non_chat_output_or_missing_finish_reason_is_not_faked(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            for result in (InferenceResult({"decision": True}), InferenceResult("text")):
                session.infer.return_value = result
                with self.subTest(result=result):
                    response = client.post("/v1/chat/completions", json=request_body())
                    self.assertEqual(response.status_code, 500)

    def test_unknown_usage_is_not_invented(self) -> None:
        session = fake_session(tokens=None)
        with TestClient(create_app(session)) as client:
            response = client.post("/v1/chat/completions", json=request_body())
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("usage", response.json())

    def test_sse_content_finish_usage_and_done(self) -> None:
        session = fake_session()
        session.stream.side_effect = lambda data, cancel: stream_chunks()
        with TestClient(create_app(session)) as client:
            response = client.post(
                "/v1/chat/completions",
                json=request_body(stream=True, stream_options={"include_usage": True}),
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        frames = events(response)
        self.assertEqual(frames[-1], "[DONE]")
        self.assertEqual(frames[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(
            "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames[:-2]),
            "Bonjour",
        )
        self.assertEqual(frames[-3]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(frames[-2]["choices"], [])
        self.assertEqual(
            frames[-2]["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        )
        for frame in frames[:-1]:
            self.assertEqual(frame["id"], frames[0]["id"])
            self.assertEqual(frame["created"], frames[0]["created"])
            self.assertEqual(frame["model"], session.config["model_id"])
        session.infer.assert_not_called()

    def test_sse_omits_usage_unless_requested(self) -> None:
        session = fake_session()
        session.stream.side_effect = lambda data, cancel: stream_chunks()
        with TestClient(create_app(session)) as client:
            response = client.post("/v1/chat/completions", json=request_body(stream=True))
        frames = events(response)
        self.assertEqual(frames[-1], "[DONE]")
        self.assertTrue(all("usage" not in frame for frame in frames[:-1]))

    def test_missing_stream_hook_is_an_explicit_error(self) -> None:
        session = fake_session()
        session.supports_streaming = False
        with TestClient(create_app(session)) as client:
            response = client.post("/v1/chat/completions", json=request_body(stream=True))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "stream_not_supported")
        session.stream.assert_not_called()

    def test_stream_error_or_missing_final_never_emits_success_finish(self) -> None:
        session = fake_session()

        def broken(data, cancel):
            yield StreamChunk(text="partial")
            raise RuntimeError("backend failed")

        def incomplete(data, cancel):
            yield StreamChunk(text="partial")

        with TestClient(create_app(session)) as client:
            for implementation in (broken, incomplete):
                session.stream.side_effect = implementation
                with self.subTest(implementation=implementation.__name__):
                    response = client.post("/v1/chat/completions", json=request_body(stream=True))
                    frames = events(response)
                    self.assertEqual(frames[-1]["error"]["type"], "server_error")
                    self.assertNotIn("[DONE]", frames)
                    self.assertFalse(any(
                        frame.get("choices") and frame["choices"][0].get("finish_reason")
                        for frame in frames
                    ))

    def test_closing_sse_cancels_and_closes_the_adapter(self) -> None:
        session = fake_session()
        closed = Event()
        cancellations = []

        def implementation(data, cancel):
            cancellations.append(cancel)
            try:
                yield StreamChunk(text="first")
                yield StreamChunk(text="never requested")
            finally:
                closed.set()

        session.stream.side_effect = implementation

        async def consume():
            stream = chat_events(session, {"messages": []}, "lfm3", "id", 1, False)
            await anext(stream)
            await anext(stream)
            await stream.aclose()

        asyncio.run(consume())
        self.assertTrue(closed.is_set())
        self.assertTrue(cancellations[0].is_set())

    def test_disconnect_closes_stream_for_both_asgi_protocols(self) -> None:
        from common.apps.openai import ChatRequest

        for protocol in ("2.3", "2.4"):
            session = fake_session()
            closed = Event()
            cancellations = []

            def implementation(data, cancel):
                cancellations.append(cancel)
                try:
                    yield StreamChunk(text="first")
                    yield StreamChunk(text="second")
                finally:
                    closed.set()

            session.stream.side_effect = implementation
            app = create_app(session)
            route = next(route for route in app.routes if getattr(route, "path", "") == "/v1/chat/completions")

            async def disconnect():
                response = await route.endpoint(ChatRequest(**request_body(stream=True)))
                disconnected = asyncio.Event()

                async def receive():
                    await disconnected.wait()
                    return {"type": "http.disconnect"}

                async def send(message):
                    if message["type"] == "http.response.body" and b'"content": "first"' in message["body"]:
                        disconnected.set()
                        if protocol == "2.4":
                            raise OSError("Client closed the connection.")

                try:
                    await response(
                        {"type": "http", "asgi": {"spec_version": protocol}}, receive, send
                    )
                except ClientDisconnect:
                    pass
                self.assertTrue(closed.is_set())
                self.assertTrue(cancellations[0].is_set())

            with self.subTest(protocol=protocol):
                asyncio.run(disconnect())

    def test_openapi_documents_json_sse_errors_and_generic_api(self) -> None:
        with TestClient(create_app(fake_session())) as client:
            schema = client.get("/openapi.json").json()
            self.assertEqual(client.get("/docs").status_code, 200)
        self.assertIn("/infer", schema["paths"])
        self.assertIn("/v1/models", schema["paths"])
        operation = schema["paths"]["/v1/chat/completions"]["post"]
        self.assertIn("application/json", operation["responses"]["200"]["content"])
        self.assertIn("text/event-stream", operation["responses"]["200"]["content"])
        self.assertNotIn("422", operation["responses"])
        self.assertIn("400", operation["responses"])
        self.assertFalse(schema["components"]["schemas"]["ChatRequest"]["additionalProperties"])
