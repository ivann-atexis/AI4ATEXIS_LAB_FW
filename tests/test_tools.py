import copy
import importlib
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from common.apps.api import create_app
from common.apps.openai import ChatRequest
from common.config import load_config
from common.tool_calling import tool_messages, tool_schemas
from common.transformers import causal_inputs
from common.types import InferenceResult, StreamChunk
from models.lfm3.tools import finish_tools, stream_tools
from tests.helpers import OfflineTestCase, fake_session
from tests.test_openai import events, request_body

TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
XML = "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
DATA = {"messages": [{"role": "user", "content": "Weather?"}], "tools": [TOOL]}


def parsed(text=XML, **options):
    return finish_tools(InferenceResult(text, 4, 8, "stop"), {**DATA, **options})


class NativeToolTests(OfflineTestCase):
    def test_parse_real_lfm_format_without_executing_anything(self) -> None:
        result = parsed("Checking.\n" + XML)
        self.assertEqual(result.output, "Checking.\n")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0].name, "weather")
        self.assertEqual(result.tool_calls[0].arguments, {"city": "Paris"})
        self.assertEqual(json.loads(result.tool_calls[0].to_dict()["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(result.output_tokens, 8)

    def test_schema_driven_types_and_nested_json(self) -> None:
        tool = copy.deepcopy(TOOL)
        tool["function"]["parameters"]["properties"].update({
            "enabled": {"type": "boolean"}, "settings": {"type": "object"},
            "tags": {"type": "array"}, "optional": {"type": ["null", "string"]},
        })
        body = (
            '<parameter=city>\n123\n</parameter><parameter=days>2</parameter>'
            '<parameter=enabled>true</parameter><parameter=settings>{"unit":"C"}</parameter>'
            '<parameter=tags>["a","b"]</parameter><parameter=optional>null</parameter>'
        )
        text = f"<tool_call><function=weather>{body}</function></tool_call>"
        result = parsed(text, tools=[tool])
        self.assertEqual(result.tool_calls[0].arguments, {
            "city": "123", "days": 2, "enabled": True,
            "settings": {"unit": "C"}, "tags": ["a", "b"], "optional": None,
        })

    def test_local_schema_references_preserve_string_type(self) -> None:
        tool = copy.deepcopy(TOOL)
        schema = tool["function"]["parameters"]
        schema["$defs"] = {"name": {"type": "string"}}
        schema["properties"]["city"] = {"$ref": "#/$defs/name"}
        self.assertEqual(parsed(XML.replace("Paris", "123"), tools=[tool]).tool_calls[0].arguments, {"city": "123"})
        self.assertEqual(
            parsed(XML.replace("Paris", '"quoted"')).tool_calls[0].arguments,
            {"city": '"quoted"'},
        )

    def test_invalid_or_truncated_model_calls_are_never_returned(self) -> None:
        cases = [
            XML[:-4], XML.replace("weather", "unknown"), XML.replace("city", "days"),
            XML.replace("Paris", "Paris</parameter><parameter=city>Again"),
            "<tool_call><function=weather></function></tool_call>",
            XML + "<tool_call>", XML + "<tool_call",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                parsed(text)

    def test_reasoning_and_code_examples_are_not_executable_calls(self) -> None:
        for text in (f"```xml\n{XML}\n```", f"<think>Example: {XML}</think>Answer"):
            with self.subTest(text=text):
                result = parsed(text)
                self.assertEqual(result.output, text)
                self.assertFalse(result.tool_calls)
        text = f"<think>Example: {XML}</think>\n{XML}"
        self.assertEqual(len(parsed(text).tool_calls), 1)
        with self.assertRaises(RuntimeError):
            parsed(f"<think>{XML}</think>\n<tool_call>broken")
        self.assertEqual(len(parsed(XML.replace("Paris", "```") + XML).tool_calls), 2)

    def test_parallel_and_forced_choice(self) -> None:
        result = parsed(XML + "\n" + XML)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertNotEqual(result.tool_calls[0].id, result.tool_calls[1].id)
        with self.assertRaises(RuntimeError):
            parsed(XML + XML, parallel_tool_calls=False)
        with self.assertRaises(RuntimeError):
            parsed("Plain answer", tool_choice="required")
        with self.assertRaises(RuntimeError):
            parsed(tool_choice={"type": "function", "function": {"name": "different"}})
        self.assertFalse(parsed(XML, tool_choice="none").tool_calls)

    def test_stream_hides_split_markup_and_preserves_usage(self) -> None:
        def backend():
            for part in ("Checking. ", "<tool", "_call>", XML[len("<tool_call>"):]):
                yield StreamChunk(text=part)
            yield StreamChunk(result=InferenceResult("Checking. " + XML, 4, 8, "stop"))
        chunks = list(stream_tools(backend(), DATA))
        self.assertEqual("".join(chunk.text for chunk in chunks), "Checking. ")
        self.assertEqual(chunks[-1].result.finish_reason, "tool_calls")
        self.assertEqual(chunks[-1].result.output_tokens, 8)

    def test_api_round_trip_preserves_calls_ids_and_decoded_arguments(self) -> None:
        session = fake_session()
        session.infer.return_value = parsed()
        with TestClient(create_app(session)) as client:
            first = client.post("/v1/chat/completions", json=request_body(tools=[TOOL]))
            self.assertEqual(first.status_code, 200)
            assistant = first.json()["choices"][0]["message"]
            call = assistant["tool_calls"][0]
            self.assertEqual(first.json()["choices"][0]["finish_reason"], "tool_calls")
            session.infer.return_value = InferenceResult("Sunny.", 12, 3, "stop")
            history = [
                request_body()["messages"][0], assistant,
                {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature":20}'},
            ]
            second = client.post(
                "/v1/chat/completions", json=request_body(tools=[TOOL], messages=history)
            )
        self.assertEqual(second.status_code, 200)
        forwarded = session.infer.call_args.args[0]
        self.assertEqual(forwarded["messages"][1]["tool_calls"][0]["function"]["arguments"], {"city": "Paris"})
        self.assertEqual(forwarded["messages"][2]["tool_call_id"], call["id"])
        self.assertEqual(second.json()["choices"][0]["message"]["content"], "Sunny.")

    def test_streaming_api_emits_indexed_tool_calls_before_finish(self) -> None:
        session = fake_session()
        result = parsed()

        def backend(data, cancel):
            yield StreamChunk(result=result)
        session.stream.side_effect = backend
        with TestClient(create_app(session)) as client:
            response = client.post(
                "/v1/chat/completions",
                json=request_body(tools=[TOOL], stream=True, stream_options={"include_usage": True}),
            )
        frames = events(response)
        self.assertEqual(frames[-1], "[DONE]")
        call = frames[1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(call["index"], 0)
        self.assertEqual(call["id"], result.tool_calls[0].id)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(frames[2]["choices"][0]["finish_reason"], "tool_calls")

    def test_tools_reach_chat_template_as_mappings(self) -> None:
        result = parsed()
        calls = [call.to_dict() for call in result.tool_calls]
        history = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": calls, "reasoning_content": "Check weather"},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": "20C"},
        ]
        tokenizer = Mock()
        causal_inputs(
            SimpleNamespace(device="cpu"), tokenizer,
            {**DATA, "messages": history}, load_config("lfm3"),
        )
        forwarded = tokenizer.apply_chat_template.call_args
        self.assertEqual(forwarded.kwargs["tools"], [TOOL])
        self.assertEqual(forwarded.args[0][1]["tool_calls"][0]["function"]["arguments"], {"city": "Paris"})
        self.assertEqual(history[1]["tool_calls"][0]["function"]["arguments"], '{"city": "Paris"}')

    def test_choice_none_and_named_choice_reach_the_template_correctly(self) -> None:
        tokenizer = Mock()
        model = SimpleNamespace(device="cpu")
        config = load_config("lfm3")
        causal_inputs(model, tokenizer, {**DATA, "tool_choice": "none"}, config)
        self.assertNotIn("tools", tokenizer.apply_chat_template.call_args.kwargs)
        second = copy.deepcopy(TOOL)
        second["function"]["name"] = "clock"
        request = {
            **DATA, "tools": [TOOL, second],
            "messages": [{"role": "system", "content": "Original"}, *DATA["messages"]],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
        }
        before = copy.deepcopy(request)
        causal_inputs(model, tokenizer, request, config)
        call = tokenizer.apply_chat_template.call_args
        self.assertEqual(call.kwargs["tools"], [TOOL, second])
        self.assertTrue(call.args[0][0]["content"].startswith("Original"))
        self.assertIn("weather", call.args[0][0]["content"])
        self.assertEqual(request, before)

    def test_invalid_reference_is_reported_as_a_backend_error(self) -> None:
        tool = copy.deepcopy(TOOL)
        tool["function"]["parameters"]["properties"]["city"] = {"$ref": "#/$defs/missing"}
        with self.assertRaisesRegex(RuntimeError, "schema"):
            parsed(tools=[tool])

    def test_broken_tool_stream_never_emits_an_executable_call(self) -> None:
        session = fake_session()
        def backend(data, cancel):
            def raw():
                yield StreamChunk(text=XML[:-4])
                yield StreamChunk(result=InferenceResult(XML[:-4], 4, 8, "length"))
            yield from stream_tools(raw(), data)
        session.stream.side_effect = backend
        with TestClient(create_app(session)) as client:
            response = client.post(
                "/v1/chat/completions", json=request_body(tools=[TOOL], stream=True)
            )
        frames = events(response)
        self.assertEqual(frames[-1]["error"]["type"], "server_error")
        self.assertNotIn("[DONE]", frames)
        self.assertFalse(any("tool_calls" in frame.get("choices", [{}])[0].get("delta", {}) for frame in frames))

    def test_invalid_histories_and_external_schemas_are_rejected_before_backend(self) -> None:
        session = fake_session()
        external = copy.deepcopy(TOOL)
        external["function"]["parameters"]["$ref"] = "https://example.com/schema"
        cases = [
            request_body(tools=[TOOL, TOOL]),
            request_body(tools=[external]),
            request_body(messages=[{"role": "tool", "content": "x"}]),
            request_body(messages=[{"role": "tool", "tool_call_id": "missing", "content": "x"}]),
            request_body(tool_choice="required"),
            request_body(tools=[TOOL], tool_choice={"type": "function", "function": {"name": "missing"}}),
        ]
        with TestClient(create_app(session)) as client:
            for body in cases:
                with self.subTest(body=body):
                    self.assertEqual(client.post("/v1/chat/completions", json=body).status_code, 400)
        session.infer.assert_not_called()

    def test_tool_references_and_current_reasoning_are_preserved(self) -> None:
        call = parsed().tool_calls[0].to_dict()
        history = [
            {"role": "assistant", "tool_calls": [call], "reasoning_content": "Need weather"},
            {"role": "tool", "tool_call_id": call["id"], "content": [
                {"type": "text", "text": "Found it"}, {"type": "tool_reference", "name": "weather"},
            ]},
        ]
        normalized = tool_messages(history)
        self.assertEqual(normalized[0]["reasoning_content"], "Need weather")
        self.assertEqual(normalized[1]["content"], history[1]["content"])

    def test_deferred_tools_preserve_the_level_of_the_flag(self) -> None:
        for nested in (False, True):
            tool = copy.deepcopy(TOOL)
            (tool["function"] if nested else tool)["defer_loading"] = True
            data = ChatRequest(**request_body(tools=[tool])).input_data()
            normalized = data["tools"][0]
            with self.subTest(nested=nested):
                if nested:
                    self.assertNotIn("defer_loading", normalized)
                    self.assertTrue(normalized["function"]["defer_loading"])
                else:
                    self.assertTrue(normalized["defer_loading"])
                    self.assertNotIn("defer_loading", normalized["function"])

    def test_broken_json_arguments_and_missing_results_are_rejected(self) -> None:
        session = fake_session()
        with TestClient(create_app(session)) as client:
            for arguments in ('{"city":"Paris","city":"Lyon"}', '[]', '{"value":NaN}'):
                call = parsed().tool_calls[0].to_dict()
                call["function"]["arguments"] = arguments
                history = [
                    {"role": "assistant", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": call["id"], "content": "result"},
                ]
                with self.subTest(arguments=arguments):
                    response = client.post(
                        "/v1/chat/completions", json=request_body(tools=[TOOL], messages=history)
                    )
                    self.assertEqual(response.status_code, 400)
            call = parsed().tool_calls[0].to_dict()
            self.assertEqual(client.post(
                "/v1/chat/completions",
                json=request_body(tools=[TOOL], messages=[{"role": "assistant", "tool_calls": [call]}]),
            ).status_code, 400)
        session.infer.assert_not_called()

    def test_model_adapter_uses_the_parser_for_json_and_stream(self) -> None:
        adapter = importlib.import_module("models.lfm3.inference")
        raw = InferenceResult(XML, 4, 8, "stop")
        with patch.object(adapter, "infer_causal", return_value=raw):
            self.assertEqual(adapter.infer(None, None, DATA, load_config("lfm3")).finish_reason, "tool_calls")

        def backend(*args):
            yield StreamChunk(text=XML)
            yield StreamChunk(result=raw)
        with patch.object(adapter, "stream_causal", side_effect=backend):
            chunks = list(adapter.stream(None, None, DATA, load_config("lfm3"), Event()))
        self.assertEqual(chunks[-1].result.finish_reason, "tool_calls")
        self.assertFalse("".join(chunk.text for chunk in chunks))
