import copy
import sys
import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from common.config import load_config
from common.runtime import ModelSession
from common.transformers import StopTextBuffer, causal_generation, infer_causal, stream_causal
from common.types import InferenceResult, StreamChunk
from tests.helpers import OfflineTestCase


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    pieces = {1: "prompt", 2: ":", 3: "hello ", 4: "world", 9: ""}

    def apply_chat_template(self, messages, **options):
        return Batch(input_ids=torch.tensor([[1, 2]]))

    def decode(self, tokens, **options):
        return "".join(self.pieces[int(token)] for token in tokens)


def fake_model(generate):
    return SimpleNamespace(
        device=torch.device("cpu"),
        generation_config=SimpleNamespace(eos_token_id=9),
        generate=generate,
    )


class StreamingTests(OfflineTestCase):
    def test_generation_overrides_are_validated_without_mutation(self) -> None:
        config = load_config("lfm3")
        before = copy.deepcopy(config)
        options = causal_generation(
            {"messages": [], "generation": {"temperature": 0, "max_new_tokens": 2, "top_p": 0.8}},
            config,
        )
        self.assertFalse(options["do_sample"])
        self.assertNotIn("temperature", options)
        self.assertNotIn("top_k", options)
        self.assertEqual(options["max_new_tokens"], 2)
        self.assertEqual(config, before)
        for values in (
            {"arbitrary_backend_flag": True},
            {"max_new_tokens": True},
            {"top_p": 0},
            {"temperature": float("inf")},
            {"temperature": None},
            {"max_new_tokens": None},
            {"stop_strings": [""]},
            {"do_sample": "false"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                causal_generation({"messages": [], "generation": values}, config)

    def test_stop_strings_are_hidden_even_across_chunks(self) -> None:
        buffer = StopTextBuffer(["<END>", "STOP"])
        text = "".join(buffer.push(part) for part in ("hello <", "EN", "D>not", " emitted"))
        text += buffer.finish()
        self.assertEqual(text, "hello ")
        self.assertTrue(buffer.stopped)
        buffer = StopTextBuffer(["<END>"])
        self.assertEqual(buffer.push("hello <") + buffer.finish(), "hello <")

    def test_finish_reason_uses_actual_limit_and_eos(self) -> None:
        config = load_config("lfm3")
        config["generation"] = {"max_new_tokens": 2, "do_sample": False}
        for tokens, reason in (([3, 4], "length"), ([3, 9], "stop")):
            model = fake_model(Mock(return_value=torch.tensor([[1, 2, *tokens]])))
            with self.subTest(tokens=tokens):
                result = infer_causal(model, Tokenizer(), "prompt", config)
                self.assertEqual(result.finish_reason, reason)
                self.assertEqual(result.input_tokens, 2)
                self.assertEqual(result.output_tokens, 2)

    def test_inference_excludes_stop_text_and_passes_tokenizer(self) -> None:
        config = load_config("lfm3")
        tokenizer = Tokenizer()
        generate = Mock(return_value=torch.tensor([[1, 2, 3, 4]]))
        result = infer_causal(
            fake_model(generate), tokenizer,
            {"prompt": "prompt", "generation": {"stop_strings": ["world"]}}, config,
        )
        self.assertEqual(result.output, "hello ")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIs(generate.call_args.kwargs["tokenizer"], tokenizer)

    def test_text_is_emitted_before_generation_finishes(self) -> None:
        received = Event()

        def generate(input_ids, streamer, **options):
            self.assertTrue(torch.is_inference_mode_enabled())
            streamer.put(input_ids)
            streamer.put(torch.tensor([3]))
            if not received.wait(3):
                raise RuntimeError("Streaming was buffered until completion.")
            streamer.put(torch.tensor([4]))
            streamer.put(torch.tensor([9]))
            streamer.end()
            return torch.tensor([[1, 2, 3, 4, 9]])

        chunks = stream_causal(fake_model(generate), Tokenizer(), "prompt", load_config("lfm3"), Event())
        try:
            first = next(chunk for chunk in chunks if chunk.text)
            self.assertEqual(first.text, "hello ")
            received.set()
            remaining = list(chunks)
        finally:
            received.set()
            chunks.close()
        self.assertEqual(first.text + "".join(chunk.text for chunk in remaining), "hello world")
        self.assertEqual(remaining[-1].result.output, "hello world")
        self.assertEqual(remaining[-1].result.output_tokens, 3)

    def test_worker_error_is_propagated_without_hanging(self) -> None:
        def generate(**options):
            raise RuntimeError("worker failed")

        chunks = stream_causal(fake_model(generate), Tokenizer(), "prompt", load_config("lfm3"), Event())
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            list(chunks)

    def test_cancellation_stops_worker_and_waits_for_cleanup(self) -> None:
        exited = Event()
        cooperative_stop = Event()
        cancel = Event()

        def generate(input_ids, streamer, stopping_criteria, **options):
            try:
                streamer.put(input_ids)
                streamer.put(torch.tensor([3]))
                deadline = time.monotonic() + 3
                while not bool(stopping_criteria(torch.tensor([[1, 2, 3]]), None).all()):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Le worker n'a pas recu l'annulation.")
                    time.sleep(0.005)
                cooperative_stop.set()
                streamer.end()
                return torch.tensor([[1, 2, 3]])
            finally:
                exited.set()

        chunks = stream_causal(fake_model(generate), Tokenizer(), "prompt", load_config("lfm3"), cancel)
        next(chunk for chunk in chunks if chunk.text)
        chunks.close()
        self.assertTrue(cancel.is_set())
        self.assertTrue(exited.is_set())
        self.assertTrue(cooperative_stop.is_set())

    def make_session(self, hook):
        loader = SimpleNamespace(load_model=Mock(return_value=(object(), None)))
        inference = SimpleNamespace(infer=Mock(return_value=InferenceResult("answer")), stream=hook)
        with patch.dict(
            sys.modules,
            {"models.lfm3.load": loader, "models.lfm3.inference": inference},
        ):
            return ModelSession("lfm3", load_config("lfm3"))

    def test_session_holds_the_same_lock_and_cleans_up_on_close(self) -> None:
        closed = Event()
        cancellation = []

        def implementation(model, processor, data, config, cancel):
            cancellation.append(cancel)
            try:
                yield StreamChunk(text="first")
                yield StreamChunk(result=InferenceResult("first", 1, 1, "stop"))
            finally:
                closed.set()

        session = self.make_session(implementation)
        chunks = session.stream("prompt")
        self.assertEqual(next(chunks).text, "first")
        self.assertTrue(session._lock.locked())
        chunks.close()
        self.assertTrue(closed.is_set())
        self.assertTrue(cancellation[0].is_set())
        self.assertFalse(session._lock.locked())
        self.assertEqual(session.infer("next").output, "answer")

    def test_session_requires_exactly_one_final_result(self) -> None:
        def incomplete(model, processor, data, config, cancel):
            yield StreamChunk(text="first")

        def extra(model, processor, data, config, cancel):
            yield StreamChunk(result=InferenceResult("first", 1, 1, "stop"))
            yield StreamChunk(text="too late")

        for hook, error in ((incomplete, RuntimeError), (extra, TypeError)):
            session = self.make_session(hook)
            with self.subTest(hook=hook.__name__), self.assertRaises(error):
                list(session.stream("prompt"))
            self.assertFalse(session._lock.locked())

    def test_cancelled_request_does_not_start_generation(self) -> None:
        hook = Mock()
        session = self.make_session(hook)
        cancellation = Event()
        cancellation.set()
        self.assertEqual(list(session.stream("prompt", cancellation)), [])
        hook.assert_not_called()

    def test_chat_capability_must_be_an_explicit_boolean(self) -> None:
        self.assertTrue(load_config("lfm3")["api"]["chat"])
        self.assertFalse(load_config("d1-2b")["api"].get("chat", False))
        for value in (True, {"chat": "true"}, {"chat": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config("lfm3", overrides={"api": value})

    def test_adapters_without_a_stream_hook_keep_working(self) -> None:
        session = self.make_session(None)
        self.assertFalse(session.supports_streaming)
        with self.assertRaises(ValueError):
            list(session.stream("prompt"))
        self.assertEqual(session.infer("prompt").output, "answer")

    def test_stream_chunk_rejects_malformed_events(self) -> None:
        with self.assertRaises(TypeError):
            StreamChunk(text=1)
        with self.assertRaises(ValueError):
            StreamChunk(text="unexpected", result=InferenceResult("final"))
