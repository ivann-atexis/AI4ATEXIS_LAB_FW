from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr

from common.runtime import ModelSession
from common.types import AppOptions
from common.utils import parse_input, with_image

LOGGER = logging.getLogger(__name__)


def history_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        for part in content
    ):
        return "\n".join(part["text"] for part in content)
    raise ValueError("Cette application de chat accepte uniquement du texte.")


def create_chat_app(session: ModelSession, options: AppOptions) -> gr.ChatInterface:
    def respond(message: str, history: list[dict[str, Any]]) -> str:
        try:
            messages = [
                {"role": item["role"], "content": history_text(item["content"])}
                for item in history
            ]
            messages.append({"role": "user", "content": message})
            result = session.infer({"messages": messages})
        except ValueError as error:
            LOGGER.warning("Entree de chat invalide : %s", error)
            raise gr.Error(str(error)) from error
        if not isinstance(result.output, str):
            raise TypeError("Une application de chat exige une sortie texte.")
        return result.output

    examples = [options.input_data] if isinstance(options.input_data, str) else None
    return gr.ChatInterface(
        fn=respond,
        title=f"ModelForge - {session.name}",
        description="Demo locale. Le modele est charge une seule fois.",
        examples=examples,
        cache_examples=False,
        concurrency_limit=1,
        analytics_enabled=False,
    )


def create_decision_app(session: ModelSession, options: AppOptions) -> gr.Blocks:
    initial = options.input_data
    state = initial if isinstance(initial, str) else initial.get("state", "")
    questions = (
        initial.get("questions", session.config.get("questions", {}))
        if isinstance(initial, dict)
        else session.config.get("questions", {})
    )

    def decide(text: str, specification: str, image: Any) -> dict[str, Any]:
        try:
            data = {"state": text, "questions": json.loads(specification)}
            if image is not None:
                data = with_image(data, image)
            return session.infer(data).to_dict()
        except ValueError as error:
            LOGGER.warning("Entree de demonstration invalide : %s", error)
            raise gr.Error(str(error)) from error

    with gr.Blocks(title=f"ModelForge - {session.name}", analytics_enabled=False) as app:
        gr.Markdown(f"# {session.name}\nDecisions structurees en une passe, sans generation.")
        text = gr.Textbox(
            label="Etat / document",
            value=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
            lines=6,
        )
        image = gr.Image(label="Image facultative", type="pil")
        specification = gr.Code(
            label="Questions (JSON)",
            language="json",
            value=json.dumps(questions, ensure_ascii=False, indent=2),
        )
        output = gr.JSON(label="Reponses et usage")
        gr.Button("Analyser").click(
            decide, inputs=[text, specification, image], outputs=output, concurrency_limit=1
        )
    return app


def create_generic_app(session: ModelSession, options: AppOptions) -> gr.Blocks:
    value = options.input_data

    def respond(text: str) -> dict[str, Any]:
        try:
            return session.infer(parse_input(text)).to_dict()
        except ValueError as error:
            LOGGER.warning("Entree de demonstration invalide : %s", error)
            raise gr.Error(str(error)) from error

    with gr.Blocks(title=f"ModelForge - {session.name}", analytics_enabled=False) as app:
        gr.Markdown(f"# {session.name}\nEntree texte ou objet JSON, sortie JSON.")
        text = gr.Textbox(
            label="Entree",
            value=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
            lines=8,
        )
        output = gr.JSON(label="Resultat")
        gr.Button("Executer").click(respond, inputs=text, outputs=output, concurrency_limit=1)
    return app


def run_chat(session: ModelSession, options: AppOptions) -> None:
    create_chat_app(session, options).launch(
        server_name=options.host, server_port=options.port, share=False
    )


def run_decisions(session: ModelSession, options: AppOptions) -> None:
    create_decision_app(session, options).launch(
        server_name=options.host, server_port=options.port, share=False
    )


def run_generic(session: ModelSession, options: AppOptions) -> None:
    create_generic_app(session, options).launch(
        server_name=options.host, server_port=options.port, share=False
    )
