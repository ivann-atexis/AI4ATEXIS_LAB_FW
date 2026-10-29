from __future__ import annotations

from copy import deepcopy
from functools import partial
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
        title=f"AI4ATEXIS_LAB_FW - {session.name}",
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

    criteria_names: list[str] = []
    criteria_inputs: list[gr.Dataframe] = []

    def add_criterion(rows: list[list[Any]], *, columns: int) -> list[list[Any]]:
        return [*rows, [False, *([""] * (columns - 1))]]

    def criteria_editor(
        name: str, rows: list[list[Any]], *, choice: bool
    ) -> gr.Dataframe:
        headers = ["Utiliser", "Categorie", "Description"] if choice else ["Utiliser", "Niveau"]
        table = gr.Dataframe(
            label=f"Criteres - {name}",
            value=rows,
            headers=headers,
            datatype=["bool", *(["str"] * (len(headers) - 1))],
            type="array",
            row_count=len(rows),
            column_count=len(headers),
            interactive=True,
            wrap=True,
        )
        gr.Button(f"+ Ajouter un critere - {name}").click(
            partial(add_criterion, columns=len(headers)),
            inputs=table,
            outputs=table,
            queue=False,
        )
        return table

    def selected_criteria(
        rows: list[list[Any]], name: str, *, choice: bool
    ) -> dict[str, str] | list[str]:
        labels: list[str] = []
        descriptions: dict[str, str] = {}
        if not isinstance(rows, list):
            raise ValueError(f"{name} : tableau de criteres attendu.")
        for row in rows:
            if not isinstance(row, list) or len(row) != (3 if choice else 2):
                raise ValueError(f"{name} : ligne de criteres invalide.")
            if not isinstance(row[0], bool):
                raise ValueError(f"{name} : cocher ou decocher Utiliser.")
            if not row[0]:
                continue
            label = row[1].strip() if isinstance(row[1], str) else ""
            if not label:
                raise ValueError(f"{name} : chaque critere actif exige un nom.")
            if label in labels:
                raise ValueError(f"{name} : les noms des criteres doivent etre uniques.")
            labels.append(label)
            if choice:
                description = row[2].strip() if isinstance(row[2], str) else ""
                if not description:
                    raise ValueError(f"{name} : chaque categorie active exige une description.")
                descriptions[label] = description
        if not labels:
            raise ValueError(f"{name} : selectionner au moins un critere.")
        return descriptions if choice else labels

    def image_example() -> tuple[str, str, str, list[list[Any]], str]:
        return (
            "Question personnalisee",
            "Reconnaissance d'animaux dans une image",
            "L'image contient-elle un chat, un chien, les deux ou aucun ? "
            "Base la reponse uniquement sur les animaux visibles dans l'image.",
            [
                [True, "chat", "Au moins un chat est visible, et aucun chien."],
                [True, "chien", "Au moins un chien est visible, et aucun chat."],
                [True, "aucun", "Ni chat ni chien n'est visible, meme si d'autres animaux sont presents."],
                [True, "les_deux", "Au moins un chat et au moins un chien sont visibles."],
            ],
            "",
        )

    def reset_form() -> tuple[str, str, str, list[list[Any]], str, None, None]:
        return "Question personnalisee", "", "", [[False, "", ""]], "", None, None

    def show_question_mode(mode: str) -> tuple[gr.Group, gr.Group, gr.Group]:
        return (
            gr.Group(visible=mode == "Questions configurees"),
            gr.Group(visible=mode == "Question personnalisee"),
            gr.Group(visible=mode == "JSON avance"),
        )

    def decide(
        text: str,
        specification: str,
        image: Any,
        mode: str,
        theme: str | None,
        question_text: str | None,
        custom_rows: list[list[Any]],
        *criteria_values: list[list[Any]],
    ) -> dict[str, Any]:
        try:
            if mode == "Questions configurees":
                selected = deepcopy(questions)
                for name, rows in zip(criteria_names, criteria_values, strict=True):
                    selected[name]["criteria"] = selected_criteria(
                        rows, name, choice=selected[name]["type"] == "choice"
                    )
            elif mode == "Question personnalisee":
                if question_text is None or not question_text.strip():
                    raise ValueError("Saisir une question.")
                instructions = question_text.strip()
                if theme is not None and theme.strip():
                    instructions = f"Theme / contexte : {theme.strip()}\n\n{instructions}"
                selected = {
                    "classification": {
                        "type": "choice",
                        "instructions": instructions,
                        "criteria": selected_criteria(custom_rows, "classification", choice=True),
                    }
                }
            elif mode == "JSON avance":
                selected = json.loads(specification)
            else:
                raise ValueError("Mode de questions inconnu.")
            if not isinstance(selected, dict) or not selected:
                raise ValueError("Fournir au moins une question nommee.")
            if not text.strip() and image is None:
                raise ValueError("Fournir un etat/document ou une image.")
            data = {"state": text, "questions": selected}
            if image is not None:
                data = with_image(data, image)
            return session.infer(data).to_dict()
        except ValueError as error:
            LOGGER.warning("Entree de demonstration invalide : %s", error)
            raise gr.Error(str(error)) from error

    default_mode, default_theme, default_question, default_rows, default_state = image_example()
    if initial == session.config.get("example_input"):
        state = default_state

    with gr.Blocks(title=f"AI4ATEXIS_LAB_FW - {session.name}", analytics_enabled=False) as app:
        gr.Markdown(f"# {session.name}\nDecisions structurees en une passe, sans generation.")
        mode = gr.Radio(
            label="Mode de questions",
            choices=["Questions configurees", "Question personnalisee", "JSON avance"],
            value=default_mode,
        )
        with gr.Row():
            example = gr.Button("Exemple image : chat / chien / aucun / les deux")
            reset = gr.Button("reset")
        gr.Markdown(
            "Cochez **Utiliser** pour retenir un critere. Double-cliquez sur une cellule "
            "pour modifier son nom ou sa description. Utilisez **+ Ajouter un critere**, "
            "remplissez la ligne puis cochez-la pour l'activer."
        )
        with gr.Group(visible=False) as configured_group:
            gr.Markdown(
                "## Questions configurees\n"
                "Pour les scores, conservez l'ordre des niveaux du plus faible au plus eleve."
            )
            for name, question in questions.items():
                criteria = question.get("criteria")
                if question.get("type") == "choice" and isinstance(criteria, dict):
                    rows = [[True, label, description] for label, description in criteria.items()]
                    choice = True
                elif question.get("type") == "score" and isinstance(criteria, list):
                    rows = [[True, label] for label in criteria]
                    choice = False
                else:
                    continue
                gr.Markdown(f"### {name}\n{question.get('instructions', '')}")
                criteria_names.append(name)
                criteria_inputs.append(criteria_editor(name, rows, choice=choice))
        with gr.Group() as custom_group:
            gr.Markdown(
                "## Votre question et vos categories\n"
                "Ce mode remplace toutes les questions configurees par une seule "
                "classification. Une seule categorie est choisie par analyse."
            )
            theme = gr.Textbox(
                label="Theme / contexte (facultatif)",
                value=default_theme,
                placeholder="Exemple : animaux, controle qualite, tri de documents...",
                info="Ajoute aux instructions ; ne remplace pas l'image ou le document a analyser.",
            )
            question_text = gr.Textbox(
                label="Question",
                value=default_question,
                placeholder="Quelle question poser au modele ?",
                lines=3,
            )
            custom_criteria = criteria_editor("personnalise", default_rows, choice=True)
        with gr.Group(visible=False) as json_group:
            gr.Markdown(
                "## Questions JSON (avance)\n"
                "Ce mode utilise uniquement le JSON ci-dessous. Les trois modes sont "
                "independants et ne se synchronisent pas."
            )
            specification = gr.Code(
                label="Questions (JSON)",
                language="json",
                value=json.dumps(questions, ensure_ascii=False, indent=2),
            )
        text = gr.Textbox(
            label="Etat / document",
            value=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
            info="Facultatif si une image est fournie.",
            lines=6,
        )
        image = gr.Image(label="Image facultative", type="pil")
        mode.change(
            show_question_mode,
            inputs=mode,
            outputs=[configured_group, custom_group, json_group],
            queue=False,
        )
        example.click(
            image_example,
            inputs=[],
            outputs=[mode, theme, question_text, custom_criteria, text],
            queue=False,
        )
        output = gr.JSON(label="Reponses et usage")
        analyze = gr.Button("Analyser").click(
            decide,
            inputs=[
                text, specification, image, mode, theme, question_text, custom_criteria,
                *criteria_inputs,
            ],
            outputs=output,
            concurrency_limit=1,
        )
        reset.click(
            reset_form,
            inputs=[],
            outputs=[mode, theme, question_text, custom_criteria, text, image, output],
            cancels=[analyze],
            queue=False,
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

    with gr.Blocks(title=f"AI4ATEXIS_LAB_FW - {session.name}", analytics_enabled=False) as app:
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
