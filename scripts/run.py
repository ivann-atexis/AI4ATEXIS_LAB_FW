from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import execute, get_config, get_input, parser_for, port_number
from common.runtime import ModelSession
from common.types import AppOptions, InferenceResult


def main(argv: list[str] | None = None) -> int:
    parser = parser_for("Executer un adaptateur AI4ATEXIS_LAB_FW.")
    parser.add_argument("--app", choices=["cli", "gradio", "api"], default="cli")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=port_number, default=7860)
    parser.add_argument("--test", action="store_true", help="Executer le smoke test reel du modele.")
    args = parser.parse_args(argv)

    def action() -> None:
        config = get_config(args)
        has_input = any(
            value is not None for value in (args.prompt, args.input, args.image)
        )
        if args.app != "cli" and has_input:
            raise ValueError("--prompt, --input et --image sont reserves a l'app CLI.")
        if args.test and args.app != "cli":
            raise ValueError("--test s'utilise avec --app cli.")
        if args.test and has_input:
            raise ValueError("--test utilise example_input du modele, sans surcharge d'entree.")
        data = get_input(args, config) if args.app != "api" else ""
        session = ModelSession(args.model, config)
        if args.test:
            module = importlib.import_module(f"models.{args.model}.test")
            result = module.test(session)
            if not isinstance(result, InferenceResult):
                raise TypeError("Le smoke test doit retourner InferenceResult.")
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            session.run_app(args.app, AppOptions(data, args.host, args.port))

    return execute(parser, args, action)


if __name__ == "__main__":
    raise SystemExit(main())
