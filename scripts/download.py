from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import execute, get_config, parser_for
from common.hub import download_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = parser_for("Telecharger un snapshot HF epingle, sans charger le modele.", with_input=False)
    args = parser.parse_args(argv)

    def action() -> None:
        config = get_config(args)
        snapshot = download_snapshot(config)
        print(
            json.dumps(
                {
                    "model_id": config["model_id"],
                    "revision": config["resolved_revision"],
                    "snapshot": str(snapshot),
                },
                indent=2,
            )
        )

    return execute(parser, args, action)


if __name__ == "__main__":
    raise SystemExit(main())
