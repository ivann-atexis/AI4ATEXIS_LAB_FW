from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.benchmark import benchmark
from common.cli import execute, get_config, get_input, nonnegative_int, parser_for, positive_int
from common.config import ROOT
from common.runtime import ModelSession
from common.utils import write_json


def main(argv: list[str] | None = None) -> int:
    parser = parser_for("Mesurer une inference complete, hors chargement et warmup.")
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--output", type=Path, help="Rapport JSON (defaut : results/).")
    args = parser.parse_args(argv)

    def action() -> None:
        config = get_config(args)
        data = get_input(args, config)
        start = time.perf_counter()
        session = ModelSession(args.model, config)
        report = benchmark(session, data, args.runs, args.warmup, time.perf_counter() - start)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = args.output or ROOT / "results" / f"{args.model}-{stamp}.json"
        write_json(path, report)
        print(json.dumps({"report": str(path), "measurement": report["measurement"]}, indent=2))

    return execute(parser, args, action)


if __name__ == "__main__":
    raise SystemExit(main())
