from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

from common.config import list_models, load_config
from common.logging import configure_logging
from common.types import Config, InputData
from common.utils import open_image, with_image

LOGGER = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Valeur attendue : entier >= 1.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Valeur attendue : entier >= 0.")
    return number


def port_number(value: str) -> int:
    port = positive_int(value)
    if port > 65535:
        raise argparse.ArgumentTypeError("Le port doit etre <= 65535.")
    return port


def parser_for(description: str, *, with_input: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model", nargs="?", help="Nom du dossier dans models/.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--config", type=Path, help="Fichier YAML alternatif complet.")
    parser.add_argument("--device", help="Surcharge device, interpretee par le chargeur.")
    parser.add_argument("--dtype", help="Surcharge dtype, interpretee par le chargeur.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--revision", help="SHA Hugging Face complet.")
    parser.add_argument("--offline", action="store_true", help="Utiliser uniquement le cache.")
    parser.add_argument("--debug", action="store_true")
    if with_input:
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--prompt", help="Entree texte.")
        group.add_argument("--input", type=Path, help="Fichier contenant une chaine ou un objet JSON.")
        parser.add_argument("--image", type=Path, help="Image locale, si l'adaptateur l'accepte.")
    return parser


def get_config(args: argparse.Namespace) -> Config:
    overrides = {
        key: getattr(args, key)
        for key in ("device", "dtype", "seed", "revision")
        if getattr(args, key) is not None
    }
    if args.offline:
        overrides["local_files_only"] = True
    return load_config(args.model, args.config, overrides)


def get_input(args: argparse.Namespace, config: Config) -> InputData:
    if args.prompt is not None:
        data = args.prompt
    elif args.input is not None:
        with args.input.open(encoding="utf-8") as handle:
            data = json.load(handle)
    elif "example_input" in config:
        data = config["example_input"]
    else:
        raise ValueError("Fournir --prompt, --input ou example_input dans la configuration.")
    if not isinstance(data, (str, dict)):
        raise ValueError("L'entree doit etre un texte ou un objet JSON.")
    if args.image is not None:
        data = with_image(data, open_image(args.image))
    return data


def execute(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    action: Callable[[], None],
) -> int:
    if args.list_models:
        print("\n".join(list_models()))
        return 0
    if not args.model:
        parser.error("Indiquer un modele ou --list-models.")
    configure_logging(args.debug)
    try:
        action()
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as error:
        LOGGER.error("%s", error, exc_info=args.debug)
        return 1
    return 0
