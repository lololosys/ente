"""Owner commands, run on the box: `python -m finder <command>`."""

import argparse
import logging

from .config import Config
from .ente_api import PublicAlbum
from .indexer import Stage, index_album, index_folders
from .models import Models, download_models
from .stages import clip_stage, cluster_faces, face_stage
from .store import Store


def album(config: Config, link) -> PublicAlbum:
    return PublicAlbum(config.museum_url, link.token, link.key)


def build_stages(config: Config, store: Store, only: str | None) -> dict[str, Stage]:
    models = Models(config.models_dir)
    stages: dict[str, Stage] = {"faces": face_stage(models, store), "clip": clip_stage(models, store)}
    return {k: v for k, v in stages.items() if only in (None, k)}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="finder")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="index the shared album")
    ix.add_argument("--only", choices=["faces", "clip", "caption", "folders"])
    ix.add_argument("--limit", type=int)
    sub.add_parser("models", help="download and verify Ente's models")
    args = p.parse_args(argv)

    config = Config.from_env()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(config.db_path)
    if args.cmd == "models":
        download_models(config.models_dir)
    if args.cmd == "index":
        download_models(config.models_dir)
        if args.only in (None, "folders"):
            index_folders([album(config, l) for l in config.parts], store)
        if args.only != "folders":
            stages = build_stages(config, store, args.only)
            if stages:
                index_album(album(config, config.main), store, stages, args.limit)
            if "faces" in stages:
                cluster_faces(store)
