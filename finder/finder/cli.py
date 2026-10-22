"""Owner commands, run on the box: `python -m finder <command>`."""

import argparse
import logging
import sys

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
    # Request lines carry signed download URLs; keep them out of the logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="finder")
    sub = p.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="index the shared album")
    ix.add_argument("--only", choices=["faces", "clip", "caption", "folders"])
    ix.add_argument("--limit", type=int)
    sub.add_parser("models", help="download and verify Ente's models")
    sf = sub.add_parser("set-folder", help="set a folder's photo ids, one per line on stdin")
    sf.add_argument("name")
    rf = sub.add_parser("remove-folder")
    rf.add_argument("name")
    sub.add_parser("clusters", help="list the people found, biggest first")
    ex = sub.add_parser("exclude", help="never return this person in find-me")
    ex.add_argument("cluster", type=int)
    inc = sub.add_parser("include", help="undo an exclusion")
    inc.add_argument("person", type=int)
    sub.add_parser("excluded", help="list excluded people")
    ff = sub.add_parser("forget-faces", help="delete all face data")
    ff.add_argument("--yes", action="store_true")
    tf = sub.add_parser("try-face", help="tuning: scores of one face against the album")
    tf.add_argument("file_id", type=int)
    tf.add_argument("face_index", type=int, nargs="?", default=0)
    args = p.parse_args(argv)

    config = Config.from_env()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(config.db_path)
    if args.cmd == "set-folder":
        store.set_folder(args.name, [int(line) for line in sys.stdin if line.strip()])
    if args.cmd == "remove-folder":
        store.set_folder(args.name, [])
    if args.cmd == "clusters":
        people: dict[int, list[int]] = {}
        for r in store.face_rows():
            people.setdefault(r.cluster or 0, []).append(r.file_id)
        for c, ids in sorted(people.items(), key=lambda kv: -len(kv[1])):
            sample = " ".join(map(str, sorted(set(ids))[:8]))
            print(f"person {c}: {len(ids)} faces, photos {sample}")
    if args.cmd == "exclude":
        print(f"excluded {store.exclude_cluster(args.cluster)}")
    if args.cmd == "include":
        store.include_person(args.person)
    if args.cmd == "excluded":
        for pid, _ in store.excluded_people():
            print(f"excluded {pid}")
    if args.cmd == "forget-faces":
        if not args.yes:
            p.error("forget-faces deletes all face data; pass --yes")
        store.forget_faces()
    if args.cmd == "try-face":
        rows = store.face_rows()
        mine = [r for r in rows if r.file_id == args.file_id][args.face_index]
        scores = sorted(((float(r.emb @ mine.emb), r.file_id) for r in rows if r.file_id != args.file_id), reverse=True)
        for score, fid in scores[:60]:
            print(f"{score:.3f} {fid}")
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
