"""The indexing loop: each stage runs once per file and is resumable."""

import logging
from collections.abc import Callable

from PIL import Image

from .ente_api import PublicAlbum, RemoteFile
from .images import load_image
from .store import Store

log = logging.getLogger("finder.index")

Stage = Callable[[RemoteFile, Image.Image], None]

FOLDER_SEPARATORS = (" · ", " / ")


def folder_name(album_name: str) -> str:
    for sep in FOLDER_SEPARATORS:
        if sep in album_name:
            return album_name.split(sep, 1)[1].strip()
    return album_name.strip()


def index_folders(parts: list[PublicAlbum], store: Store) -> None:
    for album in parts:
        store.set_folder(folder_name(album.album_name()), [f.id for f in album.files()])


def index_album(
    album: PublicAlbum,
    store: Store,
    stages: dict[str, Stage],
    limit: int | None = None,
) -> None:
    files = {f.id: f for f in album.files()}
    store.ensure_files(list(files))
    todo = sorted({fid for name in stages for fid in store.pending(name) if fid in files})
    if limit is not None:
        todo = todo[:limit]
    for n, fid in enumerate(todo, 1):
        f = files[fid]
        img = load_image(f, album)
        if img is None:
            log.warning("file %d: could not decode", fid)
            continue
        for name, stage in stages.items():
            if fid not in store.pending(name):
                continue
            try:
                stage(f, img)
            except Exception as e:  # one bad file must not stop the run
                log.warning("file %d: %s failed: %s", fid, name, type(e).__name__)
                continue
            store.mark_done(fid, name)
        if n % 25 == 0:
            log.info("%d/%d files", n, len(todo))
