"""Read-only client for an Ente public album link, mirroring
web/apps/albums/src/public-album/data/api/public-collection.ts."""

import json
from base64 import b64decode
from dataclasses import dataclass

import httpx

from .crypto import decrypt_blob, decrypt_stream, open_box

IMAGE, VIDEO, LIVE_PHOTO = 0, 1, 2


@dataclass(frozen=True)
class RemoteFile:
    id: int
    key: bytes
    file_type: int
    title: str
    thumb_header: str
    file_header: str


class PublicAlbum:
    def __init__(
        self,
        museum_url: str,
        token: str,
        key: bytes,
        transport: httpx.BaseTransport | None = None,
    ):
        self.key = key
        self._token = token
        self._device_token: str | None = None
        self._http = httpx.Client(base_url=museum_url.rstrip("/"), transport=transport, timeout=60)
        # Signed object-store URLs live on another host.
        self._blobs = httpx.Client(transport=transport, timeout=120, follow_redirects=True)

    def _headers(self) -> dict[str, str]:
        h = {"X-Auth-Access-Token": self._token, "X-Client-Package": "io.ente.finder"}
        if self._device_token:
            h["X-Auth-Link-Device-Token"] = self._device_token
        return h

    def _get(self, path: str, **params) -> httpx.Response:
        res = self._http.get(path, params=params, headers=self._headers())
        res.raise_for_status()
        if token := res.headers.get("X-Link-Device-Token"):
            self._device_token = token
        return res

    def info(self) -> dict:
        return self._get("/public-collection/info").json()

    def album_name(self) -> str:
        c = self.info()["collection"]
        return open_box(c["encryptedName"], c["nameDecryptionNonce"], self.key).decode()

    def files(self) -> list[RemoteFile]:
        by_id: dict[int, RemoteFile] = {}
        since = 0
        while True:
            body = self._get("/public-collection/diff", sinceTime=since).json()
            for change in body["diff"]:
                since = max(since, change["updationTime"])
                if change.get("isDeleted"):
                    by_id.pop(change["id"], None)
                else:
                    by_id[change["id"]] = self._decrypt(change)
            if not body["hasMore"] or not body["diff"]:
                return list(by_id.values())

    def _decrypt(self, change: dict) -> RemoteFile:
        key = open_box(change["encryptedKey"], change["keyDecryptionNonce"], self.key)
        meta_enc = change["metadata"]
        meta = json.loads(
            decrypt_blob(b64decode(meta_enc["encryptedData"]), meta_enc["decryptionHeader"], key)
        )
        return RemoteFile(
            id=change["id"],
            key=key,
            file_type=int(meta.get("fileType", IMAGE)),
            title=meta.get("title", ""),
            thumb_header=change["thumbnail"]["decryptionHeader"],
            file_header=change["file"]["decryptionHeader"],
        )

    def _download(self, file_id: int, kind: str) -> bytes:
        v3 = "download" if kind == "file" else "thumbnail"
        res = self._http.get(f"/public-collection/files/{v3}/v3/{file_id}", headers=self._headers())
        if res.status_code != 404:
            res.raise_for_status()
            blob = self._blobs.get(res.json()["url"])
        else:
            legacy = "download" if kind == "file" else "preview"
            blob = self._http.get(f"/public-collection/files/{legacy}/{file_id}", headers=self._headers())
        blob.raise_for_status()
        return blob.content

    def thumbnail(self, f: RemoteFile) -> bytes:
        return decrypt_blob(self._download(f.id, "thumbnail"), f.thumb_header, f.key)

    def original(self, f: RemoteFile) -> bytes:
        return decrypt_stream(self._download(f.id, "file"), f.file_header, f.key)
