import json
import os
from base64 import b64encode

import httpx
import nacl.bindings as nb
import nacl.utils

from finder.ente_api import PublicAlbum


def b64(b: bytes) -> str:
    return b64encode(b).decode()


def remote_file(fid: int, ckey: bytes, title: str, deleted=False) -> dict:
    fkey = os.urandom(32)
    nonce = nacl.utils.random(24)
    state = nb.crypto_secretstream_xchacha20poly1305_state()
    header = nb.crypto_secretstream_xchacha20poly1305_init_push(state, fkey)
    meta = nb.crypto_secretstream_xchacha20poly1305_push(
        state,
        json.dumps({"title": title, "fileType": 0}).encode(),
        tag=nb.crypto_secretstream_xchacha20poly1305_TAG_FINAL,
    )
    return {
        "id": fid,
        "encryptedKey": b64(nb.crypto_secretbox(fkey, nonce, ckey)),
        "keyDecryptionNonce": b64(nonce),
        "file": {"decryptionHeader": b64(b"f" * 24)},
        "thumbnail": {"decryptionHeader": b64(b"t" * 24)},
        "metadata": {"encryptedData": b64(meta), "decryptionHeader": b64(header)},
        "updationTime": fid * 10,
        "isDeleted": deleted,
    }


def test_files_pages_until_has_more_false():
    ckey = os.urandom(32)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        since = int(request.url.params["sinceTime"])
        if since == 0:
            body = {"diff": [remote_file(1, ckey, "a.jpg"), remote_file(2, ckey, "b.jpg", True)], "hasMore": True}
        else:
            body = {"diff": [remote_file(3, ckey, "c.jpg")], "hasMore": False}
        return httpx.Response(200, json=body, headers={"X-Link-Device-Token": "dev-1"})

    album = PublicAlbum("http://museum", "TOKEN", ckey, transport=httpx.MockTransport(handler))
    files = album.files()

    assert [f.id for f in files] == [1, 3]
    assert [f.title for f in files] == ["a.jpg", "c.jpg"]
    assert files[0].file_type == 0
    assert all(r.headers["X-Auth-Access-Token"] == "TOKEN" for r in seen)
    assert "X-Auth-Link-Device-Token" not in seen[0].headers
    assert seen[1].headers["X-Auth-Link-Device-Token"] == "dev-1"
    assert seen[1].url.params["sinceTime"] == "20"
