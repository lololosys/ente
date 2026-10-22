import os

import base58
import nacl.bindings as nb
import nacl.utils
from base64 import b64encode

from finder.crypto import (
    STREAM_CHUNK,
    collection_key_from_fragment,
    decrypt_blob,
    decrypt_stream,
    open_box,
)


def b64(b: bytes) -> str:
    return b64encode(b).decode()


def push(key: bytes, chunks: list[bytes]) -> tuple[bytes, str]:
    state = nb.crypto_secretstream_xchacha20poly1305_state()
    header = nb.crypto_secretstream_xchacha20poly1305_init_push(state, key)
    out = b""
    for i, c in enumerate(chunks):
        tag = (
            nb.crypto_secretstream_xchacha20poly1305_TAG_FINAL
            if i == len(chunks) - 1
            else nb.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE
        )
        out += nb.crypto_secretstream_xchacha20poly1305_push(state, c, tag=tag)
    return out, b64(header)


def test_fragment_base58_and_hex():
    key = os.urandom(32)
    assert collection_key_from_fragment(base58.b58encode(key).decode()) == key
    assert collection_key_from_fragment(key.hex()) == key


def test_open_box_round_trip():
    key, nonce = os.urandom(32), nacl.utils.random(24)
    c = nb.crypto_secretbox(b"file key bytes", nonce, key)
    assert open_box(b64(c), b64(nonce), key) == b"file key bytes"


def test_blob_round_trip():
    key = os.urandom(32)
    data, header = push(key, [b'{"title":"a.jpg"}'])
    assert decrypt_blob(data, header, key) == b'{"title":"a.jpg"}'


def test_stream_round_trip_multi_chunk():
    key = os.urandom(32)
    payload = os.urandom(9 * 1024 * 1024)
    chunks = [payload[i : i + STREAM_CHUNK] for i in range(0, len(payload), STREAM_CHUNK)]
    data, header = push(key, chunks)
    assert decrypt_stream(data, header, key) == payload
