"""libsodium primitives used by Ente public links, mirroring
web/packages/base/crypto/libsodium.ts."""

from base64 import b64decode

import base58
import nacl.bindings as nb

#: Ente encrypts files as a secretstream in plaintext chunks of this size.
STREAM_CHUNK = 4 * 1024 * 1024
_ABYTES = nb.crypto_secretstream_xchacha20poly1305_ABYTES
_TAG_FINAL = nb.crypto_secretstream_xchacha20poly1305_TAG_FINAL


def collection_key_from_fragment(fragment: str) -> bytes:
    """The URL fragment of a share link: base58 now, hex for legacy links."""
    fragment = fragment.lstrip("#")
    return base58.b58decode(fragment) if len(fragment) < 50 else bytes.fromhex(fragment)


def open_box(cipher_b64: str, nonce_b64: str, key: bytes) -> bytes:
    return nb.crypto_secretbox_open(b64decode(cipher_b64), b64decode(nonce_b64), key)


def _pull_state(header_b64: str, key: bytes):
    state = nb.crypto_secretstream_xchacha20poly1305_state()
    nb.crypto_secretstream_xchacha20poly1305_init_pull(state, b64decode(header_b64), key)
    return state


def decrypt_blob(data: bytes, header_b64: str, key: bytes) -> bytes:
    """A single-message secretstream (metadata, thumbnails)."""
    message, _ = nb.crypto_secretstream_xchacha20poly1305_pull(_pull_state(header_b64, key), data)
    return message


def decrypt_stream(data: bytes, header_b64: str, key: bytes) -> bytes:
    """A chunked secretstream (file contents)."""
    state = _pull_state(header_b64, key)
    out, offset, tag = [], 0, None
    size = STREAM_CHUNK + _ABYTES
    while tag != _TAG_FINAL:
        if offset >= len(data):
            raise ValueError("stream ended before the final chunk")
        message, tag = nb.crypto_secretstream_xchacha20poly1305_pull(state, data[offset : offset + size])
        out.append(message)
        offset += size
    return b"".join(out)
